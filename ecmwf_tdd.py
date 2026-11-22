"""ECMWF IFS -> native temperature extrema -> SQLite -> approximate CONUS DD.

Python 3.11+. All network downloads go through the official ecmwf-opendata client.
Days are 00:00-24:00 UTC. Incomplete days are excluded, never filled with zero.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import sqlite3
import sys
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

LOG = logging.getLogger("ecmwf_tdd")
UTC = timezone.utc
DEFAULT_CONFIG = Path(__file__).with_name("regions.json")


@dataclass(frozen=True)
class Config:
    id: str
    document: dict

    @property
    def points(self):
        return [p for region in self.document["regions"] for p in region["points"]]


def utc_string(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds")


def now_utc() -> str:
    return utc_string(datetime.now(UTC))


def load_config(path: Path) -> Config:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc.get("version"), str) or not doc["version"].strip():
        raise ValueError("Config needs a non-empty version")
    base = doc.get("base_f")
    if isinstance(base, bool) or not isinstance(base, (int, float)) or not math.isfinite(base):
        raise ValueError("base_f must be finite")
    regions = doc.get("regions", [])
    if not regions or any(not r.get("points") for r in regions):
        raise ValueError("Each region must contain points")
    ids, point_ids = set(), set()
    for group in [regions] + [r["points"] for r in regions]:
        weights = [x.get("weight") for x in group]
        if any(isinstance(w, bool) or not isinstance(w, (int, float))
               or not math.isfinite(w) or w <= 0 for w in weights):
            raise ValueError("All weights must be finite and positive")
        if not math.isclose(math.fsum(weights), 1.0, abs_tol=1e-9):
            raise ValueError("Region weights and each set of point weights must sum to 1")
    for region in regions:
        if not region.get("id") or region["id"] == "CONUS" or region["id"] in ids:
            raise ValueError("Region IDs must be unique; CONUS is reserved")
        ids.add(region["id"])
        for point in region["points"]:
            if not point.get("id") or point["id"] in point_ids:
                raise ValueError("Point IDs must be globally unique")
            point_ids.add(point["id"])
            for key, lower, upper in [("lat", 24, 50), ("lon", -126, -66)]:
                value = point.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) \
                        or not math.isfinite(value) or not lower <= value <= upper:
                    raise ValueError(f"Invalid CONUS coordinate: {point['id']}.{key}")
    canonical = json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return Config(hashlib.sha256(canonical.encode()).hexdigest(), doc)


def forecast_steps(days: int, cycle: int | None = None) -> list[int]:
    if not 1 <= days <= 15:
        raise ValueError("days must be between 1 and 15")
    if cycle in (6, 18) and days > 6:
        raise ValueError("06/18 UTC cycles support at most 6 days (144 hours)")
    return [s for s in list(range(3, 145, 3)) + list(range(150, 361, 6)) if s <= days * 24]


def make_client(source: str):
    from ecmwf.opendata import Client

    return Client(source=source, model="ifs", resol="0p25", maximum_retries=3, retry_after=5)


def resolve_run(client, days: int, date: str | None, cycle: int | None) -> datetime:
    steps = forecast_steps(days, cycle)
    if date:
        if cycle is None:
            raise ValueError("An explicit --date requires --cycle")
        return datetime.strptime(date, "%Y-%m-%d").replace(hour=cycle, tzinfo=UTC)
    request = dict(stream="oper", type="fc", param="2t", step=steps[-1])
    if cycle is not None:
        request["time"] = cycle
    # Check the final requested lead, not just the presence of the run directory.
    run = client.latest(**request)
    return run.replace(tzinfo=UTC) if run.tzinfo is None else run.astimezone(UTC)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(client, run: datetime, days: int, source: str, cache: Path):
    """Two high-level client calls at most: 3-hour and 6-hour interval extrema."""
    steps = forecast_steps(days, run.hour)
    folder = cache / run.strftime("%Y%m%d%H")
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for interval, group in [(3, [s for s in steps if s <= 144]),
                            (6, [s for s in steps if s > 144])]:
        if not group:
            continue
        request = dict(date=run.strftime("%Y%m%d"), time=run.hour, stream="oper", type="fc",
                       levtype="sfc", step=group, param=[f"mn2t{interval}", f"mx2t{interval}"])
        target = folder / f"extrema_{interval}h_to_{group[-1]:03d}h.grib2"
        manifest_path = target.with_suffix(".json")
        cached = None
        if target.exists() and manifest_path.exists():
            try:
                candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
                if candidate["request"] == request and candidate["sha256"] == file_hash(target):
                    cached = candidate
            except (ValueError, KeyError, OSError):
                LOG.warning("Ignoring an invalid cache manifest: %s", manifest_path)
        if cached is None:
            partial = target.with_suffix(".part")
            LOG.info("Downloading %s UTC, %sh extrema through +%sh", run, interval, group[-1])
            try:
                client.retrieve(**request, target=str(partial))
                if not partial.exists() or partial.stat().st_size == 0:
                    raise ValueError("ECMWF returned an empty file")
                partial.replace(target)
            finally:
                partial.unlink(missing_ok=True)
            cached = dict(request=request, source=source, sha256=file_hash(target),
                          bytes=target.stat().st_size, downloaded_at_utc=now_utc())
            manifest_path.write_text(json.dumps(cached, indent=2), encoding="utf-8")
        else:
            LOG.info("Reusing verified cache: %s", target)
        files.append((target, cached))
    return files


def read_samples(files, run: datetime, days: int, config: Config):
    import eccodes as ec

    expected = {(step, f"{kind}2t{3 if step <= 144 else 6}")
                for step in forecast_steps(days, run.hour) for kind in ("mn", "mx")}
    seen, samples, grid_fingerprint, nearest = set(), [], None, None
    grid_keys = ("gridType", "Ni", "Nj", "latitudeOfFirstGridPointInDegrees",
                 "longitudeOfFirstGridPointInDegrees", "latitudeOfLastGridPointInDegrees",
                 "longitudeOfLastGridPointInDegrees", "iScansNegatively", "jScansPositively",
                 "jPointsAreConsecutive")
    for path, manifest in files:
        with path.open("rb") as handle:
            while (message := ec.codes_grib_new_from_file(handle)) is not None:
                try:
                    step, param = int(ec.codes_get(message, "endStep")), ec.codes_get(message, "shortName")
                    key = (step, param)
                    if key not in expected or key in seen:
                        raise ValueError(f"Unexpected or duplicate GRIB field: {key}")
                    interval = 3 if step <= 144 else 6
                    if int(ec.codes_get(message, "startStep")) != step - interval:
                        raise ValueError(f"Unexpected temperature interval at +{step}h")
                    if ec.codes_get(message, "units") != "K":
                        raise ValueError("Expected temperatures in kelvin")
                    if int(ec.codes_get(message, "dataDate")) != int(run.strftime("%Y%m%d")) \
                            or int(ec.codes_get(message, "dataTime")) != run.hour * 100:
                        raise ValueError("GRIB initialization differs from the selected run")
                    if ec.codes_get(message, "dataType") != "fc" \
                            or ec.codes_get(message, "stream") != "oper":
                        raise ValueError("Expected an IFS oper/fc forecast")
                    fingerprint = tuple(ec.codes_get(message, k) for k in grid_keys)
                    if grid_fingerprint is None:
                        grid_fingerprint = fingerprint
                        if fingerprint[:3] != ("regular_ll", 1440, 721):
                            raise ValueError("Expected the global regular 0.25-degree grid")
                        nearest = []
                        for point in config.points:
                            match = ec.codes_grib_find_nearest(message, point["lat"], point["lon"])[0]
                            if match["distance"] > 50:
                                raise ValueError(f"Grid point too far from {point['id']}")
                            nearest.append(match)
                    elif fingerprint != grid_fingerprint:
                        raise ValueError("Grid geometry changed within the run")
                    values = ec.codes_get_values(message)
                    packing_error = float(ec.codes_get(message, "packingError"))
                    if not math.isfinite(packing_error) or not 0 <= packing_error < 10:
                        raise ValueError("Invalid GRIB packing error")
                    for point, match in zip(config.points, nearest, strict=True):
                        kelvin = float(values[match["index"]])
                        if not math.isfinite(kelvin) or not 150 < kelvin < 350:
                            raise ValueError(f"Missing or invalid temperature: {point['id']}")
                        samples.append(dict(point_id=point["id"], param=param, step=step,
                                            start_hour=step - interval, end_hour=step, kelvin=kelvin,
                                            grid_lat=match["lat"], grid_lon=match["lon"],
                                            packing_error_k=packing_error,
                                            downloaded_at_utc=manifest["downloaded_at_utc"]))
                    seen.add(key)
                finally:
                    ec.codes_release(message)
    if seen != expected:
        raise ValueError(f"Incomplete run: {len(expected - seen)} temperature fields missing")
    return samples


def degree_days(tmin_k: float, tmax_k: float, base_f: float):
    if not all(math.isfinite(v) for v in (tmin_k, tmax_k, base_f)) or tmin_k > tmax_k:
        raise ValueError("Invalid daily temperature extrema")
    mean_f = ((tmin_k + tmax_k) / 2 - 273.15) * 1.8 + 32
    hdd, cdd = max(base_f - mean_f, 0), max(mean_f - base_f, 0)
    return mean_f, hdd, cdd


def calculate(samples: list[dict], run: datetime, days: int, config: Config):
    lookup = {(s["point_id"], s["step"], s["param"]): s for s in samples}
    if len(lookup) != len(samples):
        raise ValueError("Duplicate temperature samples")
    expected = {(p["id"], step, f"{kind}2t{3 if step <= 144 else 6}")
                for p in config.points for step in forecast_steps(days, run.hour)
                for kind in ("mn", "mx")}
    if set(lookup) != expected:
        raise ValueError("Every point must have every requested temperature interval")
    cursor, end = run.replace(hour=0, minute=0, second=0, microsecond=0), run + timedelta(days=days)
    if cursor < run:
        cursor += timedelta(days=1)
    point_days, indices = [], []
    while cursor + timedelta(days=1) <= end:
        lo = int((cursor - run).total_seconds() / 3600)
        hi, date = lo + 24, cursor.date().isoformat()
        selected = [s for s in forecast_steps(days, run.hour) if lo < s <= hi]
        intervals = [(s - (3 if s <= 144 else 6), s) for s in selected]
        if not intervals or intervals[0][0] != lo or intervals[-1][1] != hi \
                or any(a[1] != b[0] for a, b in zip(intervals, intervals[1:])):
            raise ValueError(f"No continuous coverage for {date}")
        daily = {}
        for point in config.points:
            lows, highs, low_errors, high_errors = [], [], [], []
            quality = "ok"
            for step in selected:
                interval = 3 if step <= 144 else 6
                low_row = lookup[(point["id"], step, f"mn2t{interval}")]
                high_row = lookup[(point["id"], step, f"mx2t{interval}")]
                low, high = low_row["kelvin"], high_row["kelvin"]
                low_error = low_row.get("packing_error_k", 0)
                high_error = high_row.get("packing_error_k", 0)
                if low - high > low_error + high_error + 1e-9:
                    raise ValueError("Interval minimum exceeds its maximum")
                if low > high:
                    quality = "interval_packing_overlap"
                lows.append(low)
                highs.append(high)
                low_errors.append(low_error)
                high_errors.append(high_error)
            low, high = min(lows), max(highs)
            if low > high:
                tolerance = low_errors[lows.index(low)] + high_errors[highs.index(high)]
                if low - high > tolerance + 1e-9:
                    raise ValueError("Daily extrema exceed the reported packing uncertainty")
                low = high = (low + high) / 2
                quality = "daily_packing_overlap"
            mean, hdd, cdd = degree_days(low, high, config.document["base_f"])
            row = dict(valid_date_utc=date, point_id=point["id"], tmin_c=low - 273.15,
                       tmax_c=high - 273.15, tmean_f=mean, hdd=hdd, cdd=cdd, tdd=hdd + cdd,
                       quality=quality)
            daily[point["id"]] = row
            point_days.append(row)
        national_h, national_c = [], []
        for region in config.document["regions"]:
            hdd = math.fsum(daily[p["id"]]["hdd"] * p["weight"] for p in region["points"])
            cdd = math.fsum(daily[p["id"]]["cdd"] * p["weight"] for p in region["points"])
            indices.append(dict(valid_date_utc=date, region_id=region["id"], hdd=hdd, cdd=cdd,
                                tdd=hdd + cdd, weight=region["weight"]))
            national_h.append(hdd * region["weight"])
            national_c.append(cdd * region["weight"])
        hdd, cdd = math.fsum(national_h), math.fsum(national_c)
        indices.append(dict(valid_date_utc=date, region_id="CONUS", hdd=hdd, cdd=cdd,
                            tdd=hdd + cdd, weight=1.0))
        cursor += timedelta(days=1)
    if not point_days:
        raise ValueError("Selected horizon contains no complete UTC day; increase --days")
    adjusted = sum(r["quality"] != "ok" for r in point_days)
    if adjusted:
        LOG.warning("%s city-days have extrema overlapping within GRIB packing uncertainty; "
                    "see quality column. Raw interval values are retained.", adjusted)
    return point_days, indices


SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS configurations (
 config_id TEXT PRIMARY KEY, version TEXT NOT NULL, base_f REAL NOT NULL, document TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs (
 run_time_utc TEXT PRIMARY KEY, model TEXT NOT NULL, resolution TEXT NOT NULL,
 first_recorded_at_utc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS collections (
 run_time_utc TEXT NOT NULL REFERENCES runs, config_id TEXT NOT NULL REFERENCES configurations,
 horizon_hours INTEGER NOT NULL, complete_days INTEGER NOT NULL, source TEXT NOT NULL,
 collected_at_utc TEXT NOT NULL, PRIMARY KEY (run_time_utc, config_id));
CREATE TABLE IF NOT EXISTS raw_files (
 run_time_utc TEXT NOT NULL REFERENCES runs, sha256 TEXT NOT NULL, path TEXT NOT NULL,
 source TEXT NOT NULL, bytes INTEGER NOT NULL, downloaded_at_utc TEXT NOT NULL,
 request TEXT NOT NULL, PRIMARY KEY (run_time_utc, sha256));
CREATE TABLE IF NOT EXISTS temperature_samples (
 run_time_utc TEXT NOT NULL REFERENCES runs, config_id TEXT NOT NULL REFERENCES configurations,
 point_id TEXT NOT NULL, param TEXT NOT NULL, step INTEGER NOT NULL,
 start_hour INTEGER NOT NULL, end_hour INTEGER NOT NULL, kelvin REAL NOT NULL,
 grid_lat REAL NOT NULL, grid_lon REAL NOT NULL, downloaded_at_utc TEXT NOT NULL,
 packing_error_k REAL NOT NULL,
 PRIMARY KEY (run_time_utc, config_id, point_id, param, step));
CREATE TABLE IF NOT EXISTS daily_temperatures (
 run_time_utc TEXT NOT NULL REFERENCES runs, config_id TEXT NOT NULL REFERENCES configurations,
 valid_date_utc TEXT NOT NULL, point_id TEXT NOT NULL, tmin_c REAL NOT NULL,
 tmax_c REAL NOT NULL, tmean_f REAL NOT NULL, hdd REAL NOT NULL, cdd REAL NOT NULL, tdd REAL NOT NULL,
 quality TEXT NOT NULL,
 PRIMARY KEY (run_time_utc, config_id, valid_date_utc, point_id));
CREATE TABLE IF NOT EXISTS daily_indices (
 run_time_utc TEXT NOT NULL REFERENCES runs, config_id TEXT NOT NULL REFERENCES configurations,
 valid_date_utc TEXT NOT NULL, region_id TEXT NOT NULL, hdd REAL NOT NULL,
 cdd REAL NOT NULL, tdd REAL NOT NULL, weight REAL NOT NULL,
 PRIMARY KEY (run_time_utc, config_id, valid_date_utc, region_id));
"""


def connect(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    db.executescript(SCHEMA)
    version = db.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
    if version and version[0] != "1":
        db.close()
        raise ValueError("Unsupported database schema version")
    db.execute("INSERT OR IGNORE INTO metadata VALUES ('schema_version', '1')")
    db.commit()
    return db


def save(db, run: datetime, days: int, config: Config, source: str, files, samples,
         point_days, indices):
    run_id, recorded = utc_string(run), now_utc()
    with db:
        db.execute("INSERT OR IGNORE INTO configurations VALUES (?, ?, ?, ?)",
                   (config.id, config.document["version"], config.document["base_f"],
                    json.dumps(config.document, ensure_ascii=False, sort_keys=True)))
        db.execute("INSERT OR IGNORE INTO runs VALUES (?, 'ifs-oper-fc', '0p25', ?)",
                   (run_id, recorded))
        previous = db.execute("SELECT horizon_hours FROM collections WHERE run_time_utc=? "
                              "AND config_id=?", (run_id, config.id)).fetchone()
        # A shorter repeated request must not erase an already collected longer forecast.
        if previous and previous[0] > days * 24:
            LOG.info("Keeping the longer forecast already in the database")
            return
        for table in ("temperature_samples", "daily_temperatures", "daily_indices"):
            db.execute(f"DELETE FROM {table} WHERE run_time_utc=? AND config_id=?",
                       (run_id, config.id))
        db.execute("INSERT OR REPLACE INTO collections VALUES (?, ?, ?, ?, ?, ?)",
                   (run_id, config.id, days * 24,
                    len([r for r in indices if r["region_id"] == "CONUS"]), source, recorded))
        for path, manifest in files:
            db.execute("INSERT OR IGNORE INTO raw_files VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (run_id, manifest["sha256"], str(path), manifest["source"], manifest["bytes"],
                        manifest["downloaded_at_utc"], json.dumps(manifest["request"], sort_keys=True)))
        db.executemany("INSERT INTO temperature_samples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       [(run_id, config.id, r["point_id"], r["param"], r["step"], r["start_hour"],
                         r["end_hour"], r["kelvin"], r["grid_lat"], r["grid_lon"],
                         r["downloaded_at_utc"], r.get("packing_error_k", 0)) for r in samples])
        db.executemany("INSERT INTO daily_temperatures VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       [(run_id, config.id, r["valid_date_utc"], r["point_id"], r["tmin_c"],
                         r["tmax_c"], r["tmean_f"], r["hdd"], r["cdd"], r["tdd"], r["quality"])
                        for r in point_days])
        db.executemany("INSERT INTO daily_indices VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                       [(run_id, config.id, r["valid_date_utc"], r["region_id"], r["hdd"],
                         r["cdd"], r["tdd"], r["weight"]) for r in indices])


def comparison(db, config_id: str, current_run: str | None = None):
    condition = " AND run_time_utc<=?" if current_run else ""
    parameters = (config_id, current_run) if current_run else (config_id,)
    runs = [r[0] for r in db.execute("SELECT run_time_utc FROM collections WHERE config_id=? "
                                   + condition + " ORDER BY run_time_utc DESC LIMIT 2", parameters)]
    if len(runs) < 2:
        return []
    return [dict(r) for r in db.execute("""
      SELECT n.valid_date_utc, n.region_id, n.run_time_utc AS current_run_utc,
             p.run_time_utc AS previous_run_utc, n.hdd-p.hdd AS delta_hdd,
             n.cdd-p.cdd AS delta_cdd, n.tdd AS current_tdd, p.tdd AS previous_tdd,
             n.tdd-p.tdd AS delta_tdd
      FROM daily_indices n JOIN daily_indices p
        ON n.config_id=p.config_id AND n.valid_date_utc=p.valid_date_utc
       AND n.region_id=p.region_id
      WHERE n.config_id=? AND n.run_time_utc=? AND p.run_time_utc=?
      ORDER BY n.valid_date_utc, n.region_id
    """, (config_id, *runs))]


def write_csv(path: Path, rows: list[dict], columns: list[str]):
    partial = path.with_suffix(".tmp")
    with partial.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    partial.replace(path)


def export(db, config: Config, out: Path, current_run: str | None = None):
    out.mkdir(parents=True, exist_ok=True)
    columns = ["run_time_utc", "valid_date_utc", "region_id", "hdd", "cdd", "tdd", "weight"]
    rows = [dict(r) for r in db.execute("SELECT " + ",".join(columns) +
            " FROM daily_indices WHERE config_id=? ORDER BY run_time_utc,valid_date_utc,region_id",
            (config.id,))]
    write_csv(out / "daily_national.csv", [r for r in rows if r["region_id"] == "CONUS"], columns)
    write_csv(out / "daily_regions.csv", [r for r in rows if r["region_id"] != "CONUS"], columns)
    temperature_columns = ["run_time_utc", "valid_date_utc", "point_id", "tmin_c", "tmax_c",
                           "tmean_f", "hdd", "cdd", "tdd", "quality"]
    temperatures = [dict(r) for r in db.execute("SELECT " + ",".join(temperature_columns) +
                     " FROM daily_temperatures WHERE config_id=? ORDER BY run_time_utc,valid_date_utc,point_id",
                     (config.id,))]
    write_csv(out / "daily_temperatures.csv", temperatures, temperature_columns)
    deltas = comparison(db, config.id, current_run)
    delta_columns = ["valid_date_utc", "region_id", "current_run_utc", "previous_run_utc",
                     "delta_hdd", "delta_cdd", "current_tdd", "previous_tdd", "delta_tdd"]
    write_csv(out / "run_comparison.csv", deltas, delta_columns)
    national = [r for r in rows if r["region_id"] == "CONUS"]
    if not national:
        raise ValueError("No collected runs for this configuration; run collect first")
    latest = current_run or max((r["run_time_utc"] for r in national), default=None)
    current = [r for r in national if r["run_time_utc"] == latest]
    common = [r for r in deltas if r["region_id"] == "CONUS"]
    summary = dict(model="ECMWF IFS oper/fc 0.25", run_time_utc=latest, config_id=config.id,
                   weights_version=config.document["version"], base_f=config.document["base_f"],
                   units="Fahrenheit degree-days", day_boundary="UTC 00:00-24:00",
                   full_days=len(current), dates=[r["valid_date_utc"] for r in current],
                   hdd=math.fsum(r["hdd"] for r in current),
                   cdd=math.fsum(r["cdd"] for r in current),
                   tdd=math.fsum(r["tdd"] for r in current),
                   comparison_dates=[r["valid_date_utc"] for r in common],
                   delta_tdd_same_dates=math.fsum(r["delta_tdd"] for r in common) if common else None,
                   limitation="Approximate population-weighted city proxy; not an official gas-demand index")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "used_config.json").write_text(json.dumps(config.document, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    return summary


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    for command in ("collect", "export"):
        p = sub.add_parser(command)
        p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        p.add_argument("--db", type=Path, default=Path("data/weather.sqlite"))
        p.add_argument("--out", type=Path, default=Path("data/export"))
        if command == "collect":
            p.add_argument("--days", type=int, default=15)
            p.add_argument("--cycle", type=int, choices=[0, 6, 12, 18])
            p.add_argument("--date", help="Initialization date YYYY-MM-DD; requires --cycle")
            p.add_argument("--source", choices=["ecmwf", "aws", "azure", "google"], default="ecmwf")
            p.add_argument("--cache", type=Path, default=Path("data/raw"))
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        config = load_config(args.config)
        if args.command == "collect":
            forecast_steps(args.days, args.cycle)
            if args.date and args.cycle is None:
                raise ValueError("An explicit --date requires --cycle")
            client = make_client(args.source)
            run = resolve_run(client, args.days, args.date, args.cycle)
            LOG.info("Selected initialization: %s", utc_string(run))
            files = download(client, run, args.days, args.source, args.cache)
            samples = read_samples(files, run, args.days, config)
            point_days, indices = calculate(samples, run, args.days, config)
            with closing(connect(args.db)) as db:
                save(db, run, args.days, config, args.source, files, samples, point_days, indices)
                summary = export(db, config, args.out, utc_string(run))
        else:
            if not args.db.exists():
                raise ValueError("Database does not exist; run collect first")
            with closing(connect(args.db)) as db:
                summary = export(db, config, args.out)
        print(json.dumps(summary, indent=2))
        return 0
    except (OSError, ValueError, sqlite3.Error) as error:
        LOG.error("%s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
