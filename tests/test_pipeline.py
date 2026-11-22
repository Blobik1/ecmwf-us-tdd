import copy
import json
import sqlite3
from datetime import datetime, timezone

import pytest

import ecmwf_tdd as app

UTC = timezone.utc
RUN = datetime(2026, 11, 22, tzinfo=UTC)


@pytest.fixture
def config(tmp_path):
    doc = {"version": "test-v1", "base_f": 65, "regions": [
        {"id": "cold", "name": "Cold", "weight": 0.5,
         "points": [{"id": "a", "lat": 40, "lon": -100, "weight": 1.0}]},
        {"id": "warm", "name": "Warm", "weight": 0.5,
         "points": [{"id": "b", "lat": 30, "lon": -90, "weight": 1.0}]},
    ]}
    path = tmp_path / "regions.json"
    path.write_text(json.dumps(doc))
    return app.load_config(path)


def samples(config, run=RUN, days=1, temperatures=(40, 90)):
    rows = []
    for point, fahrenheit in zip(config.points, temperatures, strict=True):
        for step in app.forecast_steps(days, run.hour):
            interval = 3 if step <= 144 else 6
            for kind in ("mn", "mx"):
                rows.append(dict(point_id=point["id"], param=f"{kind}2t{interval}", step=step,
                                 start_hour=step - interval, end_hour=step,
                                 kelvin=(fahrenheit - 32) / 1.8 + 273.15,
                                 grid_lat=point["lat"], grid_lon=point["lon"],
                                 downloaded_at_utc="2026-11-22T10:00:00+00:00"))
    return rows


@pytest.mark.parametrize("fahrenheit,hdd,cdd", [(40, 25, 0), (65, 0, 0), (80, 0, 15)])
def test_degree_days_known_examples(fahrenheit, hdd, cdd):
    k = (fahrenheit - 32) / 1.8 + 273.15
    mean, actual_hdd, actual_cdd = app.degree_days(k, k, 65)
    assert mean == pytest.approx(fahrenheit)
    assert actual_hdd == pytest.approx(hdd)
    assert actual_cdd == pytest.approx(cdd)


def test_threshold_precedes_spatial_averaging(config):
    _, indices = app.calculate(samples(config), RUN, 1, config)
    national = next(row for row in indices if row["region_id"] == "CONUS")
    assert national["hdd"] == pytest.approx(12.5)
    assert national["cdd"] == pytest.approx(12.5)
    assert national["tdd"] == pytest.approx(25)  # DD of the mean temperature would wrongly be zero.


@pytest.mark.parametrize("hour,days,count,first", [(0, 2, 2, "2026-11-22"),
                          (6, 2, 1, "2026-11-23"), (12, 2, 1, "2026-11-23")])
def test_only_complete_utc_days(config, hour, days, count, first):
    run = RUN.replace(hour=hour)
    points, indices = app.calculate(samples(config, run, days), run, days, config)
    dates = sorted({r["valid_date_utc"] for r in points})
    assert len(dates) == count
    assert dates[0] == first
    assert len(indices) == count * 3


def test_missing_and_duplicate_samples_are_rejected(config):
    rows = samples(config)
    with pytest.raises(ValueError, match="Every point"):
        app.calculate(rows[:-1], RUN, 1, config)
    with pytest.raises(ValueError, match="Duplicate"):
        app.calculate(rows + [rows[0]], RUN, 1, config)


def test_reversed_extrema_rejected(config):
    rows = samples(config)
    rows[0]["kelvin"] = rows[1]["kelvin"] + 10
    with pytest.raises(ValueError, match="minimum exceeds"):
        app.calculate(rows, RUN, 1, config)


def test_small_interval_inversion_with_reported_packing_error_is_preserved(config):
    rows = samples(config)
    rows[0]["kelvin"] += 0.1
    rows[0]["packing_error_k"] = 0.125
    points, _ = app.calculate(rows, RUN, 1, config)
    assert points[0]["quality"] == "interval_packing_overlap"
    assert points[0]["hdd"] == pytest.approx(25)
    assert rows[0]["kelvin"] > rows[1]["kelvin"]  # Native raw values are not rewritten.


def test_flat_day_extrema_use_midpoint_only_within_packing_bounds(config):
    rows = samples(config)
    for row in rows:
        if row["point_id"] == "a":
            row["packing_error_k"] = 0.125
            if row["param"].startswith("mn"):
                row["kelvin"] += 0.1
    points, _ = app.calculate(rows, RUN, 1, config)
    assert points[0]["quality"] == "daily_packing_overlap"
    assert points[0]["tmin_c"] == points[0]["tmax_c"]
    assert points[0]["hdd"] == pytest.approx(24.91)


def test_bad_weight_sum_is_rejected(config, tmp_path):
    doc = copy.deepcopy(config.document)
    doc["regions"][0]["weight"] = 0.7
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="sum to 1"):
        app.load_config(path)


def test_configuration_identity_changes_with_weights(config, tmp_path):
    doc = copy.deepcopy(config.document)
    doc["regions"][0]["weight"] = 0.4
    doc["regions"][1]["weight"] = 0.6
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(doc))
    assert app.load_config(path).id != config.id


def save_case(db, config, run=RUN, days=1, temperatures=(40, 90)):
    raw = samples(config, run, days, temperatures)
    points, indices = app.calculate(raw, run, days, config)
    app.save(db, run, days, config, "ecmwf", [], raw, points, indices)
    return raw, points, indices


def test_repeated_collection_has_no_duplicates(config, tmp_path):
    with app.connect(tmp_path / "weather.sqlite") as db:
        save_case(db, config)
        save_case(db, config)
        assert db.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM temperature_samples").fetchone()[0] == 32
        assert db.execute("SELECT COUNT(*) FROM daily_temperatures").fetchone()[0] == 2
        assert db.execute("SELECT COUNT(*) FROM daily_indices").fetchone()[0] == 3


def test_shorter_retry_does_not_erase_saved_horizon(config, tmp_path):
    with app.connect(tmp_path / "weather.sqlite") as db:
        save_case(db, config, days=2)
        save_case(db, config, days=1)
        assert db.execute("SELECT horizon_hours FROM collections").fetchone()[0] == 48
        assert db.execute("SELECT COUNT(*) FROM daily_indices").fetchone()[0] == 6


def test_run_comparison_uses_common_dates(config, tmp_path):
    with app.connect(tmp_path / "weather.sqlite") as db:
        save_case(db, config, run=RUN.replace(day=21), days=2, temperatures=(50, 50))
        save_case(db, config, days=2, temperatures=(40, 40))
        rows = [r for r in app.comparison(db, config.id) if r["region_id"] == "CONUS"]
        assert len(rows) == 1
        assert rows[0]["valid_date_utc"] == "2026-11-22"
        assert rows[0]["delta_tdd"] == pytest.approx(10)
        summary = app.export(db, config, tmp_path / "exports")
        assert summary["full_days"] == 2
        assert summary["tdd"] == pytest.approx(50)
        assert summary["delta_tdd_same_dates"] == pytest.approx(10)
        assert (tmp_path / "exports/daily_national.csv").read_bytes().startswith(b"\xef\xbb\xbf")


def test_historical_collection_does_not_compare_to_a_future_run(config, tmp_path):
    with app.connect(tmp_path / "weather.sqlite") as db:
        old = RUN.replace(day=21)
        save_case(db, config, run=old, days=2)
        save_case(db, config, days=2)
        summary = app.export(db, config, tmp_path / "exports", app.utc_string(old))
        assert summary["run_time_utc"] == app.utc_string(old)
        assert summary["delta_tdd_same_dates"] is None


def test_database_replacement_rolls_back_on_failure(config, tmp_path):
    with app.connect(tmp_path / "weather.sqlite") as db:
        raw, points, indices = save_case(db, config)
        broken = copy.deepcopy(indices)
        broken[0]["hdd"] = None
        with pytest.raises(sqlite3.IntegrityError):
            app.save(db, RUN, 1, config, "ecmwf", [], raw, points, broken)
        assert db.execute("SELECT COUNT(*) FROM daily_indices").fetchone()[0] == 3
        assert db.execute("SELECT hdd FROM daily_indices WHERE region_id='CONUS'").fetchone()[0] \
            == pytest.approx(12.5)


def test_export_without_a_collected_configuration_fails(config, tmp_path):
    with app.connect(tmp_path / "weather.sqlite") as db:
        with pytest.raises(ValueError, match="No collected runs"):
            app.export(db, config, tmp_path / "exports")


def test_latest_selection_checks_final_requested_lead():
    class Client:
        def latest(self, **request):
            assert request["step"] == 360
            assert request["time"] == 0
            return datetime(2026, 11, 22)
    assert app.resolve_run(Client(), 15, None, 0) == RUN


def test_short_cycles_reject_unsupported_horizon():
    with pytest.raises(ValueError, match="144 hours"):
        app.forecast_steps(15, 6)


def test_cache_detects_corruption(tmp_path):
    class Client:
        calls = 0
        def retrieve(self, target, **request):
            self.calls += 1
            from pathlib import Path
            Path(target).write_bytes(b"test-payload")
    client = Client()
    files = app.download(client, RUN, 1, "ecmwf", tmp_path)
    app.download(client, RUN, 1, "ecmwf", tmp_path)
    assert client.calls == 1
    files[0][0].write_bytes(b"corrupted")
    app.download(client, RUN, 1, "ecmwf", tmp_path)
    assert client.calls == 2
