"""Drift: the population stability index of every input and of the score, against each
bundle's own training reference."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("xgboost")

from prism import config, tracks  # noqa: E402
from prism.api import db  # noqa: E402
from prism.api.inference import InferenceEngine  # noqa: E402
from prism.data.frames import load_frame  # noqa: E402
from prism.live import drift  # noqa: E402
from prism.training import train  # noqa: E402
from tests._india_service import build_deployment, payload_from, service, use_temp_database  # noqa: E402

INDIA = tracks.INDIA
S = {"drift_warn": 0.10, "drift_alert": 0.25, "drift_epsilon": 0.0001, "category_floor": 0.01, "drift_min_rows": 5,
     "drift_window_rows": 5000, "drift_discrete_max": 10}


@pytest.fixture(scope="module")
def frame(india_stem):
    return load_frame(INDIA, india_stem)


@pytest.fixture(scope="module")
def reference(frame):
    scores = np.random.default_rng(0).random(len(frame))
    return drift.reference(frame, scores, INDIA, S), scores


def test_psi_is_zero_for_identical_shares_and_matches_a_worked_example():
    assert drift.psi([0.25, 0.75], [0.25, 0.75], 1e-4) == 0.0
    expected = (0.4 - 0.5) * math.log(0.4 / 0.5) + (0.6 - 0.5) * math.log(0.6 / 0.5)
    assert expected == pytest.approx(0.1 * math.log(1.25) + 0.1 * math.log(1.2), abs=1e-12)
    assert drift.psi([0.5, 0.5], [0.4, 0.6], 1e-4) == pytest.approx(expected, abs=1e-9)


def test_an_empty_bin_is_floored_at_the_epsilon():
    value = drift.psi([0.5, 0.5, 0.0], [0.5, 0.4, 0.1], 1e-4)
    assert math.isfinite(value) and value > 0


def test_the_reference_has_every_input_and_the_score(reference):
    ref, _ = reference
    assert set(ref["numeric"]) == set(INDIA.numeric_columns)
    assert set(ref["categorical"]) == set(INDIA.categorical_columns)
    for entry in ref["numeric"].values():
        bins = entry["edges"] if "edges" in entry else entry["values"]
        assert len(entry["shares"]) == len(bins) + 1
        assert sum(entry["shares"]) == pytest.approx(1.0)
    assert "values" in ref["numeric"]["cvv_match_flag"] and "edges" in ref["numeric"]["amount"]
    assert sum(ref["score"]["shares"]) == pytest.approx(1.0)
    assert all("other" in entry["shares"] for entry in ref["categorical"].values())


def test_an_unseen_category_counts_as_other(reference, frame):
    ref, scores = reference
    assert "9999" not in ref["categorical"]["mcc"]["shares"]
    recent = frame.copy()
    recent["mcc"] = "9999"
    report = drift.report(ref, recent, scores, S)
    mcc = next(row for row in report["features"] if row["feature"] == "mcc")
    assert mcc["state"] == "alert"


def test_states_follow_the_configured_cut_points():
    assert drift.state(0.09, S) == "ok"
    assert drift.state(0.10, S) == "warn"
    assert drift.state(0.2499, S) == "warn"
    assert drift.state(0.25, S) == "alert"


def test_the_same_rows_show_no_drift(reference, frame):
    ref, scores = reference
    report = drift.report(ref, frame, scores, S)
    assert report["state"] == "ok"
    assert report["rows"] == len(frame)
    assert max(row["psi"] for row in report["features"]) < 1e-6


def test_a_shifted_amount_raises_the_amount_feature_first(reference, frame):
    ref, scores = reference
    recent = frame.copy()
    recent["amount"] = recent["amount"] * 3
    report = drift.report(ref, recent, scores, S)
    assert report["features"][0]["feature"] == "amount"
    assert report["features"][0]["state"] == "alert"
    assert report["state"] == "alert"
    assert [row["psi"] for row in report["features"]] == sorted((row["psi"] for row in report["features"]), reverse=True)


def test_the_offline_training_writes_the_reference_for_the_indian_track(india_splits_module, india_stem, tmp_path, monkeypatch):
    frame = load_frame(INDIA, india_stem)
    model_dir = build_deployment(india_stem, tmp_path / "models")
    monkeypatch.setitem(config.load_config()["api"], "primary_model", "xgboost")
    written = train.write_drift_reference(frame, india_splits_module, INDIA, model_dir)
    assert written == model_dir / drift.FILENAME
    ref = drift.load(model_dir)
    assert set(ref["numeric"]) == set(INDIA.numeric_columns)
    assert ref["rows"] == len(india_splits_module.X_train) + len(india_splits_module.X_val)


@pytest.fixture(scope="module")
def deployment(tmp_path_factory, india_stem):
    return build_deployment(india_stem, tmp_path_factory.mktemp("drift_models"))


def test_the_drift_endpoint_refuses_without_a_reference(deployment, tmp_path, monkeypatch, frame):
    use_temp_database(tmp_path, monkeypatch)
    engine = InferenceEngine(deployment, track="india").load()
    with service(engine) as client:
        response = client.get("/api/v1/drift")
    assert response.status_code == 409
    assert str(deployment) in response.json()["detail"]


def test_the_drift_endpoint_refuses_a_short_log_and_then_reports(deployment, tmp_path, monkeypatch, frame):
    use_temp_database(tmp_path, monkeypatch)
    monkeypatch.setitem(config.load_config()["live"], "drift_min_rows", 5)
    monkeypatch.setitem(config.load_config()["live"], "drift_refresh_seconds", 0)
    bundle = tmp_path / "with_reference"
    import shutil

    shutil.copytree(deployment, bundle)
    drift.save(drift.reference(frame, np.random.default_rng(1).random(len(frame)), INDIA, S), bundle)
    engine = InferenceEngine(bundle, track="india").load()
    with service(engine) as client:
        short = client.get("/api/v1/drift")
        for _, row in frame.iloc[:8].iterrows():
            client.post("/api/v1/predict", json=payload_from(row))
        answer = client.get("/api/v1/drift")
    assert short.status_code == 409 and "5" in short.json()["detail"]
    assert answer.status_code == 200
    body = answer.json()
    assert body["rows"] == 8
    assert {row["feature"] for row in body["features"]} == set(INDIA.numeric_columns) | set(INDIA.categorical_columns)
    assert body["score"]["state"] in ("ok", "warn", "alert")


def test_a_point_mass_column_is_binned_by_value_and_still_drifts(frame):
    ref = drift.reference(frame, np.random.default_rng(0).random(len(frame)), INDIA, S)
    entry = ref["numeric"]["is_cross_border"]
    assert entry["values"] == [0.0, 1.0] and len(entry["shares"]) == 3
    recent = frame.copy()
    rng = np.random.default_rng(2)
    recent["is_cross_border"] = (rng.random(len(recent)) < 0.30).astype(float)
    report = drift.report(ref, recent, np.random.default_rng(0).random(len(frame)), S)
    row = next(r for r in report["features"] if r["feature"] == "is_cross_border")
    assert row["state"] == "alert"


def test_a_value_unseen_in_a_discrete_column_counts_as_other(frame):
    ref = drift.reference(frame, np.random.default_rng(0).random(len(frame)), INDIA, S)
    recent = frame.copy()
    recent["is_cross_border"] = 7.0
    report = drift.report(ref, recent, np.random.default_rng(0).random(len(frame)), S)
    row = next(r for r in report["features"] if r["feature"] == "is_cross_border")
    assert row["state"] == "alert"


def _drift_service(deployment, tmp_path, monkeypatch, frame):
    use_temp_database(tmp_path, monkeypatch)
    live = config.load_config()["live"]
    monkeypatch.setitem(live, "drift_min_rows", 5)
    monkeypatch.setitem(live, "drift_refresh_seconds", 0)
    import shutil

    bundle = tmp_path / "with_reference"
    shutil.copytree(deployment, bundle)
    drift.save(drift.reference(frame, np.random.default_rng(1).random(len(frame)), INDIA, S), bundle)
    return InferenceEngine(bundle, track="india").load()


def test_the_audit_row_records_the_bundle_that_scored_it(deployment, tmp_path, monkeypatch, frame):
    engine = _drift_service(deployment, tmp_path, monkeypatch, frame)
    with service(engine) as client:
        client.post("/api/v1/predict", json=payload_from(frame.iloc[0]))
    row = db.find_transaction(str(frame.iloc[0]["transaction_id"]))
    assert row.bundle_id == "offline"


def test_score_drift_counts_only_the_rows_the_serving_bundle_scored(deployment, tmp_path, monkeypatch, frame):
    engine = _drift_service(deployment, tmp_path, monkeypatch, frame)
    for i, (_, row) in enumerate(frame.iloc[100:120].iterrows()):
        db.log_transaction(
            transaction_id=f"old-{i}", features={}, primary_model="xgboost", risk_score=0.5, flagged=False,
            latency_ms=1.0, scores=[], explanation=None, amount=float(row["amount"]),
            time_feature=float(row["tx_time_seconds"]), request=payload_from(row), bundle_id="day-001-abcdef",
        )
    with service(engine) as client:
        early = client.get("/api/v1/drift").json()
        for _, row in frame.iloc[:8].iterrows():
            client.post("/api/v1/predict", json=payload_from(row))
        later = client.get("/api/v1/drift").json()
    assert early["rows"] == 20 and early["score"] is None and early["score_rows"] == 0
    assert later["rows"] == 28 and later["score_rows"] == 8 and later["score"]["state"] in ("ok", "warn", "alert")


def test_the_drift_report_is_cached_for_the_refresh_window(deployment, tmp_path, monkeypatch, frame):
    engine = _drift_service(deployment, tmp_path, monkeypatch, frame)
    monkeypatch.setitem(config.load_config()["live"], "drift_refresh_seconds", 3600)
    with service(engine) as client:
        for _, row in frame.iloc[:6].iterrows():
            client.post("/api/v1/predict", json=payload_from(row))
        first = client.get("/api/v1/drift").json()
        for _, row in frame.iloc[6:9].iterrows():
            client.post("/api/v1/predict", json=payload_from(row))
        cached = client.get("/api/v1/drift").json()
    assert first["rows"] == cached["rows"] == 6


def test_a_continuous_column_massed_at_zero_still_drifts(frame):
    rng = np.random.default_rng(4)
    reference_rows = frame.copy()
    reference_rows["geo_risk_score"] = np.where(rng.random(len(frame)) < 0.92, 0.0, rng.uniform(1.0, 100.0, len(frame)))
    ref = drift.reference(reference_rows, rng.random(len(frame)), INDIA, S)
    assert "edges" in ref["numeric"]["geo_risk_score"]
    assert min(ref["numeric"]["geo_risk_score"]["shares"]) > 0.0, "no bin is empty by construction"
    recent = frame.copy()
    recent["geo_risk_score"] = np.where(rng.random(len(frame)) < 0.60, 0.0, rng.uniform(1.0, 100.0, len(frame)))
    report = drift.report(ref, recent, rng.random(len(frame)), S)
    row = next(r for r in report["features"] if r["feature"] == "geo_risk_score")
    assert row["state"] == "alert"
