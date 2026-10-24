"""The service on the Indian track: categorical inputs, velocity from the log, named drivers."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("xgboost")

from fastapi.testclient import TestClient  # noqa: E402

from prism import config, tracks  # noqa: E402
from prism.api import db, schemas  # noqa: E402
from prism.api.inference import InferenceEngine  # noqa: E402
from prism.api.main import app, get_engine, get_engine_optional  # noqa: E402
from prism.data import india  # noqa: E402
from prism.data.preprocess import prepare_splits, save_encoder, save_scaler  # noqa: E402
from prism.evaluation.threshold import tune_threshold  # noqa: E402
from prism.explain import artifacts  # noqa: E402
from prism.features import velocity  # noqa: E402
from prism.features.velocity import add_velocity_features  # noqa: E402
from prism.models import build  # noqa: E402

INDIA = tracks.INDIA


def payload_from(row, txn_id: str) -> dict:
    body = {}
    for column in INDIA.input_columns:
        value = row[column]
        body[column] = str(value) if (column in INDIA.categorical_columns or column == INDIA.card_column) else float(value)
    body.update({
        "transaction_id": txn_id, "merchant": str(row["merchant_id"]),
        "channel": str(row["channel_type"]), "currency": "INR",
    })
    return body


@pytest.fixture(scope="module")
def deployment(tmp_path_factory, india_stem):
    frame = add_velocity_features(india.load_india(india_stem))
    splits = prepare_splits(frame, schema=INDIA)
    model_dir = tmp_path_factory.mktemp("india_models")
    thresholds = {}
    for name in ("isolation_forest", "xgboost"):
        overrides = {"categorical_columns": splits.categorical_columns}
        if name == "xgboost":
            overrides["n_estimators"] = 60
        model = build(name, **overrides).fit(splits.X_train, splits.y_train)
        model.save(model_dir)
        thresholds[name] = tune_threshold(splits.y_val, model.risk_score(splits.X_val), strategy="fpr_budget", fpr_budget_per_10k=50)
    save_scaler(splits.scaler, model_dir / "scaler.joblib")
    save_encoder(splits.encoder, model_dir / "encoder.joblib")
    (model_dir / "thresholds.json").write_text(json.dumps(thresholds), encoding="utf-8")
    reference, cases = artifacts.build(frame, splits, schema=INDIA)
    artifacts.save(reference, cases, model_dir)
    raw_test = frame.loc[splits.X_test.index]
    fraud_row = raw_test[raw_test["is_fraud"] == 1].iloc[0]
    normal_row = raw_test[raw_test["is_fraud"] == 0].iloc[0]
    return model_dir, payload_from(normal_row, "txn-in-normal"), payload_from(fraud_row, "txn-in-fraud")


@pytest.fixture(autouse=True)
def temp_database(tmp_path, monkeypatch):
    url = f"sqlite:///{(tmp_path / 'audit.db').as_posix()}"
    monkeypatch.setitem(config.load_config()["api"], "database_url", url)
    # The deployment here records no manifest, so there is no champion for auto to serve.
    monkeypatch.setitem(config.load_config()["api"], "primary_model", "xgboost")
    db.init_db(url)
    return url


@pytest.fixture
def engine(deployment):
    model_dir, _, _ = deployment
    return InferenceEngine(model_dir, track="india").load()


@pytest.fixture
def client(engine):
    app.dependency_overrides[get_engine] = lambda: engine
    app.dependency_overrides[get_engine_optional] = lambda: engine
    try:
        with TestClient(app) as test_client:
            # Startup pins the schema to the engine it loaded (the ULB one on a
            # machine with ULB models), so the pin to the Indian engine comes after.
            schemas.set_active_schema(INDIA)
            yield test_client
    finally:
        app.dependency_overrides.clear()
        schemas.set_active_schema(None)


def test_health_and_metrics_report_the_track(client):
    assert client.get("/api/v1/health").json()["track"] == "india"
    assert client.get("/api/v1/metrics").json()["track"] == "india"


def test_engine_derives_velocity_from_the_history_it_is_given(engine, deployment):
    _, normal, _ = deployment
    alone = engine.score(normal, history=[])
    assert alone["features"]["card_tx_1h"] == 0.0 and alone["features"]["first_cnp_30d"] in (0.0, 1.0)
    t = float(normal["tx_time_seconds"])
    busy = engine.score(normal, history=[(t - 60.0, 100.0, "5411", "CP_POS_CHIP"), (t - 30.0, 50.0, "5812", "CNP_WEB")])
    assert busy["features"]["card_tx_1h"] == 2.0 and busy["features"]["card_amount_24h"] == 150.0
    assert busy["frame"].columns.tolist() == engine.feature_order


def test_predict_scores_an_indian_row_and_the_log_becomes_its_history(client, deployment):
    _, normal, _ = deployment
    first = client.post("/api/v1/predict", json=normal)
    assert first.status_code == 200, first.text
    assert [s["model_name"] for s in first.json()["scores"]] == ["xgboost"]
    logged = client.get("/api/v1/transactions?limit=1").json()[0]
    assert {s["model_name"] for s in logged["scores"]} == {"isolation_forest", "xgboost"}

    second = dict(normal, transaction_id="txn-in-normal-2", tx_time_seconds=float(normal["tx_time_seconds"]) + 120.0)
    assert client.post("/api/v1/predict", json=second).status_code == 200

    history = db.card_history(normal["card_id"], before=float(second["tx_time_seconds"]) + 1.0, window_seconds=velocity.longest_window())
    assert [h[0] for h in history] == [float(normal["tx_time_seconds"]), float(second["tx_time_seconds"])]
    assert history[0][2] == normal["mcc"] and history[0][3] == normal["channel_type"]

    row = client.get("/api/v1/transactions?limit=1").json()[0]
    assert row["card_id"] == normal["card_id"] and row["currency"] == "INR" and row["amount"] == normal["amount"]


def test_explanation_names_its_driver(client, deployment):
    _, _, fraud = deployment
    body = client.post("/api/v1/explain", json=fraud).json()
    assert body["facts"] is not None
    assert "The strongest push" in body["narrative"]
    assert all(f["interpretable"] for f in body["top_features"])
    assert "display" in body["top_features"][0]


def test_missing_input_is_422_naming_the_track(client, deployment):
    _, normal, _ = deployment
    incomplete = {k: v for k, v in normal.items() if k != "channel_type"}
    response = client.post("/api/v1/predict", json=incomplete)
    assert response.status_code == 422
    assert "india" in response.text and "channel_type" in response.text


def test_unknown_category_still_scores(client, deployment):
    _, normal, _ = deployment
    response = client.post("/api/v1/predict", json=dict(normal, gateway_provider="NEWPAY", transaction_id="txn-new-gateway"))
    assert response.status_code == 200, response.text


def test_a_ulb_payload_on_the_indian_service_is_422_not_500(client):
    ulb = {"Time": 0.0, "Amount": 10.0, **{f"V{i}": 0.0 for i in range(1, 29)}}
    response = client.post("/api/v1/predict", json=ulb)
    assert response.status_code == 422
    assert "india" in response.text


def test_a_non_numeric_feature_is_422_not_500(client, deployment):
    _, normal, _ = deployment
    response = client.post(
        "/api/v1/predict", json=dict(normal, vaai_score="high", transaction_id="txn-text-score")
    )
    assert response.status_code == 422
    assert "vaai_score" in response.text


def test_a_non_finite_feature_is_422_not_500(client, deployment):
    """float("NaN") passes a bare float() call; the old typed field rejected it
    with a 422 and this schema must too, or the row scores and the audit-log
    insert fails on a NULL amount."""
    _, normal, _ = deployment
    for column, value in (("amount", "NaN"), ("tx_time_seconds", "Infinity")):
        payload = dict(normal, transaction_id=f"txn-{column}")
        payload[column] = value
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422, response.text
        assert column in response.text


def test_an_overlong_text_input_is_422(client, deployment):
    """A categorical of any length reached the audit log's VARCHAR columns;
    SQLite stores it, Postgres would raise after scoring."""
    _, normal, _ = deployment
    for column in ("gateway_provider", "mcc"):
        payload = dict(normal, transaction_id=f"txn-long-{column}")
        payload[column] = "X" * 200
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422, response.text
        assert column in response.text


def test_history_reads_the_scored_channel_type_not_the_display_channel(client, deployment):
    """A client may omit the display field `channel`; the velocity features
    must still see the channel type that was scored."""
    _, normal_payload, _ = deployment
    first = {k: v for k, v in normal_payload.items() if k != "channel"}
    first.update({"transaction_id": "txn-ch-1", "channel_type": "CNP_WEB"})
    second = {**first, "transaction_id": "txn-ch-2", "tx_time_seconds": float(first["tx_time_seconds"]) + 60.0}
    assert client.post("/api/v1/predict", json=first).status_code == 200
    assert client.post("/api/v1/predict", json=second).status_code == 200

    history = db.card_history(first["card_id"], before=second["tx_time_seconds"], window_seconds=velocity.longest_window())
    assert history and history[-1][3] == "CNP_WEB"


def test_the_counterfactual_sweep_moves_the_limit_ratio_with_the_amount(engine, deployment):
    """amount_to_limit is derived from the amount; a sweep that lowers the
    amount but keeps the original share shows the model rows it would never
    see, and the sentence it writes is false."""
    _, _, fraud = deployment
    seen = {}

    class Recorder:
        name = "xgboost"

        def risk_score(self, frame):
            seen["frame"] = frame.copy()
            return np.full(len(frame), 0.99)

    engine.models["xgboost"] = Recorder()
    engine.thresholds["xgboost"] = 0.5
    result = engine.score(fraud, history=[])
    engine.clearing(result["frame"], result["features"])

    swept = seen["frame"]
    assert len(swept) == int(config.get("explain.counterfactual_steps", 200))
    assert np.all(np.diff(swept["amount"].to_numpy()) < 0)
    assert np.all(np.diff(swept["amount_to_limit"].to_numpy()) <= 0)
    assert swept["amount_to_limit"].nunique() > 1



def test_a_value_outside_a_contract_code_set_is_422_naming_the_field(client, deployment):
    """The shipped generator's contract (generator/spec/19) fixes the code sets a request may
    carry; a request outside one is refused before scoring, naming the field and the value."""
    _, normal, _ = deployment
    for field, value in (("channel_type", "TELEPATHY"), ("avs_match_code", "Q"), ("pos_entry_mode", "999"), ("asn_type", "satellite")):
        response = client.post("/api/v1/predict", json=dict(normal, **{field: value}, transaction_id=f"txn-contract-{field}"))
        assert response.status_code == 422, (field, response.text)
        assert field in response.text and str(value) in response.text


def test_a_value_of_the_wrong_contract_type_is_422_naming_the_field(client, deployment):
    _, normal, _ = deployment
    response = client.post("/api/v1/predict", json=dict(normal, is_cross_border="maybe", transaction_id="txn-contract-bool"))
    assert response.status_code == 422
    assert "is_cross_border" in response.text


def test_an_open_vocabulary_still_scores_and_the_contract_fields_are_the_tracks(client, deployment):
    """Identifier-like vocabularies (the acquirer, the gateway, the product) grow in production;
    a new value there scores, it does not refuse the authorisation."""
    from prism import contract, tracks

    _, normal, _ = deployment
    response = client.post("/api/v1/predict", json=dict(normal, acquirer_bin="999999", transaction_id="txn-new-acquirer"))
    assert response.status_code == 200, response.text
    assert tracks.get("india").contract_feed == "authorisation_request"
    assert tracks.get("ulb").contract_feed is None
    known = {f.name for f in contract.fields("authorisation_request")}
    assert set(tracks.get("india").input_columns) - set(tracks.INDIA_DERIVED) <= known


def test_explanation_carries_reason_codes(client, deployment):
    _, _, fraud = deployment
    body = client.post("/api/v1/explain", json=fraud).json()
    assert body["reason_codes"], "an alert's explanation ends in reason codes"
    first = body["reason_codes"][0]
    assert set(first) == {"code", "label", "share", "features"}
    assert first["code"] in ("VEL", "AMT", "NEW", "NET", "LOC", "CHN", "MER", "PRD", "TIM")
    assert 0.0 < first["share"] <= 1.0 and first["features"]
    assert sum(c["share"] for c in body["reason_codes"]) <= 1.0 + 1e-9
