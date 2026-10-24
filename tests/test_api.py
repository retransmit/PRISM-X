"""Integration tests for the FastAPI service tier (spec §6.2.2)."""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("xgboost")

from fastapi.testclient import TestClient  # noqa: E402

from prism import config  # noqa: E402
from prism.api import db  # noqa: E402
from prism.api.inference import InferenceEngine  # noqa: E402
from prism.api.main import app, get_engine, get_engine_optional  # noqa: E402
from prism.explain import artifacts
from prism.data.preprocess import prepare_splits, save_scaler  # noqa: E402
from prism.evaluation.threshold import tune_threshold  # noqa: E402
from prism.models import build  # noqa: E402

PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]
FEATURES = ["Time", *PCA_COLUMNS, "Amount"]

CREDENTIALS = {"analyst": "analyst-secret", "customer": "customer-secret"}


def as_role(role: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {CREDENTIALS[role]}"}


@pytest.fixture(autouse=True)
def verdict_credentials(monkeypatch):
    monkeypatch.setitem(config.load_config()["api"], "credentials", dict(CREDENTIALS))
    # The deployment here records no manifest, so there is no champion for auto to serve.
    monkeypatch.setitem(config.load_config()["api"], "primary_model", "xgboost")


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Train and persist XGBoost + Isolation Forest; return (dir, normal, fraud)."""
    rng = np.random.default_rng(21)
    n, n_fraud = 6_000, 60

    frame = pd.DataFrame(
        {
            **{c: rng.normal(size=n) for c in PCA_COLUMNS},
            "Time": rng.uniform(0, 172_800, size=n),
            "Amount": rng.gamma(2.0, 50.0, size=n),
        }
    )
    labels = np.zeros(n, dtype=int)
    labels[rng.choice(n, size=n_fraud, replace=False)] = 1
    frame["Class"] = labels

    mask = frame["Class"] == 1
    for column in PCA_COLUMNS[:6]:
        frame.loc[mask, column] += rng.normal(7.0, 1.0, size=n_fraud)
    frame.loc[mask, "Amount"] *= 6.0
    frame = frame[["Time", *PCA_COLUMNS, "Amount", "Class"]]

    splits = prepare_splits(frame)
    model_dir = tmp_path_factory.mktemp("models")

    thresholds = {}
    for name in ("isolation_forest", "xgboost"):
        model = build(name, **({"n_estimators": 60} if name == "xgboost" else {}))
        model.fit(splits.X_train, splits.y_train)
        model.save(model_dir)
        thresholds[name] = tune_threshold(splits.y_val, model.risk_score(splits.X_val))

    save_scaler(splits.scaler, model_dir / "scaler.joblib")
    (model_dir / "thresholds.json").write_text(json.dumps(thresholds), encoding="utf-8")
    reference, cases = artifacts.build(frame, splits)
    artifacts.save(reference, cases, model_dir)

    # Raw, unscaled rows; the service applies the scaler itself.
    raw_test = frame.loc[splits.X_test.index]
    fraud_row = raw_test[raw_test["Class"] == 1].iloc[0]
    normal_row = raw_test[raw_test["Class"] == 0].iloc[0]

    def payload(row, txn_id):
        body = {name: float(row[name]) for name in FEATURES}
        body["transaction_id"] = txn_id
        return body

    return (
        model_dir,
        payload(normal_row, "txn-normal"),
        payload(fraud_row, "txn-fraud"),
    )


@pytest.fixture(autouse=True)
def temp_database(tmp_path, monkeypatch):
    """Keep the audit log out of the project root for every test in this module."""
    url = f"sqlite:///{(tmp_path / 'audit.db').as_posix()}"
    settings = config.load_config()
    monkeypatch.setitem(settings["api"], "database_url", url)
    db.init_db(url)
    return url


@pytest.fixture
def client(trained):
    model_dir, _, _ = trained
    engine = InferenceEngine(model_dir).load()
    app.dependency_overrides[get_engine] = lambda: engine
    app.dependency_overrides[get_engine_optional] = lambda: engine
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


def test_predict_answers_with_the_champion_and_the_log_gets_every_model(client, trained):
    _, normal_payload, _ = trained
    response = client.post("/api/v1/predict", json=normal_payload)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["transaction_id"] == "txn-normal"
    assert [s["model_name"] for s in body["scores"]] == ["xgboost"]
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["latency_ms"] > 0
    row = client.get("/api/v1/transactions?limit=1").json()[0]
    assert {s["model_name"] for s in row["scores"]} == {"isolation_forest", "xgboost"}
    assert row["latency_ms"] == body["latency_ms"]


def test_a_flagged_transaction_is_explained_on_its_audit_row(client, trained):
    _, _, fraud_payload = trained
    body = client.post("/api/v1/predict", json=fraud_payload).json()
    assert body["flagged"] and body["explanation"] is None, "the answer does not wait for the explanation"
    row = client.get("/api/v1/transactions?flagged=true").json()[0]
    assert row["explanation"]["model_name"] == "xgboost"
    assert row["explanation"]["top_features"] and row["explanation"]["narrative"]


def test_explanation_reads_as_plain_english_with_measured_facts(client, trained):
    """The sentence must be built from facts the response also carries, and it
    must never name a PCA component. facts is declared on the response model;
    pydantic would strip it silently otherwise."""
    _, _, fraud_payload = trained
    client.post("/api/v1/predict", json=fraud_payload)
    explanation = client.get("/api/v1/transactions?flagged=true").json()[0]["explanation"]

    facts = explanation["facts"]
    assert facts is not None
    assert facts["similar_total"] == 50
    assert 0 <= facts["similar_fraud"] <= 50
    assert facts["n_characteristics"] == 28
    assert 0.0 <= facts["amount_pct_below"] <= 1.0

    narrative = explanation["narrative"]
    assert "most similar transactions" in narrative
    assert not re.search(r"\bV\d{1,2}\b", narrative), narrative
    assert "latent" not in narrative


def test_clearing_amount_is_a_lower_amount_the_model_clears(trained):
    """When the counterfactual fires, scoring the transaction at that amount,
    everything else unchanged, must fall below the primary threshold."""
    model_dir, _, fraud_payload = trained
    engine = InferenceEngine(model_dir).load()
    features = {k: v for k, v in fraud_payload.items() if k != "transaction_id"}
    frame = engine.score(features)["frame"]

    cleared_at = engine.clearing_amount(frame, features)
    if cleared_at is None:
        pytest.skip("counterfactual did not fire for this fixture row")
    assert cleared_at < features["Amount"]
    lowered = dict(features, Amount=cleared_at)
    assert engine.score(lowered)["risk_score"] < engine.threshold_for(engine.primary)


def test_predict_rejects_a_payload_missing_a_pca_component(client, trained):
    _, normal_payload, _ = trained
    broken = {k: v for k, v in normal_payload.items() if k != "V5"}

    response = client.post("/api/v1/predict", json=broken)
    assert response.status_code == 422
    assert "V5" in response.text


def test_explain_works_for_an_unflagged_transaction(client, trained):
    _, normal_payload, _ = trained
    response = client.post("/api/v1/explain", json=normal_payload)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model_name"] == "xgboost"
    assert isinstance(body["narrative"], str) and body["narrative"]


def test_health_reports_loaded_models(client):
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ready"
    assert body["primary_model"] == "xgboost"
    assert body["explainer"] is True


def test_metrics_counts_what_was_processed(client, trained):
    _, normal_payload, fraud_payload = trained
    client.post("/api/v1/predict", json=normal_payload)
    client.post("/api/v1/predict", json=fraud_payload)

    body = client.get("/api/v1/metrics").json()
    assert body["processed"] >= 2
    assert body["flagged"] >= 1
    assert set(body["thresholds"]) == {"isolation_forest", "xgboost"}
    assert body["mean_latency_ms"] > 0


def test_audit_log_persists_the_explanation(client, trained):
    _, _, fraud_payload = trained
    client.post("/api/v1/predict", json=fraud_payload)

    rows = client.get("/api/v1/transactions?limit=10").json()
    flagged = [r for r in rows if r["flagged"]]
    assert flagged, "a flagged transaction should be in the audit log"
    assert flagged[0]["explanation"] is not None
    assert flagged[0]["latency_ms"] > 0


def test_service_returns_503_when_no_models_are_loaded(trained, tmp_path, monkeypatch):
    """With no engine, a fully valid payload must 503 rather than 500 on a None.

    The service is pointed at an empty model directory, so the outcome does not
    depend on whether models/ has been trained on this machine.
    """
    _, normal_payload, _ = trained
    empty = tmp_path / "no_models"
    empty.mkdir()
    monkeypatch.setitem(config.load_config()["api"], "model_dir", str(empty))
    app.dependency_overrides.clear()

    with TestClient(app) as bare_client:
        response = bare_client.post("/api/v1/predict", json=normal_payload)

    assert response.status_code == 503
    assert "No models loaded" in response.text


def test_load_warms_the_explanation_path(trained):
    """Measured on the real models, the first flagged transaction after startup
    took 1.6 s end to end against about 50 ms warm, because the SHAP explainer
    and the case index build their buffers on first use. Loading must pay that
    cost once, before the first real request."""
    model_dir, _, _ = trained
    engine = InferenceEngine(model_dir).load()
    assert engine.warmed_up is True
    assert engine.mean_latency_ms == 0.0, "the warm-up must not count toward reported latency"


def test_context_fields_are_stored_and_never_change_the_score(client, trained):
    _, normal_payload, _ = trained
    plain = client.post("/api/v1/predict", json=normal_payload).json()
    with_context = client.post(
        "/api/v1/predict",
        json={
            **normal_payload,
            "transaction_id": "txn-context",
            "card_id": "card-07",
            "merchant": "Shop",
            "channel": "CNP",
            "currency": "INR",
        },
    ).json()

    assert with_context["risk_score"] == plain["risk_score"]

    rows = {r["transaction_id"]: r for r in client.get("/api/v1/transactions").json()}
    assert rows["txn-context"]["card_id"] == "card-07"
    assert rows["txn-context"]["merchant"] == "Shop"
    assert rows["txn-context"]["channel"] == "CNP"
    assert rows["txn-context"]["currency"] == "INR"
    assert rows["txn-normal"]["card_id"] is None


def test_decision_round_trip_keeps_history_and_reports_the_latest(client, trained):
    _, _, fraud_payload = trained
    client.post("/api/v1/predict", json=fraud_payload)

    first = client.post(
        "/api/v1/decisions",
        json={"transaction_id": "txn-fraud", "verdict": "fraud", "note": "card testing"},
        headers=as_role("analyst"),
    )
    assert first.status_code == 201, first.text
    assert first.json()["verdict"] == "fraud"
    assert first.json()["decided_at"]

    second = client.post(
        "/api/v1/decisions",
        json={"transaction_id": "txn-fraud", "verdict": "legitimate"},
        headers=as_role("analyst"),
    )
    assert second.status_code == 201

    row = client.get("/api/v1/transactions?flagged=true").json()[0]
    assert row["transaction_id"] == "txn-fraud"
    assert row["decisions"]["analyst"]["verdict"] == "legitimate"
    assert row["decisions"]["analyst"]["note"] is None
    assert row["decisions"]["customer"] is None


def test_customer_and_analyst_decisions_are_independent(client, trained):
    _, _, fraud_payload = trained
    client.post("/api/v1/predict", json=fraud_payload)
    client.post(
        "/api/v1/decisions",
        json={"transaction_id": "txn-fraud", "verdict": "fraud"},
        headers=as_role("customer"),
    )

    row = client.get("/api/v1/transactions?flagged=true").json()[0]
    assert row["decisions"]["customer"]["verdict"] == "fraud"
    assert row["decisions"]["analyst"] is None


def test_decision_for_an_unknown_transaction_is_404(client):
    response = client.post(
        "/api/v1/decisions",
        json={"transaction_id": "txn-nope", "verdict": "fraud"},
        headers=as_role("analyst"),
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Transaction txn-nope is not in the audit log."


def test_decision_rejects_an_unknown_verdict(client, trained):
    _, _, fraud_payload = trained
    client.post("/api/v1/predict", json=fraud_payload)
    bad_verdict = client.post(
        "/api/v1/decisions", json={"transaction_id": "txn-fraud", "verdict": "maybe"}, headers=as_role("analyst")
    )
    assert bad_verdict.status_code == 422


def test_a_verdict_without_a_credential_is_401(client, trained):
    _, _, fraud_payload = trained
    client.post("/api/v1/predict", json=fraud_payload)
    body = {"transaction_id": "txn-fraud", "verdict": "fraud"}
    anonymous = client.post("/api/v1/decisions", json=body)
    wrong = client.post("/api/v1/decisions", json=body, headers={"Authorization": "Bearer nope"})
    basic = client.post("/api/v1/decisions", json=body, headers={"Authorization": "Basic abc"})
    empty = client.post("/api/v1/decisions", json=body, headers={"Authorization": "Bearer "})
    assert anonymous.status_code == wrong.status_code == basic.status_code == empty.status_code == 401
    assert client.get("/api/v1/transactions?flagged=true").json()[0]["decisions"] == {"analyst": None, "customer": None}


def test_the_credential_decides_the_actor_not_the_body(client, trained):
    _, _, fraud_payload = trained
    client.post("/api/v1/predict", json=fraud_payload)
    response = client.post(
        "/api/v1/decisions",
        json={"transaction_id": "txn-fraud", "actor": "analyst", "verdict": "fraud"},
        headers=as_role("customer"),
    )
    assert response.status_code == 201
    assert response.json()["actor"] == "customer"


def test_verdicts_are_refused_until_credentials_are_configured(client, trained, monkeypatch):
    _, _, fraud_payload = trained
    client.post("/api/v1/predict", json=fraud_payload)
    monkeypatch.setitem(config.load_config()["api"], "credentials", {"analyst": "", "customer": ""})
    response = client.post(
        "/api/v1/decisions", json={"transaction_id": "txn-fraud", "verdict": "fraud"}, headers=as_role("analyst")
    )
    assert response.status_code == 503
    assert "config.local.yaml" in response.json()["detail"]


def test_listing_filters_by_flagged_and_by_card(client, trained):
    _, normal_payload, fraud_payload = trained
    client.post("/api/v1/predict", json={**normal_payload, "card_id": "card-01"})
    client.post("/api/v1/predict", json={**fraud_payload, "card_id": "card-02"})

    everything = client.get("/api/v1/transactions").json()
    only_flagged = client.get("/api/v1/transactions?flagged=true").json()
    only_cleared = client.get("/api/v1/transactions?flagged=false").json()
    one_card = client.get("/api/v1/transactions?card_id=card-02").json()

    assert {r["transaction_id"] for r in everything} == {"txn-normal", "txn-fraud"}
    assert [r["transaction_id"] for r in only_flagged] == ["txn-fraud"]
    assert [r["transaction_id"] for r in only_cleared] == ["txn-normal"]
    assert [r["card_id"] for r in one_card] == ["card-02"]


def test_stats_on_an_empty_log_is_all_zeros(client):
    body = client.get("/api/v1/stats").json()
    assert body["processed"] == 0
    assert body["flag_rate"] == 0.0
    assert body["per_minute"] == 0
    assert body["latency_ms"] == {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    assert body["decided"] == 0
    assert body["first_received_at"] is None
    assert body["cards"] == []


def test_stats_reflects_the_log_and_the_decisions(client, trained):
    _, normal_payload, fraud_payload = trained
    client.post("/api/v1/predict", json={**normal_payload, "card_id": "card-01"})
    client.post("/api/v1/predict", json={**fraud_payload, "card_id": "card-02"})
    client.post(
        "/api/v1/decisions",
        json={"transaction_id": "txn-fraud", "verdict": "fraud"},
        headers=as_role("analyst"),
    )
    client.post(
        "/api/v1/decisions",
        json={"transaction_id": "txn-fraud", "verdict": "fraud"},
        headers=as_role("customer"),
    )

    body = client.get("/api/v1/stats").json()
    assert body["processed"] == 2
    assert body["flagged"] == 1
    assert body["flag_rate"] == 0.5
    assert body["per_minute"] == 2
    assert body["latency_ms"]["p99"] >= body["latency_ms"]["p50"] > 0
    assert body["decided"] == 1
    assert body["confirmed_fraud"] == 1
    assert body["marked_legitimate"] == 0
    assert body["customer_responses"] == 1
    assert body["first_received_at"] <= body["last_received_at"]
    cards = {c["card_id"]: c for c in body["cards"]}
    assert cards["card-02"] == {"card_id": "card-02", "transactions": 1, "held": 1}
    assert cards["card-01"]["held"] == 0


def test_engine_latency_window_is_bounded(trained, monkeypatch):
    model_dir, normal_payload, _ = trained
    monkeypatch.setitem(config.load_config()["api"], "stats_window", 3)
    engine = InferenceEngine(model_dir).load()
    features = {k: v for k, v in normal_payload.items() if k != "transaction_id"}
    for _ in range(5):
        engine.score(features)
    assert len(engine._latencies) == 3
    assert engine.mean_latency_ms > 0


def test_timestamps_carry_an_explicit_utc_offset(client, trained):
    """SQLite hands naive datetimes back; the API must say they are UTC, or a
    browser in another timezone shows the wrong clock."""
    _, _, fraud_payload = trained
    client.post("/api/v1/predict", json=fraud_payload)
    stored = client.post(
        "/api/v1/decisions",
        json={"transaction_id": "txn-fraud", "verdict": "fraud"},
        headers=as_role("analyst"),
    ).json()
    row = client.get("/api/v1/transactions?flagged=true").json()[0]
    stats = client.get("/api/v1/stats").json()

    assert stored["decided_at"].endswith("+00:00")
    assert row["received_at"].endswith("+00:00")
    assert row["decisions"]["analyst"]["decided_at"].endswith("+00:00")
    assert stats["first_received_at"].endswith("+00:00")
    assert stats["last_received_at"].endswith("+00:00")


def test_a_retry_with_the_same_payload_returns_the_stored_decision_without_a_new_row(client, trained):
    _, normal_payload, _ = trained
    payload = {**normal_payload, "transaction_id": "txn-retry"}
    first = client.post("/api/v1/predict", json=payload)
    processed_after_first = db.counts()[0]
    second = client.post("/api/v1/predict", json=payload)

    assert first.status_code == 200 and second.status_code == 200
    assert second.json()["risk_score"] == first.json()["risk_score"]
    assert second.json()["received_at"] == first.json()["received_at"]
    assert db.counts()[0] == processed_after_first


def test_the_same_id_with_a_different_payload_is_refused(client, trained):
    _, normal_payload, fraud_payload = trained
    client.post("/api/v1/predict", json={**normal_payload, "transaction_id": "txn-shared"})
    clash = client.post("/api/v1/predict", json={**fraud_payload, "transaction_id": "txn-shared"})

    assert clash.status_code == 409
    assert "txn-shared" in clash.json()["detail"]
    rows = [r for r in client.get("/api/v1/transactions?limit=500").json() if r["transaction_id"] == "txn-shared"]
    assert len(rows) == 1 and rows[0]["amount"] == pytest.approx(normal_payload["Amount"])


def test_a_row_without_a_request_hash_answers_a_retry(client, trained):
    """Rows logged before the hash existed carry none; a retry of such a row
    is answered from the log rather than refused for ever."""
    _, normal_payload, _ = trained
    payload = {**normal_payload, "transaction_id": "txn-legacy"}
    client.post("/api/v1/predict", json=payload)
    with db.get_session() as session:
        row = session.query(db.TransactionLog).filter(db.TransactionLog.transaction_id == "txn-legacy").one()
        row.request_hash = None
        session.commit()
    again = client.post("/api/v1/predict", json=payload)
    assert again.status_code == 200 and again.json()["transaction_id"] == "txn-legacy"


def test_an_overlong_transaction_id_is_422(client, trained):
    _, normal_payload, _ = trained
    response = client.post("/api/v1/predict", json={**normal_payload, "transaction_id": "x" * 65})
    assert response.status_code == 422


def test_a_failing_completion_does_not_lose_the_audit_row(client, trained, monkeypatch):
    _, _, fraud_payload = trained
    engine = app.dependency_overrides[get_engine]()
    monkeypatch.setattr(engine, "explain", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("explainer down")))
    body = client.post("/api/v1/predict", json=fraud_payload).json()
    assert body["flagged"]
    row = client.get("/api/v1/transactions?flagged=true").json()[0]
    assert row["transaction_id"] == "txn-fraud" and row["explanation"] is None
    assert {s["model_name"] for s in row["scores"]} == {"isolation_forest", "xgboost"}



def test_clearing_reports_whether_every_lower_amount_clears(trained):
    model_dir, normal_payload, _ = trained
    engine = InferenceEngine(model_dir).load()

    class Bumpy:
        name = "xgboost"

        def risk_score(self, frame):
            amount = frame["Amount"].to_numpy()
            # Flagged above the scaled zero, and again in a pocket well below it.
            return np.where((amount > 0.0) | ((amount > -1.0) & (amount < -0.8)), 0.99, 0.01)

    engine.models["xgboost"] = Bumpy()
    engine.thresholds["xgboost"] = 0.5
    result = engine.score({**normal_payload, "Amount": 5_000.0}, history=[])
    clearing = engine.clearing(result["frame"], result["features"])
    assert clearing is not None and clearing["monotone"] is False
    assert engine.clearing_amount(result["frame"], result["features"]) == clearing["amount"]


def test_a_non_ascii_credential_is_401(client, trained):
    _, _, fraud_payload = trained
    client.post("/api/v1/predict", json=fraud_payload)
    response = client.post(
        "/api/v1/decisions",
        json={"transaction_id": "txn-fraud", "verdict": "fraud"},
        headers={b"Authorization": "Bearer \u00e4\u00f6".encode("utf-8")},
    )
    assert response.status_code == 401


def test_a_failing_challenger_keeps_the_explanation(client, trained, monkeypatch):
    _, _, fraud_payload = trained
    engine = app.dependency_overrides[get_engine]()
    monkeypatch.setattr(engine, "score_challengers", lambda frame: (_ for _ in ()).throw(RuntimeError("challenger down")))
    body = client.post("/api/v1/predict", json=fraud_payload).json()
    assert body["flagged"]
    row = client.get("/api/v1/transactions?flagged=true").json()[0]
    assert row["explanation"] is not None and row["explanation"]["narrative"]
    assert [s["model_name"] for s in row["scores"]] == ["xgboost"]


def test_placeholder_or_shared_credentials_are_refused(client, trained, monkeypatch):
    _, _, fraud_payload = trained
    client.post("/api/v1/predict", json=fraud_payload)
    body = {"transaction_id": "txn-fraud", "verdict": "fraud"}
    monkeypatch.setitem(config.load_config()["api"], "credentials", {
        "analyst": "replace-with-a-long-random-string", "customer": "replace-with-another-long-random-string",
    })
    placeholder = client.post("/api/v1/decisions", json=body, headers={"Authorization": "Bearer replace-with-a-long-random-string"})
    monkeypatch.setitem(config.load_config()["api"], "credentials", {"analyst": "same-token", "customer": "same-token"})
    shared = client.post("/api/v1/decisions", json=body, headers={"Authorization": "Bearer same-token"})
    assert placeholder.status_code == 503 and "placeholder" in placeholder.json()["detail"]
    assert shared.status_code == 503 and "same" in shared.json()["detail"]


def test_health_explains_why_no_models_are_served(client, monkeypatch):
    from prism.api import main as service

    monkeypatch.setattr(service, "engine_error", "No tuned threshold for autoencoder in models/ulb")
    app.dependency_overrides[get_engine_optional] = lambda: None
    body = client.get("/api/v1/health").json()
    assert body["status"] == "no_models"
    assert "autoencoder" in body["detail"]
