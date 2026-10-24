"""The audit log keeps what was scored, and labels reach it as the bank learns them."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("xgboost")

from sqlalchemy import create_engine, text  # noqa: E402

from prism import config, tracks  # noqa: E402
from prism.api import db, schemas  # noqa: E402
from prism.api.inference import InferenceEngine  # noqa: E402
from prism.data import india  # noqa: E402
from tests._india_service import as_role, build_deployment, payload_from, service, use_temp_database  # noqa: E402


@pytest.fixture(scope="module")
def deployment(tmp_path_factory, india_stem):
    return build_deployment(india_stem, tmp_path_factory.mktemp("live_models"))


@pytest.fixture(scope="module")
def rows(india_stem):
    return india.load_india(india_stem)


@pytest.fixture(autouse=True)
def temp_database(tmp_path, monkeypatch):
    return use_temp_database(tmp_path, monkeypatch)


@pytest.fixture
def client(deployment):
    engine = InferenceEngine(deployment, track="india").load()
    with service(engine) as test_client:
        yield test_client


def test_a_scored_request_is_kept_with_its_inputs(client, rows):
    body = payload_from(rows.iloc[0])
    assert client.post("/api/v1/predict", json=body).status_code == 200
    logged = db.logged_requests()
    assert len(logged) == 1
    kept = logged[0]
    assert kept["transaction_id"] == body["transaction_id"]
    assert kept["amount"] == body["amount"]
    assert kept["card_id"] == body["card_id"]
    assert kept["mcc"] == body["mcc"]
    assert kept["merchant"] == body["merchant"]
    assert "received_at" in kept


def test_an_old_log_gains_the_request_column(tmp_path):
    url = f"sqlite:///{(tmp_path / 'old.db').as_posix()}"
    old = create_engine(url)
    with old.begin() as conn:
        conn.execute(text(
            "CREATE TABLE transaction_log (id INTEGER PRIMARY KEY, transaction_id VARCHAR(64), "
            "received_at DATETIME, amount FLOAT, time_feature FLOAT, primary_model VARCHAR(32), "
            "risk_score FLOAT, flagged BOOLEAN, latency_ms FLOAT, scores_json TEXT, explanation_json TEXT)"
        ))
        conn.execute(text(
            "INSERT INTO transaction_log (transaction_id, received_at, amount, time_feature, primary_model, "
            "risk_score, flagged, latency_ms, scores_json) "
            "VALUES ('old-1', '2026-01-01 00:00:00', 1.0, 5.0, 'xgboost', 0.1, 0, 1.0, '[]')"
        ))
    old.dispose()
    db.init_db(url)
    with db.get_session() as session:
        value = session.execute(text("SELECT request_json FROM transaction_log")).scalar_one()
    assert value is None
    assert db.logged_requests() == []
    assert db.stream_clock() == 5.0


def label_body(transaction_id: str, label: int = 1, source: str = "CHARGEBACK_DISPUTE", at: float = 10.0) -> dict:
    """A label as the generator's feed sends it, extra fields included."""
    return {
        "transaction_id": transaction_id, "card_id": "CARD_000001", "tx_timestamp_utc": "2024-01-01T00:00:00Z",
        "discovered_label": label, "label_source": source, "discovery_time_seconds": at,
        "discovery_timestamp_utc": "2024-01-01T00:00:10Z",
    }


def scored(client, rows, n: int = 1, times: list[float] | None = None) -> list[str]:
    ids = []
    for i in range(n):
        body = payload_from(rows.iloc[i])
        if times is not None:
            body["tx_time_seconds"] = times[i]
        assert client.post("/api/v1/predict", json=body).status_code == 200
        ids.append(body["transaction_id"])
    return ids


def test_a_label_is_stored_once_and_a_repeat_answers_the_stored_row(client, rows):
    (txn,) = scored(client, rows)
    first = client.post("/api/v1/labels", json=label_body(txn), headers=as_role("label_feed"))
    again = client.post("/api/v1/labels", json=label_body(txn), headers=as_role("label_feed"))
    assert first.status_code == 201 and again.status_code == 200
    assert first.json() == again.json()
    assert first.json()["source"] == "CHARGEBACK_DISPUTE" and first.json()["label"] == 1
    assert db.label_counts() == {"CHARGEBACK_DISPUTE": 1}


def test_a_new_label_of_one_source_is_a_new_row_and_the_latest_wins(client, rows):
    (txn,) = scored(client, rows)
    client.post("/api/v1/labels", json=label_body(txn, 1, "INVESTIGATOR_ALERT", 10.0), headers=as_role("label_feed"))
    changed = client.post("/api/v1/labels", json=label_body(txn, 0, "INVESTIGATOR_ALERT", 20.0), headers=as_role("label_feed"))
    assert changed.status_code == 201
    assert db.label_counts() == {"INVESTIGATOR_ALERT": 2}
    assert db.known_labels(15.0) == {txn: 1}
    assert db.known_labels(30.0) == {txn: 0}


def test_a_label_for_an_unscored_transaction_is_refused(client):
    response = client.post("/api/v1/labels", json=label_body("TX_NEVER_SEEN"), headers=as_role("label_feed"))
    assert response.status_code == 404
    assert "TX_NEVER_SEEN" in response.json()["detail"]
    assert db.label_counts() == {}


def test_the_label_feed_credential_cannot_post_verdicts_and_the_analyst_cannot_post_labels(client, rows):
    (txn,) = scored(client, rows)
    verdict = client.post(
        "/api/v1/decisions", json={"transaction_id": txn, "verdict": "fraud"}, headers=as_role("label_feed")
    )
    label = client.post("/api/v1/labels", json=label_body(txn), headers=as_role("analyst"))
    anonymous = client.post("/api/v1/labels", json=label_body(txn))
    assert verdict.status_code == label.status_code == anonymous.status_code == 401
    assert db.label_counts() == {}


def test_labels_are_refused_until_the_feed_credential_is_configured(client, rows, monkeypatch):
    (txn,) = scored(client, rows)
    monkeypatch.setitem(config.load_config()["api"], "credentials", {"analyst": "a-secret", "customer": "c-secret"})
    response = client.post("/api/v1/labels", json=label_body(txn), headers={"Authorization": "Bearer a-secret"})
    assert response.status_code == 503
    assert "label_feed" in response.json()["detail"]


def test_precedence_keeps_a_chargeback_over_a_later_verdict(client, rows):
    (txn,) = scored(client, rows)
    db.record_label(transaction_id=txn, label=1, source="CHARGEBACK_DISPUTE", known_at=10.0)
    db.record_label(transaction_id=txn, label=0, source="analyst", known_at=20.0)
    assert db.known_labels(30.0) == {txn: 1}
    assert db.known_labels(15.0) == {txn: 1}


def test_a_verdict_outranks_the_investigator_and_the_cardholder(client, rows):
    (txn,) = scored(client, rows)
    db.record_label(transaction_id=txn, label=1, source="customer", known_at=5.0)
    db.record_label(transaction_id=txn, label=0, source="analyst", known_at=6.0)
    db.record_label(transaction_id=txn, label=1, source="INVESTIGATOR_ALERT", known_at=7.0)
    assert db.known_labels(10.0) == {txn: 0}


def test_a_label_is_not_known_before_its_discovery(client, rows):
    (txn,) = scored(client, rows)
    db.record_label(transaction_id=txn, label=1, source="CHARGEBACK_DISPUTE", known_at=10.0)
    assert db.known_labels(9.0) == {}
    assert db.known_labels(10.0) == {txn: 1}


def test_a_verdict_becomes_a_label_known_at_the_stream_clock(client, rows):
    first, _ = scored(client, rows, n=2, times=[100.0, 200.0])
    response = client.post(
        "/api/v1/decisions", json={"transaction_id": first, "verdict": "fraud"}, headers=as_role("analyst")
    )
    assert response.status_code == 201
    with db.get_session() as session:
        labels = session.query(db.Label).all()
    assert [(row.transaction_id, row.label, row.source, row.known_at) for row in labels] == [
        (first, 1, "analyst", 200.0)
    ]


def test_the_label_route_answers_404_off_the_indian_track(client, rows):
    (txn,) = scored(client, rows)
    schemas.set_active_schema(tracks.ULB)
    response = client.post("/api/v1/labels", json=label_body(txn), headers=as_role("label_feed"))
    assert response.status_code == 404
    assert "Indian track" in response.json()["detail"]
