"""The streaming daemon posts what an authorisation request carries and nothing else; the
labels travel on a feed of their own, each released only once the bank would know it.

Before this change the daemon posted every record field, including is_fraud, scenario_tag,
the syndicate identifiers and the generator's own risk score, and only generated US traffic.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from prism_generator import SimulationEngine
from prism_generator.cli import AUTH_STREAM_COLUMNS, GATEWAY_TELEMETRY_COLUMNS
from prism_generator.stream import (
    AUTHORISATION_OUTCOME_COLUMNS,
    LABEL_FIELDS,
    REQUEST_FIELDS,
    LabelFeed,
    SupervisionEngine,
    label_payload,
    request_payload,
    run_stream_daemon,
)

NEVER_IN_A_REQUEST = {
    "is_fraud", "scenario_tag", "risk_score", "base_risk", "hawkes_intensity_R",
    "syndicate_id", "botnet_cluster_id", "mule_ring_id", "beneficiary_account_id",
    "ip_subnet_prefix", "device_fingerprint_id", "asn", "isp", "ja4_signature", "mule_tier",
    "liquidation_channel", "dominant_causal_driver", "analytical_shapley_probability",
    "analytical_shapley_log_odds", "counterfactual_input_deltas", "counterfactual_mode",
    "counterfactual_twin", "normative_baseline", "explanation_narrative", "hop_origin",
    "response_code", "auth_response_code", "auth_code", "clearing_mti", "clearing_delay_hours",
    "settled_amount", "settled_amount_minor", "interchange_fee_minor", "dispute_status",
    "dispute_reason_code", "ce3_qualified", "arbitration_fee_usd", "rbi_liability_tier",
    "rbi_provisional_credit_mandate_days", "cfcfrms_1930_lien_status",
}


def _records(region: str = "IN", n: int = 400, fraud_prevalence: float = 0.10):
    engine = SimulationEngine(n_cards=120, n_merchants=40, region=region, seed=11)
    return engine.generate_batch(n_transactions=n, fraud_prevalence=fraud_prevalence, time_span_days=3)


def test_request_fields_are_the_two_contract_views_minus_the_authorisation_outcome():
    contract = set(AUTH_STREAM_COLUMNS) | set(GATEWAY_TELEMETRY_COLUMNS)
    assert set(REQUEST_FIELDS) == contract - set(AUTHORISATION_OUTCOME_COLUMNS)
    assert set(AUTHORISATION_OUTCOME_COLUMNS) == {"response_code", "auth_response_code", "auth_code"}
    assert not (set(REQUEST_FIELDS) & NEVER_IN_A_REQUEST)
    assert len(REQUEST_FIELDS) == len(set(REQUEST_FIELDS))


def test_a_request_payload_carries_request_fields_only():
    records = _records()
    assert any(r["is_fraud"] == 1 for r in records)
    for record in records:
        payload = request_payload(record)
        assert set(payload) <= set(REQUEST_FIELDS)
        assert not (set(payload) & NEVER_IN_A_REQUEST)
        assert payload["transaction_id"] == record["transaction_id"]
        json.dumps(payload)


def test_a_label_payload_says_what_the_bank_learned_and_when():
    records = _records()
    supervision = SupervisionEngine(seed=1)
    labels = [label_payload(s) for s in supervision.process_batch(records)]
    assert all(set(l) == set(LABEL_FIELDS) for l in labels)
    assert {l["discovered_label"] for l in labels} <= {0, 1, None}
    assert all("risk_score" not in l and "is_fraud_ground_truth" not in l for l in labels)


def test_the_label_feed_releases_labels_in_discovery_order_and_never_early():
    records = _records()
    supervision = SupervisionEngine(seed=1, alert_threshold=0.30)
    feed = LabelFeed()
    for s in supervision.process_batch(records):
        feed.push(s)
    finite = feed.pending
    assert finite >= 1
    released = []
    clock = min(float(r["tx_time_seconds"]) for r in records)
    horizon = clock + 400 * 86400.0
    while clock < horizon:
        for item in feed.due(clock):
            assert item["discovery_time_seconds"] <= clock
            released.append(item)
        clock += 6 * 3600.0
    times = [item["discovery_time_seconds"] for item in released]
    assert times == sorted(times)
    assert len(released) == finite and feed.pending == 0
    assert all(item["label_source"] != "UNREPORTED_DARK_FRAUD" for item in released)


def test_the_daemon_posts_requests_and_labels_on_separate_feeds():
    requests, labels = [], []

    async def take_request(payload):
        requests.append(payload)
        return True

    async def take_label(payload):
        labels.append(payload)
        return True

    asyncio.run(run_stream_daemon(
        duration_sec=2.0, target_tps=100.0, fraud_prevalence=0.10, seed=5, region="IN",
        n_cards=80, n_merchants=30, flush_labels=True,
        emit_request=take_request, emit_label=take_label,
    ))
    assert len(requests) >= 150
    assert all(not (set(p) & NEVER_IN_A_REQUEST) for p in requests)
    assert all(set(p) <= set(REQUEST_FIELDS) for p in requests)
    assert all(p["currency"] == "INR" for p in requests)
    posted = {p["transaction_id"] for p in requests}
    assert labels, "with flush_labels every label the bank would ever learn is released at the end"
    assert all(set(l) == set(LABEL_FIELDS) for l in labels)
    assert {l["transaction_id"] for l in labels} <= posted
    times = [l["discovery_time_seconds"] for l in labels]
    assert times == sorted(times)


def test_the_daemon_takes_a_calibration_profile():
    requests = []

    async def take_request(payload):
        requests.append(payload)
        return True

    asyncio.run(run_stream_daemon(
        duration_sec=2.0, target_tps=100.0, seed=5, region="IN", calibration="rbi-psi-2026-07",
        n_cards=80, n_merchants=30, emit_request=take_request,
    ))
    assert len(requests) >= 150 and all(p["currency"] == "INR" for p in requests)
    with pytest.raises(ValueError):
        asyncio.run(run_stream_daemon(
            duration_sec=1.0, target_tps=10.0, seed=5, region="US", calibration="rbi-psi-2026-07",
            n_cards=20, n_merchants=10, emit_request=take_request,
        ))


def test_the_offline_partitioner_feeds_a_detector_the_request_fields_only():
    """The partitioner's inference feed is the same thing the daemon posts: no authorisation
    outcome, no settlement, no latent intensity."""
    from prism_generator.stream import INFERENCE_ALLOWLIST, ZeroLeakageDataPartitioner

    assert set(INFERENCE_ALLOWLIST) == set(REQUEST_FIELDS)
    records = _records()
    inference, labels, graph = ZeroLeakageDataPartitioner(seed=1).partition_batch(records)
    assert len(inference) == len(records)
    for row in inference:
        assert set(row) <= set(REQUEST_FIELDS)
        assert not (set(row) & NEVER_IN_A_REQUEST)
    one_inference, one_labels, one_graph = ZeroLeakageDataPartitioner(seed=1).partition_record(records[0])
    assert set(one_inference) <= set(REQUEST_FIELDS)


def test_the_detectors_answer_decides_what_gets_investigated():
    """When the endpoint answers a request with a score, that score, not the generator's
    reference score, feeds the investigation queue: a detector that scores everything at
    0.99 has every row investigated; one that scores everything at 0.0 has no alert at all."""
    def daemon(answer):
        labels = []

        async def take_request(payload):
            return answer

        async def take_label(payload):
            labels.append(payload)
            return True

        counts = asyncio.run(run_stream_daemon(
            duration_sec=2.0, target_tps=100.0, fraud_prevalence=0.10, seed=5, region="IN",
            n_cards=80, n_merchants=30, flush_labels=True, k_daily=10_000, alert_threshold=0.5,
            emit_request=take_request, emit_label=take_label,
        ))
        return counts, labels

    counts, labels = daemon({"risk_score": 0.99})
    assert counts["labels_from_detector"] == counts["requests_sent"] > 0
    assert labels and all(l["label_source"] == "INVESTIGATOR_ALERT" for l in labels)

    counts, labels = daemon({"risk_score": 0.0})
    assert counts["labels_from_detector"] == counts["requests_sent"]
    assert all(l["label_source"] != "INVESTIGATOR_ALERT" for l in labels)

    counts, labels = daemon(True)
    assert counts["labels_from_detector"] == 0, "an answer without a score leaves the generator's reference score in charge"
