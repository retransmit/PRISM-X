"""Deterministic unit and integration tests for SupervisionEngine & Verification Latency Layer.

Verifies:
1. Investigator daily capacity constraint (K_daily) strictly enforced per calendar day.
2. Analyst review latency matches Weibull SLA bounds [0.5, 72.0] hours.
3. Customer chargeback dispute latency matches LogNormal bounds [3.0, 120.0] days.
4. Sigmoidal dark fraud non-reporting curve withholds micro-probes (infinite latency).
5. Point-in-time prequential zero-leakage query contract.
6. Alert queue priority triage strategies (RISK_SCORE, VALUE_AT_RISK, HYBRID).
7. ZeroLeakageDataPartitioner end-to-end decoupled feeds.
"""

import math
from typing import Any, Dict
import numpy as np

from prism_generator.stream import (
    INFERENCE_ALLOWLIST,
    InvestigationStatus,
    LabelSource,
    PriorityStrategy,
    SupervisionEngine,
    ZeroLeakageDataPartitioner,
)


def _make_dummy_tx(
    tx_id: str,
    day: int = 0,
    hour: float = 12.0,
    amount: float = 100.0,
    is_fraud: int = 0,
    risk_score: float = 0.50,
    scenario_tag: str = "ORGANIC_NORMAL",
    currency: str = "USD",
) -> Dict[str, Any]:
    """Helper to construct dummy transaction dictionary for testing."""
    t_sec = float(day * 86400 + hour * 3600)
    return {
        "transaction_id": tx_id,
        "card_id": f"CARD_{tx_id}",
        "tx_time_seconds": t_sec,
        "timestamp_utc": f"2026-03-{day+1:02d}T{int(hour):02d}:00:00Z",
        "amount": amount,
        "currency": currency,
        "is_fraud": is_fraud,
        "risk_score": risk_score,
        "scenario_tag": scenario_tag,
        "mcc": "5411",
        "merchant_name": "Test Merchant",
    }


def test_investigator_daily_capacity_enforcement():
    """Verifies that each simulation calendar day admits at most K_daily alerts,
    marking overflow as DROPPED_CAPACITY.
    """
    k_daily = 50
    engine = SupervisionEngine(k_daily=k_daily, alert_threshold=0.70, seed=42)

    # Day 0: 120 alert candidates (score >= 0.70)
    records_day0 = [
        _make_dummy_tx(f"D0_{i:03d}", day=0, hour=(i % 24), amount=50.0 + i, risk_score=0.70 + (i % 25) * 0.01)
        for i in range(120)
    ]
    # Day 1: 30 alert candidates (score >= 0.70)
    records_day1 = [
        _make_dummy_tx(f"D1_{i:03d}", day=1, hour=(i % 24), amount=100.0 + i, risk_score=0.75 + (i % 20) * 0.01)
        for i in range(30)
    ]
    # Day 2: 75 alert candidates (score >= 0.70)
    records_day2 = [
        _make_dummy_tx(f"D2_{i:03d}", day=2, hour=(i % 24), amount=20.0 + i, risk_score=0.80 + (i % 15) * 0.01)
        for i in range(75)
    ]

    all_records = records_day0 + records_day1 + records_day2
    results = engine.process_batch(all_records)

    assert len(results) == len(all_records)

    day0_results = results[:120]
    day1_results = results[120:150]
    day2_results = results[150:225]

    # Day 0: Exactly 50 investigated, 70 dropped
    d0_inv = [r for r in day0_results if r.investigation_status == InvestigationStatus.INVESTIGATED]
    d0_dropped = [r for r in day0_results if r.investigation_status == InvestigationStatus.DROPPED_CAPACITY]
    assert len(d0_inv) == k_daily
    assert len(d0_dropped) == 120 - k_daily

    # Day 1: Exactly 30 investigated, 0 dropped
    d1_inv = [r for r in day1_results if r.investigation_status == InvestigationStatus.INVESTIGATED]
    d1_dropped = [r for r in day1_results if r.investigation_status == InvestigationStatus.DROPPED_CAPACITY]
    assert len(d1_inv) == 30
    assert len(d1_dropped) == 0

    # Day 2: Exactly 50 investigated, 25 dropped
    d2_inv = [r for r in day2_results if r.investigation_status == InvestigationStatus.INVESTIGATED]
    d2_dropped = [r for r in day2_results if r.investigation_status == InvestigationStatus.DROPPED_CAPACITY]
    assert len(d2_inv) == k_daily
    assert len(d2_dropped) == 75 - k_daily

    # Test online streaming processing maintains daily budget count and resets on day boundary
    stream_engine = SupervisionEngine(k_daily=5, alert_threshold=0.70, seed=123)
    stream_res_day0 = [
        stream_engine.process_record(_make_dummy_tx(f"S0_{i}", day=0, risk_score=0.85))
        for i in range(10)
    ]
    assert sum(1 for r in stream_res_day0 if r.investigation_status == InvestigationStatus.INVESTIGATED) == 5
    assert sum(1 for r in stream_res_day0 if r.investigation_status == InvestigationStatus.DROPPED_CAPACITY) == 5

    # Day 1 stream should reset counter and allow 5 more
    stream_res_day1 = [
        stream_engine.process_record(_make_dummy_tx(f"S1_{i}", day=1, risk_score=0.85))
        for i in range(7)
    ]
    assert sum(1 for r in stream_res_day1 if r.investigation_status == InvestigationStatus.INVESTIGATED) == 5
    assert sum(1 for r in stream_res_day1 if r.investigation_status == InvestigationStatus.DROPPED_CAPACITY) == 2


def test_investigator_sla_latency_bounds():
    """Verifies that analyst review latency adheres to Weibull SLA distribution within [0.5, 72.0] hours."""
    engine = SupervisionEngine(weibull_k=1.35, weibull_scale_hours=18.0, seed=42)

    n_samples = 2000
    delays = np.array([engine._sample_investigation_delay_hours() for _ in range(n_samples)])

    # 1. Strict boundary containment
    assert np.all(delays >= 0.5), f"Found delays < 0.5h: {delays[delays < 0.5]}"
    assert np.all(delays <= 72.0), f"Found delays > 72.0h: {delays[delays > 72.0]}"

    # 2. Statistical properties (Dal Pozzolo et al. 2018: median ~13.73h, mean ~16.53h)
    median_val = float(np.median(delays))
    mean_val = float(np.mean(delays))

    assert 11.0 <= median_val <= 16.5, f"Unexpected median review latency: {median_val:.2f}h"
    assert 14.0 <= mean_val <= 19.5, f"Unexpected mean review latency: {mean_val:.2f}h"

    # 3. Interquartile concentration: vast majority between 4 and 36 hours
    fraction_midrange = np.mean((delays >= 4.0) & (delays <= 36.0))
    assert fraction_midrange > 0.70, f"Expected >70% in [4h, 36h], got {fraction_midrange:.2%}"


def test_chargeback_dispute_latency_bounds():
    """Verifies customer chargeback dispute maturity latency adheres to LogNormal distribution within [3.0, 120.0] days."""
    engine = SupervisionEngine(lognormal_mu_days=3.40, lognormal_sigma=0.45, seed=42)

    n_samples = 2000
    delays = np.array([engine._sample_chargeback_delay_days() for _ in range(n_samples)])

    # 1. Scheme rules bounds: [3 days, 120 days] (Visa VCR / Mastercard MasterCom statutory window)
    assert np.all(delays >= 3.0), f"Found dispute delays < 3.0 days: {delays[delays < 3.0]}"
    assert np.all(delays <= 120.0), f"Found dispute delays > 120.0 days: {delays[delays > 120.0]}"

    # 2. Statistical properties: median ~ exp(3.40) = 29.96 days, mean ~ 33.2 days
    median_val = float(np.median(delays))
    mean_val = float(np.mean(delays))

    assert 26.0 <= median_val <= 34.0, f"Unexpected median dispute latency: {median_val:.2f} days"
    assert 29.0 <= mean_val <= 38.0, f"Unexpected mean dispute latency: {mean_val:.2f} days"


def test_dark_fraud_withholding_and_micro_probe_elevation():
    """Verifies decaying sigmoidal non-reporting curve withholds micro-probing fraud
    as dark fraud (discovery_time = inf, discovered_label = None).
    """
    engine = SupervisionEngine(alert_threshold=0.99, seed=42)  # High alert threshold so all go to uninvestigated

    # 1. Micro-probes (amounts <= $1.00 or micro scenario)
    micro_records = [
        _make_dummy_tx(f"MICRO_{i:03d}", amount=0.75, is_fraud=1, risk_score=0.40, scenario_tag="ADV_MICRO_AUTH_PROBE")
        for i in range(200)
    ]
    micro_results = engine.process_batch(micro_records)

    dark_micro = [r for r in micro_results if r.label_source == LabelSource.UNREPORTED_DARK_FRAUD]
    assert len(dark_micro) / len(micro_records) >= 0.95, (
        f"Micro probes dark rate {len(dark_micro)/len(micro_records):.2%} below 95%"
    )

    # For dark fraud records, discovery time must be infinite, label None, and point-in-time check returns None
    for r in dark_micro:
        assert math.isinf(r.discovery_time_seconds)
        assert r.discovered_label is None
        assert r.is_label_available_at(query_time_seconds=1e12) is None

    # 2. Large fraudulent transactions ($5,000)
    large_records = [
        _make_dummy_tx(f"LARGE_{i:03d}", amount=5000.0, is_fraud=1, risk_score=0.40, scenario_tag="ADV_ACCOUNT_TAKEOVER")
        for i in range(200)
    ]
    large_results = engine.process_batch(large_records)

    dark_large = [r for r in large_results if r.label_source == LabelSource.UNREPORTED_DARK_FRAUD]
    # For $5,000, non-reporting probability is 1 / (1 + (5000/15)^2.5) ~ 0.0001
    assert len(dark_large) <= 2, f"Too many large frauds marked dark: {len(dark_large)}"

    cb_large = [r for r in large_results if r.label_source == LabelSource.CHARGEBACK_DISPUTE]
    assert len(cb_large) >= 198
    for r in cb_large:
        assert r.discovered_label == 1
        assert not math.isinf(r.discovery_time_seconds)
        assert r.discovery_time_seconds > r.tx_time_seconds


def test_prequential_zero_leakage_point_in_time():
    """Verifies that SupervisionRecord.is_label_available_at strictly prevents future temporal leakage."""
    engine = SupervisionEngine(k_daily=10, alert_threshold=0.70, seed=42)

    # Record 1: Investigated fraud alert (discovery in ~18 hours)
    r_inv = _make_dummy_tx("TX_INV_01", day=0, hour=10.0, amount=200.0, is_fraud=1, risk_score=0.85)
    # Record 2: Unflagged fraud disputed via chargeback (discovery in ~30 days)
    r_cb = _make_dummy_tx("TX_CB_01", day=0, hour=10.0, amount=200.0, is_fraud=1, risk_score=0.20)
    # Record 3: Micro-probe dark fraud (discovery never)
    r_dark = _make_dummy_tx("TX_DARK_01", day=0, hour=10.0, amount=0.50, is_fraud=1, risk_score=0.10, scenario_tag="ADV_MICRO_AUTH_PROBE")
    # Record 4: Legitimate uninvestigated transaction (clean maturity in 90 days)
    r_legit = _make_dummy_tx("TX_LEGIT_01", day=0, hour=10.0, amount=50.0, is_fraud=0, risk_score=0.10)

    sups = engine.process_batch([r_inv, r_cb, r_dark, r_legit])
    sup_inv, sup_cb, sup_dark, sup_legit = sups

    # Verify sup_inv
    assert sup_inv.investigation_status == InvestigationStatus.INVESTIGATED
    assert sup_inv.label_source == LabelSource.INVESTIGATOR_ALERT
    t_tx = sup_inv.tx_time_seconds
    t_disc_inv = sup_inv.discovery_time_seconds
    assert t_disc_inv > t_tx
    # Prior to discovery time: None
    assert sup_inv.is_label_available_at(t_tx) is None
    assert sup_inv.is_label_available_at(t_disc_inv - 1.0) is None
    # At or after discovery time: 1
    assert sup_inv.is_label_available_at(t_disc_inv) == 1
    assert sup_inv.is_label_available_at(t_disc_inv + 1000.0) == 1

    # Verify sup_cb
    assert sup_cb.label_source == LabelSource.CHARGEBACK_DISPUTE
    t_disc_cb = sup_cb.discovery_time_seconds
    assert t_disc_cb >= t_tx + 3.0 * 86400
    assert sup_cb.is_label_available_at(t_tx) is None
    assert sup_cb.is_label_available_at(t_disc_cb - 1.0) is None
    assert sup_cb.is_label_available_at(t_disc_cb) == 1

    # Verify sup_dark
    assert sup_dark.label_source == LabelSource.UNREPORTED_DARK_FRAUD
    assert sup_dark.is_label_available_at(t_tx) is None
    assert sup_dark.is_label_available_at(t_tx + 365 * 86400) is None

    # Verify sup_legit (90-day maturity)
    assert sup_legit.label_source == LabelSource.UNLABELLED
    t_disc_legit = sup_legit.discovery_time_seconds
    assert t_disc_legit == t_tx + 90.0 * 86400
    assert sup_legit.is_label_available_at(t_disc_legit - 1.0) is None
    assert sup_legit.is_label_available_at(t_disc_legit) == 0


def test_priority_queue_value_at_risk_triage():
    """Verifies that priority triage strategies (RISK_SCORE, VALUE_AT_RISK, HYBRID)
    rank alerts as specified under capacity constraint K_daily.
    """
    # 4 candidates on Day 0
    c1 = _make_dummy_tx("TX1", day=0, amount=10.0, risk_score=0.95)      # VaR = 9.5
    c2 = _make_dummy_tx("TX2", day=0, amount=20.0, risk_score=0.90)      # VaR = 18.0
    c3 = _make_dummy_tx("TX3", day=0, amount=5000.0, risk_score=0.75)    # VaR = 3750.0
    c4 = _make_dummy_tx("TX4", day=0, amount=10000.0, risk_score=0.72)   # VaR = 7200.0
    candidates = [c1, c2, c3, c4]

    # Strategy 1: RISK_SCORE with K_daily = 2
    # Expect c1 (0.95) and c2 (0.90) to be investigated, c3 and c4 dropped
    engine_score = SupervisionEngine(k_daily=2, alert_threshold=0.70, priority_strategy=PriorityStrategy.RISK_SCORE)
    res_score = engine_score.process_batch(candidates)
    status_map_score = {r.transaction_id: r.investigation_status for r in res_score}
    assert status_map_score["TX1"] == InvestigationStatus.INVESTIGATED
    assert status_map_score["TX2"] == InvestigationStatus.INVESTIGATED
    assert status_map_score["TX3"] == InvestigationStatus.DROPPED_CAPACITY
    assert status_map_score["TX4"] == InvestigationStatus.DROPPED_CAPACITY

    # Strategy 2: VALUE_AT_RISK with K_daily = 2
    # Expect c4 (VaR 7200) and c3 (VaR 3750) to be investigated, c1 and c2 dropped
    engine_var = SupervisionEngine(k_daily=2, alert_threshold=0.70, priority_strategy=PriorityStrategy.VALUE_AT_RISK)
    res_var = engine_var.process_batch(candidates)
    status_map_var = {r.transaction_id: r.investigation_status for r in res_var}
    assert status_map_var["TX4"] == InvestigationStatus.INVESTIGATED
    assert status_map_var["TX3"] == InvestigationStatus.INVESTIGATED
    assert status_map_var["TX1"] == InvestigationStatus.DROPPED_CAPACITY
    assert status_map_var["TX2"] == InvestigationStatus.DROPPED_CAPACITY

    # Strategy 3: HYBRID priority score verification: 0.60 * score + 0.40 * min(1.0, amount / 1000)
    engine_hybrid = SupervisionEngine(priority_strategy=PriorityStrategy.HYBRID)
    p_c1 = engine_hybrid.compute_priority_score(0.95, 10.0)
    assert abs(p_c1 - (0.60 * 0.95 + 0.40 * (10.0 / 1000.0))) < 1e-5

    p_c4 = engine_hybrid.compute_priority_score(0.72, 10000.0)
    assert abs(p_c4 - (0.60 * 0.72 + 0.40 * 1.0)) < 1e-5


def test_zero_leakage_data_partitioner_supervision_integration():
    """Verifies ZeroLeakageDataPartitioner cleanly isolates inference feed from supervision metadata."""
    partitioner = ZeroLeakageDataPartitioner(k_daily=50, alert_threshold=0.70, seed=42)

    records = [
        _make_dummy_tx(
            f"TX_{i:03d}",
            day=i // 30,
            hour=(i % 24),
            amount=25.0 * (i + 1),
            is_fraud=(1 if i % 10 == 0 else 0),
            risk_score=(0.85 if i % 10 == 0 else 0.15),
            scenario_tag=("ADV_TEST" if i % 10 == 0 else "ORGANIC_NORMAL"),
        )
        for i in range(100)
    ]

    inf_batch, lbl_batch, graph_batch = partitioner.partition_batch(records)

    assert len(inf_batch) == 100
    assert len(lbl_batch) == 100
    assert len(graph_batch) == 100

    # Strictly enforce zero target leakage in inference feed
    forbidden_inference_keys = {
        "is_fraud",
        "is_fraud_ground_truth",
        "label_source",
        "investigation_status",
        "discovery_time_seconds",
        "discovery_timestamp_utc",
        "discovered_label",
        "scenario_tag",
        "syndicate_id",
        "mule_ring_id",
        "botnet_cluster_id",
    }

    for inf_rec in inf_batch:
        for k in inf_rec.keys():
            assert k in INFERENCE_ALLOWLIST, f"Key {k} not in INFERENCE_ALLOWLIST"
            assert k not in forbidden_inference_keys, f"Leakage detected: key {k} present in inference feed"

    # Verify delayed labels contain supervision metadata
    for lbl_rec in lbl_batch:
        assert "investigation_status" in lbl_rec
        assert "label_source" in lbl_rec
        assert "discovery_time_seconds" in lbl_rec
        assert "is_fraud_ground_truth" in lbl_rec
        assert lbl_rec["label_source"] in [e.value for e in LabelSource]
        assert lbl_rec["investigation_status"] in [e.value for e in InvestigationStatus]

    # Verify single record streaming partitioning
    single_r = records[0]
    inf, lbl, grp = partitioner.partition_record(single_r)
    assert "is_fraud" not in inf
    assert "label_source" in lbl
    assert "syndicate_id" in grp
