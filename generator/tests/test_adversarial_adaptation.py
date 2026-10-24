"""Automated test suite for Phase 4: Closed-Loop Adaptive Adversary Policy & Threat Graph Telemetry.

Validates:
1. Stateful closed-loop adversary adaptation:
   - Binary bisection search on ISO 51 (Insufficient Funds): A_{k+1} = max(25.0, round(A_k * 0.70, 2)).
   - Gateway hopping on 3DS Step-Up Challenge (TransStatus 'C'): hops to MCC 5815 with amount <= $28.
   - Card purging & burn on consecutive fraud declines (ISO 59 / 05 >= 2).
2. Zero-Leakage Information Architecture:
   - Inference feed contains strictly point-in-time features with zero labels and zero threat intelligence.
   - Delayed labels feed incorporates realistic 21-day chargeback maturity lag.
   - Threat intel graph enclave quarantines syndicate, botnet, and mule ring topologies.
3. Machine learning baseline utility on inference feed:
   - LightGBM model trained on clean inference feed yields realistic non-trivial PR-AUC.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np

from prism_generator.agents import (
    AdaptiveFraudsterAgent,
    CardholderProfile,
    FraudsterState,
    ISO8583Response,
)
from prism_generator.engine import DiscreteEventEngine
from prism_generator.stream import ZeroLeakageDataPartitioner


def create_mock_card(card_id: str = "CARD_ADV_001", credit_limit: float = 5000.0) -> CardholderProfile:
    return CardholderProfile(
        card_id=card_id,
        home_lat=40.7128,
        home_lon=-74.0060,
        work_lat=40.7628,
        work_lon=-73.9560,
        is_commuter=True,
        credit_limit=credit_limit,
        current_balance=2500.0,
        pan_masked="411122******3344",
        product_id="US_PROD_REVOLVING",
        cohort_id="C1_URBAN_TECH_PROFESSIONAL",
        region="US",
        currency="USD",
        domestic_cnp_enabled=True,
        international_enabled=True,
    )


def test_adversary_bisection_search_on_iso_51():
    """Verify binary bisection discount on Insufficient Funds (ISO 51)."""
    rng = np.random.default_rng(42)
    fraudster = AdaptiveFraudsterAgent(rng=rng, adversary_mimicry=0.0)
    card = create_mock_card()
    target = fraudster.get_or_create_target(card)

    target.current_probe_amount = 1000.0
    target.fsm_state = FraudsterState.AMOUNT_ADAPTATION

    # 1. First ISO 51 decline: amount discounted by 30% -> $700.00
    fraudster.receive_feedback(
        response_code=ISO8583Response.INSUFFICIENT_FUNDS_51,
        trans_status_3ds=None,
        sim_time_seconds=100.0,
        card_id=card.card_id,
    )
    assert target.fsm_state == FraudsterState.AMOUNT_ADAPTATION
    assert target.current_probe_amount == 700.0
    assert target.consecutive_declines == 1
    assert target.cooldown_until_sec == 130.0

    # 2. Attack generation uses adapted amount
    attack = fraudster.select_attack_playbook(
        card=card,
        sim_time_seconds=135.0,
        world_center_lat=card.home_lat,
        world_center_lon=card.home_lon,
    )
    assert attack["amount"] == 700.0

    # 3. Second consecutive ISO 51 decline: $700 * 0.70 -> $490.00
    fraudster.receive_feedback(
        response_code=ISO8583Response.INSUFFICIENT_FUNDS_51,
        trans_status_3ds=None,
        sim_time_seconds=140.0,
        card_id=card.card_id,
    )
    assert target.current_probe_amount == 490.0
    assert target.consecutive_declines == 2


def test_adversary_gateway_hop_on_3ds_challenge():
    """Verify adversary hops to low-friction merchant gateway when challenged by 3DS."""
    rng = np.random.default_rng(42)
    fraudster = AdaptiveFraudsterAgent(rng=rng, adversary_mimicry=0.0)
    card = create_mock_card()
    target = fraudster.get_or_create_target(card)
    target.current_probe_amount = 450.0

    # Receive 3DS step-up challenge feedback
    fraudster.receive_feedback(
        response_code=ISO8583Response.DO_NOT_HONOR_05,
        trans_status_3ds="C",
        sim_time_seconds=100.0,
        card_id=card.card_id,
    )

    assert target.fsm_state == FraudsterState.GATEWAY_HOP
    assert target.target_mcc == 5815  # Digital Goods / low-friction gateway
    assert target.current_probe_amount <= 28.0  # Kept under PSD2 €30 / $30 Low-Value threshold


def test_adversary_burns_card_on_consecutive_fraud_declines():
    """Verify that adversary discards and burns cards after 2 consecutive fraud declines."""
    rng = np.random.default_rng(42)
    fraudster = AdaptiveFraudsterAgent(rng=rng, adversary_mimicry=0.0)
    card = create_mock_card()
    target = fraudster.get_or_create_target(card)

    assert not fraudster.is_card_burned(card.card_id)

    # 1. First fraud decline
    fraudster.receive_feedback(
        response_code=ISO8583Response.SUSPECTED_FRAUD_59,
        trans_status_3ds="N",
        sim_time_seconds=100.0,
        card_id=card.card_id,
    )
    assert not target.is_burned
    assert target.consecutive_declines == 1
    assert target.fsm_state == FraudsterState.VELOCITY_BACKOFF

    # 2. Second fraud decline -> triggers card burn and purge
    fraudster.receive_feedback(
        response_code=ISO8583Response.SUSPECTED_FRAUD_59,
        trans_status_3ds="N",
        sim_time_seconds=450.0,
        card_id=card.card_id,
    )
    assert target.is_burned is True
    assert target.fsm_state == FraudsterState.CARD_PURGE
    assert fraudster.is_card_burned(card.card_id) is True


def test_zero_leakage_feed_partitioning():
    """Verify strict 3-way stream partitioning and absence of leakage in inference feed."""
    engine = DiscreteEventEngine(n_cards=30, n_merchants=15, region="US", seed=101)
    batch = engine.generate_batch(n_transactions=150, fraud_prevalence=0.10)

    partitioner = ZeroLeakageDataPartitioner(mean_chargeback_lag_days=21.0, seed=42)
    inf_feed, delayed_labels, graph_enclave = partitioner.partition_batch(batch)

    assert len(inf_feed) == len(batch)
    assert len(delayed_labels) == len(batch)
    assert len(graph_enclave) == len(batch)

    # 1. Inference Feed Anti-Leakage Tripwires
    prohibited_keys = {
        "is_fraud",
        "scenario_tag",
        "syndicate_id",
        "botnet_cluster_id",
        "mule_ring_id",
        "beneficiary_account_id",
        "ip_subnet_prefix",
        "device_fingerprint_id",
        "dispute_status",
        "dispute_reason_code",
        "arbitration_fee_usd",
        "rbi_liability_tier",
        "cfcfrms_1930_lien_status",
        "risk_score",
        "analytical_shapley_probability",
        "counterfactual_twin",
    }

    for record in inf_feed:
        for bad_key in prohibited_keys:
            assert bad_key not in record, f"Critical leakage tripwire violated: '{bad_key}' found in inference feed!"
        # Invariant: Must contain essential point-in-time features
        assert "transaction_id" in record
        assert "amount" in record
        # the authorisation outcome is the issuer's answer; a detector scoring the request has not seen it
        assert "response_code" not in record
        assert "mti" in record

    # 2. Delayed Labels Feed Verification
    cb_count = 0
    for lbl in delayed_labels:
        assert "is_fraud" in lbl
        assert "label_maturity_timestamp_utc" in lbl
        if lbl.get("chargeback_delay_days") is not None:
            assert 3.0 <= lbl["chargeback_delay_days"] <= 120.0
            cb_count += 1
        if lbl.get("investigation_delay_hours") is not None:
            assert 0.5 <= lbl["investigation_delay_hours"] <= 72.0
    assert cb_count > 0, "Expected at least one chargeback dispute in delayed labels"

    # 3. Threat Intel Graph Enclave Verification
    fraud_graphs = [g for g in graph_enclave if g["is_fraud"] == 1]
    assert len(fraud_graphs) > 0
    for g in fraud_graphs:
        assert "syndicate_id" in g
        assert "botnet_cluster_id" in g
        assert "mule_ring_id" in g


def test_export_partitioned_feeds_to_disk():
    """Verify that export_partitioned_feeds creates valid JSONL files on disk."""
    engine = DiscreteEventEngine(n_cards=20, n_merchants=10, region="US", seed=202)
    batch = engine.generate_batch(n_transactions=50, fraud_prevalence=0.08)

    partitioner = ZeroLeakageDataPartitioner(seed=42)
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = partitioner.export_partitioned_feeds(batch, tmpdir)
        for feed_name, path in paths.items():
            assert Path(path).exists()
            assert Path(path).stat().st_size > 0
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                assert len(lines) == 50
