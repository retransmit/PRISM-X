"""Tests verifying multi-agent closed-loop feedback, monotonicity, and card freezing."""

import pytest
import numpy as np

from prism_generator import (
    DiscreteEventEngine,
    AdaptiveFraudsterAgent,
    BankDecisionEngine,
    CardholderProfile,
    ISO8583Response,
)


def test_strict_global_temporal_monotonicity():
    """Discrete-event engine must guarantee that all generated transactions are non-decreasing in time."""
    engine = DiscreteEventEngine(n_cards=100, n_merchants=50, seed=42)
    records = engine.generate_batch(n_transactions=1000, fraud_prevalence=0.05, time_span_days=15)

    assert len(records) == 1000
    times = [r["tx_time_seconds"] for r in records]

    for i in range(len(times) - 1):
        assert times[i] <= times[i + 1], (
            f"Temporal monotonicity violated at index {i}: "
            f"tx[{i}]={times[i]}s > tx[{i+1}]={times[i+1]}s"
        )


def test_adversary_amount_decay_on_insufficient_funds():
    """Fraudster must decay transaction amount upon receiving ISO 51 (Insufficient Funds)."""
    rng = np.random.default_rng(123)
    fraudster = AdaptiveFraudsterAgent(rng)
    fraudster.current_amount = 1000.0

    # Fraudster receives ISO 51
    fraudster.receive_feedback(
        response_code=ISO8583Response.INSUFFICIENT_FUNDS_51,
        trans_status_3ds=None,
        sim_time_seconds=1000.0,
    )

    # Must enter AMOUNT_ADAPTATION and decrease current_amount by ~30%
    assert fraudster.current_amount < 1000.0
    assert fraudster.current_amount == pytest.approx(700.0, abs=1e-3)


def test_adversary_aborts_session_and_hops_to_soft_gateway_on_3ds_challenge():
    """Fraudster must abort session and hop to soft gateway upon encountering 3DS Challenge."""
    rng = np.random.default_rng(456)
    fraudster = AdaptiveFraudsterAgent(rng)
    fraudster.active_merchant_tier = "TIER_A"

    # Encounters 3DS challenge 'C'
    fraudster.receive_feedback(
        response_code=ISO8583Response.DO_NOT_HONOR_05,
        trans_status_3ds="C",
        sim_time_seconds=2000.0,
    )

    assert fraudster.active_merchant_tier == "TIER_C"


def test_cardholder_unauthorized_alert_and_card_freeze():
    """Cardholder must trigger freeze after discovery latency, blocking subsequent transactions."""
    engine = DiscreteEventEngine(n_cards=10, n_merchants=20, seed=777)
    card = engine.cards[0]

    # Initially card is not frozen
    assert not card.is_frozen
    assert card.unauthorized_alert_time < 0.0

    # Trigger alert at t = 1000s
    card.trigger_unauthorized_alert(sim_time=1000.0, rng=engine.rng)
    assert card.unauthorized_alert_time > 1000.0
    assert not card.is_frozen  # Not frozen yet during detection delay

    # Before alert time, check_freeze_status returns False
    assert not card.check_freeze_status(sim_time=1005.0)

    # After alert time, card becomes frozen
    assert card.check_freeze_status(sim_time=card.unauthorized_alert_time + 1.0)
    assert card.is_frozen


def test_bank_decision_engine_pipeline():
    """BankDecisionEngine must enforce limits, kinematics, and security."""
    bank = BankDecisionEngine(max_speed_kmh=900.0)
    card = CardholderProfile(
        card_id="TEST_CARD",
        home_lat=40.75,
        home_lon=-73.98,
        work_lat=40.75,
        work_lon=-73.98,
        is_commuter=False,
        credit_limit=1000.0,
        current_balance=900.0,
        spend_mean_log=3.0,
        spend_sigma_log=0.5,
        preferred_channels=["CP_POS_CHIP"],
        circadian_peak_hour=13.0,
        last_physical_lat=40.75,
        last_physical_lon=-73.98,
        last_physical_time=0.0,
    )

    # 1. Amount exceeds available balance -> ISO 51
    resp, *_ = bank.evaluate_authorization(
        card=card,
        amount=250.0,  # 900 + 250 = 1150 > 1000
        channel="CP_POS_CHIP",
        sim_time=100.0,
        velocity_kph=0.0,
        count_1h=1,
    )
    assert resp == ISO8583Response.INSUFFICIENT_FUNDS_51

    # 2. Supersonic velocity -> ISO 59
    resp, *_ = bank.evaluate_authorization(
        card=card,
        amount=50.0,
        channel="CP_POS_CHIP",
        sim_time=100.0,
        velocity_kph=1500.0,  # Mach 1.2
        count_1h=1,
    )
    assert resp == ISO8583Response.SUSPECTED_FRAUD_59

    # 3. CVV mismatch -> ISO 82 (or legacy 63)
    resp, *_ = bank.evaluate_authorization(
        card=card,
        amount=50.0,
        channel="CNP_WEB",
        sim_time=100.0,
        velocity_kph=0.0,
        count_1h=1,
        cvv_valid=False,
    )
    assert resp in (ISO8583Response.INVALID_CVV_82, ISO8583Response.SECURITY_VIOLATION_63)
