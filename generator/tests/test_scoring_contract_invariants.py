"""Tests for scoring contract invariants across RailVerifierSwitch, BankDecisionEngine, and DiscreteEventEngine.

Verifies:
1. RailVerifierSwitch parameterization of decline and challenge thresholds.
2. BankDecisionEngine threshold alignment and cached delegation to RailVerifierSwitch.
3. DiscreteEventEngine single source of truth for spend marginals and circadian specification.
4. Separation of concerns between deterministic rails, institutional decisions, and causal SCM.
"""

import pytest

from prism_generator.rails import (
    CandidateTransactionIntent,
    RailVerifierSwitch,
)
from prism_generator.agents import BankDecisionEngine, CardholderProfile
from prism_generator.engine import DiscreteEventEngine, INDIAN_PRODUCT_SPEND_MARGINALS
from prism_generator.spec_loader import load_all_specs


def _make_dummy_card(region: str = "US") -> CardholderProfile:
    currency = "INR" if region == "IN" else "USD"
    lat = 40.7128 if region == "US" else 19.0760
    lon = -74.0060 if region == "US" else 72.8777
    return CardholderProfile(
        card_id="CARD_TEST_001",
        home_lat=lat,
        home_lon=lon,
        work_lat=lat + 0.05,
        work_lon=lon + 0.05,
        is_commuter=False,
        credit_limit=50000.0,
        current_balance=0.0,
        currency=currency,
        product_id="IN_PROD_PMJDY_RUPAY_DEBIT" if region == "IN" else "US_PROD_STANDARD",
        cohort_id="C4_SUBURBAN_FAMILY",
        region=region,
        spend_mean_log=3.8,
        spend_sigma_log=0.7,
        domestic_cnp_enabled=True,
        international_enabled=True,
        is_frozen=False,
    )


def test_rail_verifier_switch_threshold_parameterization():
    """RailVerifierSwitch must strictly obey custom tau_decline and challenge thresholds."""
    custom_decline = 0.75
    switch = RailVerifierSwitch(region="US", tau_decline=custom_decline, seed=42)
    card = _make_dummy_card("US")

    intent = CandidateTransactionIntent(
        tx_id="TX_TEST_01",
        card_id=card.card_id,
        sim_time_sec=1000.0,
        amount=50.0,
        currency="USD",
        channel_type="CNP_WEB",
        merchant_id="M_TEST",
        mcc=5411,
        merchant_lat=card.home_lat,
        merchant_lon=card.home_lon,
        is_cross_border=False,
        otp_submitted=True,
        cvv_provided=True,
        pin_entered=False,
        emv_chip_present=False,
        emv_cryptogram_valid=False,
        three_ds_requested=True,
        risk_score=0.76,  # Above custom_decline (0.75), but below default (0.88)
        vaai_score=20,
        haversine_velocity_kph=0.0,
        tx_count_1h=1,
    )

    res = switch.verify_intent(intent, card)
    assert res.approved is False
    assert res.iso_response_code == "59"
    assert res.decline_reason == "BANK_ML_DECISION_ENGINE_FRAUD_DECLINE"


def test_bank_decision_engine_cached_delegation():
    """BankDecisionEngine must cache RailVerifierSwitch and respect its configured tau_decline."""
    bank = BankDecisionEngine(tau_decline=0.80, region="US")
    card = _make_dummy_card("US")

    code, trans_status, approved_amt = bank.evaluate_authorization(
        card=card,
        amount=40.0,
        channel="CNP_WEB",
        sim_time=1000.0,
        velocity_kph=0.0,
        count_1h=1,
        cvv_valid=True,
        ml_risk_score=0.82,  # > 0.80 -> Decline
    )
    assert code.value == "59"
    assert approved_amt == 0.0

    # Ensure RailVerifierSwitch is cached by region
    assert "US" in bank._switches
    assert bank._switches["US"].tau_decline == 0.80


def test_indian_product_spend_marginals_grounded_in_spec():
    """INDIAN_PRODUCT_SPEND_MARGINALS must match spec/05_india_payment_rails.yaml exactly."""
    specs = load_all_specs()
    for product_id, in_prod in specs.indian_products.items():
        assert product_id in INDIAN_PRODUCT_SPEND_MARGINALS
        expected_mu = in_prod.spend_mean_log
        expected_sigma = in_prod.spend_sigma_log
        actual_mu, actual_sigma = INDIAN_PRODUCT_SPEND_MARGINALS[product_id]
        assert actual_mu == pytest.approx(expected_mu)
        assert actual_sigma == pytest.approx(expected_sigma)


def test_circadian_fatal_assertion_when_missing():
    """DiscreteEventEngine must raise RuntimeError if circadian spec is absent."""
    engine = DiscreteEventEngine(n_cards=10, n_merchants=5, region="US", seed=42)
    saved = engine.specs.circadian
    try:
        engine.specs.circadian = None
        with pytest.raises(RuntimeError, match="requires valid circadian specification"):
            engine._compute_circular_diurnal_distribution()
    finally:
        engine.specs.circadian = saved


def test_discrete_event_engine_routes_via_rail_switch():
    """DiscreteEventEngine must have rail_switch initialized and no shadowed self.bank."""
    engine = DiscreteEventEngine(n_cards=10, n_merchants=5, region="US", seed=42)
    assert hasattr(engine, "rail_switch")
    assert isinstance(engine.rail_switch, RailVerifierSwitch)
    assert not hasattr(engine, "bank")
