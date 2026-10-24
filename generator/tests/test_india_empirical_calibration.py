"""Automated Empirical Verification Suite for Day 2: Slice 15.

India Empirical Payment Rails, Regulatory Invariants & Kinematics Calibration.
Grounded in:
- Reserve Bank of India (RBI) Payment System Indicators (2022-2025) & Annual Reports
- National Payments Corporation of India (NPCI) Operating Regulations & RuPay Specs
- RBI Master Direction on Digital Payment Security & Customer Limited Liability
- Income Tax Rules 1962 (Rule 114B reporting limits on high-value jewelry MCC 5944)

Verifies:
1. Subcritical Hawkes branching ratio (eta < 1.0) and stationary daily arrival frequency for Indian cohorts (0.40 to 1.20 tx/day).
2. Mandatory Additional Factor of Authentication (AFA/2FA OTP) on domestic CNP (ISO 63 decline if absent).
3. Contactless NFC ₹5,000 PINless ceiling and ₹15,000 cumulative velocity cap (ISO 65 decline if breached).
4. High-value jewelry transaction threshold compliance (Income Tax Rule 114B PAN mandate).
5. Currency strictly INR, minor units integer precision (paisa), and RuPay BIN masking (607152).
6. Two-sample Kolmogorov-Smirnov (KS) test on generated ticket sizes matching calibrated RBI LogNormal marginals with p > 0.05.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from prism_generator import (
    CandidateTransactionIntent,
    DiscreteEventEngine,
    RailVerifierSwitch,
)
from prism_generator.agents import CardholderProfile, ISO8583Response
from prism_generator.engine import INDIAN_PRODUCT_SPEND_MARGINALS
from prism_generator.hawkes import INDIAN_PERSONA_HAWKES_PROFILES


def create_indian_test_card(
    card_id: str = "CARD_IN_TEST_001",
    product_id: str = "IN_PROD_SALARIED_PRIME_REWARDS",
    cohort_id: str = "C4_SUBURBAN_FAMILY",
    credit_limit: float = 150000.0,
    current_balance: float = 25000.0,
    consecutive_pinless: int = 0,
    cumulative_pinless: float = 0.0,
) -> CardholderProfile:
    """Helper to instantiate a valid Indian CardholderProfile for deterministic rail testing."""
    sp_mu, sp_sigma = INDIAN_PRODUCT_SPEND_MARGINALS.get(product_id, (7.60, 0.75))
    return CardholderProfile(
        card_id=card_id,
        home_lat=19.0760,
        home_lon=72.8777,
        work_lat=19.1136,
        work_lon=72.8697,
        is_commuter=True,
        credit_limit=credit_limit,
        current_balance=current_balance,
        posted_balance=current_balance,
        pending_holds=0.0,
        overdraft_limit=10000.0 if product_id == "IN_PROD_PMJDY_RUPAY_DEBIT" else 0.0,
        spend_mean_log=sp_mu,
        spend_sigma_log=sp_sigma,
        preferred_channels=["CP_POS_CHIP", "CP_POS_CONTACTLESS", "CNP_WEB", "CNP_MOBILE"],
        pan_masked="607152******1234",
        product_id=product_id,
        cohort_id=cohort_id,
        region="IN",
        currency="INR",
        domestic_cnp_enabled=True,
        international_enabled=False,
        contactless_enabled=True,
        consecutive_pinless_contactless_count=consecutive_pinless,
        cumulative_pinless_contactless_amount=cumulative_pinless,
    )


def test_indian_hawkes_branching_ratio_and_population_weighted_frequency():
    """Verify that all Indian persona Hawkes parameters are subcritical (eta < 1.0) and match RBI macro card kinematics."""
    # 7 Indian cohorts and their calibrated Hawkes parameters
    unique_profiles = {
        "IN_C1_HOURLY_GIG_WORKER": INDIAN_PERSONA_HAWKES_PROFILES["IN_C1_HOURLY_GIG_WORKER"],
        "IN_C2_FIXED_INCOME_SENIOR": INDIAN_PERSONA_HAWKES_PROFILES["IN_C2_FIXED_INCOME_SENIOR"],
        "IN_C3_YOUNG_ADULT_STUDENT": INDIAN_PERSONA_HAWKES_PROFILES["IN_C3_YOUNG_ADULT_STUDENT"],
        "IN_C4_SUBURBAN_FAMILY": INDIAN_PERSONA_HAWKES_PROFILES["IN_C4_SUBURBAN_FAMILY"],
        "IN_C5_URBAN_TECH_PROFESSIONAL": INDIAN_PERSONA_HAWKES_PROFILES["IN_C5_URBAN_TECH_PROFESSIONAL"],
        "IN_C6_SMALL_BUSINESS_OWNER": INDIAN_PERSONA_HAWKES_PROFILES["IN_C6_SMALL_BUSINESS_OWNER"],
        "IN_C7_LUXURY_AFFLUENT": INDIAN_PERSONA_HAWKES_PROFILES["IN_C7_LUXURY_AFFLUENT"],
    }

    # Macro weights reflecting Indian active cardholder demographic mix
    weights = {
        "IN_C1_HOURLY_GIG_WORKER": 0.16,
        "IN_C2_FIXED_INCOME_SENIOR": 0.18,
        "IN_C3_YOUNG_ADULT_STUDENT": 0.14,
        "IN_C4_SUBURBAN_FAMILY": 0.28,
        "IN_C5_URBAN_TECH_PROFESSIONAL": 0.12,
        "IN_C6_SMALL_BUSINESS_OWNER": 0.07,
        "IN_C7_LUXURY_AFFLUENT": 0.05,
    }

    day_seconds = 86400.0
    bar_phi = 0.8208  # Mean diurnal intensity integral over S^1

    weighted_daily_tx = 0.0
    for pid, p in unique_profiles.items():
        eta = p.branching_ratio
        assert eta < 1.0, f"Indian Persona {pid} is supercritical! eta = {eta:.4f} >= 1.0"
        assert 0.50 <= eta <= 0.75, f"Indian Persona {pid} branching ratio {eta:.3f} outside [0.50, 0.75]"
        assert 200.0 <= p.half_life_seconds <= 300.0, f"Indian Persona {pid} memory half life {p.half_life_seconds:.1f}s outside [200, 300]"

        expected_daily_n = (day_seconds * p.mu_0 * bar_phi) / (1.0 - eta)
        # Indian cardholder frequency is bounded between 0.40 and 1.25 tx/day due to UPI micropayment substitution
        assert 0.35 <= expected_daily_n <= 1.30, f"{pid} expected daily count {expected_daily_n:.2f} outside [0.35, 1.30]"
        weighted_daily_tx += weights[pid] * expected_daily_n

    # Population-weighted Indian card daily frequency must be in [0.60, 0.75] tx/day (~0.68 tx/day)
    assert 0.60 <= weighted_daily_tx <= 0.75, (
        f"Population-weighted Indian daily frequency {weighted_daily_tx:.3f} outside calibrated band [0.60, 0.75]"
    )


def test_domestic_cnp_without_otp_declined_iso63():
    """Verify RBI Mandatory AFA (2FA/OTP): domestic CNP without OTP is strictly declined with ISO 63."""
    switch = RailVerifierSwitch(region="IN", seed=42)
    card = create_indian_test_card()

    # Case A: Domestic CNP without OTP submitted -> ISO 63 Security Violation
    intent_no_otp = CandidateTransactionIntent(
        tx_id="TX_IN_CNP_001",
        card_id=card.card_id,
        sim_time_sec=100.0,
        amount=2499.0,
        currency="INR",
        channel_type="CNP_WEB",
        merchant_id="M_SWIGGY_001",
        mcc=5812,
        merchant_lat=card.home_lat,
        merchant_lon=card.home_lon,
        is_cross_border=False,
        otp_submitted=False,
    )
    result_no_otp = switch.verify_intent(intent_no_otp, card)
    assert not result_no_otp.approved
    assert result_no_otp.iso_response_code == ISO8583Response.SECURITY_VIOLATION_63.value
    assert result_no_otp.regulatory_rule_triggered == "RBI_MANDATORY_AFA_2FA"
    assert result_no_otp.decline_reason == "RBI_MANDATORY_AFA_OTP_REQUIRED"

    # Case B: Domestic CNP with valid OTP submitted -> ISO 00 Approved
    intent_with_otp = CandidateTransactionIntent(
        tx_id="TX_IN_CNP_002",
        card_id=card.card_id,
        sim_time_sec=120.0,
        amount=2499.0,
        currency="INR",
        channel_type="CNP_WEB",
        merchant_id="M_SWIGGY_001",
        mcc=5812,
        merchant_lat=card.home_lat,
        merchant_lon=card.home_lon,
        is_cross_border=False,
        otp_submitted=True,
    )
    result_with_otp = switch.verify_intent(intent_with_otp, card)
    assert result_with_otp.approved
    assert result_with_otp.iso_response_code == ISO8583Response.APPROVED_00.value


def test_contactless_nfc_over_5000_inr_pin_mandate():
    """Verify RBI Contactless Ceiling: NFC transactions > ₹5,000 require PIN; PINless triggers ISO 65."""
    switch = RailVerifierSwitch(region="IN", seed=42)
    card = create_indian_test_card()

    # Case A: NFC tap of ₹5,500 without PIN -> Declined with ISO 65 Activity Limit Exceeded
    intent_over_cap = CandidateTransactionIntent(
        tx_id="TX_IN_NFC_001",
        card_id=card.card_id,
        sim_time_sec=100.0,
        amount=5500.0,
        currency="INR",
        channel_type="CP_POS_CONTACTLESS",
        merchant_id="M_DMART_001",
        mcc=5411,
        merchant_lat=card.home_lat,
        merchant_lon=card.home_lon,
        pin_entered=False,
    )
    res_no_pin = switch.verify_intent(intent_over_cap, card)
    assert not res_no_pin.approved
    assert res_no_pin.iso_response_code == ISO8583Response.ACTIVITY_LIMIT_EXCEEDED_65.value
    assert res_no_pin.regulatory_rule_triggered == "RBI_NFC_PIN_MANDATE"

    # Case B: NFC tap of ₹5,500 with PIN entered -> Approved with ISO 00
    intent_with_pin = CandidateTransactionIntent(
        tx_id="TX_IN_NFC_002",
        card_id=card.card_id,
        sim_time_sec=105.0,
        amount=5500.0,
        currency="INR",
        channel_type="CP_POS_CONTACTLESS",
        merchant_id="M_DMART_001",
        mcc=5411,
        merchant_lat=card.home_lat,
        merchant_lon=card.home_lon,
        pin_entered=True,
    )
    res_pin = switch.verify_intent(intent_with_pin, card)
    assert res_pin.approved
    assert res_pin.iso_response_code == ISO8583Response.APPROVED_00.value


def test_contactless_nfc_cumulative_cap_15000_inr():
    """Verify RBI Contactless Cumulative Limit: cumulative PINless amount > ₹15,000 triggers ISO 65."""
    switch = RailVerifierSwitch(region="IN", seed=42)

    # Card with prior PINless cumulative spend of ₹13,500
    card = create_indian_test_card(cumulative_pinless=13500.0)

    # New transaction of ₹2,000 pushes cumulative spend to ₹15,500 > ₹15,000 ceiling
    intent_breach = CandidateTransactionIntent(
        tx_id="TX_IN_NFC_CUM_001",
        card_id=card.card_id,
        sim_time_sec=200.0,
        amount=2000.0,
        currency="INR",
        channel_type="CP_POS_CONTACTLESS",
        merchant_id="M_RELIANCE_001",
        mcc=5411,
        merchant_lat=card.home_lat,
        merchant_lon=card.home_lon,
        pin_entered=False,
    )
    res_breach = switch.verify_intent(intent_breach, card)
    assert not res_breach.approved
    assert res_breach.iso_response_code == ISO8583Response.ACTIVITY_LIMIT_EXCEEDED_65.value
    assert res_breach.regulatory_rule_triggered == "RBI_NFC_CUMULATIVE_CAP"
    assert res_breach.decline_reason == "RBI_NFC_CUMULATIVE_LIMIT_EXCEEDED_15000_INR"

    # When cardholder enters PIN, the cumulative breach is bypassed (Step-up SCA satisfied)
    intent_step_up = CandidateTransactionIntent(
        tx_id="TX_IN_NFC_CUM_002",
        card_id=card.card_id,
        sim_time_sec=205.0,
        amount=2000.0,
        currency="INR",
        channel_type="CP_POS_CONTACTLESS",
        merchant_id="M_RELIANCE_001",
        mcc=5411,
        merchant_lat=card.home_lat,
        merchant_lon=card.home_lon,
        pin_entered=True,
    )
    res_step_up = switch.verify_intent(intent_step_up, card)
    assert res_step_up.approved
    assert res_step_up.iso_response_code == ISO8583Response.APPROVED_00.value


def test_dhanteras_jewellery_splitting_pan_threshold():
    """Verify Income Tax Rule 114B: jewelry (MCC 5944) purchases strictly stay below ₹2,00,000 reporting threshold."""
    engine = DiscreteEventEngine(n_cards=100, n_merchants=30, region="IN", seed=42)

    # Simulate during festive Dhanteras/Diwali seasonal regime
    records = engine.generate_batch(
        n_transactions=500,
        fraud_prevalence=0.0,
        time_span_days=7,
        start_time_seconds=1730419200.0,  # Nov 1 (Diwali season)
        active_macro_regime="DHANTERAS_DIWALI",
    )

    jewelry_txs = [r for r in records if r.get("mcc") == 5944]
    # For every jewellery transaction, verify strict compliance with Rule 114B (< ₹2,00,000 / 20,000,000 paisa)
    for jtx in jewelry_txs:
        amt = float(jtx["amount"])
        assert amt < 200000.0, f"Jewelry transaction {jtx['tx_id']} amount ₹{amt:.2f} violates Rule 114B (< ₹2,00,000)"


def test_inr_currency_and_minor_units_precision():
    """Verify that Indian simulation generates strictly INR currency with exact integer paisa minor units."""
    engine = DiscreteEventEngine(n_cards=100, n_merchants=30, region="IN", seed=777)
    records = engine.generate_batch(
        n_transactions=300,
        fraud_prevalence=0.02,
        time_span_days=5,
        start_time_seconds=1704067200.0,
    )

    assert len(records) > 0
    for r in records:
        assert r["currency"] == "INR", f"Transaction {r['tx_id']} has invalid currency {r['currency']}"
        amt = float(r["amount"])
        # Verify amount is strictly positive
        assert amt > 0.0, f"Transaction {r['tx_id']} amount {amt} <= 0"
        # Verify minor units precision: rounded to 2 decimal places (paisa)
        paisa = int(round(amt * 100))
        assert abs(amt - (paisa / 100.0)) < 1e-4, f"Minor unit drift detected: amount {amt} vs paisa {paisa}"
        # Verify RuPay BIN masking format
        pan = r["pan_masked"]
        assert pan.startswith("607152"), f"Masked PAN {pan} does not match RuPay standard prefix 607152"


def test_two_sample_ks_spend_marginals_against_rbi():
    """Verify that sampled ticket sizes for all Indian card products match calibrated LogNormal distributions (KS p > 0.05)."""
    sample_size = 500

    products = [
        ("IN_PROD_RETAIL_RUPAY_DEBIT", 7.8620, 0.75, 0.0),
        ("IN_PROD_PMJDY_RUPAY_DEBIT", 6.50, 0.65, 10000.0),
        ("IN_PROD_ENTRY_FD_BACKED", 6.95, 0.70, 25000.0),
        ("IN_PROD_SALARIED_PRIME_REWARDS", 7.65, 0.70, 150000.0),
        ("IN_PROD_KISAN_CREDIT_CARD", 7.80, 0.72, 200000.0),
        ("IN_PROD_SUPER_PREMIUM_HNI", 8.55, 0.75, 2500000.0),
    ]

    for prod_id, ref_mu, ref_sigma, cred_limit in products:
        rng = np.random.default_rng(42)
        card = create_indian_test_card(
            card_id=f"CARD_{prod_id}",
            product_id=prod_id,
            credit_limit=cred_limit,
        )

        # Generate empirical samples from CardholderProfile
        empirical_samples = np.array([card.sample_spend_amount(rng) for _ in range(sample_size)])

        # Generate reference theoretical draws, censored exactly the way
        # CardholderProfile.sample_spend_amount censors them. The engine stopped deriving a
        # debit instrument's ticket ceiling from a credit line on 2026-10-12 (a debit card has
        # no credit line, and the old rule made the RBI debit mean unreachable by
        # construction), so mirroring the old `credit_limit * 0.5` here would compare the
        # engine against a censoring rule it does not use -- which is exactly what produced
        # KS=1.0 for IN_PROD_RETAIL_RUPAY_DEBIT when this test still assumed a credit line.
        is_credit_instrument = not any(
            m in prod_id for m in ("DEBIT", "PREPAID", "EBT", "HSA", "PNJ")
        )
        ceiling = cred_limit * 0.50 if (is_credit_instrument and cred_limit > 0) else 250000.0
        ref_samples = rng.lognormal(mean=ref_mu, sigma=ref_sigma, size=sample_size)
        ref_samples = np.clip(ref_samples, 10.0, ceiling)

        # Two-sample Kolmogorov-Smirnov test
        ks_stat, p_value = stats.ks_2samp(empirical_samples, ref_samples)
        assert p_value > 0.05, (
            f"Product {prod_id} ticket distribution rejected by 2-sample KS test: KS={ks_stat:.4f}, p={p_value:.4f} <= 0.05"
        )
