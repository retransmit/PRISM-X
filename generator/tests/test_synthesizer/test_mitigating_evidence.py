"""Tests verifying cryptographic mitigating evidence and elimination of hard-negative inversion."""

import pytest
from prism_generator import (
    StructuralCausalEngine,
)


def test_dhanteras_gold_high_ticket_chip_pin_mitigation():
    """Legitimate ₹1.85L gold purchase with EMV Chip+PIN must evaluate to risk < 0.01 and approve ISO 00."""
    engine = StructuralCausalEngine(base_prevalence=0.0020)

    # Dhanteras Gold transaction: ₹1,85,000 (~60x mean of ₹3,000), MCC 5944 (Jewelry)
    dhanteras_record = {
        "transaction_id": "TX_DHANTERAS_01",
        "amount": 185000.0,
        "user_avg_tx_amount_30d": 3000.0,
        "haversine_velocity_kph": 12.0,
        "channel_type": "CP_POS_CHIP",
        "tx_count_1h": 1,
        "tx_count_24h": 2,
        "tx_amount_sum_24h": 185000.0,
        "credit_limit": 500000.0,
        "ip_distance_from_home_km": 5.0,
        "is_cross_border": False,
        "avs_match_code": "Y",
        "billing_shipping_match": 1,
        "cvv_match_flag": 1,
        "mcc": 5944,
        "hour_of_day": 16,
        "is_fraud": 0,
        "emv_arqc_verified": 1,
        "emv_pin_verified": 1,
        "emv_tvr_clean": 1,
    }

    gt = engine.evaluate(dhanteras_record, scenario_tag="HARD_NEGATIVE_DHANTERAS_GOLD")

    # 1. Risk score must be very low (< 0.01) despite huge amount
    assert gt.risk_score < 0.01, f"Expected risk < 0.01 for Dhanteras gold Chip+PIN, got {gt.risk_score}"

    # 2. Mitigating evidence must have negative logit Shapley values
    assert gt.analytical_shapley_log_odds["emv_arqc_verified"] < -3.0
    assert gt.analytical_shapley_log_odds["emv_pin_verified"] < -1.0

    # 3. Exact Shapley efficiency holds
    sum_logit = sum(gt.analytical_shapley_log_odds.values())
    assert sum_logit == pytest.approx(gt.logit_z - gt.base_logit, abs=1e-4)


def test_overseas_travel_chip_mitigation():
    """Legitimate international dining with chip must evaluate to risk < 0.01."""
    engine = StructuralCausalEngine(base_prevalence=0.0020)

    # London vacation dining: 5,000 km from home, cross border, flight velocity
    travel_record = {
        "transaction_id": "TX_LONDON_01",
        "amount": 85.0,
        "user_avg_tx_amount_30d": 60.0,
        "haversine_velocity_kph": 450.0,
        "channel_type": "CP_POS_CHIP",
        "tx_count_1h": 1,
        "tx_count_24h": 3,
        "tx_amount_sum_24h": 150.0,
        "credit_limit": 10000.0,
        "ip_distance_from_home_km": 5500.0,
        "is_cross_border": True,
        "avs_match_code": "Y",
        "billing_shipping_match": 1,
        "cvv_match_flag": 1,
        "mcc": 5812,
        "hour_of_day": 20,
        "is_fraud": 0,
        "emv_arqc_verified": 1,
        "emv_pin_verified": 0,
        "emv_tvr_clean": 1,
    }

    gt = engine.evaluate(travel_record, scenario_tag="HARD_NEGATIVE_TRAVEL")

    assert gt.risk_score < 0.01, f"Expected risk < 0.01 for overseas travel with chip, got {gt.risk_score}"
    assert gt.analytical_shapley_log_odds["emv_arqc_verified"] < -3.0


def test_carding_micro_auth_probe_detection():
    """Carding probe ($1.50, CNP, no 3DS, AVS mismatch) must evaluate to risk > 0.85."""
    engine = StructuralCausalEngine(base_prevalence=0.0020)

    carding_record = {
        "transaction_id": "TX_CARDING_01",
        "amount": 1.50,
        "user_avg_tx_amount_30d": 50.0,
        "haversine_velocity_kph": 0.0,
        "channel_type": "CNP_WEB",
        "tx_count_1h": 5,
        "tx_count_24h": 6,
        "tx_amount_sum_24h": 8.0,
        "credit_limit": 5000.0,
        "ip_distance_from_home_km": 1500.0,
        "is_cross_border": False,
        "avs_match_code": "N",
        "billing_shipping_match": 0,
        "cvv_match_flag": 1,
        "mcc": 8398,
        "hour_of_day": 3,
        "is_fraud": 1,
        "emv_arqc_verified": 0,
        "emv_pin_verified": 0,
        "emv_tvr_clean": 0,
        "three_ds_authenticated": 0,
        "three_ds_attempted": 0,
    }

    gt = engine.evaluate(carding_record, scenario_tag="ADV_MICRO_AUTH_PROBE")

    assert gt.risk_score > 0.85, f"Expected risk > 0.85 for carding probe, got {gt.risk_score}"
    assert gt.dominant_causal_driver in ("is_carding_probe", "tx_count_1h", "avs_mismatch_flag", "billing_shipping_mismatch", "is_night_tx")
