"""Tests verifying Shapley efficiency axioms and causal ground-truth explanations."""

import pytest
from prism_generator import SimulationEngine, StructuralCausalEngine


def test_shapley_efficiency_probability_space():
    """Verifies that sum(phi_i_prob) == risk_score - base_risk to machine precision."""
    causal_engine = StructuralCausalEngine(base_prevalence=0.0020)
    test_record = {
        "amount": 1850.0,
        "user_avg_tx_amount_30d": 35.0,
        "haversine_velocity_kph": 650.0,
        "channel_type": "CP_POS_CHIP",
        "tx_count_1h": 4,
        "tx_count_24h": 9,
        "tx_amount_sum_24h": 2200.0,
        "credit_limit": 5000.0,
        "ip_distance_from_home_km": 1200.0,
        "is_cross_border": True,
        "avs_match_code": "N",
        "billing_shipping_match": 0,
        "cvv_match_flag": 1,
        "mcc": 5094,
        "hour_of_day": 3,
        "is_fraud": 1,
    }

    gt = causal_engine.evaluate(test_record, scenario_tag="IMPOSSIBLE_TRAVEL")

    sum_shapley_prob = sum(gt.analytical_shapley_probability.values())
    expected_delta_prob = gt.risk_score - gt.base_risk
    assert sum_shapley_prob == pytest.approx(expected_delta_prob, abs=1e-5), (
        f"Shapley efficiency violated in probability space: sum={sum_shapley_prob}, expected={expected_delta_prob}"
    )


def test_shapley_efficiency_log_odds_space():
    """Verifies that sum(phi_i_logit) == logit_z - base_logit."""
    causal_engine = StructuralCausalEngine(base_prevalence=0.0020)
    test_record = {
        "amount": 45.0,
        "user_avg_tx_amount_30d": 40.0,
        "haversine_velocity_kph": 12.0,
        "channel_type": "CP_POS_CHIP",
        "tx_count_1h": 0,
        "tx_count_24h": 1,
        "tx_amount_sum_24h": 45.0,
        "credit_limit": 5000.0,
        "ip_distance_from_home_km": 5.0,
        "is_cross_border": False,
        "avs_match_code": "Y",
        "billing_shipping_match": 1,
        "cvv_match_flag": 1,
        "mcc": 5411,
        "hour_of_day": 14,
        "is_fraud": 0,
    }

    gt = causal_engine.evaluate(test_record, scenario_tag="ORGANIC_NORMAL")

    sum_shapley_logit = sum(gt.analytical_shapley_log_odds.values())
    expected_delta_logit = gt.logit_z - gt.base_logit
    assert sum_shapley_logit == pytest.approx(expected_delta_logit, abs=1e-4), (
        f"Shapley efficiency violated in logit space: sum={sum_shapley_logit}, expected={expected_delta_logit}"
    )


def test_batch_generation_shapley_conformance():
    """Batch generation must maintain exact Shapley efficiency across all records."""
    engine = SimulationEngine(n_cards=100, n_merchants=50, seed=42)
    records = engine.generate_batch(n_transactions=500, fraud_prevalence=0.05)

    for r in records:
        prob_dict = r["analytical_shapley_probability"]
        risk_score = float(r["risk_score"])
        base_risk = float(r["base_risk"])
        sum_prob = sum(prob_dict.values())
        diff = abs(sum_prob - (risk_score - base_risk))
        assert diff < 1e-4, f"Record {r['transaction_id']} violated Shapley efficiency: diff={diff}"


def test_nonlinear_synergies_and_owen_partition():
    """Verifies that non-linear multi-feature interactions (night + cross-border + high amount)
    correctly amplify risk non-linearly while preserving exact Owen multilinear Shapley efficiency.
    """
    causal_engine = StructuralCausalEngine(base_prevalence=0.0020)
    synergy_attack_record = {
        "amount": 2500.0,
        "user_avg_tx_amount_30d": 50.0,  # 50x ratio
        "haversine_velocity_kph": 750.0,
        "channel_type": "CNP_ECOMMERCE",
        "tx_count_1h": 6,
        "tx_count_24h": 12,
        "tx_amount_sum_24h": 4000.0,
        "credit_limit": 5000.0,
        "ip_distance_from_home_km": 2500.0,
        "is_cross_border": True,
        "avs_match_code": "N",
        "billing_shipping_match": 0,
        "cvv_match_flag": 0,  # Both AVS and CVV mismatch -> triggers SYN_AVS_CVV_MISMATCH
        "mcc": 5732,  # Electronics -> triggers SYN_HIGH_AMOUNT_HIGH_RISK_MCC
        "hour_of_day": 3,  # Night -> triggers SYN_NIGHT_CROSS_BORDER and SYN_TRIAD_ATO
        "is_fraud": 1,
    }

    gt = causal_engine.evaluate(synergy_attack_record, scenario_tag="ACCOUNT_TAKEOVER")

    # Risk score must be very high for this compound attack
    assert gt.risk_score > 0.99
    assert gt.dominant_causal_driver in ("cvv_mismatch_flag", "avs_mismatch_flag", "amount_to_mean_ratio_30d", "is_cross_border_tx", "tx_count_1h")

    # Verify exact efficiency in log-odds space
    sum_logit = sum(gt.analytical_shapley_log_odds.values())
    expected_delta_logit = gt.logit_z - gt.base_logit
    assert abs(sum_logit - expected_delta_logit) < 1e-4

    # Verify exact efficiency in probability space
    sum_prob = sum(gt.analytical_shapley_probability.values())
    expected_delta_prob = gt.risk_score - gt.base_risk
    assert abs(sum_prob - expected_delta_prob) < 1e-4



def test_batched_flow_scorer_agrees_row_for_row_with_the_scalar_one():
    """The path integral scores its two endpoints one call at a time and every
    quadrature node in a single batched call. If those two scorers ever disagree, the
    reported attribution would depend on which code path computed it, so they are
    pinned to each other here, including at the clip boundaries."""
    import numpy as np
    from prism_generator import StructuralCausalEngine

    causal_engine = StructuralCausalEngine(base_prevalence=0.0020)
    rows = np.array([
        [-5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],          # lower clips
        [15.0, 900.0, 9.0, 40.0, 5.0, 1.0, 1.0, 3.5],        # upper clips
        [float(np.log(1850.0)), 650.0, 3.2, 4.0, 1.5, -1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])

    for mean_30d in (1.0, 40.0, 1_000_000.0):
        scalar = np.array([causal_engine._flow_scorer(r, mean_30d) for r in rows])
        batched = causal_engine._flow_scorer_batch(rows, mean_30d)
        np.testing.assert_allclose(batched, scalar, rtol=1e-12, atol=1e-15)


def test_batched_path_integration_matches_the_scalar_one():
    """`attribute` keeps the original per-row scorer loop behind a flag and takes a
    batched scorer otherwise; `evaluate` supplies the batched one, so both paths must
    return the same attribution for the same observation. The same holds for passing
    the already-computed abduction `u_obs` instead of recomputing it."""
    from prism_generator import StructuralCausalEngine
    from prism_generator.experimental.invertible_flow import (
        LatentAumannShapleyAttributor,
        TransactionFlowFeatureCodec,
    )

    causal_engine = StructuralCausalEngine(base_prevalence=0.0020)
    record = {
        "amount": 1850.0,
        "user_avg_tx_amount_30d": 35.0,
        "haversine_velocity_kph": 650.0,
        "channel_type": "CP_POS_CHIP",
        "tx_count_1h": 4,
        "tx_count_24h": 9,
        "tx_amount_sum_24h": 2200.0,
        "credit_limit": 5000.0,
        "ip_distance_from_home_km": 1200.0,
        "is_cross_border": True,
        "avs_match_code": "N",
        "billing_shipping_match": 0,
        "cvv_match_flag": 1,
        "mcc": 5094,
        "hour_of_day": 3,
        "is_fraud": 1,
    }
    x_obs, ctx = TransactionFlowFeatureCodec.encode(record)
    mean_30d = 35.0
    scorer = lambda v: causal_engine._flow_scorer(v, mean_30d)          # noqa: E731
    batched_scorer = lambda X: causal_engine._flow_scorer_batch(X, mean_30d)  # noqa: E731

    attributor = LatentAumannShapleyAttributor(causal_engine.flow)
    reference = attributor.attribute(x_obs, ctx, scorer)

    assert set(reference) == set(TransactionFlowFeatureCodec.FEATURE_NAMES)
    assert any(abs(v) > 0.0 for v in reference.values()), (
        "an all-zero attribution vector would make any two implementations agree vacuously"
    )

    fast = attributor.attribute(x_obs, ctx, scorer, scorer_batch_fn=batched_scorer)
    u_obs, _ = causal_engine.flow.forward(x_obs, ctx)
    reused = attributor.attribute(x_obs, ctx, scorer, u_obs=u_obs, scorer_batch_fn=batched_scorer)

    for name, value in reference.items():
        assert fast[name] == pytest.approx(value, rel=1e-9, abs=1e-15), name
        assert reused[name] == pytest.approx(value, rel=1e-9, abs=1e-15), name
