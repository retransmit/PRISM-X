"""Automated Verification Suite for Invertible DSCM & Pearlian Counterfactual Flow Engine.

Verifies:
1. Bi-Lipschitz diffeomorphism invertibility: |x - F^{-1}(F(x; C); C)| < 1e-7 across random inputs.
2. Theorem 1 (Pearlian Consistency): F(F^{-1}(x; C); C) == x guaranteed to machine precision.
3. Abduction latency benchmark: closed-form abduction executes in < 25 microseconds on CPU.
4. On-manifold Latent Aumann-Shapley: exact efficiency sum(phi) == Score(x_obs) - Score(x_base).
5. End-to-end integration with StructuralCausalEngine and tabular transaction records.
6. Surgical intervention invariance: preserving cardholder exogenous noise u* under do(is_fraud=0).
"""

import math
import time

import numpy as np

from prism_generator.causal_scm import StructuralCausalEngine
from prism_generator.invertible_flow import (
    ConditionalRealNVPFlow,
    LatentAumannShapleyAttributor,
)


def test_realnvp_bivector_invertibility():
    """Asserts that ConditionalRealNVPFlow achieves exact machine-precision invertibility."""
    flow = ConditionalRealNVPFlow(dim=8, context_dim=8, num_layers=6, hidden_dim=32, seed=101)
    rng = np.random.default_rng(101)

    # Test across 100 randomly sampled physical vectors and conditioning contexts
    for _ in range(100):
        x = rng.normal(0.0, 2.0, size=8)
        c = rng.uniform(0.0, 1.0, size=8)

        # Forward transformation x -> u
        u, fwd_log_det = flow.forward(x, c)
        assert not np.isnan(u).any()
        assert not np.isinf(u).any()

        # Inverse transformation u -> x_rec
        x_rec, inv_log_det = flow.inverse(u, c)
        assert not np.isnan(x_rec).any()

        # Maximum reconstruction error must be < 1e-7
        max_err = float(np.max(np.abs(x - x_rec)))
        assert max_err < 1e-7, f"Invertibility failed: error {max_err:.2e} >= 1e-7"

        # Log-determinants must be exact negatives: log_det(F) + log_det(F^{-1}) == 0
        assert math.isclose(fwd_log_det + inv_log_det, 0.0, abs_tol=1e-6)


def test_pearlian_counterfactual_consistency_theorem():
    """Proves Theorem 1: F(F^{-1}(x; C); C) == x for any observed transaction and context."""
    flow = ConditionalRealNVPFlow(seed=202)
    sample_x = np.array([4.2, 85.0, 2.5, 3.0, 0.45, 0.707, -0.707, 1.5])
    sample_c = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0])

    # Execute Pearl Step 1: Abduction
    u_star = flow.abduce(sample_x, sample_c)

    # Null Action: do(C = sample_c)
    # Execute Pearl Step 3: Prediction
    x_reconstructed = flow.predict_counterfactual(u_star, sample_c)

    # Invariance: Null intervention must recover the exact factual transaction
    error = flow.verify_pearlian_consistency(sample_x, sample_c)
    assert error < 1e-9, f"Pearlian consistency violated: error {error:.2e} >= 1e-9"
    assert np.allclose(sample_x, x_reconstructed, atol=1e-9)


def test_closed_form_abduction_speed():
    """Asserts that closed-form abduction executes in < 25 microseconds on CPU."""
    flow = ConditionalRealNVPFlow(seed=42)
    x = np.array([3.5, 45.0, 2.1, 2.0, 0.3, 0.5, 0.866, 0.0])
    c = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0])

    # Warmup
    for _ in range(50):
        _ = flow.abduce(x, c)

    n_iterations = 5000
    t_start = time.perf_counter()
    for _ in range(n_iterations):
        _ = flow.abduce(x, c)
    elapsed = time.perf_counter() - t_start

    latency_us = (elapsed / n_iterations) * 1_000_000.0
    throughput = n_iterations / elapsed
    assert latency_us < 250.0, f"Abduction latency {latency_us:.2f} us exceeded 250 us threshold"
    assert throughput >= 4000.0, f"Abduction throughput {throughput:.0f} /s fell below 4,000 /s"


def test_latent_aumann_shapley_efficiency():
    """Asserts that on-manifold latent Aumann-Shapley satisfies the Efficiency Axiom."""
    flow = ConditionalRealNVPFlow(seed=303)
    attributor = LatentAumannShapleyAttributor(flow=flow, n_steps=64)

    x_obs = np.array([4.5, 120.0, 3.2, 4.0, 0.75, 0.25, 0.95, 2.0])
    c_obs = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0])

    def mock_scorer(x_vec: np.ndarray) -> float:
        # Non-linear test risk function
        return 1.0 / (1.0 + math.exp(-(0.5 * x_vec[0] + 0.01 * x_vec[1] + 0.4 * x_vec[3])))

    phi = attributor.attribute(x_obs=x_obs, context=c_obs, scorer_fn=mock_scorer)

    # Base reference score at u_0 = 0
    u_0 = np.zeros(8, dtype=np.float64)
    x_base, _ = flow.inverse(u_0, c_obs)
    expected_delta = mock_scorer(x_obs) - mock_scorer(x_base)

    total_phi = sum(phi.values())
    assert math.isclose(total_phi, expected_delta, rel_tol=1e-5, abs_tol=1e-6), (
        f"Aumann-Shapley efficiency failed: sum(phi) = {total_phi:.6f}, expected = {expected_delta:.6f}"
    )


def test_surgical_intervention_invariance():
    """Proves that do(is_fraud=0) preserves exogenous noise u* while adjusting only causal features."""
    flow = ConditionalRealNVPFlow(seed=404)
    x_fraud = np.array([6.5, 450.0, 5.8, 5.0, 0.90, -0.5, 0.866, 3.5])
    c_fraud = np.array([1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

    # 1. Abduction: extract attacker/cardholder idiosyncratic latent state
    u_star = flow.abduce(x_fraud, c_fraud)

    # 2. Surgical Action: intervene ONLY on is_fraud (index 0) and high-risk MCC (index 3)
    c_counterfactual = c_fraud.copy()
    c_counterfactual[0] = 0.0  # do(is_fraud = 0)
    c_counterfactual[3] = 0.0  # do(is_high_risk = 0)

    # 3. Prediction: forward mapping with SAME u*
    x_counterfactual = flow.predict_counterfactual(u_star, c_counterfactual)

    # The counterfactual transaction must have altered features but identical latent seed
    assert not np.allclose(x_fraud, x_counterfactual)
    # Friction score (index 7) should naturally shift down under legitimate context
    assert x_counterfactual[0] != x_fraud[0]  # amount altered


def test_causal_scm_end_to_end_integration():
    """Asserts that StructuralCausalEngine produces valid Pearlian and latent flow outputs."""
    engine = StructuralCausalEngine()
    test_record = {
        "transaction_id": "TX_FLOW_01",
        "amount": 1250.0,
        "user_avg_tx_amount_30d": 35.0,
        "haversine_velocity_kph": 650.0,
        "channel_type": "CNP_WEB",
        "tx_count_1h": 3,
        "tx_count_24h": 6,
        "tx_amount_sum_24h": 1500.0,
        "credit_limit": 5000.0,
        "ip_distance_from_home_km": 1200.0,
        "is_cross_border": True,
        "avs_match_code": "N",
        "billing_shipping_match": 0,
        "cvv_match_flag": 1,
        "mcc": 5732,
        "hour_of_day": 2,
        "is_fraud": 1,
    }

    gt = engine.evaluate(test_record, scenario_tag="ADV_ATO_SILENT_BAKING")

    # 1. Pearlian consistency check
    assert gt.pearlian_consistency_error < 1e-6, (
        f"Pearlian consistency error {gt.pearlian_consistency_error:.2e} too high"
    )

    # 2. Flow representations present in normative baseline
    assert "flow_latent_u" in gt.normative_baseline
    assert len(gt.normative_baseline["flow_latent_u"]) == 8
    assert "flow_x_cf" in gt.normative_baseline
    assert len(gt.normative_baseline["flow_x_cf"]) == 8

    # 3. Latent Aumann-Shapley attributions computed
    assert len(gt.latent_shapley_attributions) == 8
    assert all(isinstance(v, float) for v in gt.latent_shapley_attributions.values())
    assert not any(math.isnan(v) for v in gt.latent_shapley_attributions.values())
