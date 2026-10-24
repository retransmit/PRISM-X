"""Tests verifying Pearl's structural counterfactual XAI ground truth and benchmark metrics."""

import pytest
import numpy as np

from prism_generator import (
    DiscreteEventEngine,
    StructuralCausalEngine,
    GroundTruthXAIEvaluator,
)


def test_counterfactual_twin_generation():
    """Causal engine must generate unperturbed twin and exact input-space deltas for attacks."""
    engine = StructuralCausalEngine()
    test_attack_record = {
        "transaction_id": "TX_TEST_01",
        "amount": 1850.0,
        "user_avg_tx_amount_30d": 45.0,
        "haversine_velocity_kph": 1200.0,
        "channel_type": "CP_POS_MAGSTRIPE",
        "tx_count_1h": 4,
        "tx_count_24h": 8,
        "tx_amount_sum_24h": 2200.0,
        "credit_limit": 5000.0,
        "ip_distance_from_home_km": 2500.0,
        "is_cross_border": True,
        "avs_match_code": "N",
        "billing_shipping_match": 0,
        "cvv_match_flag": 1,
        "mcc": 5094,
        "hour_of_day": 3,
        "is_fraud": 1,
    }

    gt = engine.evaluate(test_attack_record, scenario_tag="COUNTERFEIT_CLONE")

    # 1. Counterfactual twin must be legitimate
    assert gt.counterfactual_twin["is_fraud"] == 0
    assert gt.counterfactual_twin["amount"] == 45.0
    assert gt.counterfactual_twin["haversine_velocity_kph"] == 0.0

    # 2. Input deltas must capture the attack intervention
    deltas = gt.counterfactual_input_deltas
    assert deltas["amount"] == pytest.approx(1850.0 - 45.0, abs=1e-2)
    assert deltas["haversine_velocity_kph"] == pytest.approx(1200.0, abs=1e-2)
    assert deltas["avs_mismatch_flag"] == 1.0

    # 3. Dominant driver must identify attack lever
    assert gt.dominant_causal_driver in ("haversine_velocity_kph", "amount_to_mean_ratio_30d", "is_cross_border_tx")
    assert len(gt.explanation_narrative) > 10


def test_xai_evaluator_metrics():
    """GroundTruthXAIEvaluator must correctly compute Precision@k, Cosine Similarity, and Kendall tau."""
    evaluator = GroundTruthXAIEvaluator()

    # Ground-truth causal attribution vector (e.g. 5 features: [velocity, amount, ip_dist, hour, avs])
    phi_star = np.array([0.45, 0.35, 0.15, 0.00, 0.05])

    # Perfect explainer (matches ranking and direction)
    phi_perfect = np.array([0.50, 0.30, 0.12, 0.00, 0.08])
    res_perfect = evaluator.evaluate_instance(phi_perfect, phi_star, k_values=(2, 3))

    assert res_perfect.precision_at_k[2] == 1.0
    assert res_perfect.precision_at_k[3] == 1.0
    assert res_perfect.cosine_similarity > 0.98
    assert res_perfect.kendall_tau > 0.80

    # Inverted / spurious explainer (highlights hour and zeroes velocity)
    phi_inverted = np.array([0.00, 0.05, 0.00, 0.85, 0.10])
    res_inverted = evaluator.evaluate_instance(phi_inverted, phi_star, k_values=(2, 3))

    # Top-2 features in inverted are index 3 (hour) and index 4 (avs). Index 3 has phi_star == 0!
    assert res_inverted.precision_at_k[2] <= 0.50
    assert res_inverted.cosine_similarity < 0.30


def test_end_to_end_batch_with_counterfactuals():
    """Full batch generation must attach complete counterfactual metadata and narratives."""
    engine = DiscreteEventEngine(n_cards=50, n_merchants=20, seed=42)
    records = engine.generate_batch(n_transactions=200, fraud_prevalence=0.08)

    fraud_records = [r for r in records if r["is_fraud"] == 1]
    assert len(fraud_records) > 0

    for r in fraud_records:
        assert "counterfactual_input_deltas" in r
        assert "dominant_causal_driver" in r
        assert "explanation_narrative" in r
        assert len(r["explanation_narrative"]) > 0


def test_xai_benchmark_harness_execution():
    """XAIBenchmarkHarness must execute end-to-end with TreeSHAP and compute concordance metrics."""
    from prism_generator import XAIBenchmarkHarness

    harness = XAIBenchmarkHarness(
        n_transactions=300,
        fraud_prevalence=0.10,
        seed=999,
    )
    summary = harness.run_benchmark(model_type="lightgbm")

    assert summary.model_name == "lightgbm"
    assert "TreeSHAP" in summary.explainer_name
    assert summary.n_evaluated_samples > 0
    assert -1.0 <= summary.mean_kendall_tau <= 1.0
    assert -1.0 <= summary.mean_cosine_similarity <= 1.0
    assert 0.0 <= summary.mean_precision_at_3 <= 1.0
    assert 0.0 <= summary.auc_roc <= 1.0


def test_random_forest_benchmark_runs():
    """The RandomForest path must survive SHAP returning one array of shape (n, d, 2) for a binary forest."""
    pytest.importorskip("shap")
    from prism_generator import XAIBenchmarkHarness

    harness = XAIBenchmarkHarness(n_transactions=300, fraud_prevalence=0.08, region="US", seed=11)
    summary = harness.run_benchmark(model_type="rf")
    assert summary.model_name == "rf"
    assert summary.n_evaluated_samples > 0
    assert 0.0 <= summary.pr_auc <= 1.0
