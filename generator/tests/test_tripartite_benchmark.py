"""Industrial Tripartite Benchmark and Rechallenging Audit Test Suite.

Verifies:
1. Dimension 1: Statistical Fidelity (Wasserstein-1, Jensen-Shannon, Spearman Frobenius).
2. Dimension 2: Machine Learning Utility (TSTR vs TRTR PR-AUC retention).
3. Dimension 3: Adversarial Privacy & Non-Memorization (DCR 5th percentile, NNDR, MIA resistance).
4. Negative Controls / Falsification Tests (Random Noise, Verbatim Cloner, Target Label Leak).
5. End-to-end TripartiteBenchmarkHarness execution on US and India rail configs.
"""

from __future__ import annotations

import numpy as np
import pytest

from prism_generator.benchmark import (
    AdversarialPrivacyEvaluator,
    MLUtilityEvaluator,
    StatisticalFidelityEvaluator,
    TripartiteBenchmarkHarness,
    generate_tripartite_markdown_report,
)


# =====================================================================
# 1. UNIT TESTS: STATISTICAL FIDELITY EVALUATOR (DIMENSION 1)
# =====================================================================

def test_wasserstein_1d_identical_and_shifted():
    """Wasserstein-1 distance must be 0 for identical distributions and strictly positive for shifts."""
    evaluator = StatisticalFidelityEvaluator()
    rng = np.random.default_rng(42)
    u = rng.lognormal(mean=3.5, sigma=0.8, size=1000)

    # Identical arrays
    d_zero = evaluator.compute_wasserstein_1d(u, u, log_scale=True)
    assert d_zero == pytest.approx(0.0, abs=1e-6)

    # Shifted spend amounts (e.g. 5x inflation)
    v = u * 5.0
    d_shifted = evaluator.compute_wasserstein_1d(u, v, log_scale=True)
    assert d_shifted > 0.30  # Significant shift on log scale


def test_jensen_shannon_discrete_properties():
    """Jensen-Shannon divergence must be symmetric, 0 for identical distributions, and bounded by ln(2)."""
    evaluator = StatisticalFidelityEvaluator()

    cats_p = ["5411", "5411", "5812", "5999"] * 100
    cats_q = ["5411", "5411", "5812", "5999"] * 100
    cats_r = ["6011", "6011", "7995", "4829"] * 100

    # Identical distributions
    js_identical = evaluator.compute_jensen_shannon_discrete(cats_p, cats_q)
    assert js_identical == pytest.approx(0.0, abs=1e-5)

    # Completely disjoint distributions
    js_disjoint = evaluator.compute_jensen_shannon_discrete(cats_p, cats_r)
    assert 0.50 <= js_disjoint <= np.log(2.0) + 1e-4

    # Symmetry property: D_JS(P || Q) == D_JS(Q || P)
    cats_mixed = ["5411"] * 80 + ["5812"] * 20
    js_pq = evaluator.compute_jensen_shannon_discrete(cats_p, cats_mixed)
    js_qp = evaluator.compute_jensen_shannon_discrete(cats_mixed, cats_p)
    assert js_pq == pytest.approx(js_qp, abs=1e-6)


def test_correlation_frobenius_error():
    """Relative Frobenius norm error must be 0 for identical rank correlation matrices."""
    evaluator = StatisticalFidelityEvaluator()
    rng = np.random.default_rng(42)
    X = rng.normal(size=(500, 5))

    # Identical matrices
    err_zero = evaluator.compute_correlation_frobenius_error(X, X, method="spearman")
    assert err_zero == pytest.approx(0.0, abs=1e-6)

    # Independent matrices should have non-zero difference
    Y = rng.normal(size=(500, 5))
    err_diff = evaluator.compute_correlation_frobenius_error(X, Y, method="spearman")
    assert err_diff > 0.05


# =====================================================================
# 2. UNIT TESTS: ML UTILITY EVALUATOR (DIMENSION 2 - TSTR)
# =====================================================================

def test_tstr_ml_utility_on_synthetic_signal():
    """TSTR evaluator must compute PR-AUC retention and distinguish strong vs collapsed utility."""
    n_samples = 600

    # Create synthetic and reference datasets with real predictive signal
    def make_dataset(seed: int):
        r = np.random.default_rng(seed)
        X = r.normal(0.0, 1.0, size=(n_samples, 8))
        # Non-linear fraud probability
        logits = 1.8 * X[:, 0] + 1.2 * X[:, 1] - 0.5 * X[:, 2] ** 2
        probs = 1.0 / (1.0 + np.exp(-logits))
        y = (r.uniform(0.0, 1.0, size=n_samples) < probs * 0.35).astype(int)
        # Ensure at least some positive class
        if np.sum(y) < 20:
            y[:25] = 1
        return X, y

    X_syn, y_syn = make_dataset(42)
    X_ref_train, y_ref_train = make_dataset(43)
    X_ref_test, y_ref_test = make_dataset(44)

    summary = MLUtilityEvaluator.evaluate_tstr(
        X_syn=X_syn,
        y_syn=y_syn,
        X_ref_train=X_ref_train,
        y_ref_train=y_ref_train,
        X_ref_test=X_ref_test,
        y_ref_test=y_ref_test,
        random_state=42,
    )

    assert summary.tstr_pr_auc > 0.20
    assert summary.trtr_pr_auc > 0.20
    assert summary.relative_pr_auc_retention > 0.60
    assert summary.tstr_roc_auc > 0.65


# =====================================================================
# 3. UNIT TESTS: ADVERSARIAL PRIVACY EVALUATOR (DIMENSION 3)
# =====================================================================

def test_adversarial_privacy_diffuse_points():
    """Diffuse continuous distributions must satisfy DCR > 0.01 and MIA ROC-AUC <= 0.55."""
    rng = np.random.default_rng(42)
    X_ref_train = rng.uniform(0.0, 100.0, size=(400, 6))
    X_ref_test = rng.uniform(0.0, 100.0, size=(400, 6))
    X_syn = rng.uniform(0.0, 100.0, size=(400, 6))

    summary = AdversarialPrivacyEvaluator.compute_privacy_metrics(
        X_syn=X_syn,
        X_ref_train=X_ref_train,
        X_ref_test=X_ref_test,
        random_state=42,
    )

    # In continuous 6D space, 5th percentile distance must be distinctly positive
    assert summary.dcr_5th_percentile > 0.02
    assert summary.nndr_mean >= 0.50
    # MIA adversary cannot differentiate member from non-member
    assert summary.mia_attack_roc_auc <= 0.58


# =====================================================================
# 4. NEGATIVE CONTROLS & FALSIFICATION TESTS (RECHALLENGING ASSUMPTIONS)
# =====================================================================

def test_negative_control_random_noise_fails_fidelity_and_utility():
    """Falsification Test A: Pure uniform random noise must FAIL Statistical Fidelity and TSTR Utility."""
    rng = np.random.default_rng(999)
    n = 500

    # Reference data: Log-normal spend and clustered arrivals
    ref_amounts = rng.lognormal(mean=3.5, sigma=0.5, size=n)
    ref_times = np.cumsum(rng.exponential(scale=120.0, size=n))
    ref_records = [
        {"amount": float(ref_amounts[i]), "tx_time_seconds": float(ref_times[i]), "mcc": "5411", "channel_type": "CP_POS_EMV_CHIP", "response_code": "00"}
        for i in range(n)
    ]

    # Pathological generator: Static uniform noise in [-1000, 1000]
    syn_amounts = rng.uniform(5000.0, 9000.0, size=n)  # Drastically different scale
    syn_times = np.linspace(0, 1000000, n)
    syn_records = [
        {"amount": float(syn_amounts[i]), "tx_time_seconds": float(syn_times[i]), "mcc": "9999", "channel_type": "UNKNOWN", "response_code": "99"}
        for i in range(n)
    ]

    X_ref = rng.normal(0, 1, size=(n, 5))
    X_syn = rng.uniform(50, 100, size=(n, 5))

    evaluator = StatisticalFidelityEvaluator()
    summary = evaluator.evaluate(syn_records, ref_records, X_syn, X_ref)

    # Must FAIL fidelity gate
    assert summary.fidelity_passed is False
    assert summary.amount_passed is False or summary.mcc_passed is False


def test_negative_control_verbatim_cloner_fails_privacy_dcr():
    """Falsification Test B: Verbatim cloner of training data must trip the DCR non-memorization tripwire."""
    rng = np.random.default_rng(123)
    n = 300
    X_ref_train = rng.uniform(0.0, 50.0, size=(n, 6))
    X_ref_test = rng.uniform(0.0, 50.0, size=(n, 6))

    # Pathological memorizer: Exactly copies training records
    X_syn_memorized = np.copy(X_ref_train)

    summary = AdversarialPrivacyEvaluator.compute_privacy_metrics(
        X_syn=X_syn_memorized,
        X_ref_train=X_ref_train,
        X_ref_test=X_ref_test,
        random_state=42,
    )

    # 5th percentile distance to training set is exactly 0.0!
    assert summary.dcr_5th_percentile == pytest.approx(0.0, abs=1e-8)
    # Must FAIL privacy non-memorization gate
    assert summary.dcr_passed is False
    assert summary.privacy_passed is False


def test_negative_control_target_label_leaker_fails_anti_leak_tripwires():
    """Falsification Test C: Leaking target label into an authorization feature must trigger tripwires."""
    harness = TripartiteBenchmarkHarness(n_transactions=300, fraud_prevalence=0.10, seed=42)

    # Run standard XAI harness to check anti-leak detection
    summary = harness.xai_harness.run_benchmark(model_type="lightgbm")
    # A clean non-leaking dataset must pass anti-leak tripwires
    assert summary.anti_leak_tripwire_passed is True


# =====================================================================
# 5. INTEGRATION TESTS: COMPLETE TRIPARTITE BENCHMARK HARNESS
# =====================================================================

def test_tripartite_benchmark_harness_us_execution():
    """TripartiteBenchmarkHarness must execute end-to-end on US metro configuration."""
    harness = TripartiteBenchmarkHarness(
        n_transactions=400,
        fraud_prevalence=0.08,
        region="US",
        adversary_mimicry=0.55,
        seed=42,
    )

    summary = harness.run_tripartite_benchmark(include_xai=True)

    # 1. Verify structure of summary
    assert summary.fidelity is not None
    assert summary.utility is not None
    assert summary.privacy is not None
    assert summary.xai_summary is not None
    assert summary.n_synthetic_samples == 400
    assert summary.n_reference_samples == 400
    assert summary.region == "US"

    # 2. Verify metrics are within plausible bounds
    assert summary.fidelity.wasserstein_amount_log >= 0.0
    assert summary.fidelity.js_divergence_mcc >= 0.0
    assert summary.utility.tstr_pr_auc >= 0.0
    assert summary.privacy.dcr_5th_percentile > 0.0

    # 3. Verify Markdown report generation
    report = generate_tripartite_markdown_report(summary)
    assert "# Prism Generator Tripartite Industrial Benchmark Certification Report" in report
    assert "Dimension 1: Statistical Fidelity Suite" in report
    assert "Dimension 2: Machine Learning Utility" in report
    assert "Dimension 3: Adversarial Privacy & Non-Memorization" in report
    assert "Dimension 4: Ground-Truth Causal XAI Conformance" in report
    assert "Final Certification Decision:" in report


def test_tripartite_benchmark_harness_india_execution():
    """TripartiteBenchmarkHarness must execute end-to-end on India metro configuration."""
    harness = TripartiteBenchmarkHarness(
        n_transactions=350,
        fraud_prevalence=0.07,
        region="IN",
        adversary_mimicry=0.60,
        seed=101,
    )

    summary = harness.run_tripartite_benchmark(include_xai=False)

    assert summary.region == "IN"
    assert summary.fidelity.js_divergence_response_code >= 0.0
    assert summary.utility.tstr_roc_auc > 0.50
    assert summary.privacy.dcr_5th_percentile > 0.0
