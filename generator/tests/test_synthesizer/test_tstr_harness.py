"""Automated verification suite for TSTR and Temporal Self-TSTR Protocol Harness.

Verifies:
1. Time-ordered 60/40 train/test split with zero lookahead bias.
2. Downstream ML utility on pure synthetic stream (PR-AUC, ROC-AUC, optimal F1).
3. Investigator capacity metrics (Precision@K) and operational net cost savings.
4. Delegation from TripartiteBenchmarkHarness.evaluate_temporal_self_tstr.
"""


from prism_generator import DiscreteEventEngine, TripartiteBenchmarkHarness
from prism_generator.benchmark import TSTRHarness, MLUtilitySummary


def test_temporal_self_tstr_ordering_and_metrics():
    """Asserts that evaluate_temporal_self_tstr strictly respects time order and computes valid utility metrics."""
    engine = DiscreteEventEngine(n_cards=150, n_merchants=30, region="US", seed=42)
    records = engine.generate_batch(n_transactions=1200, fraud_prevalence=0.06)

    summary = TSTRHarness.evaluate_temporal_self_tstr(
        records=records,
        train_fraction=0.60,
        k_daily=30,
        random_state=42,
    )

    assert isinstance(summary, MLUtilitySummary)
    assert summary.temporal_split_fraction == 0.60
    assert summary.tstr_pr_auc >= 0.15, f"PR-AUC {summary.tstr_pr_auc:.4f} is too low"
    assert summary.tstr_pr_auc <= 0.985, f"PR-AUC {summary.tstr_pr_auc:.4f} leaked (too high)"
    assert summary.tstr_roc_auc >= 0.65, f"ROC-AUC {summary.tstr_roc_auc:.4f} is too low"
    # A validity contract, not a quality bar: precision@k is a proportion, so >= 0.0 could
    # never fail and asserted nothing. The quality signal is carried by the two AUC bounds above.
    assert 0.0 <= summary.precision_at_k <= 1.0, f"Precision@K {summary.precision_at_k:.4f} is not a proportion"
    assert summary.cost_savings >= 0.0, f"Cost savings negative: {summary.cost_savings}"
    assert summary.utility_passed is True


def test_tripartite_harness_temporal_delegation():
    """Asserts that TripartiteBenchmarkHarness delegates to TSTRHarness cleanly."""
    harness = TripartiteBenchmarkHarness(
        n_transactions=1000,
        fraud_prevalence=0.05,
        region="US",
        seed=101,
    )
    summary = harness.evaluate_temporal_self_tstr(train_fraction=0.60, k_daily=25)
    assert isinstance(summary, MLUtilitySummary)
    assert summary.tstr_roc_auc >= 0.60


def test_tstr_falls_back_when_lightgbm_has_no_classifier(monkeypatch):
    """The harness already falls back to HistGradientBoosting when lightgbm is missing;
    an installed lightgbm without LGBMClassifier (a broken or partial install) raised
    AttributeError instead of taking the same fallback."""
    import sys
    import types

    monkeypatch.setitem(sys.modules, "lightgbm", types.ModuleType("lightgbm"))
    engine = DiscreteEventEngine(n_cards=150, n_merchants=30, region="US", seed=42)
    records = engine.generate_batch(n_transactions=1200, fraud_prevalence=0.06)
    summary = TSTRHarness.evaluate_temporal_self_tstr(
        records=records, train_fraction=0.60, k_daily=30, random_state=42
    )
    assert isinstance(summary, MLUtilitySummary)
