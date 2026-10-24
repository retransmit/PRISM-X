"""Tests for decision-threshold tuning and the metric contract."""

from __future__ import annotations

import numpy as np
import pytest

from prism.evaluation.metrics import evaluate
from prism.evaluation.threshold import tune_threshold


@pytest.fixture
def separable_scores():
    """1,000 rows, 20 fraud, with fraud scoring visibly higher than normal."""
    rng = np.random.default_rng(7)
    y = np.zeros(1000, dtype=int)
    y[:20] = 1

    scores = np.empty(1000)
    scores[:20] = rng.uniform(0.60, 0.99, size=20)
    scores[20:] = rng.uniform(0.00, 0.40, size=980)
    return y, scores


def test_f1_threshold_separates_the_classes(separable_scores):
    """On separable data the F1-optimal cutoff must classify everything correctly.

    Asserting the property rather than a numeric range; the exact threshold is
    the minimum fraud score, which moves with the RNG.
    """
    y, scores = separable_scores
    threshold = tune_threshold(y, scores, strategy="f1")

    assert scores[20:].max() < threshold <= scores[:20].max()
    result = evaluate(y, scores, threshold)
    assert result.precision == 1.0
    assert result.recall == 1.0


def test_recall_at_strategy_meets_its_target(separable_scores):
    y, scores = separable_scores
    threshold = tune_threshold(y, scores, strategy="recall_at", target_recall=0.9)
    achieved = evaluate(y, scores, threshold).recall
    assert achieved >= 0.9


def test_recall_at_holds_even_on_an_inverted_model():
    """A model ranking fraud *below* normal still yields a usable threshold.

    Recall is always reachable at the lowest cutoff (flag everything), so the
    strategy degrades into terrible precision rather than failing, which is the
    behaviour the API needs. The precision cost is what the report should show.
    """
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.9, 0.8, 0.1, 0.2])  # deliberately inverted
    threshold = tune_threshold(y, scores, strategy="recall_at", target_recall=0.99)

    result = evaluate(y, scores, threshold)
    assert np.isfinite(threshold)
    assert result.recall >= 0.99
    assert result.precision < 1.0


def test_threshold_tuning_rejects_a_split_with_no_fraud():
    with pytest.raises(ValueError, match="no fraud"):
        tune_threshold(np.zeros(50, dtype=int), np.random.default_rng(0).random(50))


def test_perfect_separation_scores_perfectly(separable_scores):
    y, scores = separable_scores
    result = evaluate(y, scores, threshold=0.5, model="test")
    assert result.recall == 1.0
    assert result.precision == 1.0
    assert result.false_negatives == 0
    assert result.pr_auc == pytest.approx(1.0)


def test_trivial_all_negative_model_is_exposed_by_pr_auc():
    """The failure mode §3 warns about: 99.8% accurate, useless, and PR-AUC says so."""
    y = np.zeros(1000, dtype=int)
    y[:2] = 1
    scores = np.full(1000, 0.01)  # predicts "normal" for everything

    result = evaluate(y, scores, threshold=0.5, model="trivial")
    assert result.recall == 0.0
    assert result.true_positives == 0
    # PR-AUC collapses to roughly the class prevalence, not something flattering.
    assert result.pr_auc < 0.05


def test_alerts_per_true_fraud_is_inverse_precision():
    y = np.array([0, 0, 0, 1])
    scores = np.array([0.9, 0.9, 0.1, 0.9])  # 3 alerts, 1 real
    result = evaluate(y, scores, threshold=0.5)
    assert result.alerts_per_true_fraud == pytest.approx(3.0)


def test_fpr_budget_picks_the_most_sensitive_cutoff_inside_the_budget():
    from sklearn.metrics import roc_curve

    rng = np.random.default_rng(5)
    negatives = rng.uniform(0.0, 0.6, size=20_000)
    positives = rng.uniform(0.3, 1.0, size=200)
    y = np.r_[np.zeros_like(negatives), np.ones_like(positives)].astype(int)
    scores = np.r_[negatives, positives]

    threshold = tune_threshold(y, scores, strategy="fpr_budget", fpr_budget_per_10k=5)
    fpr = float(((scores >= threshold) & (y == 0)).sum() / (y == 0).sum())
    assert fpr <= 5 / 10_000
    fpr_curve, tpr_curve, cutoffs = roc_curve(y, scores)
    best_recall = tpr_curve[fpr_curve <= 5 / 10_000].max()
    assert float(((scores >= threshold) & (y == 1)).sum() / (y == 1).sum()) == pytest.approx(best_recall)


def test_fpr_budget_that_nothing_meets_flags_nothing():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    threshold = tune_threshold(y, scores, strategy="fpr_budget", fpr_budget_per_10k=0)
    assert (scores >= threshold).sum() == 0
    assert threshold <= 1.0


def test_default_strategy_follows_the_active_track(monkeypatch):
    from prism import tracks

    y = np.array([0] * 50 + [1] * 5)
    scores = np.linspace(0.0, 1.0, 55)
    tracks.set_active("india")
    try:
        budget = tune_threshold(y, scores)
    finally:
        tracks.set_active(None)
    f1 = tune_threshold(y, scores, strategy="f1")
    assert budget >= f1


def test_the_budget_alone_sets_a_cutoff_when_no_fraud_is_known():
    from prism.evaluation.threshold import budget_threshold

    negatives = np.linspace(0.0, 0.99, 10_000)
    cutoff = budget_threshold(negatives, fpr_budget_per_10k=5)
    assert int((negatives >= cutoff).sum()) == 5
    # a budget below one row flags nothing
    assert int((np.linspace(0.0, 0.5, 100) >= budget_threshold(np.linspace(0.0, 0.5, 100), fpr_budget_per_10k=5)).sum()) == 0
