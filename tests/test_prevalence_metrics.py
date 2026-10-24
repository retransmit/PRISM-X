"""Test-window rates re-weighted to one fraud in 80,847 (design D2, D6)."""

from __future__ import annotations

import numpy as np
import pytest

from prism.evaluation import metrics

R = 1.0 / 80_847.0


def test_adjusted_precision_is_bayes_over_the_two_rates():
    p = metrics.adjusted_precision(tpr=0.8, fpr=0.001, prevalence=R)
    assert p == pytest.approx(0.8 * R / (0.8 * R + 0.001 * (1 - R)))
    assert metrics.adjusted_precision(tpr=0.0, fpr=0.0, prevalence=R) == 0.0


def test_adjusted_curve_matches_the_boosted_window_pointwise():
    y = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.95])
    curve = metrics.adjusted_curve(y, scores, prevalence=2 / 6)
    assert len(curve["recall"]) == len(curve["precision"]) == len(curve["thresholds"])
    # at the window's own prevalence (2 of 6) the adjusted precision is the
    # ordinary precision of the window: at 0.8, two hits and one false alarm
    at_08 = curve["thresholds"].index(pytest.approx(0.8))
    assert curve["precision"][at_08] == pytest.approx(2 / 3)


def test_fpr_per_10k_counts_legitimate_alerts():
    y = np.array([0] * 10_000 + [1] * 10)
    scores = np.zeros(10_010)
    scores[:7] = 0.9
    scores[-10:] = 0.9
    assert metrics.fpr_per_10k(y, scores, threshold=0.5) == pytest.approx(7.0)


def test_value_caught_reports_rupees_above_and_below_the_floor():
    y = np.array([1, 1, 1, 0])
    scores = np.array([0.9, 0.2, 0.9, 0.9])
    amounts = np.array([20_000.0, 500.0, 5_000.0, 100.0])
    out = metrics.value_caught(y, scores, threshold=0.5, amounts=amounts, floor=10_000.0)
    assert out["fraud_value"] == 25_500.0
    assert out["fraud_value_caught"] == 25_000.0
    assert out["share_of_fraud_value_caught"] == pytest.approx(25_000 / 25_500)
    assert out["fraud_value_above_floor"] == 20_000.0
    assert out["share_above_floor_caught"] == pytest.approx(1.0)


def test_track_extras_are_serialisable_floats():
    y = np.array([0] * 200 + [1] * 4)
    scores = np.r_[np.linspace(0, 0.5, 200), [0.9, 0.8, 0.1, 0.95]]
    amounts = np.ones(204) * 1000.0
    extras = metrics.track_extras(y, scores, threshold=0.6, amounts=amounts, prevalence=R, floor=10_000.0)
    assert extras["prevalence_target"] == R
    assert extras["recall_at_threshold"] == pytest.approx(0.75)
    assert extras["fpr_per_10k"] == 0.0
    assert extras["adjusted_precision_at_prevalence"] == pytest.approx(1.0)
    assert extras["adjusted_alerts_per_true_fraud"] == pytest.approx(1.0)
    assert all(isinstance(v, float) for v in extras.values())


def test_results_are_strict_json_even_when_nothing_is_caught():
    """A model that flags nothing has no alerts-per-catch ratio. json.dumps would
    write Infinity, which no browser or strict parser accepts; the files the API
    serves must say null and stay valid JSON."""
    import json

    from prism.evaluation.metrics import evaluate

    y = np.array([0] * 50 + [1] * 3)
    scores = np.zeros(53)
    result = evaluate(y, scores, threshold=0.5, model="quiet")
    extras = metrics.track_extras(y, scores, threshold=0.5, amounts=np.ones(53), prevalence=R, floor=10_000.0)
    text = json.dumps({**result.as_dict(), **extras}, allow_nan=False)
    assert '"alerts_per_true_fraud": null' in text
    assert '"adjusted_alerts_per_true_fraud": null' in text


def test_track_extras_report_the_precision_over_the_registry_range():
    y = np.array([1] * 20 + [0] * 980)
    scores = np.array([0.9] * 20 + [0.1] * 970 + [0.95] * 10)
    extras = metrics.track_extras(
        y, scores, 0.5, np.ones(1000), prevalence=1 / 80_847, floor=10_000.0,
        prevalence_range=(1 / 121_303, 1 / 47_018),
    )
    assert extras["adjusted_precision_low"] < extras["adjusted_precision_at_prevalence"] < extras["adjusted_precision_high"]
