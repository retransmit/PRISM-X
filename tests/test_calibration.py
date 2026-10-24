"""Tests for score calibration.

Regression guard for a bug found during the first end-to-end run: a min-max
calibrator clipped every genuine anomaly to exactly 1.0, because at 0.2% fraud
the 99th percentile of training scores is still a normal transaction. The
tuned threshold came out as 1.0000 and alerts could not be ranked.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from prism.models.base import ScoreCalibrator


@pytest.fixture
def anomaly_scores():
    """Raw scores at realistic prevalence: 0.2% sitting far out in the tail."""
    rng = np.random.default_rng(3)
    normal = rng.normal(0.0, 1.0, size=9_980)
    anomalous = rng.normal(8.0, 1.5, size=20)
    scores = np.concatenate([normal, anomalous])
    labels = np.concatenate([np.zeros(9_980, dtype=int), np.ones(20, dtype=int)])
    return scores, labels


def test_calibration_is_strictly_monotonic(anomaly_scores):
    scores, _ = anomaly_scores
    calibrator = ScoreCalibrator().fit(scores)

    ordered = np.sort(scores)
    calibrated = calibrator.transform(ordered)
    assert np.all(np.diff(calibrated) >= 0), "calibration reordered the scores"


def test_calibration_preserves_pr_auc(anomaly_scores):
    """A monotone transform must not change a rank-based metric."""
    scores, labels = anomaly_scores
    calibrated = ScoreCalibrator().fit(scores).transform(scores)

    assert average_precision_score(labels, calibrated) == pytest.approx(
        average_precision_score(labels, scores), rel=1e-9
    )


def test_anomalies_do_not_saturate(anomaly_scores):
    """The bug: every anomaly pinned to 1.0, so alerts could not be ranked."""
    scores, labels = anomaly_scores
    calibrated = ScoreCalibrator().fit(scores).transform(scores)

    anomaly_values = calibrated[labels == 1]
    assert np.all(anomaly_values < 1.0), "anomaly scores saturated at 1.0"
    # They must remain distinguishable from one another, not all equal.
    assert len(np.unique(anomaly_values)) > 1


def test_output_stays_in_the_open_unit_interval(anomaly_scores):
    scores, _ = anomaly_scores
    calibrator = ScoreCalibrator().fit(scores)
    extreme = calibrator.transform(np.array([-1e9, 0.0, 1e9]))

    assert np.all(extreme > 0.0)
    assert np.all(extreme < 1.0)
    assert np.all(np.isfinite(extreme))


def test_ordinary_traffic_scores_below_the_midpoint(anomaly_scores):
    """Most normal transactions should look unremarkable on the 0-1 scale."""
    scores, labels = anomaly_scores
    calibrated = ScoreCalibrator().fit(scores).transform(scores)
    assert np.median(calibrated[labels == 0]) < 0.5


def test_constant_scores_do_not_divide_by_zero():
    calibrator = ScoreCalibrator().fit(np.full(500, 2.5))
    out = calibrator.transform(np.full(10, 2.5))
    assert np.all(np.isfinite(out))


def test_calibrator_survives_a_state_round_trip(anomaly_scores):
    scores, _ = anomaly_scores
    original = ScoreCalibrator().fit(scores)
    restored = ScoreCalibrator.from_state(original.state())

    assert np.allclose(original.transform(scores), restored.transform(scores))


def test_extreme_scores_keep_their_order():
    """Beyond 30 standard units the logistic form tied every score at one value;
    the strongest anomalies were one unordered block."""
    calibrator = ScoreCalibrator().fit(np.linspace(-1.0, 1.0, 1000))
    far = calibrator.center + calibrator.scale * np.array([40.0, 50.0, 60.0, 1e6])
    out = calibrator.transform(far)
    assert np.all(np.diff(out) > 0)
    assert np.all(out < 1.0)


def test_a_model_calibrated_under_the_old_form_is_refused():
    with pytest.raises(ValueError, match="Retrain"):
        ScoreCalibrator.from_state({"center": 0.0, "scale": 1.0})
