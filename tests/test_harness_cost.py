"""The cost model: a missed fraud costs its amount, an alert costs a review, a false
positive costs the friction of a held legitimate payment; everything per 10,000 rows."""

from __future__ import annotations

import numpy as np

from prism.harness import cost


def test_the_four_row_example():
    y_true = np.array([1, 1, 0, 0])
    flagged = np.array([1, 0, 1, 0])  # one caught, one missed, one false positive, one clean
    amounts = np.array([500.0, 2000.0, 300.0, 80.0])
    out = cost.cost_per_10k(
        y_true, flagged, amounts, rows=4, review_inr_per_alert=150.0, friction_inr_per_false_positive=60.0
    )
    scale = 10_000 / 4
    assert out["loss_inr"] == 2000.0 * scale
    assert out["review_inr"] == 2 * 150.0 * scale
    assert out["friction_inr"] == 1 * 60.0 * scale
    assert out["total_inr"] == out["loss_inr"] + out["review_inr"] + out["friction_inr"]
    assert out["alerts_per_10k"] == 2 * scale
    assert out["missed_fraud_per_10k"] == 1 * scale


def test_no_alerts_costs_every_fraud_and_nothing_else():
    out = cost.cost_per_10k(
        np.array([1, 0, 1]), np.array([0, 0, 0]), np.array([100.0, 50.0, 25.0]), rows=3,
        review_inr_per_alert=150.0, friction_inr_per_false_positive=60.0,
    )
    assert out["loss_inr"] == 125.0 * (10_000 / 3)
    assert out["review_inr"] == 0.0 and out["friction_inr"] == 0.0
    assert out["total_inr"] == out["loss_inr"]


def test_settings_come_from_the_config_and_are_marked_as_hypotheses():
    s = cost.settings()
    assert s["hypothesis"] is True
    assert s["review_inr_per_alert"] > 0 and s["friction_inr_per_false_positive"] > 0
