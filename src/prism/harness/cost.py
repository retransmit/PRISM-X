"""The cost of a model's decisions on a test window, per 10,000 transactions: the amount of
every missed fraud, a review per alert, a friction cost per false positive. The rates are
hypotheses in config.yaml, marked as such, until an issuer's figures replace them."""

from __future__ import annotations

import numpy as np

from prism import config


def settings() -> dict:
    return dict(config.get("harness.cost"))


def cost_per_10k(
    y_true: np.ndarray,
    flagged: np.ndarray,
    amounts: np.ndarray,
    *,
    rows: int,
    review_inr_per_alert: float,
    friction_inr_per_false_positive: float,
) -> dict[str, float]:
    y = np.asarray(y_true).astype(int)
    f = np.asarray(flagged).astype(int)
    a = np.asarray(amounts, dtype=float)
    scale = 10_000.0 / float(rows)
    missed = (y == 1) & (f == 0)
    false_positive = (y == 0) & (f == 1)
    loss = float(a[missed].sum()) * scale
    review = float(f.sum()) * float(review_inr_per_alert) * scale
    friction = float(false_positive.sum()) * float(friction_inr_per_false_positive) * scale
    return {
        "loss_inr": loss,
        "review_inr": review,
        "friction_inr": friction,
        "total_inr": loss + review + friction,
        "alerts_per_10k": float(f.sum()) * scale,
        "missed_fraud_per_10k": float(missed.sum()) * scale,
    }
