"""Limit ratios: the card's limit as context for this payment.

Every card in the Indian stream carries its own credit limit, so the raw
limit identifies the card and a model can memorise which cards were
attacked. The share of the limit in use and the payment as a share of the
limit carry what the limit says about this payment and nothing about which
card it is. Both are computed here for training and for serving.
"""

from __future__ import annotations

import pandas as pd

LIMIT_COLUMN = "credit_limit"
BALANCE_COLUMN = "available_balance"
RATIO_COLUMNS = ("utilization", "amount_to_limit")
# Ratios are clipped: an overdrawn card or a payment above its limit is a
# fact worth keeping, an unbounded value is not.
CEILING = 1.5


def limit_ratios(amount: float, credit_limit: float, available_balance: float) -> dict[str, float]:
    limit = float(credit_limit)
    if limit <= 0.0:
        return {"utilization": CEILING, "amount_to_limit": CEILING}
    utilization = 1.0 - float(available_balance) / limit
    return {
        "utilization": float(min(max(utilization, 0.0), CEILING)),
        "amount_to_limit": float(min(max(float(amount) / limit, 0.0), CEILING)),
    }


def add_limit_ratios(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    limit = out[LIMIT_COLUMN].astype(float)
    positive = limit > 0.0
    safe_limit = limit.where(positive, 1.0)
    utilization = (1.0 - out[BALANCE_COLUMN].astype(float) / safe_limit).clip(0.0, CEILING)
    share = (out["amount"].astype(float) / safe_limit).clip(0.0, CEILING)
    out["utilization"] = utilization.where(positive, CEILING)
    out["amount_to_limit"] = share.where(positive, CEILING)
    return out
