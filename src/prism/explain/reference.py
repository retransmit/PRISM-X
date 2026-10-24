"""Population reference statistics behind the plain-English explanation.

Every "larger than X% of transactions" or "unusual on k of 28 characteristics"
in the narrative is measured against these numbers. They are fitted on the
training split only, so the sentence never leans on data the model is scored
against, and they are persisted beside the models so serving describes the
same population the models were fitted on.

Two things the dataset cannot support are deliberately absent: there is no
card identifier, so nothing here is "unusual for this cardholder", and Time is
seconds since the first row with no known clock offset, so only the position
in the daily cycle is used, never an hour of the day.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd

from prism import config, tracks

SECONDS_PER_DAY = 86_400
HOURS_PER_DAY = 24


@dataclass(frozen=True)
class ReferenceStats:
    amounts: np.ndarray        # sorted raw training amounts
    cycle_share: np.ndarray    # share of training rows in each hour of the daily cycle
    v_low: pd.Series           # lower edge of the normal band, per banded numeric column
    v_high: pd.Series          # upper edge of the normal band, per banded numeric column
    band: float                # fraction of training rows the band covers, e.g. 0.995
    base_rate: float
    n_rows: int
    amount_column: str = "Amount"
    time_column: str = "Time"
    # share of training rows per value, per categorical column (empty on ULB)
    categorical_share: dict[str, dict[str, float]] = field(default_factory=dict)
    # the narrower band the driver sentence compares against (None on an
    # artifact written before it existed; numeric_driver_band then falls back)
    driver_low: pd.Series | None = None
    driver_high: pd.Series | None = None
    driver_band: float | None = None

    @classmethod
    def from_training(
        cls,
        raw_train: pd.DataFrame,
        y_train,
        schema: tracks.TrackSchema | None = None,
        band: float | None = None,
    ) -> "ReferenceStats":
        """Fit on the raw (unscaled) training rows only."""
        schema = schema or tracks.active()
        band = float(band if band is not None else config.get("explain.unusual_band", 0.995))
        # On ULB the banded columns are V1..V28: every numeric that is not the
        # amount or the clock. On India they are the named numerics, the
        # velocity features included.
        banded = [
            c for c in schema.numeric_columns
            if c not in (schema.amount_column, schema.time_column) and c in raw_train.columns
        ]

        amounts = np.sort(raw_train[schema.amount_column].to_numpy(dtype=float))
        clock = raw_train[schema.time_column].to_numpy(dtype=float)
        hours = ((clock % SECONDS_PER_DAY) // 3_600).astype(int)
        cycle_share = np.bincount(hours, minlength=HOURS_PER_DAY) / len(hours)

        tail = (1.0 - band) / 2.0
        driver_band = float(config.get("explain.driver_band", 0.95))
        driver_tail = (1.0 - driver_band) / 2.0
        shares = {
            c: {str(k): float(v) for k, v in raw_train[c].astype(str).value_counts(normalize=True).items()}
            for c in schema.categorical_columns
            if c in raw_train.columns
        }
        return cls(
            amounts=amounts,
            cycle_share=cycle_share,
            v_low=raw_train[banded].quantile(tail),
            v_high=raw_train[banded].quantile(1.0 - tail),
            band=band,
            base_rate=float(np.mean(np.asarray(y_train))),
            n_rows=len(raw_train),
            amount_column=schema.amount_column,
            time_column=schema.time_column,
            categorical_share=shares,
            driver_low=raw_train[banded].quantile(driver_tail),
            driver_high=raw_train[banded].quantile(1.0 - driver_tail),
            driver_band=driver_band,
        )

    @property
    def characteristics(self) -> list[str]:
        return list(self.v_low.index)

    def numeric_band(self, feature: str) -> tuple[float, float]:
        return float(self.v_low[feature]), float(self.v_high[feature])

    def numeric_driver_band(self, feature: str) -> tuple[float, float]:
        """The range the driver sentence quotes; the unusual band when the
        artifact predates the driver band."""
        if self.driver_low is None or self.driver_high is None:
            return self.numeric_band(feature)
        return float(self.driver_low[feature]), float(self.driver_high[feature])

    def describe(self, raw_row: Mapping[str, float]) -> dict:
        """Facts about one raw transaction, relative to the training population."""
        amount = float(raw_row[self.amount_column])
        hour = int((float(raw_row[self.time_column]) % SECONDS_PER_DAY) // 3_600)
        values = pd.Series({c: float(raw_row[c]) for c in self.characteristics})
        unusual = int(((values < self.v_low) | (values > self.v_high)).sum())
        return {
            "amount": amount,
            "amount_pct_below": float(np.searchsorted(self.amounts, amount) / len(self.amounts)),
            "cycle_share": float(self.cycle_share[hour]),
            "typical_cycle_share": 1.0 / HOURS_PER_DAY,
            "unusual_count": unusual,
            "n_characteristics": len(self.characteristics),
            "band": self.band,
        }

    # joblib is a pickle. The only file ever loaded here is the one this class
    # wrote into models/ during training, the same trust model as scaler.joblib
    # and isolation_forest.joblib beside it; nothing is loaded from a request.
    def save(self, path: Path) -> Path:
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path) -> "ReferenceStats":
        return joblib.load(path)
