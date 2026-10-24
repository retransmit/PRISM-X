"""The --cv diagnostic: folds that respect time, preprocessing fitted inside
each fold, and no stacking of resampling on class weight."""

from __future__ import annotations

import numpy as np
import pytest

from prism import tracks
from prism.data import india
from prism.features.velocity import add_velocity_features
from prism.models.base import FraudModel
from prism.training.resampling import cross_val_scores, model_factory


def test_the_resampled_factory_pins_the_class_weight():
    pytest.importorskip("xgboost")
    assert model_factory("xgboost", resample=True)()._explicit_pos_weight == 1.0
    assert model_factory("xgboost", resample=False)()._explicit_pos_weight is None


def test_time_ordered_folds_never_fit_on_rows_later_than_the_ones_they_score(india_stem):
    frame = add_velocity_features(india.load_india(india_stem))
    seen: list[tuple[float, float]] = []

    class Clock(FraudModel):
        name = "clock"

        def fit(self, X, y=None):
            # The clock is not a feature; read it from the frame by the rows' index.
            self.latest_fit = float(frame.loc[X.index, "tx_time_seconds"].max())
            return self

        def risk_score(self, X):
            seen.append((self.latest_fit, float(frame.loc[X.index, "tx_time_seconds"].min())))
            return np.linspace(0.0, 1.0, len(X))

        def save(self, model_dir):
            raise NotImplementedError

        @classmethod
        def load(cls, model_dir):
            raise NotImplementedError

    cross_val_scores(Clock, frame, tracks.INDIA, n_splits=3, resample=False, verbose=False)
    assert seen and all(latest_fit < earliest_scored for latest_fit, earliest_scored in seen)


def test_folds_fit_their_own_preprocessing(india_stem):
    """A fold's scaler is fitted on its fitting rows: the fitting rows of a
    fold have mean zero on a scaled column, which rows scaled by another
    fold's scaler do not."""
    frame = add_velocity_features(india.load_india(india_stem))
    means: list[float] = []

    class Mean(FraudModel):
        name = "mean"

        def fit(self, X, y=None):
            means.append(float(X["amount"].mean()))
            return self

        def risk_score(self, X):
            return np.linspace(0.0, 1.0, len(X))

        def save(self, model_dir):
            raise NotImplementedError

        @classmethod
        def load(cls, model_dir):
            raise NotImplementedError

    cross_val_scores(Mean, frame, tracks.INDIA, n_splits=3, resample=False, verbose=False)
    assert means and all(abs(m) < 1e-9 for m in means)
