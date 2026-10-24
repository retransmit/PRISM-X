"""Regression tests for the two rules the evaluation's validity rests on.

If either of these fails, every metric in the report is wrong. They run on
synthetic data so they need no Kaggle download.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism import config, tracks
from prism.data.preprocess import prepare_splits
from prism.explain import artifacts
from prism.training.resampling import resample_training_fold

PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]


@pytest.fixture
def imbalanced_df() -> pd.DataFrame:
    """20,000 rows at roughly the real fraud prevalence."""
    rng = np.random.default_rng(0)
    n, n_fraud = 20_000, 40

    data = {name: rng.normal(size=n) for name in PCA_COLUMNS}
    data["Time"] = rng.uniform(0, 172_800, size=n)
    data["Amount"] = rng.gamma(2.0, 50.0, size=n)

    labels = np.zeros(n, dtype=int)
    labels[rng.choice(n, size=n_fraud, replace=False)] = 1
    data["Class"] = labels

    return pd.DataFrame(data)[["Time", *PCA_COLUMNS, "Amount", "Class"]]


def test_scaler_is_fitted_on_training_rows_only(imbalanced_df):
    """The scaler must not have seen validation or test statistics."""
    splits = prepare_splits(imbalanced_df)
    scale_columns = list(tracks.ULB.scale_columns)

    raw_train = imbalanced_df.loc[splits.X_train.index, scale_columns]

    # Scaler means match the training rows exactly...
    assert np.allclose(splits.scaler.mean_, raw_train.mean().to_numpy())

    # ...and not the full dataset, which is what leakage would look like.
    full_mean = imbalanced_df[scale_columns].mean().to_numpy()
    assert not np.allclose(splits.scaler.mean_, full_mean, atol=1e-9)


def test_splits_are_disjoint(imbalanced_df):
    splits = prepare_splits(imbalanced_df)
    train, val, test = (
        set(splits.X_train.index),
        set(splits.X_val.index),
        set(splits.X_test.index),
    )
    assert not train & val
    assert not train & test
    assert not val & test
    assert len(train | val | test) == len(imbalanced_df)


def test_stratification_preserves_fraud_rate(imbalanced_df):
    splits = prepare_splits(imbalanced_df)
    overall = imbalanced_df["Class"].mean()
    for y in (splits.y_train, splits.y_val, splits.y_test):
        assert y.mean() == pytest.approx(overall, rel=0.25)


def test_test_set_is_never_resampled(imbalanced_df):
    """Resampling the training fold must leave the test split untouched."""
    splits = prepare_splits(imbalanced_df)
    test_rows_before = len(splits.X_test)
    test_frauds_before = int(splits.y_test.sum())

    X_res, y_res = resample_training_fold(splits.X_train, splits.y_train)

    # The training fold grew...
    assert len(X_res) >= len(splits.X_train)
    assert int(y_res.sum()) >= int(splits.y_train.sum())

    # ...and the test split is byte-identical.
    assert len(splits.X_test) == test_rows_before
    assert int(splits.y_test.sum()) == test_frauds_before


def test_prepare_splits_is_deterministic(imbalanced_df):
    """Same seed, same partition; benchmark.py depends on this."""
    first = prepare_splits(imbalanced_df)
    second = prepare_splits(imbalanced_df)
    assert list(first.X_test.index) == list(second.X_test.index)


def test_exact_duplicates_cannot_straddle_the_split(imbalanced_df):
    """The real file carries 1,081 exact duplicate rows, 19 of them fraud. Left
    in, 298 test rows are byte-for-byte copies of training rows and the model
    is scored on data it has already seen. Dropping duplicates before the split
    is the only thing that rules that out."""
    doubled = pd.concat([imbalanced_df, imbalanced_df], ignore_index=True)
    splits = prepare_splits(doubled)

    kept = len(splits.X_train) + len(splits.X_val) + len(splits.X_test)
    assert kept == len(imbalanced_df), "every duplicate dropped, nothing else"
    assert splits.duplicates_dropped == len(imbalanced_df)

    def rows(X, y):
        return set(zip(map(tuple, X.to_numpy()), y.to_numpy()))

    train = rows(splits.X_train, splits.y_train)
    assert not train & rows(splits.X_val, splits.y_val)
    assert not train & rows(splits.X_test, splits.y_test)


def test_explanation_reference_is_fitted_on_training_rows_only(imbalanced_df):
    """The narrative's "larger than X% of transactions" and "N of the 50 most
    similar" are measured against training rows only. Fitting them on the full
    frame would let the sentence describe a test row against itself."""
    splits = prepare_splits(imbalanced_df)
    reference, cases = artifacts.build(imbalanced_df, splits)
    assert reference.n_rows == len(splits.X_train) < len(imbalanced_df)
    assert cases.n_rows == len(splits.X_train)


def test_cv_threshold_is_not_tuned_on_the_fold_it_is_scored_on(imbalanced_df, monkeypatch):
    """Per fold, the threshold was tuned on the validation rows and those same
    rows were then scored with it. Tuning must use rows the fold score never
    sees: an inner holdout carved from the training portion."""
    from prism.models import build
    from prism.training import resampling

    tuned_on, scored_on = [], []
    real_tune, real_eval = resampling.tune_threshold, resampling.evaluate

    def spy_tune(y, scores, *args, **kwargs):
        tuned_on.append(set(y.index))
        return real_tune(y, scores, *args, **kwargs)

    def spy_eval(y, scores, threshold, **kwargs):
        scored_on.append(set(y.index))
        return real_eval(y, scores, threshold, **kwargs)

    monkeypatch.setattr(resampling, "tune_threshold", spy_tune)
    monkeypatch.setattr(resampling, "evaluate", spy_eval)

    # The raw rows with their target: each fold fits its own preprocessing.
    resampling.cross_val_scores(
        lambda: build("isolation_forest"), imbalanced_df, tracks.ULB, n_splits=3, resample=False, verbose=False
    )

    assert len(tuned_on) == len(scored_on) == 3
    for tuned, scored in zip(tuned_on, scored_on):
        assert not (tuned & scored), "threshold tuned on rows it was then scored on"


def test_resampling_keeps_categorical_codes_valid(india_stem):
    """SMOTE interpolates; a category code halfway between two channels is not a
    channel. On the Indian track the resampler is SMOTENC, which copies codes."""
    from prism.data import india
    from prism.data.preprocess import prepare_splits
    from prism.features.velocity import add_velocity_features

    splits = prepare_splits(add_velocity_features(india.load_india(india_stem)), schema=tracks.INDIA)
    X_res, y_res = resample_training_fold(
        splits.X_train, splits.y_train, categorical_columns=splits.categorical_columns
    )
    assert int(y_res.sum()) > int(splits.y_train.sum())
    for column in splits.categorical_columns:
        seen = set(splits.X_train[column].unique().tolist())
        assert set(pd.Series(X_res[column]).unique().tolist()) <= seen, column
