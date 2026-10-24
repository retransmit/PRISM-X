"""The explanation's reference material is fitted on training rows only and
lives beside the models, so serving describes the population the models saw."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.data.preprocess import prepare_splits
from prism.explain import artifacts

PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(9)
    n, n_fraud = 6_000, 30
    data = {name: rng.normal(size=n) for name in PCA_COLUMNS}
    data["Time"] = rng.uniform(0, 172_800, size=n)
    data["Amount"] = rng.gamma(2.0, 50.0, size=n)
    labels = np.zeros(n, dtype=int)
    labels[rng.choice(n, size=n_fraud, replace=False)] = 1
    data["Class"] = labels
    return pd.DataFrame(data)[["Time", *PCA_COLUMNS, "Amount", "Class"]]


def test_both_artifacts_are_fitted_on_exactly_the_training_rows(frame):
    splits = prepare_splits(frame)
    reference, cases = artifacts.build(frame, splits)
    assert reference.n_rows == len(splits.X_train)
    assert cases.n_rows == len(splits.X_train)


def test_reference_amounts_are_the_raw_training_amounts(frame):
    """Raw, not scaled: the sentence quotes amounts in currency, and it must
    never have seen a validation or test amount."""
    splits = prepare_splits(frame)
    reference, _ = artifacts.build(frame, splits)
    raw_train = frame.loc[splits.X_train.index, "Amount"].to_numpy()
    assert np.array_equal(reference.amounts, np.sort(raw_train))
    held_out = frame.loc[splits.X_test.index, "Amount"].max()
    assert held_out not in reference.amounts or (raw_train == held_out).any()


def test_round_trips_through_the_model_directory(frame, tmp_path):
    splits = prepare_splits(frame)
    reference, cases = artifacts.build(frame, splits)
    artifacts.save(reference, cases, tmp_path)
    loaded = artifacts.load(tmp_path)
    assert loaded is not None
    loaded_reference, loaded_cases = loaded
    assert loaded_reference.n_rows == reference.n_rows
    assert loaded_cases.n_rows == cases.n_rows
    probe = splits.X_test.iloc[[0]]
    assert loaded_cases.query(probe) == cases.query(probe)


def test_load_reports_absence_rather_than_raising(tmp_path):
    """A model directory trained before this feature existed must still serve;
    the explanation degrades, the service does not refuse to start."""
    assert artifacts.load(tmp_path) is None


def test_ulb_build_is_unchanged_by_the_schema_parameter(frame):
    from prism import tracks

    splits = prepare_splits(frame)
    reference, cases = artifacts.build(frame, splits, schema=tracks.ULB)
    assert reference.characteristics == PCA_COLUMNS
    assert cases.features == PCA_COLUMNS + ["Amount"]
