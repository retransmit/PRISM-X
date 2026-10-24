"""Population reference statistics behind the plain-English explanation.

Every clause of the narrative that says "larger than X% of transactions" or
"unusual on k of 28 characteristics" is computed against these statistics,
so they must be fitted on training rows only and must describe a row exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.explain.reference import ReferenceStats

PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]


@pytest.fixture
def raw_train() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(3)
    n = 4_000
    data = {name: rng.normal(size=n) for name in PCA_COLUMNS}
    # Only the first hour of each day holds transactions: a sharp daily cycle.
    data["Time"] = rng.uniform(0, 3_600, size=n) + rng.integers(0, 2, size=n) * 86_400
    data["Amount"] = np.linspace(1.0, 4_000.0, n)
    frame = pd.DataFrame(data)[["Time", *PCA_COLUMNS, "Amount"]]
    y = pd.Series(np.zeros(n, dtype=int))
    y.iloc[:8] = 1
    return frame, y


def test_amount_is_ranked_against_the_training_population(raw_train):
    frame, y = raw_train
    stats = ReferenceStats.from_training(frame, y)
    facts = stats.describe({"Time": 100.0, "Amount": 2_000.5, **{v: 0.0 for v in PCA_COLUMNS}})
    assert facts["amount_pct_below"] == pytest.approx(0.5, abs=0.01)


def test_daily_cycle_share_reflects_when_transactions_happen(raw_train):
    frame, y = raw_train
    stats = ReferenceStats.from_training(frame, y)
    assert stats.cycle_share.sum() == pytest.approx(1.0)
    busy = stats.describe({"Time": 1_800.0, "Amount": 10.0, **{v: 0.0 for v in PCA_COLUMNS}})
    quiet = stats.describe({"Time": 50_000.0, "Amount": 10.0, **{v: 0.0 for v in PCA_COLUMNS}})
    assert busy["cycle_share"] == pytest.approx(1.0)
    assert quiet["cycle_share"] == 0.0
    assert busy["typical_cycle_share"] == pytest.approx(1 / 24)


def test_unusual_count_uses_the_configured_band(raw_train):
    frame, y = raw_train
    stats = ReferenceStats.from_training(frame, y, band=0.99)
    ordinary = {"Time": 0.0, "Amount": 10.0, **{v: 0.0 for v in PCA_COLUMNS}}
    assert stats.describe(ordinary)["unusual_count"] == 0
    extreme = dict(ordinary, V1=50.0, V14=-50.0, V28=50.0)
    facts = stats.describe(extreme)
    assert facts["unusual_count"] == 3
    assert facts["n_characteristics"] == 28
    assert facts["band"] == 0.99


def test_base_rate_and_row_count_come_from_the_rows_given(raw_train):
    frame, y = raw_train
    stats = ReferenceStats.from_training(frame, y)
    assert stats.n_rows == len(frame)
    assert stats.base_rate == pytest.approx(8 / 4_000)


def test_round_trips_through_disk(raw_train, tmp_path):
    frame, y = raw_train
    stats = ReferenceStats.from_training(frame, y)
    stats.save(tmp_path / "reference.joblib")
    loaded = ReferenceStats.load(tmp_path / "reference.joblib")
    row = {"Time": 1_800.0, "Amount": 3_000.0, "V14": -50.0, **{v: 0.0 for v in PCA_COLUMNS if v != "V14"}}
    assert loaded.describe(row) == stats.describe(row)
