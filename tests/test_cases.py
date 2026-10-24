"""Case-based evidence: how many of the nearest training transactions were fraud.

"Of the 50 most similar transactions the system learned from, 45 were fraud"
is the one clause of the explanation that needs no feature names at all, so
the index behind it has to be exact about which rows it holds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.explain.cases import CaseIndex

COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]


@pytest.fixture
def clustered() -> tuple[pd.DataFrame, pd.Series]:
    """Legitimate rows around the origin, fraud rows around +6 on every axis."""
    rng = np.random.default_rng(5)
    legit = rng.normal(0.0, 1.0, size=(1_000, len(COLUMNS)))
    fraud = rng.normal(6.0, 0.5, size=(60, len(COLUMNS)))
    X = pd.DataFrame(np.vstack([legit, fraud]), columns=COLUMNS)
    y = pd.Series(np.r_[np.zeros(1_000, dtype=int), np.ones(60, dtype=int)])
    return X, y


def test_query_near_the_fraud_cluster_returns_mostly_fraud(clustered):
    X, y = clustered
    index = CaseIndex.fit(X, y, k=50)
    facts = index.query(pd.DataFrame([[6.0] * len(COLUMNS)], columns=COLUMNS))
    assert facts["similar_total"] == 50
    assert facts["similar_fraud"] >= 45


def test_query_near_the_legitimate_cluster_returns_none(clustered):
    X, y = clustered
    index = CaseIndex.fit(X, y, k=50)
    facts = index.query(pd.DataFrame([[0.0] * len(COLUMNS)], columns=COLUMNS))
    assert facts["similar_fraud"] == 0
    assert facts["base_rate"] == pytest.approx(60 / 1_060)


def test_index_holds_exactly_the_rows_it_was_fitted_on(clustered):
    X, y = clustered
    index = CaseIndex.fit(X, y, k=10)
    assert index.n_rows == len(X)
    assert index.k == 10


def test_round_trips_through_disk(clustered, tmp_path):
    X, y = clustered
    index = CaseIndex.fit(X, y, k=25)
    index.save(tmp_path / "cases.joblib")
    loaded = CaseIndex.load(tmp_path / "cases.joblib")
    probe = pd.DataFrame([[5.5] * len(COLUMNS)], columns=COLUMNS)
    assert loaded.query(probe) == index.query(probe)
    assert loaded.n_rows == len(X)


def test_similarity_ignores_time(clustered):
    """Time is seconds since the dataset's first row. A live transaction carries
    a Time far outside the training range, and if it counted toward distance it
    would dominate every query and match nothing. Similarity is about what the
    transaction looks like, not when it happened."""
    X, y = clustered
    index = CaseIndex.fit(X, y, k=50)
    assert "Time" not in index.features

    at_fraud_cluster = pd.DataFrame([[6.0] * len(COLUMNS)], columns=COLUMNS)
    at_fraud_cluster["Time"] = -1_000.0
    assert index.query(at_fraud_cluster)["similar_fraud"] >= 45


def test_an_index_pickled_before_the_expander_still_answers(tmp_path):
    """A deployment's models/ can predate the categorical expander; the service must
    load that case index and answer, not crash in its warm-up."""
    import joblib

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 4)), columns=["Time", "V1", "V2", "Amount"])
    y = (rng.random(200) < 0.1).astype(int)
    index = CaseIndex.fit(X, y, k=10)
    index.__dict__.pop("_expander")
    path = tmp_path / "old_cases.joblib"
    joblib.dump(index, path)
    loaded = CaseIndex.load(path)
    assert loaded.query(X.iloc[[0]])["similar_total"] == 10
