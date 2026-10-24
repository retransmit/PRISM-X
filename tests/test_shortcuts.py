"""What one column gives away about the label, measured before training."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism import tracks
from prism.evaluation import shortcuts


def _frame(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = pd.Series((rng.random(n) < 0.1).astype(int))
    frame = pd.DataFrame({
        "Time": rng.uniform(0, 1e5, n),
        **{f"V{i}": rng.normal(size=n) for i in range(1, 29)},
        "Amount": rng.gamma(2.0, 50.0, n),
        "Class": y,
    })
    return frame, y


def test_a_column_that_is_the_label_scores_an_auc_of_one():
    frame, y = _frame()
    frame["V1"] = y * 10.0 + frame["V1"] * 0.01
    rows = shortcuts.audit(frame, y, tracks.ULB)
    assert rows[0]["feature"] == "V1" and rows[0]["auc"] == pytest.approx(1.0)
    assert shortcuts.offenders(rows, ceiling=0.95, allowed={}) == [rows[0]]
    assert shortcuts.offenders(rows, ceiling=0.95, allowed={"V1": "a stated reason"}) == []


def test_an_inverted_column_counts_the_same_as_a_direct_one():
    frame, y = _frame()
    frame["V2"] = -y * 10.0
    by_feature = {r["feature"]: r for r in shortcuts.audit(frame, y, tracks.ULB)}
    assert by_feature["V2"]["auc"] == pytest.approx(1.0)
    assert all(r["kind"] == "numeric" for r in by_feature.values())


def test_one_sided_categorical_values_are_listed_above_the_support_floor():
    y = pd.Series([1] * 30 + [0] * 70)
    column = pd.Series(["botnet"] * 30 + ["home"] * 60 + ["office"] * 10)
    fraud_only, legit_only = shortcuts.one_sided_values(column, y, min_support=25)
    assert fraud_only == ["botnet"]
    assert legit_only == ["home"]


def test_the_audit_is_written_as_json(tmp_path):
    frame, y = _frame()
    rows = shortcuts.audit(frame, y, tracks.ULB)
    path = shortcuts.write(rows, tmp_path / "shortcut_audit.json")
    assert path.exists() and len(rows) == len(tracks.ULB.feature_columns)
