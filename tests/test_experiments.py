"""Tests for the Phase 3 experiment matrix (SMOTE vs cost-sensitive weighting)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.data.preprocess import prepare_splits
from prism.evaluation.experiments import (
    DEFAULT_ARMS,
    ExperimentArm,
    comparison_table,
    run_matrix,
)

PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]


@pytest.fixture(scope="module")
def splits():
    rng = np.random.default_rng(5)
    n, n_fraud = 6_000, 60

    frame = pd.DataFrame(
        {
            **{c: rng.normal(size=n) for c in PCA_COLUMNS},
            "Time": rng.uniform(0, 172_800, size=n),
            "Amount": rng.gamma(2.0, 50.0, size=n),
        }
    )
    labels = np.zeros(n, dtype=int)
    labels[rng.choice(n, size=n_fraud, replace=False)] = 1
    frame["Class"] = labels

    mask = frame["Class"] == 1
    for column in PCA_COLUMNS[:6]:
        frame.loc[mask, column] += rng.normal(5.0, 1.0, size=n_fraud)

    return prepare_splits(frame[["Time", *PCA_COLUMNS, "Amount", "Class"]])


def test_runs_every_requested_arm(splits):
    arms = (
        ExperimentArm("isolation_forest", resample=False),
        ExperimentArm("xgboost", resample=False),
    )
    results = run_matrix(splits, arms)

    assert len(results) == 2
    assert {r["model"] for r in results} == {"isolation_forest", "xgboost"}
    assert all("pr_auc" in r for r in results)


def test_smote_is_refused_for_unsupervised_models(splits):
    """Requesting SMOTE for Isolation Forest must record that it was not applied."""
    results = run_matrix(splits, (ExperimentArm("isolation_forest", resample=True),))
    assert results[0]["resample_requested"] is True
    assert results[0]["resample_applied"] is False


def test_smote_is_applied_for_xgboost(splits):
    results = run_matrix(splits, (ExperimentArm("xgboost", resample=True),))
    assert results[0]["resample_applied"] is True
    assert results[0]["train_positives"] > int(splits.y_train.sum())


def test_arm_labels_distinguish_the_two_imbalance_strategies():
    assert ExperimentArm("xgboost", resample=False).label == "xgboost+weighted"
    assert ExperimentArm("xgboost", resample=True).label == "xgboost+smote"


def test_default_arms_cover_the_phase_three_comparison():
    labels = {arm.label for arm in DEFAULT_ARMS}
    assert "xgboost+weighted" in labels
    assert "xgboost+smote" in labels


def test_comparison_table_ranks_by_pr_auc(splits):
    results = run_matrix(
        splits,
        (
            ExperimentArm("isolation_forest", resample=False),
            ExperimentArm("xgboost", resample=False),
        ),
    )
    table = comparison_table(results)
    lines = [line for line in table.splitlines() if line.startswith("|")]

    best = max(results, key=lambda r: r["pr_auc"])
    assert best["label"] in lines[2], "highest PR-AUC must be the first data row"
    assert "PR-AUC" in lines[0]


def test_resampled_training_pins_the_class_weight():
    from prism.training.resampling import imbalance_overrides
    assert imbalance_overrides(True) == {"scale_pos_weight": 1.0}
    assert imbalance_overrides(False) == {}


def test_smote_arm_does_not_also_reweight(splits):
    """SMOTE and scale_pos_weight answer the same question. XGBoostModel derives
    scale_pos_weight from the class ratio of whatever it is fitted on, so on a
    resampled set it counted the minority class twice: 10x oversampling and a
    10x gradient weight on top. The resampled arm pins the weight to 1.0; the
    weighted arm keeps the ratio."""
    smote = run_matrix(splits, (ExperimentArm("xgboost", resample=True),))[0]
    weighted = run_matrix(splits, (ExperimentArm("xgboost", resample=False),))[0]
    assert smote["scale_pos_weight"] == 1.0
    assert weighted["scale_pos_weight"] > 1.0


def test_comparison_table_says_n_a_for_a_model_that_flags_nothing():
    """On the Indian track the Isolation Forest cannot meet the false-positive
    budget and flags nothing; its alerts per catch are None, and the table
    must say so instead of crashing after every arm has been fitted."""
    from prism.evaluation.experiments import comparison_table

    def record(label, pr_auc, alerts):
        return {
            "label": label, "resample_applied": False, "threshold": 1.0, "pr_auc": pr_auc,
            "precision": 0.0, "recall": 0.0, "f1": 0.0, "false_negatives": 93,
            "alerts_per_true_fraud": alerts,
        }

    table = comparison_table([record("isolation_forest+weighted", 0.0169, None), record("xgboost+weighted", 0.92, 1.1)])
    assert "| n/a |" in table and "| 1.1 |" in table


def test_a_paired_bootstrap_interval_spans_zero_for_identical_arms_and_not_for_a_better_one():
    from prism.evaluation.metrics import paired_bootstrap

    rng = np.random.default_rng(0)
    y = (rng.random(2_000) < 0.05).astype(int)
    good = y * 0.8 + rng.random(2_000) * 0.3
    poor = rng.random(2_000)
    same = paired_bootstrap(y, good, good, resamples=200, seed=1)
    assert same["low"] == same["high"] == 0.0
    better = paired_bootstrap(y, good, poor, resamples=200, seed=1)
    assert better["low"] > 0.0 and 0.0 <= better["share_b_at_least_a"] < 0.05
