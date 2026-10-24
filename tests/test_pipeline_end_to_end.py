"""Full pipeline wiring test on synthetic data.

Covers what the unit tests individually do not: that config, preprocessing, the
model interface, calibration, threshold tuning, metrics and persistence
actually compose. Uses Isolation Forest because it needs only scikit-learn:
this test must stay runnable without torch, xgboost or the Kaggle download.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.data.preprocess import load_scaler, prepare_splits, save_scaler
from prism.evaluation.metrics import evaluate
from prism.evaluation.threshold import pr_curve_points, tune_threshold
from prism.models import build

PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]


@pytest.fixture(scope="module")
def separable_df() -> pd.DataFrame:
    """30,000 rows at 0.2% fraud, with fraud genuinely off the normal manifold.

    Detectability is deliberate: a pipeline that silently mangles features would
    still score near the no-skill baseline here, so a high PR-AUC is meaningful
    evidence that the wiring is intact.
    """
    rng = np.random.default_rng(42)
    n, n_fraud = 30_000, 60

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

    fraud_mask = frame["Class"] == 1
    for column in PCA_COLUMNS[:6]:
        frame.loc[fraud_mask, column] += rng.normal(5.0, 1.0, size=n_fraud)
    frame.loc[fraud_mask, "Amount"] *= 6.0

    return frame[["Time", *PCA_COLUMNS, "Amount", "Class"]]


@pytest.fixture(scope="module")
def trained(separable_df):
    splits = prepare_splits(separable_df)
    model = build("isolation_forest").fit(splits.X_train, splits.y_train)
    return splits, model


def test_feature_order_survives_preprocessing(trained):
    """Column order must match what the API sends; see api.inference.FEATURE_ORDER."""
    splits, _ = trained
    assert list(splits.X_train.columns) == ["Time", *PCA_COLUMNS, "Amount"]


def test_risk_scores_are_bounded_and_unsaturated(trained):
    splits, model = trained
    scores = model.risk_score(splits.X_val)

    assert scores.min() > 0.0
    assert scores.max() < 1.0
    # Regression guard: a clipping calibrator collapsed the tail to a single value.
    assert len(np.unique(scores)) > len(scores) // 2


def test_detects_separable_fraud_far_above_baseline(trained):
    splits, model = trained

    threshold = tune_threshold(splits.y_val, model.risk_score(splits.X_val))
    result = evaluate(
        splits.y_test, model.risk_score(splits.X_test), threshold, model=model.name
    )
    baseline = float(splits.y_test.mean())

    assert 0.0 < threshold < 1.0
    assert result.pr_auc > 0.5, f"PR-AUC {result.pr_auc:.4f}, pipeline likely broken"
    assert result.pr_auc > baseline * 50
    assert result.recall > 0.5


def test_pr_curve_is_serialisable(trained):
    splits, model = trained
    curve = pr_curve_points(splits.y_test, model.risk_score(splits.X_test))

    assert len(curve["precision"]) == len(curve["recall"]) > 1
    assert len(curve["thresholds"]) == len(curve["precision"]) - 1


def test_persistence_round_trip_is_score_identical(trained, tmp_path):
    splits, model = trained
    before = model.risk_score(splits.X_test)

    model.save(tmp_path)
    save_scaler(splits.scaler, tmp_path / "scaler.joblib")

    reloaded = type(model).load(tmp_path)
    assert np.allclose(before, reloaded.risk_score(splits.X_test))
    assert np.allclose(load_scaler(tmp_path / "scaler.joblib").mean_, splits.scaler.mean_)
