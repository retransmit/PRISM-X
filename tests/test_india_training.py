"""train_one on the Indian frame end to end, with the extras the report leads with."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism import tracks
from prism.data import india
from prism.data.preprocess import prepare_splits
from prism.evaluation import benchmark
from prism.features.velocity import add_velocity_features
from prism.training import train


@pytest.fixture(scope="module")
def frame(india_stem):
    return add_velocity_features(india.load_india(india_stem))


def test_train_one_returns_the_extras_and_persists_per_track(frame, tmp_path):
    splits = prepare_splits(frame, schema=tracks.INDIA)
    amounts = frame.loc[splits.X_test.index, "amount"].to_numpy()
    threshold, result, extras = train.train_one(
        "isolation_forest", splits, resample=False, model_dir=tmp_path, amounts=amounts, schema=tracks.INDIA
    )
    assert 0.0 < threshold <= 1.0
    assert result.model == "isolation_forest"
    assert extras["prevalence_target"] == pytest.approx(float(tracks.INDIA.setting("prevalence_target")))
    assert 0.0 <= extras["fpr_per_10k"] <= 10_000.0 and 0.0 <= extras["recall_at_threshold"] <= 1.0
    assert "share_above_floor_caught" in extras and "adjusted_precision_at_prevalence" in extras
    # the champion is chosen on this, so a track with extras of its own must still carry it
    assert 0.0 <= extras["validation_pr_auc"] <= 1.0
    assert (tmp_path / "isolation_forest.joblib").exists()


def test_ulb_train_one_carries_only_the_validation_pr_auc(tmp_path):
    rng = np.random.default_rng(2)
    n = 3000
    ulb = pd.DataFrame({**{c: rng.normal(size=n) for c in tracks.PCA_COLUMNS},
                        "Time": rng.uniform(0, 172_800, size=n), "Amount": rng.gamma(2.0, 50.0, size=n)})
    ulb["Class"] = (rng.random(n) < 0.02).astype(int)
    ulb.loc[ulb.Class == 1, "V1"] += 6.0
    splits = prepare_splits(ulb[tracks.ULB.feature_columns + ["Class"]], schema=tracks.ULB)
    _, _, extras = train.train_one("isolation_forest", splits, resample=False, model_dir=tmp_path, schema=tracks.ULB)
    assert set(extras) == {"validation_pr_auc"} and 0.0 <= extras["validation_pr_auc"] <= 1.0


def test_markdown_table_adds_the_indian_columns_when_extras_are_given():
    from prism.evaluation.metrics import EvalResult

    result = EvalResult("xgboost", 0.5, 0.9, 0.99, 0.5, 0.8, 0.6, 100, 2, 1, 4)
    table = benchmark.markdown_table([result], extras={"xgboost": {
        "adjusted_precision_at_prevalence": 0.012, "fpr_per_10k": 3.0, "share_above_floor_caught": 0.97,
    }})
    assert "Precision at 1 in 80,847" in table and "0.0120" in table and "3.0" in table and "0.970" in table
