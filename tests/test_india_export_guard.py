"""On the real export: no single feature the Indian track models may separate the
classes on its own (the generator's spec/07 gate applied to Prism's own feature set, velocity
included). Skips without the export, like the ULB dataset tests."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from prism import config, tracks
from prism.data import india
from prism.data.frames import load_frame

STEM = config.get_path("tracks.india.export_stem")
pytestmark = pytest.mark.skipif(
    not india.export_paths(STEM)["auth"].exists(), reason=f"no Indian export at {STEM}"
)


def test_no_single_feature_is_a_label_shortcut():
    frame = load_frame(tracks.INDIA)
    n_train = int(len(frame) * 0.7)
    train, test = frame.iloc[:n_train], frame.iloc[n_train:]
    y_train, y_test = train["is_fraud"].to_numpy(), test["is_fraud"].to_numpy()
    worst = (0.0, "")
    for column in tracks.INDIA.feature_columns:
        if column in tracks.INDIA.categorical_columns:
            rate = train.groupby(column)["is_fraud"].mean()
            score = test[column].map(rate).fillna(float(y_train.mean())).to_numpy()
        else:
            score = test[column].to_numpy(dtype=float)
        if len(np.unique(score)) < 2:
            continue
        auc = roc_auc_score(y_test, score)
        worst = max(worst, (max(auc, 1.0 - auc), column))
    assert worst[0] <= 0.95, f"{worst[1]} alone separates the classes: ROC-AUC {worst[0]:.3f}"
