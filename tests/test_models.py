"""Tests for the two heavyweight models and the SHAP engine.

Kept separate from the rest of the suite because these need torch, xgboost and
shap. Each block skips rather than fails when its dependency is absent, so the
core leakage and threshold tests stay runnable on a bare scikit-learn install.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.data.preprocess import prepare_splits
from prism.evaluation.metrics import evaluate
from prism.evaluation.threshold import tune_threshold
from prism.models import build

PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]


@pytest.fixture(scope="module")
def splits():
    """Small separable dataset: big enough to learn, small enough to stay fast."""
    rng = np.random.default_rng(11)
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
    frame.loc[mask, "Amount"] *= 6.0

    return prepare_splits(frame[["Time", *PCA_COLUMNS, "Amount", "Class"]])


# --------------------------------------------------------------------------
# XGBoost
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def xgb_model(splits):
    pytest.importorskip("xgboost")
    model = build("xgboost", n_estimators=60, max_depth=4)
    return model.fit(splits.X_train, splits.y_train)


def test_xgboost_derives_scale_pos_weight_from_the_data(splits):
    pytest.importorskip("xgboost")
    model = build("xgboost", n_estimators=10).fit(splits.X_train, splits.y_train)

    positives = int(splits.y_train.sum())
    expected = (len(splits.y_train) - positives) / positives
    assert model.booster.get_params()["scale_pos_weight"] == pytest.approx(expected)


def test_xgboost_rejects_unsupervised_use(splits):
    pytest.importorskip("xgboost")
    with pytest.raises(ValueError, match="supervised"):
        build("xgboost", n_estimators=10).fit(splits.X_train, None)


def test_xgboost_beats_the_no_skill_baseline(splits, xgb_model):
    scores = xgb_model.risk_score(splits.X_test)
    assert scores.min() >= 0.0 and scores.max() <= 1.0

    threshold = tune_threshold(splits.y_val, xgb_model.risk_score(splits.X_val))
    result = evaluate(splits.y_test, scores, threshold, model="xgboost")
    assert result.pr_auc > float(splits.y_test.mean()) * 20
    assert result.recall > 0.5


def test_xgboost_round_trips_through_native_json(splits, xgb_model, tmp_path):
    before = xgb_model.risk_score(splits.X_test)
    saved = xgb_model.save(tmp_path)
    assert saved.suffix == ".json", "should persist as pickle-free native JSON"

    reloaded = type(xgb_model).load(tmp_path)
    assert np.allclose(before, reloaded.risk_score(splits.X_test))
    assert reloaded.feature_names == list(splits.X_train.columns)


# --------------------------------------------------------------------------
# Autoencoder
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def autoencoder(splits):
    pytest.importorskip("torch")
    model = build("autoencoder", epochs=15, batch_size=256)
    return model.fit(splits.X_train, splits.y_train)


def test_autoencoder_requires_labels_to_isolate_normals(splits):
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="train_on_normal_only"):
        build("autoencoder", epochs=1).fit(splits.X_train, None)


def test_autoencoder_reconstructs_fraud_worse_than_normal(splits, autoencoder):
    """The core premise: unseen-shaped rows should not rebuild cleanly."""
    scores = autoencoder.risk_score(splits.X_test)
    fraud_mean = scores[splits.y_test.to_numpy() == 1].mean()
    normal_mean = scores[splits.y_test.to_numpy() == 0].mean()
    assert fraud_mean > normal_mean


def test_autoencoder_scores_stay_in_the_open_unit_interval(splits, autoencoder):
    scores = autoencoder.risk_score(splits.X_test)
    assert scores.min() > 0.0 and scores.max() < 1.0
    assert np.all(np.isfinite(scores))


def test_autoencoder_round_trips_through_state_dict(splits, autoencoder, tmp_path):
    before = autoencoder.risk_score(splits.X_test)
    saved = autoencoder.save(tmp_path)
    assert saved.suffix == ".pt"

    reloaded = type(autoencoder).load(tmp_path)
    assert np.allclose(before, reloaded.risk_score(splits.X_test), atol=1e-6)


# --------------------------------------------------------------------------
# SHAP
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shap_engine(splits, xgb_model):
    pytest.importorskip("shap")
    from prism.explain import ShapEngine

    return ShapEngine.for_model(xgb_model, list(splits.X_train.columns))


def test_shap_attributions_are_additive(splits, xgb_model, shap_engine):
    """base_value + sum(shap) must reconstruct the model's raw margin exactly.

    This is the property that makes SHAP defensible to a regulator rather than
    merely suggestive; if it fails, the explanation is decorative.
    """
    row = splits.X_test.iloc[[0]]
    explanation = shap_engine.explain(row)

    margin = float(xgb_model.booster.predict(row, output_margin=True)[0])
    assert explanation.prediction == pytest.approx(margin, abs=1e-3)


def test_shap_explains_every_feature(splits, shap_engine):
    explanation = shap_engine.explain(splits.X_test.iloc[[0]])
    assert len(explanation.contributions) == len(splits.X_train.columns)
    assert {c.feature for c in explanation.contributions} == set(splits.X_train.columns)


def test_shap_marks_pca_components_as_uninterpretable(splits, shap_engine):
    """The caveat that has to reach the report, enforced in code."""
    explanation = shap_engine.explain(splits.X_test.iloc[[0]])
    by_name = {c.feature: c for c in explanation.contributions}

    assert by_name["Amount"].interpretable is True
    assert by_name["Time"].interpretable is True
    assert all(not by_name[f"V{i}"].interpretable for i in range(1, 29))


def test_shap_payload_is_json_shaped_for_the_dashboard(splits, shap_engine):
    payload = shap_engine.explain(splits.X_test.iloc[[0]]).as_dict(k=5)

    assert payload["model"] == "xgboost"
    assert len(payload["top_features"]) == 5
    assert 0.0 <= payload["explained_fraction"] <= 1.0
    assert 0.0 <= payload["interpretable_fraction"] <= 1.0
    # Top features must be ordered by absolute influence.
    magnitudes = [abs(f["shap_value"]) for f in payload["top_features"]]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_shap_rejects_multi_row_input(splits, shap_engine):
    with pytest.raises(ValueError, match="one row"):
        shap_engine.explain(splits.X_test.iloc[:3])


def test_the_autoencoder_is_explained_by_its_reconstruction_error(splits, autoencoder):
    pytest.importorskip("shap")
    from prism.explain import ShapEngine

    engine = ShapEngine.for_model(autoencoder, list(splits.X_train.columns))
    explanation = engine.explain(splits.X_test.iloc[[0]])
    assert explanation.base_value == 0.0
    assert all(c.shap_value >= 0.0 for c in explanation.contributions), "a squared error is never negative"
    assert explanation.prediction == pytest.approx(float(autoencoder.squared_errors(splits.X_test.iloc[[0]]).sum()), rel=1e-5)


def test_explainer_names_indian_features_and_carries_the_raw_category(india_stem):
    pytest.importorskip("xgboost")
    from prism import tracks
    from prism.data import india
    from prism.explain.shap_engine import ShapEngine
    from prism.features.velocity import add_velocity_features

    frame = add_velocity_features(india.load_india(india_stem))
    india_splits = prepare_splits(frame, schema=tracks.INDIA)
    model = build("xgboost", n_estimators=40, max_depth=3, categorical_columns=india_splits.categorical_columns)
    model.fit(india_splits.X_train, india_splits.y_train)

    engine = ShapEngine.for_model(model, india_splits.feature_names, schema=tracks.INDIA)
    index = india_splits.X_test.index[0]
    raw = frame.loc[index].to_dict()
    explanation = engine.explain(india_splits.X_test.loc[[index]], raw=raw)

    assert len(explanation.contributions) == len(india_splits.feature_names)
    assert all(c.interpretable for c in explanation.contributions)
    channel = next(c for c in explanation.contributions if c.feature == "channel_type")
    assert channel.display == raw["channel_type"]
    assert channel.label == "channel"
    assert next(c for c in explanation.contributions if c.feature == "amount").display is None
    assert explanation.prediction == pytest.approx(explanation.base_value + sum(c.shap_value for c in explanation.contributions), abs=1e-4)
    assert "display" in explanation.as_dict()["top_features"][0]
