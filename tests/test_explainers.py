"""Every model explained: additive in its own space, keyed by the track's feature names,
and equal to the model's own raw output on the row."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from prism import tracks
from prism.explain.shap_engine import ShapEngine
from prism.models import MODEL_NAMES, build

QUICK = {
    "isolation_forest": {"n_estimators": 30},
    "autoencoder": {"epochs": 2},
    "xgboost": {"n_estimators": 30, "max_depth": 3},
}
DEPENDENCY = {"autoencoder": "torch", "xgboost": "xgboost"}


@pytest.fixture(scope="module")
def fitted(india_splits_module):
    splits = india_splits_module
    models = {}
    for name in MODEL_NAMES:
        if name in DEPENDENCY:
            pytest.importorskip(DEPENDENCY[name])
        models[name] = build(name, categorical_columns=splits.categorical_columns, **QUICK[name]).fit(
            splits.X_train, splits.y_train
        )
    return models


def _raw_output(name, model, row):
    """The model's own output in the explainer's space."""
    if name == "xgboost":
        p = float(model.risk_score(row)[0])
        return float(np.log(p / (1.0 - p)))
    if name == "isolation_forest":
        from sklearn.ensemble._iforest import _average_path_length

        forest = model.forest
        scale = float(_average_path_length([forest.max_samples_])[0])
        path_length = -np.log2(-forest.score_samples(model.expander.transform(row))[0]) * scale
        return float(-path_length)
    if name == "autoencoder":
        return float(model.squared_errors(row).sum())
    raise AssertionError(name)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_is_explained_additively_in_its_own_space(name, fitted, india_splits_module):
    splits = india_splits_module
    model = fitted[name]
    engine = ShapEngine.for_model(model, list(splits.X_train.columns), background=splits.X_train.head(50), schema=tracks.INDIA)
    fraud_rows = np.flatnonzero(np.asarray(splits.y_test) == 1)[:3]
    for i in fraud_rows:
        row = splits.X_test.iloc[[i]]
        explanation = engine.explain(row)
        assert [c.feature for c in explanation.contributions] == list(splits.X_train.columns)
        total = explanation.base_value + sum(c.shap_value for c in explanation.contributions)
        assert total == pytest.approx(explanation.prediction, abs=1e-6)
        assert explanation.prediction == pytest.approx(_raw_output(name, model, row), rel=1e-4, abs=1e-5)


def test_the_isolation_forest_explainer_is_tree_based_and_fast(fitted, india_splits_module):
    splits = india_splits_module
    engine = ShapEngine.for_model(fitted["isolation_forest"], list(splits.X_train.columns), schema=tracks.INDIA)
    started = time.perf_counter()
    for i in range(20):
        engine.explain(splits.X_test.iloc[[i]])
    assert (time.perf_counter() - started) / 20 < 0.2, "TreeSHAP, not the sampled kernel explainer"
    scores = fitted["isolation_forest"].risk_score(splits.X_test)
    worst, best = int(np.argmax(scores)), int(np.argmin(scores))

    def push(i):
        return sum(c.shap_value for c in engine.explain(splits.X_test.iloc[[i]]).contributions)

    assert push(worst) > push(best), "a positive push means more suspicious"


def test_a_categorical_contribution_lands_on_its_own_column(fitted, india_splits_module):
    splits = india_splits_module
    for name in ("isolation_forest", "autoencoder"):
        engine = ShapEngine.for_model(fitted[name], list(splits.X_train.columns), schema=tracks.INDIA)
        explanation = engine.explain(splits.X_test.iloc[[0]])
        by_name = {c.feature: c for c in explanation.contributions}
        for column in splits.categorical_columns:
            assert column in by_name, (name, column)
    from prism.explain.shap_engine import group_expanded
    from prism.models.base import CategoricalExpander

    fitted_on = pd.DataFrame({"a": [0.0, 1.0], "c1": [1, 0], "b": [0.0, 1.0], "c2": [2, 3]})
    expander = CategoricalExpander(["c2", "c1"]).fit(fitted_on)
    values = np.array([[1.0, 2.0, 10.0, 20.0, 100.0, 200.0]])  # a, b, the c2 block (2), the c1 block (2)
    grouped = group_expanded(values, expander, ["a", "c1", "b", "c2"])
    np.testing.assert_allclose(grouped, [[1.0, 300.0, 2.0, 30.0]])
