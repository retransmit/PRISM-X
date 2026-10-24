"""Categoricals reach each model in the representation it can use (design D3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism import tracks
from prism.data import india
from prism.data.preprocess import UNKNOWN_CODE, prepare_splits
from prism.evaluation.metrics import evaluate
from prism.evaluation.threshold import tune_threshold
from prism.features.velocity import add_velocity_features
from prism.models import build
from prism.models.base import CategoricalExpander

CATS = tracks.INDIA.categorical_columns


@pytest.fixture(scope="module")
def splits(india_stem):
    return prepare_splits(add_velocity_features(india.load_india(india_stem)), schema=tracks.INDIA)


def test_expander_widens_by_the_codes_seen_in_training_and_ignores_unknowns(splits):
    expander = CategoricalExpander(CATS).fit(splits.X_train)
    numeric = len(splits.feature_names) - len(CATS)
    expected = numeric + sum(len(codes) for codes in expander.categories.values())
    assert expander.output_dim == expected
    matrix = expander.transform(splits.X_train)
    assert matrix.shape == (len(splits.X_train), expected) and matrix.dtype == np.float32
    unknown = splits.X_test.iloc[[0]].copy()
    unknown["gateway_provider"] = UNKNOWN_CODE
    row = expander.transform(unknown)[0]
    start = numeric + sum(len(expander.categories[c]) for c in CATS[:CATS.index("gateway_provider")])
    assert row[start:start + len(expander.categories["gateway_provider"])].sum() == 0.0
    assert CategoricalExpander.from_state(expander.state()).categories == expander.categories


def test_isolation_forest_learns_the_indian_frame(splits):
    model = build("isolation_forest", categorical_columns=CATS).fit(splits.X_train, splits.y_train)
    scores = model.risk_score(splits.X_test)
    assert scores.shape == (len(splits.X_test),) and 0.0 < scores.min() and scores.max() < 1.0


def test_xgboost_uses_native_categories_and_beats_the_baseline(splits, tmp_path):
    pytest.importorskip("xgboost")
    model = build("xgboost", n_estimators=80, max_depth=4, categorical_columns=CATS)
    model.fit(splits.X_train, splits.y_train)
    assert model.booster.get_params()["enable_categorical"] is True
    assert set(model.categories) == set(CATS)
    frame = model.with_categories(splits.X_test)
    assert str(frame["channel_type"].dtype) == "category"

    threshold = tune_threshold(splits.y_val, model.risk_score(splits.X_val), strategy="f1")
    result = evaluate(splits.y_test, model.risk_score(splits.X_test), threshold, model="xgboost")
    assert result.pr_auc > float(splits.y_test.mean()) * 5

    unknown = splits.X_test.iloc[[0]].copy()
    unknown["mcc"] = UNKNOWN_CODE
    assert 0.0 <= float(model.risk_score(unknown)[0]) <= 1.0

    model.save(tmp_path)
    reloaded = type(model).load(tmp_path)
    assert reloaded.categories == model.categories
    assert np.allclose(reloaded.risk_score(splits.X_test), model.risk_score(splits.X_test))


def test_autoencoder_expands_categoricals_and_round_trips(splits, tmp_path):
    pytest.importorskip("torch")
    model = build("autoencoder", epochs=2, categorical_columns=CATS).fit(splits.X_train, splits.y_train)
    scores = model.risk_score(splits.X_test)
    assert scores.shape == (len(splits.X_test),)
    model.save(tmp_path)
    reloaded = type(model).load(tmp_path)
    assert np.allclose(reloaded.risk_score(splits.X_test), scores)


def test_isolation_forest_round_trips_its_expander(splits, tmp_path):
    model = build("isolation_forest", categorical_columns=CATS).fit(splits.X_train, splits.y_train)
    model.save(tmp_path)
    reloaded = type(model).load(tmp_path)
    assert np.allclose(reloaded.risk_score(splits.X_test), model.risk_score(splits.X_test))


def test_an_unknown_code_becomes_missing_without_a_pandas_deprecation(splits):
    """pandas 3 deprecates building a Categorical from values outside its
    categories and will raise; when it does, every unseen category would be
    refused instead of scored as missing (review focus 1)."""
    import warnings

    pytest.importorskip("xgboost")
    model = build("xgboost", categorical_columns=splits.categorical_columns, n_estimators=10)
    model.fit(splits.X_train, splits.y_train)
    frame = splits.X_test.head(2).copy()
    frame.iloc[0, frame.columns.get_loc("mcc")] = UNKNOWN_CODE
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        seen = model.with_categories(frame)
    assert seen["mcc"].isna().tolist() == [True, False]
