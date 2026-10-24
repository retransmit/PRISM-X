"""The Indian split is by time and its encoder is fitted on the training window only (design D4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism import tracks
from prism.data import india
from prism.data.preprocess import (
    UNKNOWN_CODE,
    load_encoder,
    prepare_splits,
    save_encoder,
    transform_features,
)
from prism.features.velocity import add_velocity_features


@pytest.fixture(scope="module")
def frame(india_stem) -> pd.DataFrame:
    return add_velocity_features(india.load_india(india_stem))


@pytest.fixture(scope="module")
def splits(frame):
    return prepare_splits(frame, schema=tracks.INDIA)


def test_windows_are_consecutive_in_time(frame, splits):
    clock = frame["tx_time_seconds"]
    assert clock.loc[splits.X_train.index].max() <= clock.loc[splits.X_val.index].min()
    assert clock.loc[splits.X_val.index].max() <= clock.loc[splits.X_test.index].min()
    n = len(frame)
    assert len(splits.X_train) == int(n * 0.7) and len(splits.X_val) == int(n * 0.1)
    assert len(splits.X_train) + len(splits.X_val) + len(splits.X_test) == n


def test_model_inputs_are_the_schema_features_in_order(splits):
    assert splits.feature_names == tracks.INDIA.feature_columns
    assert splits.track == "india"
    assert splits.categorical_columns == tracks.INDIA.categorical_columns
    for column in tracks.INDIA.categorical_columns:
        assert pd.api.types.is_integer_dtype(splits.X_train[column]), column


def test_encoder_and_scaler_know_only_the_training_window(frame, splits):
    train_channels = set(frame.loc[splits.X_train.index, "channel_type"])
    position = list(tracks.INDIA.categorical_columns).index("channel_type")
    assert set(splits.encoder.categories_[position]) == train_channels
    # A value that exists only in the test window must be unknown to the
    # encoder: an encoder fitted on the whole frame would know it.
    later = frame.copy()
    later.loc[later.index[-50:], "gateway_provider"] = "TESTPAY"
    leaky = prepare_splits(later, schema=tracks.INDIA)
    gateway = list(tracks.INDIA.categorical_columns).index("gateway_provider")
    assert "TESTPAY" not in set(leaky.encoder.categories_[gateway])
    assert (leaky.X_test["gateway_provider"] == UNKNOWN_CODE).sum() == 50
    raw_train = frame.loc[splits.X_train.index, list(tracks.INDIA.scale_columns)]
    assert np.allclose(splits.scaler.mean_, raw_train.mean().to_numpy())
    assert not np.allclose(splits.scaler.mean_, frame[list(tracks.INDIA.scale_columns)].mean().to_numpy())


def test_a_category_never_seen_in_training_gets_the_unknown_code(frame, splits):
    row = frame.loc[splits.X_test.index[0]].to_dict()
    row["gateway_provider"] = "NEWPAY"
    transformed = transform_features(
        row, splits.scaler, splits.feature_names, list(splits.scale_columns),
        encoder=splits.encoder, categorical_columns=splits.categorical_columns,
    )
    assert int(transformed["gateway_provider"].iloc[0]) == UNKNOWN_CODE
    assert list(transformed.columns) == splits.feature_names


def test_transform_matches_the_training_frame_byte_for_byte(frame, splits):
    index = splits.X_train.index[7]
    transformed = transform_features(
        frame.loc[index].to_dict(), splits.scaler, splits.feature_names, list(splits.scale_columns),
        encoder=splits.encoder, categorical_columns=splits.categorical_columns,
    )
    assert np.allclose(transformed.to_numpy(dtype=float), splits.X_train.loc[[index]].to_numpy(dtype=float))


def test_encoder_round_trips_and_is_optional(splits, tmp_path):
    written = save_encoder(splits.encoder, tmp_path / "encoder.joblib")
    assert load_encoder(written).categories_[0].tolist() == splits.encoder.categories_[0].tolist()
    assert load_encoder(tmp_path / "absent.joblib") is None


def test_a_window_without_fraud_is_refused(frame):
    quiet = frame.copy()
    quiet.loc[quiet.index[int(len(quiet) * 0.8):], "is_fraud"] = 0
    with pytest.raises(ValueError, match="test window"):
        prepare_splits(quiet, schema=tracks.INDIA)


def test_ulb_splits_are_unchanged():
    ulb = pd.DataFrame({
        **{c: np.random.default_rng(0).normal(size=2000) for c in tracks.PCA_COLUMNS},
        "Time": np.arange(2000, dtype=float), "Amount": np.ones(2000),
    })
    ulb["Class"] = (np.arange(2000) % 100 == 0).astype(int)
    ulb = ulb[tracks.ULB.feature_columns + ["Class"]]
    stratified = prepare_splits(ulb, schema=tracks.ULB)
    assert stratified.encoder is None and stratified.track == "ulb"
    assert len(stratified.X_test) == 400
