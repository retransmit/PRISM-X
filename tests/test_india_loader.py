"""The Indian source: a generator export joined on transaction_id, pre-authorisation columns only."""

from __future__ import annotations

import pandas as pd
import pytest

from prism import tracks
from prism.data import india
from prism.features import ratios


def test_load_joins_the_views_and_keeps_only_what_an_issuer_has_before_answering(india_stem):
    frame = india.load_india(india_stem)
    assert len(frame) == 3000
    assert list(frame.index) == list(range(3000)), "time-ordered rows with a fresh RangeIndex"
    assert frame["tx_time_seconds"].is_monotonic_increasing
    for outcome in india.OUTCOME_COLUMNS:
        assert outcome not in frame.columns
    expected = {*tracks.INDIA.identifier_columns, *india.NUMERIC_RAW, *india.CATEGORICAL_RAW, *ratios.RATIO_COLUMNS, "is_fraud"}
    assert set(frame.columns) == expected
    assert set(frame["is_fraud"].unique()) <= {0, 1}
    assert frame["is_fraud"].sum() > 0


def test_codes_keep_their_leading_zeros_and_empty_eci_is_a_category(india_stem):
    frame = india.load_india(india_stem)
    assert frame["pos_entry_mode"].isin({"051", "071", "012", "102"}).all()
    assert frame["eci"].isin({"05", "07", india.EMPTY_CATEGORY}).all()
    assert pd.api.types.is_string_dtype(frame["mcc"]) and frame["mcc"].str.len().eq(4).all()


def test_numerics_are_floats_and_the_boolean_is_zero_or_one(india_stem):
    frame = india.load_india(india_stem)
    for column in india.NUMERIC_RAW:
        assert pd.api.types.is_float_dtype(frame[column]), column
    assert set(frame["is_cross_border"].unique()) <= {0.0, 1.0}
    assert frame[list(india.NUMERIC_RAW)].isna().sum().sum() == 0


def test_a_view_with_a_foreign_id_is_refused(india_stem, tmp_path):
    """Equal row counts with different ids would inner-join to fewer rows and
    nobody would notice; the loader must say so."""
    import shutil

    for path in india.export_paths(india_stem).values():
        shutil.copy(path, tmp_path / path.name)
    gateway = tmp_path / "fixture_in_gateway_telemetry.csv"
    lines = gateway.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].replace("TX_00000000", "TX_99999999", 1)
    gateway.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="transaction_id"):
        india.load_india(tmp_path / "fixture_in")


def test_a_missing_view_is_named(tmp_path):
    with pytest.raises(FileNotFoundError, match="gateway_telemetry"):
        india.load_india(tmp_path / "nothing_in")


def test_export_paths_follow_the_cli_naming(tmp_path):
    paths = india.export_paths(tmp_path / "train_in")
    assert paths["auth"].name == "train_in_auth_stream.csv"
    assert paths["gateway"].name == "train_in_gateway_telemetry.csv"
    assert paths["master"].name == "train_in.csv"
    assert paths["calibration"].name == "train_in_calibration.json"


def test_an_absent_eci_loads_as_the_contracts_empty_string(india_stem):
    """A card-present payment has no 3-D Secure attempt and the wire carries an empty ECI
    (generator/spec/19); the loaded frame keeps that value, so the category the models learn
    is the one a live request sends."""
    frame = india.load_india(india_stem)
    card_present = frame[frame["channel_type"].str.startswith("CP")]
    assert len(card_present) > 0
    assert set(card_present["eci"].astype(str)) == {""}
    assert "NONE" not in set(frame["eci"].astype(str))
