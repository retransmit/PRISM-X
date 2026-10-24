"""The explanation's reference material on the Indian track: named bands, category shares."""

from __future__ import annotations

import pytest

from prism import tracks
from prism.data import india
from prism.data.preprocess import prepare_splits
from prism.explain import artifacts
from prism.features.velocity import add_velocity_features


@pytest.fixture(scope="module")
def built(india_stem):
    frame = add_velocity_features(india.load_india(india_stem))
    splits = prepare_splits(frame, schema=tracks.INDIA)
    reference, cases = artifacts.build(frame, splits, schema=tracks.INDIA)
    return frame, splits, reference, cases


def test_reference_bands_cover_the_named_numerics_and_shares_cover_the_categoricals(built):
    frame, splits, reference, _ = built
    assert reference.amount_column == "amount" and reference.time_column == "tx_time_seconds"
    assert "vaai_score" in reference.characteristics and "amount" not in reference.characteristics
    low, high = reference.numeric_band("vaai_score")
    assert low < high
    assert set(reference.categorical_share) == set(tracks.INDIA.categorical_columns)
    assert sum(reference.categorical_share["channel_type"].values()) == pytest.approx(1.0)
    assert reference.n_rows == len(splits.X_train)


def test_describe_reads_the_indian_columns(built):
    frame, splits, reference, _ = built
    facts = reference.describe(frame.loc[splits.X_test.index[0]].to_dict())
    assert 0.0 <= facts["amount_pct_below"] <= 1.0
    assert facts["n_characteristics"] == len(reference.characteristics)


def test_case_index_measures_distance_over_one_hot_categoricals(built):
    _, splits, _, cases = built
    assert "tx_time_seconds" not in cases.features
    assert cases.n_rows == len(splits.X_train)
    verdict = cases.query(splits.X_test.iloc[[0]])
    assert verdict["similar_total"] == cases.k and 0 <= verdict["similar_fraud"] <= cases.k


def test_reference_keeps_a_narrower_band_for_the_driver_sentence(built):
    """At the 99.5 percent band nearly every driver sits inside its range and the
    sentence says nothing; the driver clause uses its own, narrower band (D7)."""
    from prism import config

    _, _, reference, _ = built
    assert reference.driver_band == float(config.get("explain.driver_band"))
    assert reference.driver_band < reference.band
    low, high = reference.numeric_band("vaai_score")
    low_d, high_d = reference.numeric_driver_band("vaai_score")
    assert low <= low_d and high_d <= high and (low_d, high_d) != (low, high)
