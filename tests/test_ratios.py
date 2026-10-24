"""Limit ratios: what the card's limit says about this payment, without the limit itself."""

from __future__ import annotations

import pandas as pd
import pytest

from prism import tracks
from prism.features import ratios


def test_ratios_are_shares_of_the_limit_and_clipped():
    found = ratios.limit_ratios(amount=2_500.0, credit_limit=10_000.0, available_balance=4_000.0)
    assert found == {"utilization": pytest.approx(0.6), "amount_to_limit": pytest.approx(0.25)}
    over = ratios.limit_ratios(amount=30_000.0, credit_limit=10_000.0, available_balance=-2_000.0)
    assert over["utilization"] == pytest.approx(1.2) and over["amount_to_limit"] == 1.5
    far_over = ratios.limit_ratios(amount=100.0, credit_limit=10_000.0, available_balance=-9_000.0)
    assert far_over["utilization"] == 1.5
    zero = ratios.limit_ratios(amount=100.0, credit_limit=0.0, available_balance=0.0)
    assert zero == {"utilization": 1.5, "amount_to_limit": 1.5}


def test_frame_ratios_match_the_row_function():
    frame = pd.DataFrame({"amount": [2_500.0, 100.0], "credit_limit": [10_000.0, 0.0], "available_balance": [4_000.0, 0.0]})
    out = ratios.add_limit_ratios(frame)
    assert out.loc[0, "utilization"] == pytest.approx(0.6) and out.loc[0, "amount_to_limit"] == pytest.approx(0.25)
    assert out.loc[1, "utilization"] == 1.5 and out.loc[1, "amount_to_limit"] == 1.5


def test_the_indian_track_has_no_card_identifier_and_no_persona_among_its_features():
    features = set(tracks.INDIA.feature_columns)
    assert {"utilization", "amount_to_limit"} <= features
    assert not {"credit_limit", "available_balance", "cohort_id"} & features
    assert {"credit_limit", "available_balance"} <= set(tracks.INDIA.input_columns)
    assert "cohort_id" not in tracks.INDIA.input_columns
