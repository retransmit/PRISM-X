"""Velocity features see only the card's strictly earlier rows, offline and online alike."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism import tracks
from prism.features import velocity


def test_derived_names_match_the_schema():
    """The velocity pass produces exactly the velocity columns the schema
    derives; with the limit ratios they make up everything the service derives."""
    from prism.features import ratios

    assert velocity.DERIVED == tracks.INDIA_VELOCITY
    assert set(velocity.DERIVED) | set(ratios.RATIO_COLUMNS) == set(tracks.INDIA.derived_columns)
    assert set(velocity.DERIVED).isdisjoint(ratios.RATIO_COLUMNS)


def test_no_history_means_first_everything():
    row = velocity.velocity_row([], t=1_000_000.0, mcc="5411", channel="CNP_WEB")
    assert row["card_tx_1h"] == 0.0 and row["card_tx_24h"] == 0.0 and row["card_amount_24h"] == 0.0
    assert row["seconds_since_last"] == velocity.velocity_windows()["no_history_seconds"]
    assert row["first_mcc_30d"] == 1.0 and row["first_cnp_30d"] == 1.0
    assert row["hour_of_day"] == float((1_000_000 % 86_400) // 3_600)


def test_windows_count_what_they_say_and_ignore_the_future():
    t = 10_000_000.0
    history = [
        (t - 600.0, 100.0, "5411", "CP_POS_CHIP"),       # inside the hour and the half hour
        (t - 7_000.0, 50.0, "5812", "CNP_WEB"),          # inside the day, outside the hour
        (t - 40 * 86_400.0, 999.0, "5732", "CNP_WEB"),   # older than the month window
        (t, 1.0, "5732", "CNP_WEB"),                     # same clock value: not history
        (t + 5.0, 1.0, "5732", "CNP_WEB"),               # the future: never history
    ]
    row = velocity.velocity_row(history, t=t, mcc="5732", channel="CNP_MOBILE")
    assert row["card_tx_1h"] == 1.0
    assert row["card_tx_24h"] == 2.0
    assert row["card_amount_24h"] == 150.0
    assert row["seconds_since_last"] == 600.0
    assert row["card_distinct_mcc_30m"] == 1.0
    assert row["first_mcc_30d"] == 1.0, "the only 5732 is older than 30 days"
    assert row["first_cnp_30d"] == 0.0, "a CNP purchase 7,000 seconds ago counts"


def test_offline_pass_equals_row_by_row_online_computation():
    rng = np.random.default_rng(3)
    n = 400
    frame = pd.DataFrame({
        "card_id": rng.choice(["A", "B", "C"], size=n),
        "tx_time_seconds": np.sort(rng.uniform(0, 5 * 86_400, size=n)) + 1_704_067_200.0,
        "amount": rng.gamma(2.0, 500.0, size=n),
        "mcc": rng.choice(["5411", "5812", "5732"], size=n),
        "channel_type": rng.choice(["CP_POS_CHIP", "CNP_WEB"], size=n),
    })
    frame.loc[5, "tx_time_seconds"] = frame.loc[4, "tx_time_seconds"]  # a tie
    frame.loc[5, "card_id"] = frame.loc[4, "card_id"]
    shuffled = frame.sample(frac=1.0, random_state=1)
    derived = velocity.add_velocity_features(shuffled)
    assert list(derived.index) == list(shuffled.index), "row order and index are preserved"

    month = velocity.velocity_windows()["month_seconds"]
    for position in rng.choice(n, size=40, replace=False):
        row = frame.loc[position]
        earlier = frame[(frame.card_id == row.card_id) & (frame.tx_time_seconds < row.tx_time_seconds)
                        & (frame.tx_time_seconds >= row.tx_time_seconds - month)]
        history = list(zip(earlier.tx_time_seconds, earlier.amount, earlier.mcc, earlier.channel_type))
        expected = velocity.velocity_row(history, float(row.tx_time_seconds), str(row.mcc), str(row.channel_type))
        for name in velocity.DERIVED:
            assert derived.loc[position, name] == pytest.approx(expected[name]), (position, name)


def test_longest_window_is_the_month():
    assert velocity.longest_window() == velocity.velocity_windows()["month_seconds"]


def test_the_columns_the_velocity_features_read_are_named_once():
    """The service and the training pass must read the same two columns; a
    rename that reaches one and not the other would silently derive nonsense."""
    from prism import tracks
    from prism.features import velocity

    assert velocity.MCC_COLUMN in tracks.INDIA.categorical_columns
    assert velocity.CHANNEL_COLUMN in tracks.INDIA.categorical_columns
    assert velocity.MCC_COLUMN in tracks.INDIA.input_columns and velocity.CHANNEL_COLUMN in tracks.INDIA.input_columns
