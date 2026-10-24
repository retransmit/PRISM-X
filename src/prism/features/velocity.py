"""Point-in-time velocity features for a card (design D3).

One function, ``velocity_row``, computes a row's features from the card's
earlier transactions. Training calls it over the export in a single pass per
card; the service calls it over the rows the audit log holds for the card.
Only strictly earlier rows count: a row is never its own history, and two rows
with the same clock value do not see each other, so a replay produces the same
numbers the training pass did.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np
import pandas as pd

from prism import config, tracks

DERIVED: tuple[str, ...] = tracks.INDIA_VELOCITY

# The two categoricals the features read, named once for the training pass
# and the service alike.
MCC_COLUMN = "mcc"
CHANNEL_COLUMN = "channel_type"

SECONDS_PER_DAY = 86_400.0
SECONDS_PER_HOUR = 3_600.0


def velocity_windows() -> dict[str, float]:
    settings = dict(config.get("features.velocity"))
    return {key: float(value) for key, value in settings.items()}


def longest_window() -> float:
    return velocity_windows()["month_seconds"]


def velocity_row(
    history: Sequence[tuple[float, float, str, str]], t: float, mcc: str, channel: str
) -> dict[str, float]:
    """Features for one transaction at clock ``t`` from ``history``: tuples of
    ``(time_seconds, amount, mcc, channel)`` for the card, in any order."""
    windows = velocity_windows()
    hour, day = windows["hour_seconds"], windows["day_seconds"]
    half_hour, month = windows["half_hour_seconds"], windows["month_seconds"]

    tx_1h = tx_24h = 0
    amount_24h = 0.0
    last: float | None = None
    mcc_30m: set[str] = set()
    seen_mcc = seen_cnp = False
    for earlier_t, earlier_amount, earlier_mcc, earlier_channel in history:
        if earlier_t >= t:
            continue
        age = t - earlier_t
        if age > month:
            continue
        if age <= hour:
            tx_1h += 1
        if age <= day:
            tx_24h += 1
            amount_24h += float(earlier_amount)
        if age <= half_hour:
            mcc_30m.add(str(earlier_mcc))
        if str(earlier_mcc) == str(mcc):
            seen_mcc = True
        if str(earlier_channel).startswith("CNP"):
            seen_cnp = True
        if last is None or earlier_t > last:
            last = earlier_t

    return {
        "hour_of_day": float((t % SECONDS_PER_DAY) // SECONDS_PER_HOUR),
        "card_tx_1h": float(tx_1h),
        "card_tx_24h": float(tx_24h),
        "card_amount_24h": float(amount_24h),
        "seconds_since_last": float(t - last) if last is not None else windows["no_history_seconds"],
        "card_distinct_mcc_30m": float(len(mcc_30m)),
        "first_mcc_30d": 0.0 if seen_mcc else 1.0,
        "first_cnp_30d": 1.0 if (str(channel).startswith("CNP") and not seen_cnp) else 0.0,
    }


def add_velocity_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Every row's features from its card's earlier rows, in one pass per card.

    Row order and index are preserved; only the derived columns are added.
    """
    month = longest_window()
    times = frame["tx_time_seconds"].to_numpy(dtype=float)
    amounts = frame["amount"].to_numpy(dtype=float)
    mccs = frame[MCC_COLUMN].astype(str).to_numpy()
    channels = frame[CHANNEL_COLUMN].astype(str).to_numpy()
    cards = frame["card_id"].astype(str).to_numpy()

    columns = {name: np.zeros(len(frame), dtype=float) for name in DERIVED}
    order = np.lexsort((times, cards))
    history: deque[tuple[float, float, str, str]] = deque()
    current_card: str | None = None
    for position in order:
        if cards[position] != current_card:
            history = deque()
            current_card = cards[position]
        t = times[position]
        while history and t - history[0][0] > month:
            history.popleft()
        row = velocity_row(history, t, mccs[position], channels[position])
        for name in DERIVED:
            columns[name][position] = row[name]
        history.append((t, amounts[position], mccs[position], channels[position]))

    out = frame.copy()
    for name in DERIVED:
        out[name] = columns[name]
    return out
