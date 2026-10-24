"""Folds on the stream's clock: rolling-origin backtests with an expanding training window,
and novel-attack folds that hide one fraud scenario from training and validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DAY = 86_400.0


@dataclass(frozen=True)
class Fold:
    name: str
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    train_end: float
    val_end: float
    test_end: float
    held_out_scenario: str | None = None


def rolling_origin(times: np.ndarray, *, train_days: float, val_days: float, test_days: float, step_days: float) -> list[Fold]:
    """Train on everything before the cut-off, validate on the next window, test on the one
    after, and roll the origin forward while a full test window still fits in the stream."""
    times = np.asarray(times, dtype=float)
    if len(times) == 0:
        return []
    start = float(times.min())
    # The stream spans whole days (a 60-day run ends just short of the 60-day mark), so a
    # test window that ends on the stream's last day still counts as a full window.
    end = start + float(np.ceil((float(times.max()) - start) / DAY)) * DAY
    made: list[Fold] = []
    train_end = start + train_days * DAY
    index = 1
    while train_end + (val_days + test_days) * DAY <= end + 1e-9:
        val_end = train_end + val_days * DAY
        test_end = val_end + test_days * DAY
        train = np.flatnonzero(times < train_end)
        val = np.flatnonzero((times >= train_end) & (times < val_end))
        test = np.flatnonzero((times >= val_end) & (times < test_end))
        if len(train) and len(val) and len(test):
            made.append(Fold(f"fold-{index}", train, val, test, train_end, val_end, test_end))
            index += 1
        train_end += step_days * DAY
    return made


def scenarios(frame: pd.DataFrame) -> list[str]:
    """The fraud scenarios a stream holds, each a candidate novel attack."""
    fraud = frame["is_fraud"].astype(int).to_numpy() == 1
    return sorted(frame.loc[fraud, "scenario_tag"].astype(str).unique().tolist())


def leave_one_scenario_out(frame: pd.DataFrame, fold: Fold, scenario: str) -> Fold:
    """The fold with the scenario's rows removed from training and validation; the test
    window keeps every row, so the held-out recall is measured on rows of that scenario."""
    keep = frame["scenario_tag"].astype(str).to_numpy() != scenario
    return Fold(
        name=f"{fold.name}-without-{scenario}",
        train=fold.train[keep[fold.train]],
        val=fold.val[keep[fold.val]],
        test=fold.test,
        train_end=fold.train_end,
        val_end=fold.val_end,
        test_end=fold.test_end,
        held_out_scenario=scenario,
    )
