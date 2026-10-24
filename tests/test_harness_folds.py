"""Folds: time-ordered, never overlapping, and a novel-attack fold that hides one fraud
scenario from training and validation while the test window keeps it."""

from __future__ import annotations

import numpy as np
import pandas as pd

from prism.harness import folds

DAY = 86_400.0


def _times(days: int, per_day: int = 20, start: float = 1_704_067_200.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.sort(start + rng.uniform(0.0, days * DAY, size=days * per_day))


def test_rolling_origin_windows_are_ordered_disjoint_and_expanding():
    times = _times(60)
    made = folds.rolling_origin(times, train_days=24, val_days=6, test_days=10, step_days=10)
    assert [f.name for f in made] == ["fold-1", "fold-2", "fold-3"]
    for f in made:
        assert len(f.train) and len(f.val) and len(f.test)
        assert times[f.train].max() < times[f.val].min() <= times[f.val].max() < times[f.test].min()
        assert not (set(f.train) & set(f.val)) and not (set(f.val) & set(f.test)) and not (set(f.train) & set(f.test))
        assert f.train_end < f.val_end < f.test_end
        assert times[f.train].max() < f.train_end <= times[f.val].min()
        assert times[f.val].max() < f.val_end <= times[f.test].min()
        assert times[f.test].max() < f.test_end
        assert f.held_out_scenario is None
    assert len(made[0].train) < len(made[1].train) < len(made[2].train)
    assert made[1].train_end - made[0].train_end == 10 * DAY
    # a scaled frame later in the stream never enters an earlier fold's training rows
    assert made[0].train.max() < made[0].val.min()


def test_a_stream_too_short_for_one_fold_yields_none():
    assert folds.rolling_origin(_times(20), train_days=24, val_days=6, test_days=10, step_days=10) == []


def test_leave_one_scenario_out_hides_the_scenario_from_training_and_validation_only():
    times = _times(60)
    frame = pd.DataFrame({"tx_time_seconds": times, "is_fraud": 0, "scenario_tag": "ORGANIC_NORMAL"})
    rng = np.random.default_rng(1)
    fraud = rng.choice(len(frame), size=120, replace=False)
    frame.loc[fraud[:60], ["is_fraud", "scenario_tag"]] = [1, "INTENT_OMEGA_PROBE"]
    frame.loc[fraud[60:], ["is_fraud", "scenario_tag"]] = [1, "INTENT_OMEGA_HARVEST"]
    assert folds.scenarios(frame) == ["INTENT_OMEGA_HARVEST", "INTENT_OMEGA_PROBE"]
    base = folds.rolling_origin(times, train_days=24, val_days=6, test_days=10, step_days=10)[-1]
    held = folds.leave_one_scenario_out(frame, base, "INTENT_OMEGA_PROBE")
    assert held.held_out_scenario == "INTENT_OMEGA_PROBE"
    assert held.name == f"{base.name}-without-INTENT_OMEGA_PROBE"
    assert (frame.loc[held.train, "scenario_tag"] != "INTENT_OMEGA_PROBE").all()
    assert (frame.loc[held.val, "scenario_tag"] != "INTENT_OMEGA_PROBE").all()
    assert np.array_equal(held.test, base.test)
    assert (frame.loc[held.train, "scenario_tag"] == "INTENT_OMEGA_HARVEST").any()
    assert len(held.train) == len(base.train) - int((frame.loc[base.train, "scenario_tag"] == "INTENT_OMEGA_PROBE").sum())
    assert (held.train_end, held.val_end, held.test_end) == (base.train_end, base.val_end, base.test_end)
