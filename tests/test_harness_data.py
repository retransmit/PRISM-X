"""The harness's data: several generator seeds, each a stream the shipped command writes."""

from __future__ import annotations

import sys

from prism import config
from prism.harness import data


def test_seeds_and_stems_come_from_the_config(monkeypatch):
    monkeypatch.setitem(config.load_config()["harness"], "seeds", [3, 5])
    monkeypatch.setitem(config.load_config()["harness"], "stem_pattern", "data/raw/india/seeds/{seed}/train_in")
    assert data.seeds() == [3, 5]
    assert data.stem(5) == config.PROJECT_ROOT / "data" / "raw" / "india" / "seeds" / "5" / "train_in"


def test_the_generation_command_is_the_shipped_generators_with_every_setting_from_the_config(monkeypatch):
    harness = config.load_config()["harness"]
    monkeypatch.setitem(harness, "rows", 1200)
    monkeypatch.setitem(harness, "cards", 40)
    monkeypatch.setitem(harness, "merchants", 15)
    monkeypatch.setitem(harness, "days", 12)
    monkeypatch.setitem(harness, "fraud_rate", 0.02)
    monkeypatch.setitem(harness, "calibration", "rbi-psi-2026-07")
    command = data.generate_command(7)
    # the shipped generator's CLI through the running interpreter, so no activated
    # environment or PATH entry is needed on the host that runs it
    assert command[:4] == [sys.executable, "-m", "prism_generator.cli", "generate"]
    command = command[4:]
    for flag, value in (
        ("-n", "1200"), ("-c", "40"), ("-m", "15"), ("--days", "12"), ("--fraud-rate", "0.02"),
        ("--seed", "7"), ("--calibration", "rbi-psi-2026-07"), ("--region", "IN"),
    ):
        assert command[command.index(flag) + 1] == value, flag
    assert "--export-institutional-views" in command
    assert command[command.index("-o") + 1].endswith("train_in.csv")


def test_load_seed_carries_the_frame_and_the_master_columns_the_harness_needs(india_stem, monkeypatch):
    monkeypatch.setattr(data, "stem", lambda seed: india_stem)
    frame = data.load_seed(0)
    for column in ("transaction_id", "scenario_tag", "risk_score", "is_fraud", "tx_time_seconds", "amount", "card_id", "card_tx_24h"):
        assert column in frame.columns, column
    assert frame["tx_time_seconds"].is_monotonic_increasing
    assert set(frame["scenario_tag"].unique()) >= {"ORGANIC_NORMAL"}
    assert frame["risk_score"].between(0.0, 1.0).all()
    assert set(data.MASTER_COLUMNS) == {"transaction_id", "is_fraud", "scenario_tag", "risk_score"}
