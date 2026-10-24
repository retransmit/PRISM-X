"""The label timeline: when the bank would have learned each label, from the shipped
generator's supervision model, never from the ground truth directly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.harness import timeline

pytest.importorskip("prism_generator")

SETTINGS = {"k_daily": 50, "alert_threshold": 0.7, "seed": 1}


def _frame(n: int = 400, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(0.0, 30 * 86_400.0, size=n)) + 1_704_067_200.0
    fraud = rng.random(n) < 0.08
    return pd.DataFrame({
        "transaction_id": [f"TX_{i:06d}" for i in range(n)],
        "card_id": [f"CARD_{i % 40:04d}" for i in range(n)],
        "tx_time_seconds": times,
        "amount": rng.lognormal(7.0, 0.8, size=n).round(2),
        "is_fraud": fraud.astype(int),
        "scenario_tag": np.where(fraud, "INTENT_OMEGA_HARVEST", "ORGANIC_NORMAL"),
        "risk_score": np.where(fraud, rng.uniform(0.5, 1.0, size=n), rng.uniform(0.0, 0.6, size=n)),
    })


def test_discovery_never_precedes_the_transaction_and_dark_fraud_is_never_known():
    frame = _frame()
    tl = timeline.build(frame, SETTINGS)
    assert list(tl.columns) == ["transaction_id", "is_fraud", "discovery_time_seconds", "label_source"]
    assert len(tl) == len(frame)
    merged = tl.merge(frame[["transaction_id", "tx_time_seconds"]], on="transaction_id")
    known = merged["discovery_time_seconds"].notna()
    assert (merged.loc[known, "discovery_time_seconds"] >= merged.loc[known, "tx_time_seconds"]).all()
    assert set(tl["label_source"]) <= {"INVESTIGATOR_ALERT", "CHARGEBACK_DISPUTE", "UNLABELLED", "UNREPORTED_DARK_FRAUD"}
    assert tl.loc[tl["label_source"] == "UNREPORTED_DARK_FRAUD", "discovery_time_seconds"].isna().all()
    assert (tl["is_fraud"].to_numpy() == frame["is_fraud"].to_numpy()).all()


def test_known_fraud_at_a_cutoff_is_the_fraud_discovered_by_then():
    frame = _frame()
    tl = timeline.build(frame, SETTINGS)
    cutoff = float(frame["tx_time_seconds"].quantile(0.5))
    known = timeline.known_fraud_at(tl, cutoff)
    assert known.index.equals(tl.index)
    discovered = tl["discovery_time_seconds"].notna() & (tl["discovery_time_seconds"] <= cutoff) & (tl["is_fraud"] == 1)
    assert known.equals(discovered.rename("known_fraud"))
    assert 0 < known.sum() < int(tl["is_fraud"].sum()), "some fraud is known and some still unknown at the midpoint"
    dark = int((tl["label_source"] == "UNREPORTED_DARK_FRAUD").sum())
    assert timeline.known_fraud_at(tl, float("inf")).sum() == int(tl["is_fraud"].sum()) - dark


def test_the_timeline_is_cached_beside_the_export_and_reproducible(tmp_path, monkeypatch):
    from prism.harness import data

    frame = _frame()
    monkeypatch.setattr(data, "stem", lambda seed: tmp_path / "train_in")
    first = timeline.load_or_build(9, frame)
    assert (tmp_path / "train_in_labels.csv").exists()
    second = timeline.load_or_build(9, frame)
    pd.testing.assert_frame_equal(first, second)
    rebuilt = timeline.build(frame, timeline.supervision_settings())
    pd.testing.assert_frame_equal(first.reset_index(drop=True), rebuilt.reset_index(drop=True))


def test_a_stale_cache_is_rebuilt_when_the_stream_changed(tmp_path, monkeypatch):
    from prism.harness import data

    monkeypatch.setattr(data, "stem", lambda seed: tmp_path / "train_in")
    first = timeline.load_or_build(9, _frame(n=300))
    second = timeline.load_or_build(9, _frame(n=320))
    assert len(first) == 300 and len(second) == 320


def test_a_cache_with_the_same_ids_but_a_changed_stream_or_settings_is_rebuilt(tmp_path, monkeypatch):
    from prism import config
    from prism.harness import data

    monkeypatch.setattr(data, "stem", lambda seed: tmp_path / "train_in")
    first = timeline.load_or_build(9, _frame(seed=1))
    changed = _frame(seed=2)  # the same transaction ids, other times, amounts and labels
    second = timeline.load_or_build(9, changed)
    assert (second["is_fraud"].to_numpy() == changed["is_fraud"].to_numpy()).all()
    assert not first.equals(second)
    investigated = int((second["label_source"] == "INVESTIGATOR_ALERT").sum())
    monkeypatch.setitem(config.load_config()["harness"]["supervision"], "k_daily", 1)
    third = timeline.load_or_build(9, changed)
    assert int((third["label_source"] == "INVESTIGATOR_ALERT").sum()) < investigated


def test_the_supervision_seed_comes_from_the_config(monkeypatch):
    from prism import config

    monkeypatch.setitem(config.load_config()["harness"]["supervision"], "seed", 7)
    assert timeline.supervision_settings()["seed"] == 7
