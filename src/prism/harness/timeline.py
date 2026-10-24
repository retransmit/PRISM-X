"""When the bank would have learned each label: the shipped generator's supervision model
(an investigator queue with a daily capacity, chargeback delays, unreported fraud) run over
a stream once and cached beside its export. Training reads it; evaluation never does.

The cache is keyed by a fingerprint of the stream's content and the supervision settings,
never by the transaction ids alone: a regenerated stream carries the same ids and the same
row count, and serving it the old timeline would silently label the operational regime
with another stream's discoveries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prism import config, tracks
from prism.harness import data

COLUMNS = ["transaction_id", "is_fraud", "discovery_time_seconds", "label_source"]
RECORD_COLUMNS = ("transaction_id", "card_id", "tx_time_seconds", "amount", "currency", "is_fraud", "scenario_tag", "risk_score")


def supervision_settings() -> dict:
    """The supervision model's knobs, every one from config.yaml (harness.supervision)."""
    return dict(config.get("harness.supervision"))


def _records(frame: pd.DataFrame) -> pd.DataFrame:
    """What the supervision model sees of each row, in stream order."""
    records = frame.copy()
    if "currency" not in records.columns:
        records["currency"] = tracks.INDIA.currency
    return records[list(RECORD_COLUMNS)].astype({"is_fraud": int, "transaction_id": str, "card_id": str})


def fingerprint(frame: pd.DataFrame, settings: dict) -> str:
    """A digest of the stream's content the model reads and of the settings it runs under."""
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(_records(frame), index=False).to_numpy().tobytes())
    digest.update(json.dumps(settings, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()


def build(frame: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """The timeline of one stream: the supervision model sees each record as the bank does
    (its time, amount, currency, the reference risk score and the scenario it came from) and
    says when, if ever, its label becomes known. Never-discovered rows carry NaN."""
    from prism_generator.stream import SupervisionEngine

    engine = SupervisionEngine(
        k_daily=int(settings["k_daily"]),
        alert_threshold=float(settings["alert_threshold"]),
        seed=int(settings["seed"]),
    )
    supervised = engine.process_batch(_records(frame).to_dict("records"))
    discovery = [
        np.nan if s.discovery_time_seconds is None or not np.isfinite(s.discovery_time_seconds) else float(s.discovery_time_seconds)
        for s in supervised
    ]
    return pd.DataFrame({
        "transaction_id": [s.transaction_id for s in supervised],
        "is_fraud": [int(s.is_fraud_ground_truth) for s in supervised],
        "discovery_time_seconds": discovery,
        "label_source": [s.label_source.value for s in supervised],
    })[COLUMNS]


def cache_path(seed: int) -> Path:
    stem = data.stem(seed)
    return stem.with_name(stem.name + "_labels.csv")


def cache_key_path(seed: int) -> Path:
    return cache_path(seed).with_suffix(".json")


def load_or_build(seed: int, frame: pd.DataFrame) -> pd.DataFrame:
    """The cached timeline of the seed when its recorded fingerprint matches this frame and
    these settings, else a fresh one, written beside the export with its key."""
    settings = supervision_settings()
    key = fingerprint(frame, settings)
    path, key_path = cache_path(seed), cache_key_path(seed)
    if path.exists() and key_path.exists():
        recorded = json.loads(key_path.read_text(encoding="utf-8"))
        if recorded.get("fingerprint") == key:
            cached = pd.read_csv(path, dtype={"transaction_id": str, "label_source": str})
            if len(cached) == len(frame):
                return cached[COLUMNS]
    built = build(frame, settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    built.to_csv(path, index=False)
    key_path.write_text(
        json.dumps({"fingerprint": key, "rows": int(len(built)), "supervision": settings}, indent=2), encoding="utf-8"
    )
    return built


def known_fraud_at(timeline: pd.DataFrame, cutoff_seconds: float) -> pd.Series:
    """Which rows the bank knows to be fraud at the cut-off: discovered, and discovered by then."""
    discovered = timeline["discovery_time_seconds"].notna() & (timeline["discovery_time_seconds"] <= cutoff_seconds)
    return (discovered & (timeline["is_fraud"].astype(int) == 1)).rename("known_fraud")
