"""Model bundles: every retrain writes a full model directory of its own.

``<live.model_dir>/<track>/bundles/<bundle id>/`` holds what the engine loads (the
models, the scaler, the encoder, the manifest, the explanation reference) plus the drift
reference and the backtest scores. ``<live.model_dir>/<track>/current.json`` names the
bundle a live service serves; without it the offline directory ``models/<track>/`` serves.
Only a service started with the live switch on (``PRISM_LIVE=1``) reads the pointer, so
the offline deployment and the benchmarks never serve a bundle by accident.
Promotion rewrites the pointer through a temporary file and ``os.replace``, so a
reader sees the old pointer or the new one and never half of one, and appends the
retrain's event to ``bundles/events.jsonl``.
"""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from prism import config, tracks
from prism.data.preprocess import transform_features

OFFLINE = "offline"
POINTER = "current.json"
EVENTS = "events.jsonl"


def live_enabled() -> bool:
    """Whether this process follows the live loop's pointer (live.enabled, or PRISM_LIVE)."""
    value = config.get("live.enabled", False)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def live_dir(schema: tracks.TrackSchema) -> Path:
    return config.get_path("live.model_dir") / schema.name


def root(schema: tracks.TrackSchema) -> Path:
    return live_dir(schema) / "bundles"


def pointer_path(schema: tracks.TrackSchema) -> Path:
    return live_dir(schema) / POINTER


def new_id(as_of: float, origin: float = 0.0) -> str:
    """``day-<stream day>-<6 hex>``: the day of the stream the bundle was trained at,
    counted from ``origin`` (the stream's first transaction), and a random suffix."""
    day = int((float(as_of) - float(origin)) // 86_400)
    return f"day-{day:03d}-{secrets.token_hex(3)}"


def serving_dir(schema: tracks.TrackSchema) -> tuple[str, Path]:
    """The bundle id and directory the pointer names, or the offline directory."""
    pointer = pointer_path(schema)
    if not pointer.exists():
        return OFFLINE, schema.model_dir
    try:
        record = json.loads(pointer.read_text(encoding="utf-8"))
        bundle_id = str(record["bundle_id"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"The pointer {pointer} cannot be read ({exc!r}); rewrite it or delete it") from exc
    if bundle_id == OFFLINE:
        return OFFLINE, schema.model_dir
    return bundle_id, root(schema) / bundle_id


def promote(schema: tracks.TrackSchema, bundle_id: str, event: dict) -> None:
    """Point the service at ``bundle_id`` and record the event that promoted it."""
    pointer = pointer_path(schema)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    staged = pointer.with_name(pointer.name + ".tmp")
    staged.write_text(json.dumps({
        "bundle_id": bundle_id,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
    }), encoding="utf-8")
    os.replace(staged, pointer)
    record_event(schema, event)


def record_event(schema: tracks.TrackSchema, event: dict) -> None:
    """Append one retrain's event, whatever its status."""
    log = root(schema) / EVENTS
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, sort_keys=True) + "\n")


def events(schema: tracks.TrackSchema) -> list[dict]:
    log = root(schema) / EVENTS
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]


def champion_scorer(
    model_dir: Path, schema: tracks.TrackSchema
) -> tuple[str, Callable[[pd.DataFrame], np.ndarray]]:
    """The champion the engine would serve from ``model_dir``, and a function scoring
    raw frame rows (the derived columns already present) through that directory's
    encoder and scaler, exactly as the service transforms one request."""
    from prism.api.inference import InferenceEngine

    engine = InferenceEngine(model_dir=Path(model_dir), track=schema.name).load()
    model = engine.models[engine.primary]

    def score(frame: pd.DataFrame) -> np.ndarray:
        transformed = transform_features(
            frame,
            engine.scaler,
            engine.feature_order,
            engine.scale_columns,
            encoder=engine.encoder,
            categorical_columns=schema.categorical_columns,
        )
        return np.asarray(model.risk_score(transformed), dtype=float)

    return engine.primary, score
