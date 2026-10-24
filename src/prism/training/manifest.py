"""The record of what a model directory holds and how it was made.

Serving reads the thresholds from here and refuses a model this file does not
vouch for. Training writes it, merging a partial run into the previous
record only when the preprocessing the kept models were fitted under is the
preprocessing this run fitted; otherwise it refuses before writing anything,
because an old model behind a new scaler or encoder scores wrongly without
raising, and a model served at a default cutoff alerts at the wrong rate.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np

from prism.models import ARTIFACT_FILES

FILENAME = "manifest.json"
VERSION = 1


class PreprocessingMismatch(RuntimeError):
    """Models on disk were fitted under other preprocessing than this run's."""


def _canonical(scaler: Any, encoder: Any | None) -> str:
    parts: dict[str, Any] = {}
    if scaler is not None:
        parts["scaler"] = {
            "features": [str(f) for f in getattr(scaler, "feature_names_in_", [])],
            "mean": np.asarray(scaler.mean_, dtype=float).tolist(),
            "scale": np.asarray(scaler.scale_, dtype=float).tolist(),
        }
    if encoder is not None:
        parts["encoder"] = {
            "features": [str(f) for f in getattr(encoder, "feature_names_in_", [])],
            "categories": [[str(v) for v in cats] for cats in encoder.categories_],
            "unknown_value": repr(getattr(encoder, "unknown_value", None)),
        }
    return json.dumps(parts, sort_keys=True)


def preprocessing_fingerprint(scaler: Any, encoder: Any | None) -> str:
    """A digest of the transformation the fitted scaler and encoder apply: the
    features they take, the means and scales, the categories and the unknown
    code. Incidental attributes (sample counts, array dtypes) stay out, so a
    saved fit and a fresh fit of the same data agree."""
    return hashlib.sha256(_canonical(scaler, encoder).encode("utf-8")).hexdigest()


def load(model_dir: Path) -> dict[str, Any] | None:
    path = Path(model_dir) / FILENAME
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def write(model_dir: Path, record: dict[str, Any]) -> Path:
    path = Path(model_dir) / FILENAME
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def check_compatible(existing: dict[str, Any] | None, fingerprint: str, retraining: Sequence[str]) -> None:
    """Raise unless every model the run keeps was fitted under ``fingerprint``."""
    if existing is None:
        return
    stale = sorted(
        name for name, entry in existing.get("models", {}).items()
        if name not in retraining and entry.get("fingerprint") != fingerprint
    )
    if stale:
        raise PreprocessingMismatch(
            "This run fits a scaler or encoder different from the one these models were trained "
            f"under: {', '.join(stale)}. Retrain everything (--model all) or remove them first."
        )


def artifact_digests(model_dir: Path, names: Sequence[str]) -> dict[str, dict[str, str]]:
    """The file and the SHA-256 of each named model's fitted state on disk."""
    model_dir = Path(model_dir)
    digests: dict[str, dict[str, str]] = {}
    for name in names:
        artifact = ARTIFACT_FILES[name]
        path = model_dir / artifact
        if path.exists():
            digests[name] = {"artifact": artifact, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return digests


def merge(
    existing: dict[str, Any] | None,
    *,
    track: str,
    fingerprint: str,
    trained: dict[str, dict[str, Any]],
    primary: str,
    digests: dict[str, dict[str, str]] | None = None,
    known: Sequence[str] | None = None,
) -> dict[str, Any]:
    """The manifest after a run: the kept entries plus this run's models,
    each with the digest of the file it was saved to. With ``known`` (the model
    registry), an entry for a model the registry no longer has is dropped, so a
    retired model neither keeps a threshold nor stands as a champion."""
    now = datetime.now(timezone.utc).isoformat()
    models = dict((existing or {}).get("models", {}))
    if known is not None:
        models = {name: entry for name, entry in models.items() if name in set(known)}
    for name, entry in trained.items():
        models[name] = {**entry, **(digests or {}).get(name, {}), "fingerprint": fingerprint, "trained_at": now}
    champion = choose_champion(models)
    if primary == "auto":
        # The champion the evidence picks; without evidence for every model (a one-model
        # retrain over entries from before champions existed) the record keeps the primary
        # it had rather than crowning whatever was just retrained.
        primary = champion["model"] if champion else (existing or {}).get("primary_model", primary)
    return {
        "version": VERSION,
        "track": track,
        "written_at": now,
        "preprocessing": {"fingerprint": fingerprint},
        "primary_model": primary,
        "champion": champion,
        "models": models,
    }


CHAMPION_METRIC = "validation_pr_auc"


def choose_champion(models: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """The model with the best validation PR-AUC, with every candidate's value beside it, so
    the choice and its evidence travel with the bundle. None unless every model in the
    bundle recorded the metric: a choice among some of them is not a choice."""
    if not models or any(entry.get(CHAMPION_METRIC) is None for entry in models.values()):
        return None
    candidates = {name: float(entry[CHAMPION_METRIC]) for name, entry in models.items()}
    best = max(candidates, key=candidates.get)
    return {"model": best, "metric": CHAMPION_METRIC, "value": candidates[best], "candidates": candidates}


def thresholds(record: dict[str, Any]) -> dict[str, float]:
    return {name: float(entry["threshold"]) for name, entry in record["models"].items()}


def merge_rows(
    existing: list[dict[str, Any]] | None,
    fresh: list[dict[str, Any]],
    *,
    replacing: Sequence[str],
    known: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """A result list after a run: the rows of the models the run kept, then
    this run's rows, so a partial run never shortens the record; with ``known``,
    rows of models the registry no longer has are dropped."""
    kept = [
        row for row in (existing or [])
        if row.get("model") not in set(replacing) and (known is None or row.get("model") in set(known))
    ]
    return kept + list(fresh)


def adopt_legacy(model_dir: Path, fingerprint: str, *, track: str, primary: str) -> dict[str, Any] | None:
    """A manifest for a directory trained before the manifest existed.

    Such a directory carries thresholds.json alone. Its models are adopted,
    with this run's fingerprint, only when the scaler and encoder on disk apply
    the transformation this run fitted; otherwise they are recorded as
    ``legacy``, which a partial run refuses and a full retrain replaces. None
    when there is nothing to adopt.
    """
    model_dir = Path(model_dir)
    threshold_file = model_dir / "thresholds.json"
    scaler_file = model_dir / "scaler.joblib"
    if not threshold_file.exists() or not scaler_file.exists():
        return None
    encoder_file = model_dir / "encoder.joblib"
    on_disk = preprocessing_fingerprint(
        joblib.load(scaler_file), joblib.load(encoder_file) if encoder_file.exists() else None
    )
    stamp = fingerprint if on_disk == fingerprint else "legacy"
    legacy = json.loads(threshold_file.read_text(encoding="utf-8"))
    record = merge(
        None,
        track=track,
        fingerprint=stamp,
        # Whether an adopted model was trained on resampled rows is not known.
        trained={name: {"threshold": float(value), "resample_applied": None} for name, value in legacy.items()},
        primary=primary,
        digests=artifact_digests(model_dir, list(legacy)),
    )
    record["preprocessing"]["fingerprint"] = fingerprint
    return record
