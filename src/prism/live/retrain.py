"""The retrain job: a challenger trained on the labels known so far, promoted on evidence.

    python -m prism.live.retrain --as-of <stream seconds>

At stream time ``as_of`` the backtest window ends a label lag before the clock, so most
of its fraud has been reported; the validation window is the days before it and the
training window everything before that (plan Decision 5). The three models are trained
into a new bundle under operational labels (a row is fraud once a fraud label for it is
known at ``as_of``), the bundle's champion is chosen on validation as offline training
chooses it, and it is scored with the serving champion on the same backtest rows. The
challenger is promoted only when the 95 percent paired interval of the PR-AUC difference
lies above zero (Decision 6). Every retrain appends its event to the bundles' event log,
whatever its status.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from prism import config, tracks
from prism.api import db
from prism.data.preprocess import SplitData, apply_preprocessing, fit_preprocessing, save_encoder, save_scaler
from prism.evaluation import shortcuts
from prism.evaluation.metrics import paired_bootstrap
from prism.explain import artifacts
from prism.live import bundles, drift
from prism.live.frame import labelled_frame
from prism.models import MODEL_NAMES
from prism.training import manifest
from prism.training.train import train_one

DAY = 86_400.0
# The backtest rows and both models' scores on them, kept in the bundle so the run's
# report can score the same rows against the truth (the job itself never sees it).
BACKTEST_FILE = "backtest.json"
AUDIT_FILE = "shortcut_audit.json"


def settings() -> dict:
    return dict(config.get("live"))


def windows(as_of: float, s: dict, origin: float = 0.0) -> dict[str, tuple[float, float]]:
    """Half-open stream-second intervals: training from ``origin``, then validation, then
    the backtest, which ends ``label_lag_days`` before ``as_of``."""
    backtest_end = float(as_of) - float(s["label_lag_days"]) * DAY
    backtest_start = backtest_end - float(s["backtest_days"]) * DAY
    val_start = backtest_start - float(s["val_days"]) * DAY
    return {
        "train": (float(origin), val_start),
        "val": (val_start, backtest_start),
        "backtest": (backtest_start, backtest_end),
    }


def _mask(frame: pd.DataFrame, window: tuple[float, float], schema: tracks.TrackSchema) -> np.ndarray:
    times = frame[schema.time_column].to_numpy(dtype=float)
    return (times >= window[0]) & (times < window[1])


def _skip(event: dict, reason: str, schema: tracks.TrackSchema) -> dict:
    event.update({"status": "skipped", "reason": reason})
    bundles.record_event(schema, event)
    return event


def _train_bundle(
    frame: pd.DataFrame, masks: dict[str, np.ndarray], schema: tracks.TrackSchema, bundle_dir, as_of: float,
    w: dict[str, tuple[float, float]],
) -> None:
    """The three models, the preprocessing, the explanation reference and the manifest,
    fitted on the training rows, thresholds tuned on the validation rows."""
    target = schema.target_column
    X = frame[list(schema.feature_columns)]
    y = frame[target].astype(int)
    X_train, encoder, scaler = fit_preprocessing(X[masks["train"]], schema)
    splits = SplitData(
        X_train=X_train,
        y_train=y[masks["train"]],
        X_val=apply_preprocessing(X[masks["val"]], schema, encoder, scaler),
        y_val=y[masks["val"]],
        X_test=apply_preprocessing(X[masks["backtest"]], schema, encoder, scaler),
        y_test=y[masks["backtest"]],
        scaler=scaler,
        scale_columns=list(schema.scale_columns),
        encoder=encoder,
        categorical_columns=tuple(schema.categorical_columns),
        track=schema.name,
    )
    bundle_dir.mkdir(parents=True, exist_ok=False)
    save_scaler(scaler, bundle_dir / "scaler.joblib")
    if encoder is not None:
        save_encoder(encoder, bundle_dir / "encoder.joblib")
    reference, cases = artifacts.build(frame, splits, schema=schema)
    artifacts.save(reference, cases, bundle_dir)

    trained: dict[str, dict[str, Any]] = {}
    for name in sorted(MODEL_NAMES):
        threshold, _result, extras = train_one(name, splits, resample=False, model_dir=bundle_dir, schema=schema)
        trained[name] = {
            "threshold": float(threshold),
            "resample_applied": False,
            "validation_pr_auc": float(extras["validation_pr_auc"]),
        }
    names = sorted(trained)
    record = manifest.merge(
        None,
        track=schema.name,
        fingerprint=manifest.preprocessing_fingerprint(scaler, encoder),
        trained=trained,
        primary="auto",
        digests=manifest.artifact_digests(bundle_dir, names),
        known=MODEL_NAMES,
    )
    record.update({
        "as_of": float(as_of),
        "train_start": w["train"][0],
        "train_end": w["train"][1],
        "val_end": w["val"][1],
        "training_rows": int(masks["train"].sum()),
        "training_known_fraud": int(splits.y_train.sum()),
    })
    manifest.write(bundle_dir, record)
    (bundle_dir / "thresholds.json").write_text(json.dumps(manifest.thresholds(record), indent=2), encoding="utf-8")


def retrain(as_of: float, schema: tracks.TrackSchema, s: dict | None = None) -> dict:
    """Train a challenger at ``as_of`` and promote it only on evidence; returns the event."""
    s = s or settings()
    started = time.perf_counter()
    frame = labelled_frame(as_of, schema)
    target = schema.target_column
    origin = float(frame[schema.time_column].min()) if len(frame) else float(as_of)
    w = windows(as_of, s, origin)
    masks = {name: _mask(frame, window, schema) for name, window in w.items()}
    known = {name: int(frame.loc[mask, target].sum()) for name, mask in masks.items()}
    champion_bundle, champion_dir = bundles.serving_dir(schema)
    event: dict[str, Any] = {
        "as_of": float(as_of),
        "stream_day": (float(as_of) - origin) / DAY,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "windows": {name: list(window) for name, window in w.items()},
        "rows": {name: int(mask.sum()) for name, mask in masks.items()},
        "known_fraud": known,
        "status": None,
        "reason": None,
        "challenger_bundle": None,
        "challenger_model": None,
        "challenger_pr_auc": None,
        "champion_bundle": champion_bundle,
        "champion_model": None,
        "champion_pr_auc": None,
        "difference": None,
    }

    # The serving bundle was fitted and tuned on its own windows; a backtest inside them
    # would score the champion on rows it learned from. The offline directory records no
    # windows: its training export is another stream.
    serving = manifest.load(champion_dir) or {}
    if serving.get("val_end") is not None and float(serving["val_end"]) > w["backtest"][0]:
        return _skip(
            event,
            f"the backtest window starts at stream second {w['backtest'][0]:.0f}, which overlaps the serving "
            f"bundle {champion_bundle}'s validation window (it ends at {float(serving['val_end']):.0f})",
            schema,
        )
    floor = int(s["min_known_fraud"])
    for name in ("train", "val", "backtest"):
        if known[name] < floor:
            label = {"train": "training", "val": "validation", "backtest": "backtest"}[name]
            return _skip(
                event,
                f"the {label} window holds {known[name]} known frauds among {int(masks[name].sum())} rows, "
                f"fewer than min_known_fraud {floor}",
                schema,
            )

    # The shortcut audit of every training run (design 7.2 item 12): a single column that
    # ranks the known labels above the ceiling gives them away, and the retrain refuses.
    training_rows = frame.loc[masks["train"]]
    audit_rows = shortcuts.audit(training_rows, training_rows[target], schema)
    ceiling = schema.setting("shortcut_auc_ceiling", None)
    if ceiling is not None:
        allowed = dict(schema.setting("allowed_shortcuts", {}) or {})
        found = shortcuts.offenders(audit_rows, ceiling=float(ceiling), allowed=allowed)
        if found:
            named = ", ".join(f"{r['feature']} (AUC {r['auc']:.3f})" for r in found)
            return _skip(
                event,
                f"the shortcut audit refuses the training window: a single column ranks its known labels above "
                f"{float(ceiling):.2f}: {named}",
                schema,
            )

    bundle_id = bundles.new_id(as_of, origin)
    bundle_dir = bundles.root(schema) / bundle_id
    try:
        _train_bundle(frame, masks, schema, bundle_dir, as_of, w)
        shortcuts.write(audit_rows, bundle_dir / AUDIT_FILE)
        challenger_model, challenger = bundles.champion_scorer(bundle_dir, schema)
        champion_model, champion = bundles.champion_scorer(champion_dir, schema)
        reference_rows = frame.loc[masks["train"] | masks["val"]]
        drift.save(drift.reference(reference_rows, challenger(frame.loc[masks["val"]]), schema, s), bundle_dir)
    except Exception:
        shutil.rmtree(bundle_dir, ignore_errors=True)
        raise

    rows = frame.loc[masks["backtest"]]
    y = rows[target].astype(int).to_numpy()
    challenger_scores, champion_scores = challenger(rows), champion(rows)
    (bundle_dir / BACKTEST_FILE).write_text(json.dumps({
        "transaction_id": rows["transaction_id"].astype(str).tolist(),
        "challenger": [float(x) for x in challenger_scores],
        "champion": [float(x) for x in champion_scores],
    }), encoding="utf-8")
    difference = paired_bootstrap(
        y, challenger_scores, champion_scores, resamples=int(s["bootstrap_resamples"]), seed=config.seed()
    )
    promoted = difference["low"] > 0.0
    event.update({
        "status": "promoted" if promoted else "kept",
        "reason": None if promoted else (
            f"the paired interval of the PR-AUC difference [{difference['low']:.4f}, {difference['high']:.4f}] "
            "does not lie above zero"
        ),
        "challenger_bundle": bundle_id,
        "challenger_model": challenger_model,
        "challenger_pr_auc": float(average_precision_score(y, challenger_scores)),
        "champion_model": champion_model,
        "champion_pr_auc": float(average_precision_score(y, champion_scores)),
        "difference": {"low": difference["low"], "high": difference["high"]},
        "train_seconds": time.perf_counter() - started,
    })
    if promoted:
        bundles.promote(schema, bundle_id, event)
    else:
        bundles.record_event(schema, event)
    return event


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a challenger on the labels known so far.")
    parser.add_argument("--as-of", type=float, required=True, help="the stream clock, in seconds")
    parser.add_argument(
        "--database-url", default=None, help="the audit log to learn from (default: live.database_url)"
    )
    args = parser.parse_args()
    tracks.set_active(tracks.INDIA.name)
    db.init_db(args.database_url or config.get("live.database_url"))
    event = retrain(args.as_of, tracks.INDIA)
    print(json.dumps(event, indent=2))


if __name__ == "__main__":
    main()
