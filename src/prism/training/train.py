"""Module 2: training entrypoint.

    python -m prism.training.train --model all
    python -m prism.training.train --model xgboost --resample --cv
    python -m prism.training.train --track india --model all

On resampling: SMOTE and ``scale_pos_weight`` are two answers to the same
question, and stacking them double-counts the minority class. The defaults here
are deliberate:

  * xgboost          -> cost-sensitive weighting, no SMOTE (default)
  * isolation_forest -> unsupervised; SMOTE is meaningless, never applied
  * autoencoder      -> trains on normals only; SMOTE is meaningless, never applied

``--resample`` switches XGBoost to the SMOTE arm instead. Running both and
comparing PR-AUC is Phase 3 of the proposal timeline; that comparison is a
result worth reporting, not an implementation detail.

On the Indian track (``--track india``) the test metrics carry the extra lines
of design D6: precision re-weighted to one fraud in 80,847, alerts per 10,000
legitimate transactions, and the share of fraud value caught above the
discussion paper's Rs 10,000 floor.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prism import config, tracks
from prism.data import india
from prism.data.frames import calibration_report_path, load_frame
from prism.data.loader import describe_balance
from prism.data.preprocess import SplitData, prepare_splits, save_encoder, save_scaler
from prism.evaluation.experiments import RESAMPLEABLE
from prism.evaluation import shortcuts
from prism.evaluation.metrics import evaluate, track_extras
from sklearn.metrics import average_precision_score
from prism.evaluation.threshold import tune_threshold
from prism.explain import artifacts
from prism.models import MODEL_NAMES, build
from prism.training import manifest
from prism.training.resampling import (
    cross_val_scores,
    imbalance_overrides,
    model_factory,
    resample_training_fold,
)


def train_one(name, splits, *, resample, model_dir, amounts=None, schema=None):
    schema = schema or tracks.get(splits.track)
    print(f"\n=== {name} ===")
    overrides = imbalance_overrides(resample and name in RESAMPLEABLE)
    if splits.categorical_columns:
        overrides["categorical_columns"] = splits.categorical_columns
    model = build(name, **overrides)

    X_train, y_train = splits.X_train, splits.y_train
    if resample and name in RESAMPLEABLE:
        before = int(y_train.sum())
        X_train, y_train = resample_training_fold(X_train, y_train, categorical_columns=splits.categorical_columns)
        print(f"  SMOTE: {before:,} -> {int(y_train.sum()):,} positives (train only)")
    elif resample:
        print(f"  SMOTE skipped: not applicable to {name}")

    started = time.perf_counter()
    model.fit(X_train, y_train)
    print(f"  fitted in {time.perf_counter() - started:.1f}s")

    # Threshold from validation, metrics from test. Test is read once. The
    # policy is the track's (design D5), not the process's active track.
    val_scores = model.risk_score(splits.X_val)
    threshold = tune_threshold(splits.y_val, val_scores, strategy=schema.setting("threshold_strategy", None))
    # The champion is chosen on this number, never on the test window.
    validation_pr_auc = float(average_precision_score(np.asarray(splits.y_val).astype(int), val_scores))
    test_scores = model.risk_score(splits.X_test)
    result = evaluate(splits.y_test, test_scores, threshold, model=name)
    print(f"  {result}")
    print(f"  alerts per true fraud: {result.alerts_per_true_fraud:.1f}")

    extras: dict[str, float] = {"validation_pr_auc": validation_pr_auc}
    prevalence = schema.setting("prevalence_target", None)
    if prevalence is not None and amounts is not None:
        extras.update(track_extras(
            splits.y_test, test_scores, threshold, amounts,
            prevalence=float(prevalence), floor=float(schema.setting("value_floor")),
            prevalence_range=schema.setting("prevalence_range", None),
        ))
        print(
            f"  at 1 in {1 / float(prevalence):,.0f}: precision {extras['adjusted_precision_at_prevalence']:.4f}, "
            f"fpr {extras['fpr_per_10k']:.1f}/10k, value caught {extras['share_of_fraud_value_caught']:.1%} "
            f"(above floor {extras['share_above_floor_caught']:.1%})"
        )

    saved_to = model.save(model_dir)
    print(f"  saved -> {saved_to.name}")
    return threshold, result, extras


def write_drift_reference(df: pd.DataFrame, splits: SplitData, schema: tracks.TrackSchema, model_dir: Path) -> Path:
    """The drift reference beside the models: the training and validation rows, and the
    validation scores of the champion the directory serves (live-loop plan, Decision 9)."""
    from prism.live import bundles, drift

    rows = df.loc[splits.X_train.index.append(splits.X_val.index)]
    _champion, score = bundles.champion_scorer(model_dir, schema)
    scores = score(df.loc[splits.X_val.index])
    return drift.save(drift.reference(rows, scores, schema, dict(config.get("live"))), model_dir)


# The generator's report names the generator in its prose fields. The copy
# beside the metrics keeps every measurement and none of the names: this
# repository describes the generator without naming it, so the pattern is
# held encoded and no file spells the names.
GENERATOR_NAMES = re.compile(
    base64.b64decode("ZnJhdWR4Wy1fIF0/c3ludGhlc2lbc3pdZXJ8ZnJhdWR4YWl8ZnJhdWR4").decode(), re.IGNORECASE
)


def _sanitised(value: Any) -> Any:
    if isinstance(value, str):
        return GENERATOR_NAMES.sub("the generator", value)
    if isinstance(value, list):
        return [_sanitised(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitised(item) for key, item in value.items()}
    return value


def copy_calibration_report(report: Path, metrics_dir: Path) -> Path:
    """The generator's calibration report beside the metrics, measurements
    intact, names replaced."""
    data = json.loads(Path(report).read_text(encoding="utf-8"))
    target = Path(metrics_dir) / "calibration.json"
    target.write_text(json.dumps(_sanitised(data), indent=2), encoding="utf-8")
    return target


def write_report_artifacts(
    schema: tracks.TrackSchema, df: pd.DataFrame, splits: SplitData, metrics_dir: Path
) -> None:
    """The split and, on a track trained from an export, the exports: the
    files the results generator reads beside test_results.json, so the report
    never quotes a log line or a number computed by hand."""
    metrics_dir = Path(metrics_dir)
    split = {
        **splits.as_dict(),
        "loaded": describe_balance(df, target=schema.target_column),
        # The knobs this run was trained under, so the report renders the
        # run's shares and budget, not whatever config.yaml says later.
        "settings": dict(config.get(f"tracks.{schema.name}")),
    }
    (metrics_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"  split -> {metrics_dir / 'split.json'}")

    if schema.setting("export_stem", None) is None:
        return
    exports: dict[str, Any] = {
        "training": india.export_facts(config.get_path(f"tracks.{schema.name}.export_stem")),
        "demo": None,
    }
    if schema.setting("demo_stem", None) is not None:
        demo_stem = config.get_path(f"tracks.{schema.name}.demo_stem")
        try:
            exports["demo"] = india.export_facts(demo_stem)
        except FileNotFoundError:
            print(f"  demo export absent at {demo_stem}; exports.json carries the training export only")
    (metrics_dir / "exports.json").write_text(json.dumps(exports, indent=2), encoding="utf-8")
    print(f"  exports -> {metrics_dir / 'exports.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Prism fraud detection models.")
    parser.add_argument(
        "--model", default="all", choices=["all", *sorted(MODEL_NAMES)],
        help="which model to train (default: all)",
    )
    parser.add_argument(
        "--resample", action="store_true",
        help="use SMOTE instead of cost-sensitive weighting where applicable",
    )
    parser.add_argument(
        "--cv", action="store_true",
        help="also run cross-validation, forward-chaining on a time-ordered track (slower)",
    )
    tracks.add_track_argument(parser)
    args = parser.parse_args()
    schema = tracks.activate_from_args(args)

    model_dir = schema.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = schema.metrics_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)

    df = load_frame(schema)
    balance = describe_balance(df, target=schema.target_column)
    print(
        f"[{schema.name}] loaded {balance['n_total']:,} transactions, "
        f"{balance['n_fraud']:,} fraud ({balance['fraud_rate']:.5%}), "
        f"imbalance 1:{balance['imbalance_ratio']:.1f}"
    )

    splits = prepare_splits(df, schema=schema)
    print(splits.summary())
    # Refuse a partial retrain that would leave other models behind new
    # preprocessing, before anything on disk is replaced.
    fingerprint = manifest.preprocessing_fingerprint(splits.scaler, splits.encoder)
    names = sorted(MODEL_NAMES) if args.model == "all" else [args.model]
    existing = manifest.load(model_dir)
    if existing is None:
        # A directory trained before the manifest existed: its models are
        # adopted only if its preprocessing is the one this run fitted.
        existing = manifest.adopt_legacy(
            model_dir, fingerprint, track=schema.name, primary=config.get("api.primary_model", "auto")
        )
    manifest.check_compatible(existing, fingerprint, retraining=names)

    training_rows = df.loc[splits.X_train.index]
    audit_rows = shortcuts.audit(training_rows, training_rows[schema.target_column], schema)
    shortcuts.write(audit_rows, metrics_dir / "shortcut_audit.json")
    print(f"  shortcut audit: strongest single column {audit_rows[0]['feature']} at AUC {audit_rows[0]['auc']:.3f}")
    ceiling = schema.setting("shortcut_auc_ceiling", None)
    if ceiling is not None:
        allowed = dict(schema.setting("allowed_shortcuts", {}) or {})
        found = shortcuts.offenders(audit_rows, ceiling=float(ceiling), allowed=allowed)
        if found:
            named = ", ".join(f"{r['feature']} (AUC {r['auc']:.3f})" for r in found)
            raise SystemExit(
                f"Refusing to train on the {schema.name} track: a single column ranks the training labels "
                f"above {float(ceiling):.2f}: {named}. The stream gives the label away; fix the source, or "
                f"allow the feature with a reason under tracks.{schema.name}.allowed_shortcuts."
            )
    write_report_artifacts(schema, df, splits, metrics_dir)
    save_scaler(splits.scaler, model_dir / "scaler.joblib")
    if splits.encoder is not None:
        save_encoder(splits.encoder, model_dir / "encoder.joblib")
    reference, cases = artifacts.build(df, splits, schema=schema)
    artifacts.save(reference, cases, model_dir)
    print(f"  explanation reference on {reference.n_rows:,} training rows -> {model_dir}")

    report = calibration_report_path(schema)
    if report is not None:
        copied = copy_calibration_report(report, metrics_dir)
        print(f"  calibration report -> {copied}")

    amounts = df.loc[splits.X_test.index, schema.amount_column].to_numpy(dtype=float)
    thresholds: dict[str, float] = {}
    validation: dict[str, float] = {}
    results: list[dict] = []

    for name in names:
        threshold, result, extras = train_one(
            name, splits, resample=args.resample, model_dir=model_dir, amounts=amounts, schema=schema
        )
        thresholds[name] = threshold
        validation[name] = float(extras["validation_pr_auc"])
        results.append({**result.as_dict(), **extras})

        if args.cv:
            print(f"  cross-validating {name}...")
            resampled = args.resample and name in RESAMPLEABLE
            cv = cross_val_scores(
                model_factory(name, resample=resampled, categorical_columns=splits.categorical_columns),
                df.loc[splits.X_train.index],
                schema,
                resample=resampled,
            )
            if cv:
                mean_pr = sum(row["pr_auc"] for row in cv) / len(cv)
                print(f"  CV mean PR-AUC over {len(cv)} folds: {mean_pr:.4f}")

    record = manifest.merge(
        existing,
        track=schema.name,
        fingerprint=fingerprint,
        trained={
            name: {
                "threshold": thresholds[name],
                "resample_applied": bool(args.resample and name in RESAMPLEABLE),
                "validation_pr_auc": validation[name],
            }
            for name in names
        },
        primary=config.get("api.primary_model", "auto"),
        digests=manifest.artifact_digests(model_dir, names),
        known=MODEL_NAMES,
    )
    manifest.write(model_dir, record)
    # The flat map the service and the benchmark read: every model the
    # manifest vouches for, not only this run's.
    (model_dir / "thresholds.json").write_text(
        json.dumps(manifest.thresholds(record), indent=2), encoding="utf-8"
    )
    # The record of every model's test metrics; a partial run replaces its
    # own rows and keeps the others, as the manifest does with thresholds.
    results_path = metrics_dir / "test_results.json"
    previous = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else None
    results_path.write_text(
        json.dumps(manifest.merge_rows(previous, results, replacing=names, known=MODEL_NAMES), indent=2), encoding="utf-8"
    )

    if schema.contract_feed is not None:
        # The live loop's drift reference; the ULB track has no stream to drift.
        try:
            print(f"  drift reference -> {write_drift_reference(df, splits, schema, model_dir)}")
        except RuntimeError as exc:
            print(f"  drift reference not written: the directory cannot be served yet ({exc})")

    print(f"\nThresholds -> {model_dir / 'thresholds.json'}")
    print(f"Test metrics -> {metrics_dir / 'test_results.json'}")


if __name__ == "__main__":
    main()
