"""Module 3: the Phase 3 experiment matrix.

    python -m prism.evaluation.experiments

SMOTE and cost-sensitive weighting are two answers to the same question, and
stacking them double-counts the minority class. The spec's Phase 3 deliverable
is the *comparison* between them, so it is produced by one code path rather
than assembled by hand across several runs of ``train.py``.

Oversampling is only meaningful where a supervised minority class exists:
Isolation Forest is unsupervised and the autoencoder trains on normals only, so
requesting SMOTE for either records ``resample_applied: False`` rather than
silently doing nothing. The distinction between *requested* and *applied* is
kept in the output so the report can state which arms were genuinely comparable.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from prism import config, tracks
from prism.data.preprocess import SplitData
from prism.evaluation.metrics import evaluate, paired_bootstrap, track_extras
from prism.evaluation.threshold import tune_threshold
from prism.models import build

# NOTE: prism.training.resampling is imported lazily inside run_arm, not here.
# It imports prism.evaluation.metrics, so a module-level import would close the
# cycle evaluation/__init__ -> experiments -> training.resampling -> evaluation
# and fail with a half-initialised module.

# Models for which oversampling is conceptually applicable at all. Single source
# of truth: prism.training.train imports this rather than keeping its own copy,
# so the two entry points cannot drift into resampling different models.
RESAMPLEABLE: frozenset[str] = frozenset({"xgboost"})


@dataclass(frozen=True)
class ExperimentArm:
    model: str
    resample: bool = False

    @property
    def label(self) -> str:
        strategy = "smote" if self.resample else "weighted"
        return f"{self.model}+{strategy}"


DEFAULT_ARMS: tuple[ExperimentArm, ...] = (
    ExperimentArm("isolation_forest"),
    ExperimentArm("autoencoder"),
    ExperimentArm("xgboost", resample=False),
    ExperimentArm("xgboost", resample=True),
)


def run_arm(arm: ExperimentArm, splits: SplitData, amounts=None) -> dict[str, Any]:
    """Train one arm, tune the threshold on validation, evaluate once on test."""
    return run_arm_scored(arm, splits, amounts=amounts)[0]


def run_arm_scored(arm: ExperimentArm, splits: SplitData, amounts=None) -> tuple[dict[str, Any], np.ndarray]:
    """``run_arm`` and the arm's test scores, which the paired bootstrap compares."""
    schema = tracks.get(splits.track)
    from prism.training.resampling import imbalance_overrides, resample_training_fold  # see note above

    X_train, y_train = splits.X_train, splits.y_train

    resample_applied = arm.resample and arm.model in RESAMPLEABLE
    if resample_applied:
        X_train, y_train = resample_training_fold(X_train, y_train, categorical_columns=splits.categorical_columns)

    started = time.perf_counter()
    overrides = imbalance_overrides(resample_applied)
    if splits.categorical_columns:
        overrides["categorical_columns"] = splits.categorical_columns
    model = build(arm.model, **overrides).fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started

    threshold = tune_threshold(
        splits.y_val, model.risk_score(splits.X_val), strategy=schema.setting("threshold_strategy", None)
    )
    test_scores = model.risk_score(splits.X_test)
    result = evaluate(splits.y_test, test_scores, threshold, model=arm.label)

    record = result.as_dict()
    prevalence = schema.setting("prevalence_target", None)
    if prevalence is not None and amounts is not None:
        record.update(track_extras(
            splits.y_test, test_scores, threshold, amounts,
            prevalence=float(prevalence), floor=float(schema.setting("value_floor")),
            prevalence_range=schema.setting("prevalence_range", None),
        ))
    record.update(
        {
            "label": arm.label,
            # `model` is overwritten deliberately: EvalResult.model carries the
            # arm label for display, but the matrix needs the bare model name to
            # group the two XGBoost arms together.
            "model": arm.model,
            "resample_requested": arm.resample,
            "resample_applied": resample_applied,
            "train_rows": int(len(X_train)),
            "train_positives": int(y_train.sum()),
            "fit_seconds": fit_seconds,
            "scale_pos_weight": getattr(model, "fitted_pos_weight", None),
        }
    )
    return record, np.asarray(test_scores, dtype=float)


def run_matrix(
    splits: SplitData, arms: tuple[ExperimentArm, ...] | None = None, amounts=None
) -> list[dict[str, Any]]:
    selected = arms if arms is not None else DEFAULT_ARMS

    results: list[dict[str, Any]] = []
    scored: dict[str, np.ndarray] = {}
    for arm in selected:
        print(f"=== {arm.label} ===")
        record, scored[arm.label] = run_arm_scored(arm, splits, amounts=amounts)
        print(
            f"  PR-AUC={record['pr_auc']:.4f}  P={record['precision']:.4f}  "
            f"R={record['recall']:.4f}  missed={record['false_negatives']}  "
            f"({record['fit_seconds']:.1f}s)"
        )
        results.append(record)
    # Two XGBoost arms scored on the same split differ by sampling alone
    # unless a paired interval of their PR-AUC difference excludes zero.
    weighted = next((r for r in results if r["model"] == "xgboost" and not r["resample_applied"]), None)
    resampled = next((r for r in results if r["model"] == "xgboost" and r["resample_applied"]), None)
    if weighted and resampled:
        interval = paired_bootstrap(
            splits.y_test, scored[weighted["label"]], scored[resampled["label"]], seed=config.seed()
        )
        weighted["difference_ci"] = {"against": "resampled", **interval}
        print(
            f"  PR-AUC difference, weighted minus resampled: {interval['low']:+.4f} to {interval['high']:+.4f} "
            f"(95 percent, paired bootstrap)"
        )
    return results


def comparison_table(results: list[dict[str, Any]]) -> str:
    header = (
        "| Arm | SMOTE | Threshold | PR-AUC | Precision | Recall | F1 | "
        "Missed fraud | Alerts per catch |\n"
        "|---|---|---|---|---|---|---|---|---|"
    )
    # A model that flags nothing has no alerts per catch (None); the table
    # says so rather than crashing after every arm has been fitted.
    rows = [
        f"| {r['label']} | {'yes' if r['resample_applied'] else 'no'} | "
        f"{r['threshold']:.4f} | **{r['pr_auc']:.4f}** | {r['precision']:.4f} | "
        f"{r['recall']:.4f} | {r['f1']:.4f} | {r['false_negatives']} | "
        f"{'n/a' if r['alerts_per_true_fraud'] is None else f'{r['alerts_per_true_fraud']:.1f}'} |"
        for r in sorted(results, key=lambda r: r["pr_auc"], reverse=True)
    ]
    return "\n".join([header, *rows])


def main() -> None:
    import argparse

    from prism.data.frames import load_frame
    from prism.data.preprocess import prepare_splits

    parser = argparse.ArgumentParser(description="Run the SMOTE versus weighting experiment matrix.")
    tracks.add_track_argument(parser)
    schema = tracks.activate_from_args(parser.parse_args())

    frame = load_frame(schema)
    splits = prepare_splits(frame, schema=schema)
    print(splits.summary())
    print()

    amounts = frame.loc[splits.X_test.index, schema.amount_column].to_numpy(dtype=float)
    results = run_matrix(splits, amounts=amounts)
    table = comparison_table(results)

    metrics_dir = schema.metrics_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "experiment_matrix.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (metrics_dir / "experiment_matrix.md").write_text(table, encoding="utf-8")

    print("\n" + table)
    print(f"\nWritten -> {metrics_dir / 'experiment_matrix.md'}")


if __name__ == "__main__":
    main()
