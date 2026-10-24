"""The harness's runs: one (seed, fold, model, label mode) at a time, each fitting its
preprocessing on its own training rows, tuning its threshold on its validation window and
scoring its test window once; the driver over every seed, fold, model and mode, plus the
novel-attack folds; results.json and scenario.json as the evidence.

    python -m prism.harness.run [--seeds 42,7] [--models xgboost,autoencoder] [--label-modes operational,full]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prism import tracks
from prism.data import india
from prism.data.preprocess import apply_preprocessing, fit_preprocessing
from prism.evaluation.metrics import evaluate, track_extras
from prism.evaluation.threshold import budget_threshold, tune_threshold
from prism.harness import cost, data, folds, timeline
from prism.models import MODEL_NAMES
from prism.training.resampling import model_factory

OUT_DIR_NAME = "harness"
LABEL_MODES = ("operational", "full")


def settings() -> dict:
    return data.settings()


def window_settings() -> dict[str, float]:
    s = settings()
    return {f"{k}_days": float(s[f"{k}_days"]) for k in ("train", "val", "test", "step")}


def aligned_timeline(frame: pd.DataFrame, tl: pd.DataFrame) -> pd.DataFrame:
    """The timeline in the frame's row order, so a mask over it indexes the frame."""
    ids = frame["transaction_id"].astype(str)
    aligned = tl.set_index(tl["transaction_id"].astype(str)).loc[ids].reset_index(drop=True)
    if len(aligned) != len(frame):
        raise ValueError("the timeline does not cover the frame's rows")
    return aligned


def training_labels(truth: np.ndarray, tl: pd.DataFrame, cutoff: float, mode: str) -> np.ndarray:
    """The labels a model trains or tunes on: the truth (full), or only the fraud the bank
    knew by the cut-off, everything else legitimate (operational)."""
    if mode == "full":
        return truth
    if mode != "operational":
        raise ValueError(f"unknown label mode {mode!r}; the modes are {list(LABEL_MODES)}")
    return truth & timeline.known_fraud_at(tl, cutoff).to_numpy().astype(int)


def _scoring_p99_ms(model, X: pd.DataFrame, rows: int) -> float:
    timings = []
    for i in range(min(int(rows), len(X))):
        one = X.iloc[[i]]
        started = time.perf_counter()
        model.risk_score(one)
        timings.append((time.perf_counter() - started) * 1000.0)
    return float(np.percentile(timings, 99)) if timings else float("nan")


def _explain_p99_ms(model, X_train: pd.DataFrame, X: pd.DataFrame, rows: int, schema: tracks.TrackSchema) -> float:
    """The model's own explainer on up to ``rows`` test rows, one at a time; NaN when the
    model has no explainer."""
    from prism.explain.shap_engine import ShapEngine

    try:
        engine = ShapEngine.for_model(model, list(X_train.columns), background=X_train.head(100), schema=schema)
    except (NotImplementedError, ValueError):
        return float("nan")
    timings = []
    for i in range(min(int(rows), len(X))):
        one = X.iloc[[i]]
        started = time.perf_counter()
        engine.explain(one)
        timings.append((time.perf_counter() - started) * 1000.0)
    return float(np.percentile(timings, 99)) if timings else float("nan")


def fold_matrices(frame: pd.DataFrame, fold: folds.Fold, schema: tracks.TrackSchema):
    """The fold's three matrices: the encoder and the scaler are fitted on the training rows
    alone and applied to the validation and test rows."""
    X = frame[list(schema.feature_columns)]
    X_train, encoder, scaler = fit_preprocessing(X.iloc[fold.train], schema)
    X_val = apply_preprocessing(X.iloc[fold.val], schema, encoder, scaler)
    X_test = apply_preprocessing(X.iloc[fold.test], schema, encoder, scaler)
    return X_train, X_val, X_test, encoder, scaler


def run_one(
    frame: pd.DataFrame,
    tl: pd.DataFrame,
    fold: folds.Fold,
    model_name: str,
    mode: str,
    schema: tracks.TrackSchema,
    s: dict,
) -> dict[str, Any]:
    """One model on one fold under one label mode. ``tl`` is the timeline aligned to the frame."""
    truth = frame[schema.target_column].astype(int).to_numpy()
    amounts = frame[schema.amount_column].to_numpy(dtype=float)

    X_train, X_val, X_test, _encoder, _scaler = fold_matrices(frame, fold, schema)
    y_train = training_labels(truth, tl, fold.train_end, mode)[fold.train]
    # The threshold is tuned when the model is deployed, at the end of the validation
    # window, with the labels the bank knows by then.
    y_val = training_labels(truth, tl, fold.val_end, mode)[fold.val]
    y_test = truth[fold.test]
    if int(y_train.sum()) == 0:
        raise ValueError(f"{fold.name} {mode}: no fraud is known in the training window, nothing to learn from")

    model = model_factory(model_name, resample=False, categorical_columns=schema.categorical_columns)()
    started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started
    strategy = schema.setting("threshold_strategy", None)
    budget = schema.setting("fpr_budget_per_10k", None)
    val_scores = np.asarray(model.risk_score(X_val), dtype=float)
    budget_alone = int(y_val.sum()) == 0 and strategy == "fpr_budget" and budget is not None
    if budget_alone:
        # No fraud the bank knows of in the validation window (its labels have not matured):
        # the budget alone sets the cutoff over the rows it believes legitimate.
        threshold = budget_threshold(val_scores[y_val == 0], fpr_budget_per_10k=float(budget))
    else:
        threshold = float(tune_threshold(y_val, val_scores, strategy=strategy, fpr_budget_per_10k=budget))
    scores = np.asarray(model.risk_score(X_test), dtype=float)
    flagged = (scores >= threshold).astype(int)

    result = evaluate(y_test, scores, threshold, model=model_name).as_dict()
    extras = track_extras(
        y_test,
        scores,
        threshold,
        amounts[fold.test],
        prevalence=float(schema.setting("prevalence_target")),
        floor=float(schema.setting("value_floor")),
        prevalence_range=schema.setting("prevalence_range", None),
    )
    costs = cost.cost_per_10k(
        y_test,
        flagged,
        amounts[fold.test],
        rows=len(fold.test),
        review_inr_per_alert=float(s["cost"]["review_inr_per_alert"]),
        friction_inr_per_false_positive=float(s["cost"]["friction_inr_per_false_positive"]),
    )
    test_days = (fold.test_end - fold.val_end) / folds.DAY
    alerts_per_day = float(flagged.sum()) / test_days

    row: dict[str, Any] = {
        **result,
        **extras,
        "model": model_name,
        "label_mode": mode,
        "fold": fold.name,
        "held_out_scenario": fold.held_out_scenario,
        "train_end": fold.train_end,
        "val_end": fold.val_end,
        "test_end": fold.test_end,
        "training_rows": int(len(fold.train)),
        "training_fraud_rows": int(y_train.sum()),
        "validation_rows": int(len(fold.val)),
        "validation_fraud_rows": int(y_val.sum()),
        "test_rows": int(len(fold.test)),
        "test_fraud_rows": int(y_test.sum()),
        "threshold": threshold,
        "threshold_from_budget_alone": bool(budget_alone),
        "fit_seconds": float(fit_seconds),
        "alerts_per_day": alerts_per_day,
        "alerts_per_day_over_capacity": alerts_per_day / float(s["supervision"]["k_daily"]),
        "cost_loss_inr_per_10k": costs["loss_inr"],
        "cost_review_inr_per_10k": costs["review_inr"],
        "cost_friction_inr_per_10k": costs["friction_inr"],
        "cost_total_inr_per_10k": costs["total_inr"],
        "alerts_per_10k": costs["alerts_per_10k"],
        "missed_fraud_per_10k": costs["missed_fraud_per_10k"],
        "scoring_p99_ms": _scoring_p99_ms(model, X_test, int(s["scoring_rows"])),
        "explain_p99_ms": _explain_p99_ms(model, X_train, X_test, int(s["scoring_rows"]), schema),
    }
    if fold.held_out_scenario is not None:
        held = frame["scenario_tag"].astype(str).to_numpy()[fold.test] == fold.held_out_scenario
        held_fraud = held & (y_test == 1)
        row["held_out_fraud_rows"] = int(held_fraud.sum())
        row["held_out_recall"] = float(flagged[held_fraud].sum() / held_fraud.sum()) if held_fraud.any() else None
    return row


def timeline_facts(seed: int, tl: pd.DataFrame, frame: pd.DataFrame) -> dict[str, Any]:
    """What the timeline says about one stream: how much fraud the bank ever learns of,
    through which channel, and how long after the payment."""
    fraud = tl["is_fraud"].astype(int) == 1
    discovered = (fraud & tl["discovery_time_seconds"].notna()).to_numpy()
    delays_days = (
        tl.loc[discovered, "discovery_time_seconds"].to_numpy() - frame.loc[discovered, "tx_time_seconds"].to_numpy()
    ) / folds.DAY
    return {
        "seed": seed,
        "rows": int(len(tl)),
        "fraud_rows": int(fraud.sum()),
        "never_reported": int((fraud & (tl["label_source"] == "UNREPORTED_DARK_FRAUD")).sum()),
        "investigated": int((fraud & (tl["label_source"] == "INVESTIGATOR_ALERT")).sum()),
        "chargebacks": int((fraud & (tl["label_source"] == "CHARGEBACK_DISPUTE")).sum()),
        "median_discovery_days": float(np.median(delays_days)) if len(delays_days) else None,
    }


def run_seed(
    seed: int, models: list[str], modes: list[str], schema: tracks.TrackSchema, s: dict
) -> tuple[list[dict], list[dict], dict]:
    frame = data.load_seed(seed)
    tl = aligned_timeline(frame, timeline.load_or_build(seed, frame))
    made = folds.rolling_origin(frame[schema.time_column].to_numpy(dtype=float), **window_settings())
    if not made:
        raise ValueError(
            f"seed {seed}: the stream is shorter than one fold "
            f"({s['train_days']} + {s['val_days']} + {s['test_days']} days)"
        )
    rows: list[dict] = []
    scenario_rows: list[dict] = []
    for fold in made:
        for model_name in models:
            for mode in modes:
                row = run_one(frame, tl, fold, model_name, mode, schema, s)
                row["seed"] = seed
                rows.append(row)
                print(
                    f"[harness] seed {seed} {fold.name} {model_name} {mode}: PR-AUC {row['pr_auc']:.4f} "
                    f"recall {row['recall']:.3f} at {row['fpr_per_10k']:.1f} false alerts per 10k"
                )
    last = made[-1]
    for scenario in folds.scenarios(frame):
        held = folds.leave_one_scenario_out(frame, last, scenario)
        for model_name in models:
            row = run_one(frame, tl, held, model_name, "full", schema, s)
            row["seed"] = seed
            scenario_rows.append(row)
            recall = row["held_out_recall"]
            print(
                f"[harness] seed {seed} {held.name} {model_name}: held-out recall "
                f"{recall if recall is not None else 'n/a'} on {row['held_out_fraud_rows']} rows"
            )
    return rows, scenario_rows, timeline_facts(seed, tl, frame)


def _public(row: dict) -> dict:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def _write_rows(out: Path, common: dict, rows: list[dict], scenario_rows: list[dict]) -> None:
    (out / "results.json").write_text(
        json.dumps({**common, "rows": [_public(r) for r in rows]}, indent=2), encoding="utf-8"
    )
    (out / "scenario.json").write_text(
        json.dumps({**common, "rows": [_public(r) for r in scenario_rows]}, indent=2), encoding="utf-8"
    )


def summarize(out_dir: Path) -> Path:
    """The summary files from the results already written: after a change to the summary
    alone, or after a run that stopped before its summary, nothing has to be re-scored."""
    from prism.harness import summary

    out = Path(out_dir)
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    scenario = json.loads((out / "scenario.json").read_text(encoding="utf-8"))
    summary.write(results["rows"], scenario["rows"], out, results["settings"])
    return out


def run_all(seeds: list[int], models: list[str], modes: list[str], out_dir: Path | None = None) -> Path:
    """Every seed in turn, the rows written after each one so a run that dies keeps what it
    scored (a results file without a matching summary is refused by the report), and the
    summary once every seed is in."""
    schema = tracks.get("india")
    s = settings()
    out = Path(out_dir) if out_dir is not None else schema.metrics_dir / OUT_DIR_NAME
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    scenario_rows: list[dict] = []
    timelines: list[dict] = []
    common = {
        "settings": {**s, "seeds": list(seeds), "models": list(models), "label_modes": list(modes)},
        "generator_revision": india.generator_revision(),
        "timelines": timelines,
    }
    for seed in seeds:
        seed_rows, seed_scenarios, facts = run_seed(seed, models, modes, schema, s)
        rows += seed_rows
        scenario_rows += seed_scenarios
        timelines.append(facts)
        _write_rows(out, common, rows, scenario_rows)
    from prism.harness import summary

    summary.write(rows, scenario_rows, out, s)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the evaluation harness over the Indian seed streams.")
    parser.add_argument("--seeds", type=str, default=None, help="comma-separated; default: the config's")
    parser.add_argument("--models", type=str, default=",".join(MODEL_NAMES))
    parser.add_argument("--label-modes", type=str, default=None, help="comma-separated; default: the config's")
    parser.add_argument(
        "--summary-only", action="store_true",
        help="rebuild summary.json, summary.md and scenario.md from the results already written",
    )
    args = parser.parse_args()
    tracks.set_active("india")
    if args.summary_only:
        out = summarize(tracks.get("india").metrics_dir / OUT_DIR_NAME)
        print(f"[harness] summary -> {out}")
        return
    s = settings()
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else data.seeds()
    modes = args.label_modes.split(",") if args.label_modes else [str(m) for m in s["label_modes"]]
    out = run_all(seeds, args.models.split(","), modes)
    print(f"[harness] -> {out}")


if __name__ == "__main__":
    main()
