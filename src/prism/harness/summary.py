"""Means with intervals over the fold values, pairwise comparisons on common folds in every
label regime, and the markdown tables the report embeds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

METRICS = (
    "pr_auc",
    "recall",
    "precision",
    "fpr_per_10k",
    "adjusted_precision_at_prevalence",
    "share_of_fraud_value_caught",
    "share_above_floor_caught",
    "alerts_per_day",
    "alerts_per_day_over_capacity",
    "cost_total_inr_per_10k",
    "scoring_p99_ms",
    "explain_p99_ms",
)
PAIRWISE_METRICS = ("pr_auc", "recall", "cost_total_inr_per_10k")


def interval(values, *, resamples: int, seed: int = 0) -> dict[str, float]:
    """The mean of the fold values with a 95 percent percentile-bootstrap interval of the
    mean; a missing value (None, NaN) is left out and the count says how many remain."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(int(resamples), len(v)), replace=True).mean(axis=1)
    return {
        "mean": float(v.mean()),
        "low": float(np.percentile(means, 2.5)),
        "high": float(np.percentile(means, 97.5)),
        "n": int(len(v)),
    }


def aggregate(rows: list[dict], metrics, *, resamples: int) -> list[dict]:
    """One entry per (model, label mode): the interval of each metric over its fold values,
    and how many of those folds had their cutoff set by the budget alone."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["model"], r["label_mode"]), []).append(r)
    out = []
    for (model, mode), group in sorted(groups.items()):
        entry: dict[str, Any] = {
            "model": model,
            "label_mode": mode,
            "n_folds": len(group),
            "n_budget_alone": int(sum(1 for g in group if g.get("threshold_from_budget_alone"))),
        }
        for m in metrics:
            entry[m] = interval([g.get(m) for g in group], resamples=resamples)
        out.append(entry)
    return out


def pairwise(rows: list[dict], metric: str, *, resamples: int, label_mode: str = "full") -> list[dict]:
    """Differences of a metric between two models on the folds both ran under one label
    regime, with the interval of the mean difference and the share of folds on which each
    model is at least the other."""
    by_model: dict[str, dict[tuple, float]] = {}
    for r in rows:
        if r["label_mode"] == label_mode and r.get(metric) is not None:
            by_model.setdefault(r["model"], {})[(r["seed"], r["fold"])] = float(r[metric])
    names = sorted(by_model)
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            common = sorted(set(by_model[a]) & set(by_model[b]))
            if not common:
                continue
            diffs = np.asarray([by_model[a][k] - by_model[b][k] for k in common], dtype=float)
            out.append({
                "model_a": a,
                "model_b": b,
                "metric": metric,
                "label_mode": label_mode,
                "n_folds": len(common),
                "difference": interval(diffs, resamples=resamples),
                "share_a_at_least_b": float(np.mean(diffs >= 0.0)),
                "share_b_at_least_a": float(np.mean(diffs <= 0.0)),
            })
    return out


def pairwise_all(rows: list[dict], metric: str, *, resamples: int) -> list[dict]:
    """The pairwise comparisons in every label regime the rows carry."""
    modes = sorted({r["label_mode"] for r in rows})
    return [p for mode in modes for p in pairwise(rows, metric, resamples=resamples, label_mode=mode)]


def _fmt(x: dict[str, float], digits: int = 4) -> str:
    if x["n"] == 0:
        return "n/a"
    return f"{x['mean']:.{digits}f} [{x['low']:.{digits}f}, {x['high']:.{digits}f}]"


def summary_table(models: list[dict]) -> str:
    lines = [
        "| Model | Labels | Folds | Cutoffs from the budget alone | PR-AUC | Recall at budget | Precision | "
        "False alerts per 10k | Alerts per day | Cost per 10k (INR) | Scoring p99 (ms) | Explanation p99 (ms) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in models:
        lines.append(
            f"| {m['model']} | {m['label_mode']} | {m['n_folds']} | {m['n_budget_alone']} | {_fmt(m['pr_auc'])} | "
            f"{_fmt(m['recall'], 3)} | {_fmt(m['precision'], 3)} | {_fmt(m['fpr_per_10k'], 1)} | "
            f"{_fmt(m['alerts_per_day'], 1)} | {_fmt(m['cost_total_inr_per_10k'], 0)} | {_fmt(m['scoring_p99_ms'], 2)} | "
            f"{_fmt(m.get('explain_p99_ms', {'n': 0}), 2)} |"
        )
    return "\n".join(lines)


def scenario_table(scenarios: list[dict]) -> str:
    lines = [
        "| Model | Held-out scenario | Seeds | Held-out fraud rows | Held-out recall | PR-AUC |",
        "|---|---|---|---|---|---|",
    ]
    for s in scenarios:
        lines.append(
            f"| {s['model']} | {s['held_out_scenario']} | {s['n_seeds']} | {s['held_out_fraud_rows']} | "
            f"{_fmt(s['held_out_recall'], 3)} | {_fmt(s['pr_auc'])} |"
        )
    return "\n".join(lines)


def scenario_summary(scenario_rows: list[dict], *, resamples: int) -> list[dict]:
    """One entry per (model, held-out scenario) over the seeds: the scenario folds run on
    each seed's last fold, so the values are means over seeds."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in scenario_rows:
        groups.setdefault((r["model"], str(r["held_out_scenario"])), []).append(r)
    return [
        {
            "model": model,
            "held_out_scenario": scenario,
            "n_seeds": len({g["seed"] for g in group}),
            "n_runs": len(group),
            "held_out_fraud_rows": int(sum(int(g.get("held_out_fraud_rows", 0)) for g in group)),
            "held_out_recall": interval([g.get("held_out_recall") for g in group], resamples=resamples),
            "pr_auc": interval([g.get("pr_auc") for g in group], resamples=resamples),
        }
        for (model, scenario), group in sorted(groups.items())
    ]


def write(rows: list[dict], scenario_rows: list[dict], out: Path, s: dict) -> None:
    """summary.json, summary.md and scenario.md beside the results."""
    resamples = int(s["bootstrap_resamples"])
    models = aggregate(rows, METRICS, resamples=resamples)
    pairs = [p for metric in PAIRWISE_METRICS for p in pairwise_all(rows, metric, resamples=resamples)]
    scenarios = scenario_summary(scenario_rows, resamples=resamples)
    out = Path(out)
    (out / "summary.json").write_text(
        json.dumps(
            {"models": models, "pairwise": pairs, "scenarios": scenarios, "metrics": list(METRICS)},
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "summary.md").write_text(summary_table(models) + "\n", encoding="utf-8")
    (out / "scenario.md").write_text(scenario_table(scenarios) + "\n", encoding="utf-8")
