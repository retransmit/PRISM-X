"""The summary: a mean with a 95 percent interval over fold values, and pairwise
differences on common folds only."""

from __future__ import annotations

import json

import pytest

from prism.harness import summary


def _rows():
    rows = []
    for seed in (1, 2):
        for fold in ("fold-1", "fold-2"):
            for model, base in (("a", 0.8), ("b", 0.6)):
                rows.append({
                    "seed": seed, "fold": fold, "model": model, "label_mode": "full",
                    "pr_auc": base + 0.01 * seed, "recall": base - 0.1, "cost_total_inr_per_10k": 1000.0 * (1 + seed),
                })
    rows.append({"seed": 3, "fold": "fold-1", "model": "a", "label_mode": "full", "pr_auc": 0.95, "recall": 0.5, "cost_total_inr_per_10k": 900.0})
    return rows


def test_intervals_cover_the_mean_and_shrink_with_more_values():
    few = summary.interval([0.5, 0.7], resamples=500, seed=0)
    many = summary.interval([0.5, 0.7] * 20, resamples=500, seed=0)
    assert few["low"] <= few["mean"] <= few["high"] and few["n"] == 2
    assert (many["high"] - many["low"]) < (few["high"] - few["low"])
    assert summary.interval([], resamples=10)["n"] == 0
    assert summary.interval([None, 0.4], resamples=10)["n"] == 1


def test_aggregate_and_pairwise_use_only_the_folds_both_models_ran():
    rows = _rows()
    agg = {(r["model"], r["label_mode"]): r for r in summary.aggregate(rows, ["pr_auc", "recall"], resamples=200)}
    assert agg[("a", "full")]["n_folds"] == 5 and agg[("b", "full")]["n_folds"] == 4
    assert agg[("a", "full")]["pr_auc"]["mean"] > agg[("b", "full")]["pr_auc"]["mean"]
    assert "cost_total_inr_per_10k" not in agg[("a", "full")]
    pairs = summary.pairwise(rows, "pr_auc", resamples=200)
    pair = next(p for p in pairs if p["model_a"] == "a" and p["model_b"] == "b")
    assert pair["n_folds"] == 4, "seed 3 has no fold for b and must not count"
    assert pair["difference"]["low"] > 0.0 and pair["share_a_at_least_b"] == 1.0
    assert pair["difference"]["mean"] == pytest.approx(0.2)
    assert summary.pairwise(rows, "pr_auc", resamples=10, label_mode="operational") == []


def test_write_produces_the_three_artifacts(tmp_path):
    rows = _rows()
    for r in rows:
        r.update({m: 0.5 for m in summary.METRICS if m not in r})
    scenario_rows = [
        dict(r, held_out_scenario="INTENT_OMEGA_PROBE", held_out_recall=0.4, held_out_fraud_rows=12)
        for r in rows if r["model"] == "a"
    ]
    summary.write(rows, scenario_rows, tmp_path, {"bootstrap_resamples": 100, "cost": {"hypothesis": True}})
    out = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert {m["model"] for m in out["models"]} == {"a", "b"}
    assert out["pairwise"] and out["scenarios"]
    assert out["scenarios"][0]["held_out_recall"]["mean"] == 0.4
    assert out["scenarios"][0]["n_seeds"] == 3 and out["scenarios"][0]["n_runs"] == 5
    assert {p["label_mode"] for p in out["pairwise"]} == {"full"}
    assert out["metrics"] == list(summary.METRICS)
    text = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "| Model |" in text and "| a | full | 5 |" in text
    assert "INTENT_OMEGA_PROBE" in (tmp_path / "scenario.md").read_text(encoding="utf-8")


def test_budget_alone_cutoffs_are_counted_and_pairwise_covers_every_label_mode():
    rows = []
    for seed in (1, 2):
        for model, base in (("a", 0.8), ("b", 0.6)):
            rows.append({"seed": seed, "fold": "fold-1", "model": model, "label_mode": "full", "pr_auc": base, "threshold_from_budget_alone": False})
            rows.append({"seed": seed, "fold": "fold-1", "model": model, "label_mode": "operational", "pr_auc": base - 0.3, "threshold_from_budget_alone": seed == 2})
    agg = {(r["model"], r["label_mode"]): r for r in summary.aggregate(rows, ["pr_auc"], resamples=50)}
    assert agg[("a", "operational")]["n_budget_alone"] == 1 and agg[("a", "full")]["n_budget_alone"] == 0
    pairs = summary.pairwise_all(rows, "pr_auc", resamples=50)
    assert {p["label_mode"] for p in pairs} == {"full", "operational"}
    assert all(p["n_folds"] == 2 for p in pairs)


def test_the_explanation_p99_is_a_summary_metric_with_its_own_column():
    assert "explain_p99_ms" in summary.METRICS
    rows = [{"seed": 1, "fold": "fold-1", "model": "a", "label_mode": "full", **{m: 0.5 for m in summary.METRICS}}]
    header = summary.summary_table(summary.aggregate(rows, summary.METRICS, resamples=10)).splitlines()[0]
    assert "Scoring p99 (ms)" in header and "Explanation p99 (ms)" in header


def test_scenarios_are_summarised_over_seeds():
    scenario_rows = [
        {"seed": s, "fold": "fold-3-without-X", "model": "a", "label_mode": "full", "held_out_scenario": "X",
         "held_out_recall": 0.5, "held_out_fraud_rows": 10, "pr_auc": 0.4}
        for s in (1, 2, 3)
    ]
    out = summary.scenario_summary(scenario_rows, resamples=50)
    assert out[0]["n_seeds"] == 3 and out[0]["held_out_fraud_rows"] == 30
    assert "| Seeds |" in summary.scenario_table(out)
