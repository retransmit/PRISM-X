"""Render each track's results section from its measured artifacts.

    python -m prism.reporting.results [--track ulb|india]

The failure this exists to prevent: a number in the prose disagreeing with the
same number in a file, because the experiments were re-run after the chapter
was written. Nothing here is typed by hand: if a value cannot be read from an
artifact it does not appear, the pipeline's own tables are embedded verbatim,
and the README's Results section is checked against this output by a test.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from prism import config, tracks

# Per track: artifact filename -> the command that writes it. A .md artifact
# is a table the pipeline already rendered; it is embedded as it is.
ARTIFACTS: dict[str, dict[str, str]] = {
    "ulb": {
        "eda.json": "python -m prism.data.eda",
        "split.json": "python -m prism.training.train --track ulb --model all",
        "experiment_matrix.json": "python -m prism.evaluation.experiments --track ulb",
        "experiment_matrix.md": "python -m prism.evaluation.experiments --track ulb",
        "benchmark.json": "python -m prism.evaluation.benchmark --track ulb",
        "benchmark.md": "python -m prism.evaluation.benchmark --track ulb",
        "latency.json": "python -m prism.evaluation.latency --track ulb --explain --http",
        "latency_explain.json": "python -m prism.evaluation.latency --track ulb --explain --http",
        "latency_http.json": "python -m prism.evaluation.latency --track ulb --explain --http",
    },
    "india": {
        "split.json": "python -m prism.training.train --track india --model all",
        "exports.json": "python -m prism.training.train --track india --model all",
        "calibration.json": "python -m prism.training.train --track india --model all",
        "experiment_matrix.json": "python -m prism.evaluation.experiments --track india",
        "experiment_matrix.md": "python -m prism.evaluation.experiments --track india",
        "benchmark.json": "python -m prism.evaluation.benchmark --track india",
        "benchmark.md": "python -m prism.evaluation.benchmark --track india",
        "latency.json": "python -m prism.evaluation.latency --track india --explain --http",
        "latency_explain.json": "python -m prism.evaluation.latency --track india --explain --http",
        "latency_http.json": "python -m prism.evaluation.latency --track india --explain --http",
        "harness/results.json": "python -m prism.harness.run",
        "harness/summary.json": "python -m prism.harness.run",
        "harness/summary.md": "python -m prism.harness.run",
        "harness/scenario.json": "python -m prism.harness.run",
        "harness/scenario.md": "python -m prism.harness.run",
        "live/run.json": "python -m prism.live.run",
        "live/events.json": "python -m prism.live.run",
    },
}

# The harness and the live loop are runs of their own: a track renders without either
# and says so, but a directory with a file missing is refused like any other partial run.
OPTIONAL_PREFIXES = ("harness/", "live/")

TITLES = {
    "ulb": "ULB track (real data)",
    "india": "Indian track (synthetic export, RBI-calibrated)",
}

OUTPUT = config.PROJECT_ROOT / "docs" / "report" / "results.md"

# A number ends in a digit: a comma after it is punctuation, not a thousands separator.
NUMBER = re.compile(r"\d(?:[\d,]*\d)?(?:\.\d+)?(?:e-?\d+)?%?")


class ArtifactError(Exception):
    """The artifacts cannot be rendered as they are; the message says why."""


class MissingArtifact(ArtifactError):
    """A required artifact has not been generated; the message names the command."""


class MixedRuns(ArtifactError):
    """Two artifacts disagree on a number they share, so they come from
    different runs; the message names both files."""


def load_artifacts(schema: tracks.TrackSchema, metrics_dir: Path | None = None) -> dict[str, Any]:
    """Every artifact the track's section needs, or MissingArtifact naming the
    first absent one and the command that writes it. A partial directory is
    refused whole: a section rendered from two runs would mix their numbers."""
    directory = Path(metrics_dir) if metrics_dir is not None else schema.metrics_dir
    loaded: dict[str, Any] = {}
    absent = {
        prefix: not any(
            (directory / f).exists() for f in ARTIFACTS[schema.name] if f.startswith(prefix)
        )
        for prefix in OPTIONAL_PREFIXES
    }
    for filename, command in ARTIFACTS[schema.name].items():
        path = directory / filename
        if not path.exists():
            if any(filename.startswith(prefix) and absent[prefix] for prefix in OPTIONAL_PREFIXES):
                continue
            raise MissingArtifact(f"{path} is missing. Generate it with:  {command}")
        text = path.read_text(encoding="utf-8")
        stem, suffix = filename.rsplit(".", 1)
        stem = stem.replace("/", "_")
        if suffix == "json":
            loaded[stem] = json.loads(text)
        else:
            loaded[f"{stem}_md"] = text.strip()
    _check_one_run(loaded)
    return loaded


def _check_one_run(loaded: dict[str, Any]) -> None:
    """Refuse a directory whose files disagree on the numbers they share.

    A re-run overwrites in place, so every file is present after a partial
    re-run; the counts the files carry in common are the only evidence that
    they describe one run.
    """
    split = loaded["split"]
    test_rows, test_fraud = int(split["test"]["rows"]), int(split["test"]["fraud"])
    for row in loaded["benchmark"]:
        counted = sum(int(row[key]) for key in ("true_negatives", "false_positives", "false_negatives", "true_positives"))
        fraud = int(row["false_negatives"]) + int(row["true_positives"])
        if counted != test_rows or fraud != test_fraud:
            raise MixedRuns(
                f"benchmark.json ({row['model']}: {counted:,} rows, {fraud} fraud) and split.json "
                f"({test_rows:,} test rows, {test_fraud} fraud) come from different runs; re-run the pipeline."
            )
    loaded_counts = split["loaded"]
    if "eda" in loaded:
        eda = loaded["eda"]
        if int(eda["n_total"]) != int(loaded_counts["n_total"]) or int(eda["n_fraud"]) != int(loaded_counts["n_fraud"]):
            raise MixedRuns(
                f"eda.json ({int(eda['n_total']):,} rows, {int(eda['n_fraud'])} fraud) and split.json "
                f"({int(loaded_counts['n_total']):,} rows, {int(loaded_counts['n_fraud'])} fraud) come from different runs."
            )
    if "exports" in loaded:
        training = loaded["exports"]["training"]
        if int(training["rows"]) != int(loaded_counts["n_total"]):
            raise MixedRuns(
                f"exports.json ({int(training['rows']):,} rows) and split.json "
                f"({int(loaded_counts['n_total']):,} rows) come from different runs."
            )
        generated = (training.get("calibration") or {}).get("generated")
        if generated is not None and loaded["calibration"].get("generated") != generated:
            raise MixedRuns("calibration.json and exports.json carry different generated blocks; they come from different exports.")
    # Every model that appears in both files must carry the same threshold
    # and PR-AUC: the experiment matrix and the benchmark are one run or none.
    benchmark_rows = {row["model"]: row for row in loaded["benchmark"]}
    for row in loaded["experiment_matrix"]:
        counterpart = benchmark_rows.get(row["model"])
        if counterpart is None or row["model"] == "xgboost":
            continue
        if bool(row.get("resample_applied")) != bool(counterpart.get("resample_applied", False)):
            continue
        for key in ("threshold", "pr_auc"):
            if round(float(row[key]), 6) != round(float(counterpart[key]), 6):
                raise MixedRuns(
                    f"experiment_matrix.json ({row['model']} {key} {float(row[key]):.4f}) and benchmark.json "
                    f"({row['model']} {key} {float(counterpart[key]):.4f}) come from different runs."
                )
    weighted = next((r for r in loaded["experiment_matrix"] if r["model"] == "xgboost" and not r["resample_applied"]), None)
    resampled = next((r for r in loaded["experiment_matrix"] if r["model"] == "xgboost" and r["resample_applied"]), None)
    if weighted and resampled and "difference_ci" not in weighted:
        raise ArtifactError(
            "experiment_matrix.json carries no difference_ci for the two XGBoost arms: it predates the paired "
            "bootstrap. Re-run `python -m prism.evaluation.experiments` before rendering."
        )
    # The deployed XGBoost is compared with the arm that produced it: the
    # benchmark row says whether it was trained on resampled rows.
    deployed = next((row for row in loaded["benchmark"] if row["model"] == "xgboost"), None)
    arm_resampled = bool(deployed.get("resample_applied", False)) if deployed else False
    arm_name = "resampled" if arm_resampled else "weighted"
    arm = next(
        (
            row for row in loaded["experiment_matrix"]
            if row["model"] == "xgboost" and bool(row["resample_applied"]) == arm_resampled
        ),
        None,
    )
    if arm and deployed:
        for key in ("threshold", "pr_auc"):
            if round(float(arm[key]), 6) != round(float(deployed[key]), 6):
                raise MixedRuns(
                    f"experiment_matrix.json (xgboost+{arm_name} {key} {float(arm[key]):.4f}) and benchmark.json "
                    f"(xgboost {key} {float(deployed[key]):.4f}) come from different runs."
                )
    if "harness_results" in loaded:
        _check_one_harness_run(loaded)
    if "live_run" in loaded:
        _check_one_live_run(loaded)


def _check_one_live_run(loaded: dict[str, Any]) -> None:
    """The runner writes its log after every stream day; only a log it closed at the end of
    the stream describes a whole run, and its retrains must be the ones the event log kept."""
    run = loaded["live_run"]
    if not run.get("complete"):
        played = sum(day["requests"] + day.get("failed", 0) for day in run["days"])
        raise ArtifactError(
            f"live/run.json is a run in progress or one that stopped: {played:,} of {run['rows']:,} rows "
            f"over {len(run['days'])} stream days. Let python -m prism.live.run finish."
        )
    recorded = {(float(e["as_of"]), e["status"]) for e in loaded["live_events"]}
    missing = [e for e in run["retrains"] if e["status"] != "failed" and (float(e["as_of"]), e["status"]) not in recorded]
    if missing:
        raise MixedRuns(
            f"live/run.json lists {len(missing)} retrains that live/events.json does not; "
            "the two files come from different runs."
        )


def _check_one_harness_run(loaded: dict[str, Any]) -> None:
    """The harness writes its rows after every seed and its summary at the end, so a run
    that stopped early, or a summary rebuilt from other rows, leaves files that disagree on
    the counts they share."""
    rows = loaded["harness_results"]["rows"]
    summary = loaded["harness_summary"]
    for entry in summary["models"]:
        counted = sum(1 for r in rows if r["model"] == entry["model"] and r["label_mode"] == entry["label_mode"])
        if int(entry["n_folds"]) != counted:
            raise MixedRuns(
                f"harness/summary.json ({entry['model']} {entry['label_mode']}: {entry['n_folds']} folds) and "
                f"harness/results.json ({counted} rows) come from different runs; re-run the harness."
            )
    summarised = {(e["model"], e["held_out_scenario"]) for e in summary["scenarios"]}
    run = {(r["model"], r["held_out_scenario"]) for r in loaded["harness_scenario"]["rows"]}
    if summarised != run:
        raise MixedRuns(
            "harness/summary.json and harness/scenario.json disagree on the held-out scenarios "
            f"({sorted(summarised ^ run)}); re-run the harness."
        )


def _pct(share: float) -> str:
    """A requested prevalence as the CLI took it: 0.005 -> 0.5%, 0.03 -> 3%."""
    return f"{share * 100:g}%"


def _dataset_paragraph(eda: dict[str, Any]) -> str:
    rate = float(eda["fraud_rate"])
    return (
        f"**Dataset** (`eda.json`): {eda['n_total']:,} transactions, {eda['n_fraud']:,} fraud, a fraud "
        f"rate of {rate:.4%} (1 : {eda['imbalance_ratio']:.1f}). Median amount is "
        f"{eda['median_amount_fraud']:.2f} for fraud against {eda['median_amount_normal']:.2f} for "
        f"legitimate transactions. At this prevalence the no-skill baseline for PR-AUC is {rate:.4f}, "
        f"not 0.5, and accuracy is not reported anywhere."
    )


def _export_paragraphs(schema: tracks.TrackSchema, artifacts: dict[str, Any]) -> list[str]:
    exports, calibration = artifacts["exports"], artifacts["calibration"]
    training = exports["training"]
    generated = training["calibration"]["generated"]
    registry = float(generated["registry_fraud_prevalence"])
    lines = [
        f"**Training export** (`exports.json`, `calibration.json`, profile `{calibration['profile']}`): "
        f"{training['rows']:,} card authorisations over {training['cards']:,} cards, "
        f"{training['merchants']:,} merchants and {training['span_days']:.1f} days, generated at a "
        f"requested fraud prevalence of {_pct(generated['requested_fraud_prevalence'])}, which is "
        f"{generated['boost_factor']:.1f} times the registry rate of 1 in {round(1 / registry):,} "
        f"({registry:.4e}); at the registry rate the export would hold "
        f"{generated['expected_fraud_rows_at_target']:.1f} fraud rows, so the boost is stated beside "
        f"every number below."
    ]
    revision = training.get("generator_revision")
    lines[0] += (
        f" The export comes from generator revision `{revision[:7]}` (`generator/SOURCE`)."
        if revision else " The export's generator revision unrecorded: no copy of the generator was present."
    )
    for target in training["calibration"]["gated"]:
        verdict = "passes" if target["status"] == "PASS" else "fails"
        side = "inside" if target["status"] == "PASS" else "outside"
        lines.append(
            f"The report's gated target `{target['id']}` {verdict}: the {target['statistic']} is "
            f"Rs {target['observed']:,.0f} against the published Rs {target['target']:,.0f}, {side} the "
            f"{target['tolerance_relative']:.0%} tolerance. Every other target is reported rather than gated."
        )
    return lines


def _demo_paragraph(exports: dict[str, Any]) -> str | None:
    demo = exports.get("demo")
    if not demo:
        return None
    calibration = demo.get("calibration") or {}
    generated = calibration.get("generated") or {}
    text = (
        f"**Demo export** (`exports.json`): {demo['rows']:,} rows over {demo['cards']:,} cards, "
        f"{demo['merchants']:,} merchants and {demo['span_days']:.1f} days, {demo['fraud_rows']:,} fraud rows"
    )
    if generated:
        text += (
            f" at a requested {_pct(generated['requested_fraud_prevalence'])} "
            f"({generated['boost_factor']:,.0f} times the registry rate)"
        )
    text += ", generated with a different seed from the training export so every row the simulator replays is unseen."
    for target in calibration.get("gated", []):
        verdict = "passes" if target["status"] == "PASS" else "fails"
        text += (
            f" Its calibration report {verdict} the gated target `{target['id']}` "
            f"(Rs {target['observed']:,.0f} observed against Rs {target['target']:,.0f}), so it feeds the "
            f"boards and nothing in this section."
        )
    return text


def _split_paragraph(schema: tracks.TrackSchema, split: dict[str, Any]) -> str:
    rows = (
        f"{split['train']['rows']:,} training, {split['val']['rows']:,} validation and "
        f"{split['test']['rows']:,} test rows ({split['train']['fraud']:,}, {split['val']['fraud']:,} and "
        f"{split['test']['fraud']:,} fraud)"
    )
    dropped = int(split["duplicates_dropped"])
    # The run's own settings, recorded by training; config.yaml may have moved since.
    settings = split["settings"]
    if split["rule"] == "time-ordered":
        train = float(settings["train_fraction"])
        val = float(settings["val_fraction"])
        shares = f"{round(train * 100)}/{round(val * 100)}/{round((1 - train - val) * 100)}"
        duplicates = "no duplicate rows" if dropped == 0 else f"{dropped:,} duplicate rows dropped before splitting"
        prevalence = split["test"]["fraud"] / split["test"]["rows"]
        return (
            f"**Split** (`split.json`), time-ordered {shares} on the transaction clock: {rows}; {duplicates}. "
            f"Thresholds are tuned on the validation window to a false-positive budget of "
            f"{settings['fpr_budget_per_10k']} alerts per 10,000 legitimate transactions "
            f"(`tracks.{schema.name}.fpr_budget_per_10k`), not to F1, which weighs a missed fraud and a false "
            f"alert equally and moves with the prevalence, so at the registry rate its optimum sits wherever the "
            f"validation window happens to show no false alert; the test window is scored once. The test "
            f"window's own prevalence, {prevalence:.4f}, is "
            f"the no-skill baseline for the PR-AUC below, and accuracy is not reported anywhere."
        )
    return (
        f"**Split** (`split.json`): {dropped:,} exact duplicate rows are dropped before splitting, leaving "
        f"{rows}. Thresholds are tuned on validation; the test split is scored once."
    )


def _matrix_section(schema: tracks.TrackSchema, artifacts: dict[str, Any], window: str) -> list[str]:
    arms = {(row["model"], bool(row["resample_applied"])): row for row in artifacts["experiment_matrix"]}
    weighted, resampled = arms.get(("xgboost", False)), arms.get(("xgboost", True))
    method = "SMOTENC" if schema.categorical_columns else "SMOTE"
    lines = [f"**Experiment matrix** (`experiment_matrix.md`), {window}:", "", artifacts["experiment_matrix_md"], ""]
    if weighted and resampled:
        interval = weighted.get("difference_ci")
        if interval is not None and float(interval["low"]) <= 0.0 <= float(interval["high"]):
            lines.append(
                f"The two arms are not distinguishable on this split: a paired bootstrap of the PR-AUC "
                f"difference (weighting minus {method}) runs from {float(interval['low']):+.4f} to "
                f"{float(interval['high']):+.4f}, and {method} scores at least as well in "
                f"{float(interval['share_b_at_least_a']):.0%} of resamples."
            )
        else:
            if weighted["pr_auc"] >= resampled["pr_auc"]:
                sentence = (
                    f"Cost-sensitive weighting edges out {method} on the same model ({weighted['pr_auc']:.4f} "
                    f"against {resampled['pr_auc']:.4f})"
                )
            else:
                sentence = (
                    f"{method} edges out cost-sensitive weighting on the same model ({resampled['pr_auc']:.4f} "
                    f"against {weighted['pr_auc']:.4f})"
                )
            if interval is not None:
                sentence += (
                    f"; a paired bootstrap of the difference (weighting minus {method}) runs from "
                    f"{float(interval['low']):+.4f} to {float(interval['high']):+.4f}"
                )
            lines.append(sentence + ".")
    deployed = next((row for row in artifacts["benchmark"] if row["model"] == "xgboost"), None)
    if deployed is not None:
        arm_name = "resampled" if deployed.get("resample_applied", False) else "weighted"
        lines.append(f"The deployed XGBoost is the {arm_name} arm.")
    return lines


def _benchmark_section(schema: tracks.TrackSchema, artifacts: dict[str, Any], window: str) -> list[str]:
    rows = artifacts["benchmark"]
    lines = [f"**Benchmark** (`benchmark.md`), the deployed models on the {window}:", "", artifacts["benchmark_md"], ""]
    adjusted = [row for row in rows if "adjusted_precision_at_prevalence" in row]
    if adjusted:
        best = max(adjusted, key=lambda row: row["pr_auc"])
        target = float(best["prevalence_target"])
        span = ""
        if best.get("prevalence_range") and "adjusted_precision_low" in best:
            low_rate, high_rate = (float(p) for p in best["prevalence_range"])
            span = (
                f" Over the registry's monthly span, one in {round(1 / low_rate):,} to one in "
                f"{round(1 / high_rate):,}, that precision runs from {best['adjusted_precision_low']:.1%} to "
                f"{best['adjusted_precision_high']:.1%}."
            )
        lines.append(
            f"\"Precision at 1 in {round(1 / target):,}\" is the prevalence-adjusted figure from the design "
            f"(TPR times r over TPR times r plus FPR times one minus r, at r = {target:.4e}), \"FPR per 10k\" "
            f"is false positives per 10,000 legitimate transactions, and \"Value above floor caught\" is the "
            f"share of fraud value above Rs {best['floor']:,.0f} (`tracks.{schema.name}.value_floor`) that "
            f"the model flags. The rate is the registry's for all domestic payments, most of them UPI (the "
            f"research note gives the shares); a card-only rate is not published, so it is a reference, not "
            f"the card rate. At it the {best['model']} model's precision is "
            f"{best['adjusted_precision_at_prevalence']:.1%} with {best['fpr_per_10k']:.1f} alerts per 10,000 "
            f"legitimate transactions; that is what a false-positive budget buys at one fraud in "
            f"{round(1 / target):,}, and the report must say so.{span}"
        )
    tail = "its recall is zero and its alerts per catch are not defined, which the table says rather than hides."
    for row in rows:
        caught, raised = int(row["true_positives"]), int(row["false_positives"])
        if caught == 0 and raised == 0:
            # threshold 1.0 is the tuner's sentinel for a budget no cutoff met
            if schema.split == "time_ordered" and float(row["threshold"]) >= 1.0:
                lines.append(
                    f"The {row['model']} flags nothing at its threshold: the tuner could not meet the budget on "
                    f"the validation window and returned the cutoff that flags nothing, so {tail}"
                )
            else:
                lines.append(f"The {row['model']} flags nothing at its threshold: {tail}")
        elif caught == 0:
            lines.append(
                f"The {row['model']} catches nothing at its threshold while raising {raised} false positives: {tail}"
            )
    return lines


def _latency_paragraph(name: str, filename: str, report: dict[str, Any], lead: str) -> str:
    verdict = (
        "within budget"
        if report["within_budget"]
        else f"over budget: the p99 exceeds the {report['budget_ms']:.0f} ms target"
    )
    return (
        f"**{name}** (`{filename}`), {lead} {report['n']} rows: mean {report['mean_ms']:.2f} ms, "
        f"p50 {report['p50_ms']:.2f} ms, p95 {report['p95_ms']:.2f} ms, **p99 {report['p99_ms']:.2f} ms** "
        f"against the {report['budget_ms']:.0f} ms budget; the single slowest request took "
        f"{report['max_ms']:.2f} ms. The budget is judged on p99, so this path is {verdict}."
    )


def _http_paragraph(reports: dict[str, Any]) -> str:
    """The service's own figure: the HTTP round trip a caller sees, against a
    server in its own process, flagged and unflagged separately."""
    parts = []
    failing = []
    for label, key in (("not flagged", "unflagged"), ("flagged", "flagged")):
        report = reports.get(key)
        if report is None:
            continue
        parts.append(
            f"{label}, {report['n']} requests: p50 {report['p50_ms']:.2f} ms, p95 {report['p95_ms']:.2f} ms, "
            f"**p99 {report['p99_ms']:.2f} ms**"
        )
        if not report["within_budget"]:
            failing.append(label)
    budget = next(iter(reports.values()))["budget_ms"]
    if failing:
        verdict = (
            f"the service exceeds the budget ({' and '.join(failing)}): the work done after the answer "
            "shares the single worker's process with the next request, which a worker process of its own removes"
        )
    else:
        verdict = "the service is within the budget"
    return (
        f"**HTTP latency** (`latency_http.json`), the round trip a caller sees against the service running in "
        f"its own process, every test fraud plus a random sample in time order, {'; '.join(parts)}; against the "
        f"{budget:.0f} ms budget, judged on p99, {verdict}."
    )


def _interval(x: dict[str, Any], digits: int = 4) -> str:
    if not x or not x.get("n"):
        return "n/a"
    return f"{x['mean']:.{digits}f} [{x['low']:.{digits}f}, {x['high']:.{digits}f}]"


def _harness_paragraphs(schema: tracks.TrackSchema, artifacts: dict[str, Any]) -> list[str]:
    """The evaluation harness, every figure from its own files: the protocol, the summary
    table with intervals, the best model under the labels the bank would have had and its
    gap to the full-label upper bound, the pairwise differences in that regime, the
    novel-attack folds, the supervision timelines, the cost hypothesis."""
    if not any(f.startswith("harness/") for f in ARTIFACTS[schema.name]):
        return []
    if "harness_results" not in artifacts:
        return [
            "**Harness**: the evaluation harness has not run for this track; `python -m prism.harness.run` "
            f"writes `reports/metrics/{schema.name}/harness/` and this section is rendered from it."
        ]
    results = artifacts["harness_results"]
    summary = artifacts["harness_summary"]
    s = results["settings"]
    rows = results["rows"]
    seeds = [str(x) for x in s["seeds"]]
    per_seed = [len({r["fold"] for r in rows if r["seed"] == seed}) for seed in {r["seed"] for r in rows}]
    folds_per_seed = max(per_seed) if per_seed else 0
    supervision = s["supervision"]
    operational_rows = [r for r in rows if r["label_mode"] == "operational"]
    budget_alone = sum(1 for r in operational_rows if r.get("threshold_from_budget_alone"))
    lines = [
        f"**Harness** (`harness/results.json`, `harness/summary.json`), the comparison the design asks for, on "
        f"{len(seeds)} seeds of the generator's stream ({', '.join(seeds)}), each {int(s['rows']):,} rows over "
        f"{s['days']} days at a requested fraud rate of {s['fraud_rate']}: rolling-origin backtests with a training "
        f"window of {s['train_days']} days growing by {s['step_days']} days per fold, a {s['val_days']}-day validation "
        f"window and a {s['test_days']}-day test window ({folds_per_seed} fold{'s' if folds_per_seed != 1 else ''} "
        "per seed); the encoder and the scaler are fitted inside each training window, the threshold is tuned at "
        "the false-positive budget on the validation window, and each test window is scored once. Labels come in "
        "two regimes: operational, a fraud counts as fraud only once the bank's supervision model (an investigator "
        f"queue of {supervision['k_daily']} cases a day at an alert threshold of {supervision['alert_threshold']}, "
        "chargebacks with their delays, fraud that is never reported) has discovered it by the window's end; full, "
        "every label known, the upper bound. Evaluation always uses the truth. Each value is the mean over the "
        "seed-by-fold runs with a 95 percent bootstrap interval of that mean. The scoring p99 is the model's own "
        "single-row call on preprocessed rows, not the service path of `latency.json`."
    ]
    if operational_rows:
        lines[0] += (
            f" In {budget_alone} of {len(operational_rows)} operational runs no fraud was known in the validation "
            "window and the cutoff came from the budget alone."
        )
    lines += ["", artifacts["harness_summary_md"], ""]
    modes = [m for m in ("operational", "full") if any(e["label_mode"] == m and e["pr_auc"].get("n") for e in summary["models"])]
    if modes:
        lead = modes[0]
        entries = [e for e in summary["models"] if e["label_mode"] == lead and e["pr_auc"].get("n")]
        best = max(entries, key=lambda e: e["pr_auc"]["mean"])
        name = best["model"]
        if lead == "operational":
            full = next((e for e in summary["models"] if e["model"] == name and e["label_mode"] == "full"), None)
            sentence = (
                f"Under operational labels, the regime the bank would have had, the best model is the {name} at "
                f"PR-AUC {_interval(best['pr_auc'])} with recall at the budget {_interval(best['recall'], 3)}"
            )
            if full is not None:
                sentence += (
                    f"; with every label known its PR-AUC is {_interval(full['pr_auc'])} and its recall "
                    f"{_interval(full['recall'], 3)}: the price of learning from what the bank knew at the time"
                )
            lines.append(sentence + ".")
            full_entries = [e for e in summary["models"] if e["label_mode"] == "full" and e["pr_auc"].get("n")]
            if full_entries:
                best_full = max(full_entries, key=lambda e: e["pr_auc"]["mean"])
                if best_full["model"] != name:
                    lines.append(
                        f"With every label known the {best_full['model']} leads instead, at PR-AUC "
                        f"{_interval(best_full['pr_auc'])}."
                    )
        else:
            lines.append(
                f"With every label known, the only regime run, the best model is the {name} at PR-AUC "
                f"{_interval(best['pr_auc'])} with recall at the budget {_interval(best['recall'], 3)}."
            )
        for p in summary["pairwise"]:
            if p["metric"] != "pr_auc" or p["label_mode"] != lead or name not in (p["model_a"], p["model_b"]):
                continue
            other = p["model_b"] if p["model_a"] == name else p["model_a"]
            d = p["difference"]
            if p["model_a"] == name:
                mean, low, high, share = d["mean"], d["low"], d["high"], p["share_a_at_least_b"]
            else:
                mean, low, high, share = -d["mean"], -d["high"], -d["low"], p["share_b_at_least_a"]
            separated = low > 0.0 or high < 0.0
            lines.append(
                f"Under {lead} labels, against the {other} the difference in PR-AUC on the {p['n_folds']} folds "
                f"both ran is {mean:.2f} [{low:.2f}, {high:.2f}] on the {name}'s side, which is at least as good "
                f"on {share:.0%} of them"
                + ("." if separated else "; the interval includes zero, so this harness does not separate the two.")
            )
        lines.append("")
    lines += [
        "Novel attacks (`harness/scenario.json`): on each seed's last fold, one fraud scenario at a time has its "
        "labelled rows removed from the training and validation windows (the card histories the velocity features "
        "count keep them) while the test window keeps every row, so the held-out recall is the share of that "
        "scenario's fraud rows the model still catches at its budget threshold, averaged over the seeds.",
        "",
        artifacts["harness_scenario_md"],
        "",
    ]
    weakest: dict[str, dict] = {}
    for sc in summary["scenarios"]:
        if not sc["held_out_recall"].get("n"):
            continue
        current = weakest.get(sc["model"])
        if current is None or sc["held_out_recall"]["mean"] < current["held_out_recall"]["mean"]:
            weakest[sc["model"]] = sc
    for model, sc in sorted(weakest.items()):
        lines.append(
            f"Trained without {sc['held_out_scenario']}, the {model} catches {_interval(sc['held_out_recall'], 3)} "
            f"of its rows in the test window ({sc['held_out_fraud_rows']} held-out fraud rows over {sc['n_seeds']} "
            "seeds), its weakest novel attack."
        )
    if weakest:
        lines.append("")
    facts = []
    for t in results["timelines"]:
        delay = (
            f"median discovery {t['median_discovery_days']:.1f} days after the payment"
            if t.get("median_discovery_days") is not None
            else "no fraud discovered"
        )
        facts.append(
            f"seed {t['seed']}, {t['fraud_rows']} fraud rows, {t['investigated']} discovered by investigators, "
            f"{t['chargebacks']} by chargebacks, {t['never_reported']} never reported, {delay}"
        )
    cost = s["cost"]
    revision = results.get("generator_revision")
    lines += [
        "The supervision model's timelines, which the operational labels follow: " + "; ".join(facts) + ".",
        "",
        "The cost column counts a missed fraud at its amount, a review at "
        f"{float(cost['review_inr_per_alert']):.0f} INR per alert and a false decline at "
        f"{float(cost['friction_inr_per_false_positive']):.0f} INR of friction per false positive, per 10,000 "
        "transactions; the two rates are hypotheses in `config.yaml` until an issuer's figures replace them. "
        + (
            f"The streams were written by the shipped generator at revision `{revision[:12]}`."
            if revision
            else "The streams were written by the shipped generator (revision unrecorded)."
        ),
    ]
    return lines


def _stream_day(event: dict[str, Any]) -> str:
    return str(int(event["stream_day"])) if event.get("stream_day") is not None else "n/a"


def _live_paragraphs(schema: tracks.TrackSchema, artifacts: dict[str, Any]) -> list[str]:
    """The live loop, every figure from its run log: the stream and the bank, the labels the
    bank learned, each retrain with its verdict, the served champion per window against the
    truth, and drift at each retrain."""
    if not any(f.startswith("live/") for f in ARTIFACTS[schema.name]):
        return []
    if "live_run" not in artifacts:
        return [
            "**Live loop**: the live loop has not run for this track; `python -m prism.live.run` against a "
            f"service on the live stream writes `reports/metrics/{schema.name}/live/` and this section is "
            "rendered from it."
        ]
    run = artifacts["live_run"]
    s = run["settings"]
    days = run["days"]
    labels = run["labels"]
    posted = labels["posted"]
    fraud = sum(day["fraud"] for day in days)
    requests = sum(day["requests"] for day in days)
    revision = run.get("generator_revision")
    lines = [
        f"**Live loop** (`live/run.json`, `live/events.json`): a stream the models never saw, written by the "
        f"shipped generator at revision {revision or 'unrecorded'} (seed {s['seed']}, {int(s['rows']):,} rows "
        f"requested over {s['days']} days, {int(s['cards']):,} cards, {int(s['merchants']):,} merchants, fraud "
        f"rate {s['fraud_rate']}), was replayed through the service in time order: {requests:,} payments over "
        f"{len(days)} stream days, {fraud:,} of them fraud, {run.get('failed_requests', 0):,} refused. The bank "
        f"was the generator's supervision model: its investigators reviewed Prism's alerts and nothing else, "
        f"up to {s['supervision']['k_daily']} a day; chargebacks arrived at their own delays; unreported fraud "
        "never did. A challenger was trained every "
        f"{s['retrain_every_days']} days from day {s['first_retrain_day']} on the labels known by then, "
        f"backtested against the serving champion on a {s['backtest_days']}-day window ending "
        f"{s['label_lag_days']} days before the clock, and promoted only when the 95 percent paired interval of "
        f"the PR-AUC difference ({int(s['bootstrap_resamples']):,} resamples) lay above zero.",
        "",
        "Labels the bank learned and sent: "
        + ", ".join(
            f"{posted.get(source, 0):,} {name}"
            for source, name in (
                ("CHARGEBACK_DISPUTE", "chargebacks"),
                ("INVESTIGATOR_ALERT", "investigator verdicts"),
                ("UNLABELLED", "clean maturities"),
            )
        )
        + f"; {labels['never_reported']:,} fraud rows were never reported and {labels['pending_at_end']:,} labels "
        f"were still to come when the stream ended; {labels['refused']:,} labels were refused.",
        "",
    ]
    retrains = run["retrains"]
    promoted = sum(1 for e in retrains if e["status"] == "promoted")
    lines += [
        f"Retrains: {promoted} of {len(retrains)} retrains promoted.",
        "",
        "| Stream day | Verdict | Challenger PR-AUC | Champion PR-AUC | Paired interval "
        "| Challenger, truth | Champion, truth | Why |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def figure(value: Any) -> str:
        return f"{value:.4f}" if value is not None else "n/a"

    for e in retrains:
        difference = e.get("difference")
        interval = f"[{difference['low']:.4f}, {difference['high']:.4f}]" if difference else "n/a"
        truth = e.get("truth") or {}
        why = "the interval lies above zero" if e["status"] == "promoted" else (e.get("reason") or "n/a")
        lines.append(
            f"| {_stream_day(e)} | {e['status']} | {figure(e.get('challenger_pr_auc'))} | "
            f"{figure(e.get('champion_pr_auc'))} | {interval} | {figure(truth.get('challenger_pr_auc'))} | "
            f"{figure(truth.get('champion_pr_auc'))} | {why} |"
        )
    lines += [
        "",
        "The promotion reads the operational labels: on the backtest window the incumbent's alerts were "
        "investigated within hours, while the fraud it missed waits for a chargeback or is never reported, so "
        "those labels favour the incumbent over a challenger that catches different fraud. The truth columns "
        "score the same rows against the generator's ground truth, which the runner holds and the retrain "
        "job never sees.",
    ]
    lines += [
        "",
        "The served champion against the truth, per window (the runner computes these; the service never sees "
        "the truth):",
        "",
        "| Days | Payments | Fraud | PR-AUC | Recall | Alerts | Serving |",
        "|---|---|---|---|---|---|---|",
    ]
    for w in run["windows"]:
        pr_auc = f"{w['pr_auc']:.4f}" if w["pr_auc"] is not None else "n/a"
        recall = f"{w['recall']:.4f}" if w["recall"] is not None else "n/a"
        lines.append(
            f"| {w['start_day']:.0f} to {w['end_day']:.0f} | {w['rows']:,} | {w['fraud']:,} | {pr_auc} | {recall} "
            f"| {w['alerts']:,} | {', '.join(w['bundles'])} |"
        )
    drift_lines = []
    for d, e in zip(run["drift"], retrains):
        if d["status"] != 200:
            drift_lines.append(f"day {_stream_day(e)}: not measured ({d.get('detail')})")
            continue
        top = d["features"][0] if d["features"] else None
        named = f", highest input {top['feature']} at {top['psi']:.4f} ({top['state']})" if top else ""
        score = d.get("score")
        scored = (
            f"score PSI {score['psi']:.4f} ({score['state']})" if score
            else f"score PSI not measured ({d.get('score_rows', 0)} rows from the serving bundle)"
        )
        drift_lines.append(f"day {_stream_day(e)}: {scored}{named}")
    if drift_lines:
        lines += ["", "Drift of the serving bundle at each retrain, over its last rows: " + "; ".join(drift_lines) + "."]
    return lines


def render_results(schema: tracks.TrackSchema, artifacts: dict[str, Any]) -> str:
    """One track's results section, every number read from its artifacts."""
    time_ordered = schema.split == "time_ordered"
    window = "test window" if time_ordered else "test split"
    lines: list[str] = [
        f"## Results: {TITLES[schema.name]}",
        "",
        f"Every number here is read from `reports/metrics/{schema.name}/` by `prism.reporting.results`; "
        "nothing is typed from memory.",
        "",
    ]
    if "eda" in artifacts:
        lines += [_dataset_paragraph(artifacts["eda"]), ""]
    if "exports" in artifacts:
        lines += [*_export_paragraphs(schema, artifacts), ""]
    lines += [_split_paragraph(schema, artifacts["split"]), ""]
    lines += [*_matrix_section(schema, artifacts, window), ""]
    lines += [*_benchmark_section(schema, artifacts, window), ""]
    lines += [
        _latency_paragraph(
            "Latency", "latency.json", artifacts["latency"],
            "unseen rows through the scoring path, champion only, in the engine over",
        ),
        "",
        _latency_paragraph(
            "Explanation latency", "latency_explain.json", artifacts["latency_explain"],
            "the explanation alone, which runs after the answer, per row over",
        ),
        "",
        _http_paragraph(artifacts["latency_http"]),
        "",
    ]
    harness = _harness_paragraphs(schema, artifacts)
    if harness:
        lines += [*harness, ""]
    live = _live_paragraphs(schema, artifacts)
    if live:
        lines += [*live, ""]
    demo = _demo_paragraph(artifacts["exports"]) if "exports" in artifacts else None
    if demo:
        lines += [demo, ""]
    return "\n".join(lines).rstrip() + "\n"


def readme_results_section(readme: str) -> dict[str, str]:
    """The README's Results section, split into its ULB part and its Indian
    subsection so each track is checked against its own numbers."""
    section = readme[readme.index("## Results"):]
    marker = "### Indian track"
    if marker in section:
        cut = section.index(marker)
        return {"ulb": section[:cut], "india": section[cut:]}
    return {"ulb": section, "india": ""}


def numbers_in(text: str) -> list[str]:
    """Every numeric token worth checking: three characters or more, or one
    carrying a decimal point. Shorter integers match trivially anywhere."""
    return [token for token in NUMBER.findall(text) if len(token) >= 3 or "." in token]


def numbers_missing(generated: str, reference: str) -> list[str]:
    """The generated numbers the reference text does not contain as whole
    numbers: 28.35 is not found inside 128.35, nor 0.1143 inside 0.11430."""
    return sorted({
        token for token in numbers_in(generated)
        # not inside a longer number (128.35, 0.11430, 1,000) and not another
        # number's percent; a comma or full stop after it is punctuation
        if re.search(rf"(?<!\d)(?<!\d[.,]){re.escape(token)}(?![.,]?\d)(?!%)", reference) is None
    })


TABLES = (
    ("experiment_matrix_md", "experiment_matrix.md"),
    ("benchmark_md", "benchmark.md"),
    ("harness_summary_md", "harness/summary.md"),
    ("harness_scenario_md", "harness/scenario.md"),
)


def tables_missing(artifacts: dict[str, Any], reference: str) -> list[str]:
    """The pipeline tables the reference text does not carry verbatim."""
    return [name for key, name in TABLES if key in artifacts and artifacts[key] not in reference]


def results_file_sections(text: str) -> dict[str, str]:
    """The committed results file split by track, on its own headings."""
    sections: dict[str, str] = {}
    for name in TITLES:
        heading = f"## Results: {TITLES[name]}"
        if heading not in text:
            continue
        start = text.index(heading)
        rest = text[start + len(heading):]
        end = rest.find("\n## Results: ")
        sections[name] = heading + (rest if end < 0 else rest[:end])
    return sections


REPORT_START = "<!-- results:start -->"
REPORT_END = "<!-- results:end -->"


def paste_into_report(report_path: Path, generated: str) -> None:
    """Replace the block between the report's results markers with the
    generated sections, verbatim from the first track heading on; the
    generated file's own title and header stay out. Idempotent."""
    report = Path(report_path).read_text(encoding="utf-8")
    if REPORT_START not in report or REPORT_END not in report:
        raise ValueError(f"{report_path} has no {REPORT_START} / {REPORT_END} block to paste into")
    if "## Results:" not in generated:
        raise ValueError("the generated text has no '## Results:' track heading to paste")
    body = generated[generated.index("## Results:"):].strip("\n")
    before = report[: report.index(REPORT_START) + len(REPORT_START)]
    after = report[report.index(REPORT_END):]
    Path(report_path).write_text(f"{before}\n{body}\n{after}", encoding="utf-8")


def paste_into_readme(readme_path: Path, generated: str) -> None:
    """Replace the block between the README's results markers with the
    generated sections, their track headings one level down so they sit
    under the README's own "## Results". Idempotent."""
    readme = Path(readme_path).read_text(encoding="utf-8")
    if REPORT_START not in readme or REPORT_END not in readme:
        raise ValueError(f"{readme_path} has no {REPORT_START} / {REPORT_END} block to paste into")
    if "## Results:" not in generated:
        raise ValueError("the generated text has no '## Results:' track heading to paste")
    body = generated[generated.index("## Results:"):].strip("\n").replace("## Results: ", "### ")
    before = readme[: readme.index(REPORT_START) + len(REPORT_START)]
    after = readme[readme.index(REPORT_END):]
    Path(readme_path).write_text(f"{before}\n{body}\n{after}", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the results section from the measured artifacts.")
    parser.add_argument("--track", choices=["ulb", "india"], default=None, help="one track only (default: both)")
    parser.add_argument(
        "--report", default=None,
        help="also paste the generated sections between the results markers of this report file",
    )
    parser.add_argument(
        "--readme", default=None,
        help="also paste the generated sections between the results markers of this README",
    )
    args = parser.parse_args()
    if (args.report or args.readme) and args.track:
        # one track's section pasted over the block would delete the other track's chapter half
        parser.error("--report and --readme paste both tracks; drop --track")
    names = [args.track] if args.track else ["ulb", "india"]
    sections = [render_results(tracks.get(name), load_artifacts(tracks.get(name))) for name in names]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Results\n\n"
        "Generated by `python -m prism.reporting.results` from `reports/metrics/<track>/`; regenerate "
        "after any pipeline command with --readme README.md.\n\n"
    )
    text = header + "\n".join(sections)
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"Written -> {OUTPUT}")
    if args.report:
        paste_into_report(Path(args.report), text)
        print(f"Pasted -> {args.report}")
    if args.readme:
        paste_into_readme(Path(args.readme), text)
        print(f"Pasted -> {args.readme}")


if __name__ == "__main__":
    main()
