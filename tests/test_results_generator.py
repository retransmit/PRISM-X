"""The results section must be generated, never transcribed, and the README
must agree with it."""

from __future__ import annotations

import json

import pytest

from prism import config, tracks
from prism.reporting.results import (
    ArtifactError,
    MissingArtifact,
    MixedRuns,
    load_artifacts,
    numbers_missing,
    readme_results_section,
    render_results,
    results_file_sections,
    tables_missing,
)


def _write(directory, name, content) -> None:
    text = content if isinstance(content, str) else json.dumps(content)
    (directory / name).write_text(text, encoding="utf-8")


def _latency(p99: float, mean: float, p50: float, p95: float, maximum: float) -> dict:
    return {"n": 500, "mean_ms": mean, "p50_ms": p50, "p95_ms": p95, "p99_ms": p99,
            "max_ms": maximum, "budget_ms": 100.0, "within_budget": p99 <= 100.0}


def _row(model, threshold, pr_auc, roc_auc, precision, recall, f1, tn, fp, fn, tp, alerts, **extras) -> dict:
    return {"model": model, "threshold": threshold, "pr_auc": pr_auc, "roc_auc": roc_auc,
            "precision": precision, "recall": recall, "f1": f1, "true_negatives": tn,
            "false_positives": fp, "false_negatives": fn, "true_positives": tp,
            "alerts_per_true_fraud": alerts, **extras}


ULB_MATRIX_MD = "\n".join([
    "| Arm | SMOTE | Threshold | PR-AUC | Precision | Recall | F1 | Missed fraud | Alerts per catch |",
    "|---|---|---|---|---|---|---|---|---|",
    "| xgboost+weighted | no | 0.7166 | **0.8169** | 0.9730 | 0.7579 | 0.8521 | 23 | 1.0 |",
    "| xgboost+smote | yes | 0.8287 | **0.8155** | 0.9722 | 0.7368 | 0.8383 | 25 | 1.0 |",
    "| isolation_forest+weighted | no | 0.7244 | **0.1143** | 0.2283 | 0.2211 | 0.2246 | 74 | 4.4 |",
])
ULB_BENCHMARK_MD = "\n".join([
    "| Model | Threshold | PR-AUC | ROC-AUC | Precision | Recall | F1 | Missed fraud | Alerts per catch |",
    "|---|---|---|---|---|---|---|---|---|",
    "| xgboost | 0.7166 | **0.8169** | 0.9681 | 0.9730 | 0.7579 | 0.8521 | 23 | 1.0 |",
    "| isolation_forest | 0.7244 | **0.1143** | 0.9308 | 0.2283 | 0.2211 | 0.2246 | 74 | 4.4 |",
])
INDIA_MATRIX_MD = "\n".join([
    "| Arm | SMOTE | Threshold | PR-AUC | Precision | Recall | F1 | Missed fraud | Alerts per catch |",
    "|---|---|---|---|---|---|---|---|---|",
    "| xgboost+weighted | no | 0.9797 | **0.9234** | 0.9286 | 0.6989 | 0.7975 | 28 | 1.1 |",
    "| xgboost+smote | yes | 0.9900 | **0.8689** | 0.8611 | 0.6667 | 0.7515 | 31 | 1.2 |",
    "| isolation_forest+weighted | no | 1.0000 | **0.0169** | 0.0000 | 0.0000 | 0.0000 | 93 | n/a |",
])
INDIA_BENCHMARK_MD = "\n".join([
    "| Model | Threshold | PR-AUC | ROC-AUC | Precision | Recall | F1 | Missed fraud | Alerts per catch | Precision at 1 in 80,847 | FPR per 10k | Value above floor caught |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|",
    "| xgboost | 0.9797 | **0.9234** | 0.9995 | 0.9286 | 0.6989 | 0.7975 | 28 | 1.1 | 0.0300 | 2.8 | 1.000 |",
    "| isolation_forest | 1.0000 | **0.0169** | 0.8037 | 0.0000 | 0.0000 | 0.0000 | 93 | n/a | 0.0000 | 0.0 | 0.000 |",
])


@pytest.fixture
def ulb_dir(tmp_path):
    d = tmp_path / "ulb"
    d.mkdir()
    _write(d, "eda.json", {
        "n_total": 284807, "n_normal": 284315, "n_fraud": 492, "fraud_rate": 0.001727486,
        "imbalance_ratio": 577.876, "median_amount_fraud": 9.25, "median_amount_normal": 22.0,
        "mean_amount_fraud": 122.21, "mean_amount_normal": 88.29, "max_amount_fraud": 2125.87, "figures": [],
    })
    _write(d, "split.json", {
        "track": "ulb", "rule": "stratified", "train": {"rows": 181584, "fraud": 302},
        "val": {"rows": 45396, "fraud": 76}, "test": {"rows": 56746, "fraud": 95},
        "duplicates_dropped": 1081, "n_features": 30, "categorical_columns": [],
        "loaded": {"n_total": 284807, "n_fraud": 492, "fraud_rate": 0.001727486, "imbalance_ratio": 577.876},
        "settings": {"split": "stratified", "test_size": 0.2, "val_size": 0.2, "threshold_strategy": "f1"},
    })
    _write(d, "experiment_matrix.json", [
        {"label": "xgboost+weighted", "model": "xgboost", "resample_applied": False, "threshold": 0.7166, "pr_auc": 0.8169, "difference_ci": {"against": "resampled", "low": 0.0005, "high": 0.0025, "share_b_at_least_a": 0.02}},
        {"label": "xgboost+smote", "model": "xgboost", "resample_applied": True, "threshold": 0.8287, "pr_auc": 0.8155},
        {"label": "isolation_forest+weighted", "model": "isolation_forest", "resample_applied": False, "threshold": 0.7244, "pr_auc": 0.1143},
    ])
    _write(d, "experiment_matrix.md", ULB_MATRIX_MD)
    _write(d, "benchmark.json", [
        _row("xgboost", 0.7166, 0.8169, 0.9681, 0.9730, 0.7579, 0.8521, 56649, 2, 23, 72, 1.0),
        _row("isolation_forest", 0.7244, 0.1143, 0.9308, 0.2283, 0.2211, 0.2246, 56580, 71, 74, 21, 4.4),
    ])
    _write(d, "benchmark.md", ULB_BENCHMARK_MD)
    _write(d, "latency.json", _latency(28.35, 25.31, 24.96, 26.95, 76.67))
    _write(d, "latency_explain.json", _latency(33.97, 30.53, 31.45, 33.04, 70.34))
    _write(d, "latency_http.json", {
        "unflagged": _latency(77.23, 42.85, 40.10, 65.24, 95.08),
        "flagged": _latency(78.66, 46.08, 44.00, 67.69, 92.18),
    })
    return d


def _calibration(overall: str, requested: float, boost: float, expected: float, observed: float, status: str) -> dict:
    return {
        "profile": "rbi-psi-2026-07", "overall": overall,
        "generated": {"rows": 90000, "fraud_rows": 479, "requested_fraud_prevalence": requested,
                      "registry_fraud_prevalence": 1.2369e-05, "boost_factor": boost,
                      "expected_fraud_rows_at_target": expected},
        "gated": [{"id": "mean_credit_card_ticket_inr",
                   "statistic": "mean amount of legitimate rows on credit card products",
                   "target": 3440.0, "observed": observed, "tolerance_relative": 0.25, "status": status}],
    }


@pytest.fixture
def india_dir(tmp_path):
    d = tmp_path / "india"
    d.mkdir()
    _write(d, "split.json", {
        "track": "india", "rule": "time-ordered", "train": {"rows": 62999, "fraud": 333},
        "val": {"rows": 9000, "fraud": 53}, "test": {"rows": 18001, "fraud": 93},
        "duplicates_dropped": 0, "n_features": 25, "categorical_columns": list(tracks.INDIA.categorical_columns),
        "loaded": {"n_total": 90000, "n_fraud": 479, "fraud_rate": 0.005322, "imbalance_ratio": 186.9},
        # a budget the committed config does not use, so a render that read
        # config.yaml instead of the artifact would print the wrong number
        "settings": {"split": "time_ordered", "train_fraction": 0.7, "val_fraction": 0.1,
                     "threshold_strategy": "fpr_budget", "fpr_budget_per_10k": 7,
                     "prevalence_target": 1.2369e-05, "value_floor": 10000},
    })
    _write(d, "exports.json", {
        "training": {"stem": "data/raw/india/train_in", "rows": 90000, "fraud_rows": 479, "cards": 2000,
                     "merchants": 300, "span_days": 64.8,
                     "calibration": _calibration("PASS", 0.005, 404.236, 1.11321, 3960.48, "PASS")},
        "demo": {"stem": "data/raw/india/demo_in", "rows": 20000, "fraud_rows": 744, "cards": 600,
                 "merchants": 150, "span_days": 17.0,
                 "calibration": _calibration("FAIL", 0.03, 2425.418, 0.24738, 4368.54, "FAIL")},
    })
    _write(d, "calibration.json", {"profile": "rbi-psi-2026-07", "region": "IN", "as_of": "2026-07",
                                   "overall": "PASS", "targets": [],
                                   "generated": _calibration("PASS", 0.005, 404.236, 1.11321, 3960.48, "PASS")["generated"]})
    _write(d, "experiment_matrix.json", [
        {"label": "xgboost+weighted", "model": "xgboost", "resample_applied": False, "threshold": 0.9797, "pr_auc": 0.9234, "difference_ci": {"against": "resampled", "low": 0.0005, "high": 0.0025, "share_b_at_least_a": 0.02}},
        {"label": "xgboost+smote", "model": "xgboost", "resample_applied": True, "threshold": 0.9153, "pr_auc": 0.8689},
        {"label": "isolation_forest+weighted", "model": "isolation_forest", "resample_applied": False, "threshold": 1.0, "pr_auc": 0.0169},
    ])
    _write(d, "experiment_matrix.md", INDIA_MATRIX_MD)
    _write(d, "benchmark.json", [
        _row("xgboost", 0.9797, 0.9234, 0.9995, 0.9286, 0.6989, 0.7975, 17903, 5, 28, 65, 1.1,
             prevalence_target=1.2369e-05, fpr_per_10k=2.8, adjusted_precision_at_prevalence=0.0300,
             adjusted_alerts_per_true_fraud=33.3, floor=10000.0, share_above_floor_caught=1.0),
        _row("isolation_forest", 1.0, 0.0169, 0.8037, 0.0, 0.0, 0.0, 17908, 0, 93, 0, None,
             prevalence_target=1.2369e-05, fpr_per_10k=0.0, adjusted_precision_at_prevalence=0.0,
             adjusted_alerts_per_true_fraud=None, floor=10000.0, share_above_floor_caught=0.0),
    ])
    _write(d, "benchmark.md", INDIA_BENCHMARK_MD)
    _write(d, "latency.json", _latency(38.39, 34.50, 34.05, 36.58, 95.01))
    _write(d, "latency_explain.json", _latency(44.27, 39.62, 40.20, 42.33, 86.58))
    _write(d, "latency_http.json", {
        "unflagged": _latency(124.80, 68.61, 62.00, 106.54, 160.45) | {"within_budget": False},
        "flagged": _latency(124.66, 69.40, 63.00, 110.86, 125.32) | {"within_budget": False},
    })
    return d


def test_missing_artifact_names_the_command_that_creates_it(tmp_path):
    with pytest.raises(MissingArtifact, match="prism.data.eda"):
        load_artifacts(tracks.ULB, tmp_path)
    with pytest.raises(MissingArtifact, match="train --track india"):
        load_artifacts(tracks.INDIA, tmp_path)


def test_a_partial_directory_is_refused(ulb_dir):
    (ulb_dir / "latency_explain.json").unlink()
    with pytest.raises(MissingArtifact, match="latency --track ulb --explain"):
        load_artifacts(tracks.ULB, ulb_dir)


def test_ulb_results_quote_the_dataset_split_and_baseline(ulb_dir):
    text = render_results(tracks.ULB, load_artifacts(tracks.ULB, ulb_dir))
    for expected in ("284,807", "492 fraud", "0.1727%", "1 : 577.9", "9.25", "22.00", "0.0017",
                     "1,081", "181,584", "45,396", "56,746", "302, 76 and 95 fraud"):
        assert expected in text, expected
    assert "no-skill" in text


def test_tables_are_embedded_from_the_markdown_artifacts(ulb_dir):
    artifacts = load_artifacts(tracks.ULB, ulb_dir)
    text = render_results(tracks.ULB, artifacts)
    assert artifacts["benchmark_md"] in text and artifacts["experiment_matrix_md"] in text
    assert "Cost-sensitive weighting edges out SMOTE on the same model (0.8169 against 0.8155)" in text


def test_latency_is_judged_on_p99_with_a_verdict(ulb_dir):
    text = render_results(tracks.ULB, load_artifacts(tracks.ULB, ulb_dir))
    assert "**p99 28.35 ms**" in text and "**p99 33.97 ms**" in text
    assert "within budget" in text and "76.67 ms" in text


def test_an_over_budget_p99_is_reported_as_exceeding(ulb_dir):
    """An over-budget result must say so, not be quietly reworded."""
    path = ulb_dir / "latency.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update({"p99_ms": 148.3, "within_budget": False})
    path.write_text(json.dumps(data), encoding="utf-8")
    text = render_results(tracks.ULB, load_artifacts(tracks.ULB, ulb_dir))
    assert "148.30 ms" in text and "exceeds" in text


def test_india_results_lead_with_the_export_and_the_adjusted_precision(india_dir):
    text = render_results(tracks.INDIA, load_artifacts(tracks.INDIA, india_dir))
    for expected in ("90,000", "2,000 cards", "300 merchants", "64.8 days", "0.5%", "404.2 times",
                     "1 in 80,847", "1.2369e-05", "1.1 fraud rows", "Rs 3,960", "Rs 3,440", "25% tolerance",
                     "62,999", "333, 53 and 93 fraud", "no duplicate rows", "SMOTENC",
                     "0.9234 against 0.8689", "3.0%", "2.8 alerts", "Rs 10,000", "**p99 38.39 ms**", "**p99 44.27 ms**",
                     "7 alerts per 10,000", "70/10/20", "0.0052", "accuracy is not reported"):
        assert expected in text, expected


def test_a_model_that_flags_nothing_is_said_so(india_dir):
    text = render_results(tracks.INDIA, load_artifacts(tracks.INDIA, india_dir))
    assert "The isolation_forest flags nothing at its threshold" in text
    assert "could not meet the budget" in text


def test_a_model_that_catches_nothing_but_raises_alerts_is_not_said_to_flag_nothing(india_dir):
    """TP 0 with FP 7 at a threshold below the tuner's sentinel flagged seven
    rows and did meet the budget; the sentence must say what happened."""
    path = india_dir / "benchmark.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows[1].update({"threshold": 0.93, "false_positives": 7, "true_negatives": 17901})
    path.write_text(json.dumps(rows), encoding="utf-8")
    # The same run's matrix carries the same threshold for the same model.
    matrix_path = india_dir / "experiment_matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    for row in matrix:
        if row["model"] == "isolation_forest":
            row["threshold"] = 0.93
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    text = render_results(tracks.INDIA, load_artifacts(tracks.INDIA, india_dir))
    assert "The isolation_forest catches nothing at its threshold while raising 7 false positives" in text
    assert "flags nothing" not in text and "could not meet the budget" not in text


def test_the_demo_export_is_described_and_kept_out_of_the_table(india_dir):
    text = render_results(tracks.INDIA, load_artifacts(tracks.INDIA, india_dir))
    for expected in ("20,000 rows", "600 cards", "150 merchants", "17.0 days", "744 fraud rows", "3%",
                     "2,425 times", "Rs 4,369", "fails"):
        assert expected in text, expected


def test_never_reports_accuracy(ulb_dir, india_dir):
    """The one metric this project refuses to quote."""
    for schema, directory in ((tracks.ULB, ulb_dir), (tracks.INDIA, india_dir)):
        text = render_results(schema, load_artifacts(schema, directory)).lower()
        assert text.count("accuracy") == text.count("accuracy is not reported")


def test_output_contains_no_placeholders(ulb_dir, india_dir):
    for schema, directory in ((tracks.ULB, ulb_dir), (tracks.INDIA, india_dir)):
        text = render_results(schema, load_artifacts(schema, directory))
        for banned in ("TODO", "TBD", "XXX", "FIXME", "<insert", "None"):
            assert banned not in text, banned


def test_numbers_missing_finds_a_changed_number():
    assert numbers_missing("p99 28.35 ms over 500 rows", "p99 29.69 ms over 500 rows") == ["28.35"]
    assert numbers_missing("1 in 80,847 at 1.2369e-05, **0.8169**", "1 in 80,847 at 1.2369e-05 and 0.8169") == []
    # punctuation after a number is not part of it
    assert numbers_missing("judged on p99, so 0.5 is not 333, fraud", "p99 is 0.5 and 333 fraud") == []
    # a number inside a longer number is not a match
    assert numbers_missing("p99 28.35 ms", "p99 128.35 ms") == ["28.35"]
    assert numbers_missing("PR-AUC 0.1143", "PR-AUC 0.11430") == ["0.1143"]
    # ordinary punctuation after a number still counts as a match
    assert numbers_missing("is 0.0017 and (302 fraud) at 0.5", "is 0.0017, not (302, 76) and 0.5.") == []


def test_tables_missing_names_the_table_the_readme_lacks(ulb_dir):
    artifacts = load_artifacts(tracks.ULB, ulb_dir)
    assert tables_missing(artifacts, ULB_BENCHMARK_MD) == ["experiment_matrix.md"]
    assert tables_missing(artifacts, ULB_MATRIX_MD + "\n\n" + ULB_BENCHMARK_MD) == []


def test_a_directory_mixed_from_two_runs_is_refused(ulb_dir, india_dir):
    """A re-run overwrites in place, so every file is present; the numbers
    the files share must agree or the section would mix two runs."""

    def rewrite(directory, name, change):
        path = directory / name
        data = json.loads(path.read_text(encoding="utf-8"))
        change(data)
        path.write_text(json.dumps(data), encoding="utf-8")

    rewrite(ulb_dir, "benchmark.json", lambda rows: rows[0].update({"false_negatives": 24}))
    with pytest.raises(MixedRuns, match="benchmark.json.*split.json"):
        load_artifacts(tracks.ULB, ulb_dir)
    rewrite(ulb_dir, "benchmark.json", lambda rows: rows[0].update({"false_negatives": 23}))

    rewrite(ulb_dir, "eda.json", lambda eda: eda.update({"n_total": 284808}))
    with pytest.raises(MixedRuns, match="eda.json"):
        load_artifacts(tracks.ULB, ulb_dir)
    rewrite(ulb_dir, "eda.json", lambda eda: eda.update({"n_total": 284807}))

    rewrite(ulb_dir, "experiment_matrix.json", lambda rows: rows[0].update({"pr_auc": 0.8170}))
    with pytest.raises(MixedRuns, match="experiment_matrix.json"):
        load_artifacts(tracks.ULB, ulb_dir)
    rewrite(ulb_dir, "experiment_matrix.json", lambda rows: rows[0].update({"pr_auc": 0.8169}))
    load_artifacts(tracks.ULB, ulb_dir)

    rewrite(india_dir, "exports.json", lambda exports: exports["training"].update({"rows": 90001}))
    with pytest.raises(MixedRuns, match="exports.json"):
        load_artifacts(tracks.INDIA, india_dir)
    rewrite(india_dir, "exports.json", lambda exports: exports["training"].update({"rows": 90000}))

    rewrite(india_dir, "calibration.json", lambda report: report["generated"].update({"boost_factor": 400.0}))
    with pytest.raises(MixedRuns, match="calibration.json"):
        load_artifacts(tracks.INDIA, india_dir)
    rewrite(india_dir, "calibration.json", lambda report: report["generated"].update({"boost_factor": 404.236}))
    load_artifacts(tracks.INDIA, india_dir)


def test_results_file_sections_split_the_committed_file_by_track():
    text = (
        "# Results\n\nGenerated.\n\n## Results: ULB track (real data)\n\nulb 0.8169\n\n"
        "## Results: Indian track (synthetic export, RBI-calibrated)\n\nindia 0.9234\n"
    )
    sections = results_file_sections(text)
    assert "0.8169" in sections["ulb"] and "0.9234" not in sections["ulb"]
    assert "0.9234" in sections["india"]


def test_readme_results_section_splits_at_the_indian_subsection():
    readme = "# Prism\n\n## Results\n\nulb text 0.8169\n\n### Indian track (synthetic export)\n\nindia text 0.9234\n"
    parts = readme_results_section(readme)
    assert "0.8169" in parts["ulb"] and "0.9234" not in parts["ulb"]
    assert "0.9234" in parts["india"]


def test_readme_results_quote_the_generated_numbers():
    """Every number and every table the generator prints from the committed
    metric files appears in the README's Results section. A re-run that
    changes a file makes this fail until the README is regenerated. Without
    the metrics the test skips and says so; it never passes on prose alone."""
    readme = (config.PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    parts = readme_results_section(readme)
    for schema in (tracks.ULB, tracks.INDIA):
        try:
            artifacts = load_artifacts(schema)
        except MissingArtifact as exc:
            pytest.skip(f"{schema.name}: the committed metrics are missing ({exc}); run the pipeline and commit reports/metrics")
        generated = render_results(schema, artifacts)
        missing = numbers_missing(generated, parts[schema.name])
        assert not missing, f"README Results ({schema.name}) does not carry: {missing}"
        absent = tables_missing(artifacts, parts[schema.name])
        assert not absent, f"README Results ({schema.name}) does not carry the tables: {absent}"


def test_paste_into_report_replaces_the_marked_block(tmp_path):
    """The report's results chapter is the generated file pasted verbatim
    between two markers, never retyped; the file's own title and header stay out."""
    from prism.reporting.results import paste_into_report

    report = tmp_path / "report.md"
    report.write_text(
        "# Title\n\n# 6 Results\n\n<!-- results:start -->\nold numbers\n<!-- results:end -->\n\n# 7 Next\n",
        encoding="utf-8",
    )
    generated = "# Results\n\nGenerated by x.\n\n## Results: ULB track (real data)\n\nulb 0.8169\n"

    paste_into_report(report, generated)

    pasted = report.read_text(encoding="utf-8")
    assert "old numbers" not in pasted
    assert "## Results: ULB track (real data)\n\nulb 0.8169" in pasted
    assert "Generated by x." not in pasted and "# Results\n" not in pasted
    assert pasted.startswith("# Title\n") and pasted.endswith("# 7 Next\n")
    assert pasted.count("<!-- results:start -->") == 1 and pasted.count("<!-- results:end -->") == 1
    # pasting twice is idempotent
    paste_into_report(report, generated)
    assert report.read_text(encoding="utf-8") == pasted


def test_paste_into_report_refuses_a_file_without_markers(tmp_path):
    from prism.reporting.results import paste_into_report

    report = tmp_path / "report.md"
    report.write_text("# Title\n", encoding="utf-8")
    with pytest.raises(ValueError, match="results:start"):
        paste_into_report(report, "# Results\n\n## Results: ULB track (real data)\n\nx\n")


def test_report_paste_refuses_a_single_track(tmp_path, monkeypatch):
    """--track renders one track; pasting that over the report's block would
    silently delete the other track's half of the chapter."""
    from prism.reporting import results

    monkeypatch.setattr(results, "OUTPUT", tmp_path / "results.md")
    report = tmp_path / "report.md"
    report.write_text("<!-- results:start -->\nboth tracks\n<!-- results:end -->\n", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["results", "--track", "ulb", "--report", str(report)])
    with pytest.raises(SystemExit):
        results.main()
    assert "both tracks" in report.read_text(encoding="utf-8")


def test_paste_into_report_refuses_text_without_a_track_heading(tmp_path):
    from prism.reporting.results import paste_into_report

    report = tmp_path / "report.md"
    report.write_text("<!-- results:start -->" + chr(10) + "x" + chr(10) + "<!-- results:end -->" + chr(10), encoding="utf-8")
    with pytest.raises(ValueError, match="Results:"):
        paste_into_report(report, "nothing generated here")


def test_paste_into_readme_replaces_the_marked_block_with_demoted_headings(tmp_path):
    """The README's Results section is the generated file pasted between two
    markers, its track headings one level down so they sit under "## Results"."""
    from prism.reporting.results import paste_into_readme

    readme = tmp_path / "README.md"
    readme.write_text(
        "# Prism\n\n## Results\n\nEvery number below is generated.\n\n<!-- results:start -->\nold numbers\n<!-- results:end -->\n",
        encoding="utf-8",
    )
    generated = (
        "# Results\n\nGenerated by x.\n\n## Results: ULB track (real data)\n\nulb 0.8169\n\n"
        "## Results: Indian track (synthetic export, RBI-calibrated)\n\nindia 0.9234\n"
    )

    paste_into_readme(readme, generated)

    pasted = readme.read_text(encoding="utf-8")
    assert "old numbers" not in pasted and "Generated by x." not in pasted
    assert "### ULB track (real data)\n\nulb 0.8169" in pasted
    assert "### Indian track (synthetic export, RBI-calibrated)\n\nindia 0.9234" in pasted
    assert "## Results: " not in pasted


def test_a_resampled_deployment_is_checked_against_the_resampled_arm(ulb_dir):
    """With --resample deployed, benchmark.json says so and the generator
    compares it with the SMOTE arm; comparing it with the weighted arm refused
    a valid run as mixed."""
    benchmark = json.loads((ulb_dir / "benchmark.json").read_text(encoding="utf-8"))
    for row in benchmark:
        if row["model"] == "xgboost":
            row.update({"resample_applied": True, "threshold": 0.8287, "pr_auc": 0.8155})
    (ulb_dir / "benchmark.json").write_text(json.dumps(benchmark), encoding="utf-8")

    text = render_results(tracks.ULB, load_artifacts(tracks.ULB, ulb_dir))

    assert "The deployed XGBoost is the resampled arm." in text
    assert "so the deployed XGBoost uses weighting" not in text


def test_a_weighted_deployment_is_named_as_such(ulb_dir):
    text = render_results(tracks.ULB, load_artifacts(tracks.ULB, ulb_dir))
    assert "The deployed XGBoost is the weighted arm." in text


def test_the_comparison_sentence_says_no_difference_when_the_interval_spans_zero(ulb_dir):
    matrix = json.loads((ulb_dir / "experiment_matrix.json").read_text(encoding="utf-8"))
    for row in matrix:
        if row["model"] == "xgboost" and not row["resample_applied"]:
            row["difference_ci"] = {"against": "resampled", "low": -0.0187, "high": 0.0178, "share_b_at_least_a": 0.39}
    (ulb_dir / "experiment_matrix.json").write_text(json.dumps(matrix), encoding="utf-8")
    text = render_results(tracks.ULB, load_artifacts(tracks.ULB, ulb_dir))
    assert "not distinguishable on this split" in text
    assert "edges out" not in text


def test_the_comparison_sentence_keeps_the_winner_when_the_interval_excludes_zero(ulb_dir):
    matrix = json.loads((ulb_dir / "experiment_matrix.json").read_text(encoding="utf-8"))
    for row in matrix:
        if row["model"] == "xgboost" and not row["resample_applied"]:
            row["difference_ci"] = {"against": "resampled", "low": 0.0244, "high": 0.0876, "share_b_at_least_a": 0.0}
    (ulb_dir / "experiment_matrix.json").write_text(json.dumps(matrix), encoding="utf-8")
    text = render_results(tracks.ULB, load_artifacts(tracks.ULB, ulb_dir))
    assert "edges out" in text and "+0.0244 to +0.0876" in text


def test_the_http_latency_is_rendered_and_judged(ulb_dir):
    """The service's figure is the HTTP round trip, flagged and unflagged;
    latency.json is the champion's scoring path and the explanation runs
    after the answer, and the generator says so."""
    _write(ulb_dir, "latency_http.json", {
        "unflagged": _latency(77.23, 42.85, 40.1, 65.24, 95.08),
        "flagged": _latency(118.66, 46.08, 44.0, 67.69, 132.18) | {"within_budget": False},
    })
    text = render_results(tracks.ULB, load_artifacts(tracks.ULB, ulb_dir))
    assert "**HTTP latency** (`latency_http.json`)" in text
    assert "p99 77.23 ms" in text and "p99 118.66 ms" in text
    assert "exceeds the budget" in text and "the service" in text
    assert "scoring path, champion only" in text and "after the answer" in text


def test_a_matrix_without_the_interval_is_refused_as_stale(ulb_dir):
    """experiment_matrix.json from before the paired bootstrap carries no
    difference_ci; the generator refuses it rather than name a winner."""
    matrix = json.loads((ulb_dir / "experiment_matrix.json").read_text(encoding="utf-8"))
    for row in matrix:
        row.pop("difference_ci", None)
    (ulb_dir / "experiment_matrix.json").write_text(json.dumps(matrix), encoding="utf-8")
    with pytest.raises(ArtifactError, match="difference_ci"):
        load_artifacts(tracks.ULB, ulb_dir)


def test_every_model_in_both_files_must_agree(ulb_dir):
    """The run check compared only XGBoost, so a stale Isolation Forest row
    slipped through; every model that appears in both files must agree."""
    matrix = json.loads((ulb_dir / "experiment_matrix.json").read_text(encoding="utf-8"))
    for row in matrix:
        if row["model"] == "xgboost" and not row["resample_applied"]:
            row["difference_ci"] = {"against": "resampled", "low": -0.01, "high": 0.01, "share_b_at_least_a": 0.4}
        if row["model"] == "isolation_forest":
            row["pr_auc"] = 0.2000
    (ulb_dir / "experiment_matrix.json").write_text(json.dumps(matrix), encoding="utf-8")
    with pytest.raises(MixedRuns, match="isolation_forest"):
        load_artifacts(tracks.ULB, ulb_dir)


def test_the_training_paragraph_names_the_generator_revision(india_dir):
    """The report says which generator made the export, or that the revision is unrecorded."""
    from prism.reporting.results import _export_paragraphs

    def _set(revision):
        path = india_dir / "exports.json"
        exports = json.loads(path.read_text(encoding="utf-8"))
        exports["training"]["generator_revision"] = revision
        path.write_text(json.dumps(exports), encoding="utf-8")

    _set("0123456789abcdef0123456789abcdef01234567")
    text = "\n".join(_export_paragraphs(tracks.INDIA, load_artifacts(tracks.INDIA, india_dir)))
    assert "generator revision `0123456`" in text
    _set(None)
    text = "\n".join(_export_paragraphs(tracks.INDIA, load_artifacts(tracks.INDIA, india_dir)))
    assert "generator revision unrecorded" in text


def _harness_dir(india_dir):
    harness = india_dir / "harness"
    harness.mkdir()
    _write(harness, "results.json", {
        "settings": {"seeds": [42, 7], "rows": 90000, "days": 60, "fraud_rate": 0.005, "train_days": 24, "val_days": 6,
                     "test_days": 10, "step_days": 10, "label_modes": ["operational", "full"],
                     "supervision": {"k_daily": 50, "alert_threshold": 0.7},
                     "cost": {"hypothesis": True, "review_inr_per_alert": 150, "friction_inr_per_false_positive": 60},
                     "models": ["xgboost", "isolation_forest"]},
        "generator_revision": "0123456789abcdef0123456789abcdef01234567",
        "timelines": [{"seed": 42, "rows": 90000, "fraud_rows": 461, "never_reported": 40, "investigated": 300,
                       "chargebacks": 121, "median_discovery_days": 1.3}],
        "rows": [
            {"seed": seed, "fold": f"fold-{fold}", "model": model, "label_mode": mode,
             "threshold_from_budget_alone": mode == "operational" and fold == 1}
            for seed in (42, 7) for fold in (1, 2, 3)
            for model in ("xgboost", "isolation_forest") for mode in ("full", "operational")
        ],
    })

    def entry(model, mode, pr, rec):
        return {"model": model, "label_mode": mode, "n_folds": 6, "n_budget_alone": 2 if mode == "operational" else 0,
                "pr_auc": {"mean": pr, "low": pr - 0.06, "high": pr + 0.06, "n": 6},
                "recall": {"mean": rec, "low": rec - 0.1, "high": rec + 0.1, "n": 6},
                "precision": {"mean": 0.9, "low": 0.8, "high": 0.95, "n": 6},
                "fpr_per_10k": {"mean": 2.5, "low": 1.0, "high": 4.0, "n": 6},
                "alerts_per_day": {"mean": 3.1, "low": 2.0, "high": 4.0, "n": 6},
                "cost_total_inr_per_10k": {"mean": 120000.0, "low": 90000.0, "high": 150000.0, "n": 6},
                "scoring_p99_ms": {"mean": 1.2, "low": 1.0, "high": 1.5, "n": 6}}

    _write(harness, "summary.json", {
        "models": [entry("xgboost", "full", 0.61, 0.42), entry("xgboost", "operational", 0.55, 0.35),
                   entry("isolation_forest", "full", 0.11, 0.05), entry("isolation_forest", "operational", 0.10, 0.04)],
        "pairwise": [
            {"model_a": "isolation_forest", "model_b": "xgboost", "metric": "pr_auc", "label_mode": "full", "n_folds": 6,
             "difference": {"mean": -0.5, "low": -0.6, "high": -0.4, "n": 6}, "share_a_at_least_b": 0.0, "share_b_at_least_a": 1.0},
            {"model_a": "isolation_forest", "model_b": "xgboost", "metric": "pr_auc", "label_mode": "operational", "n_folds": 6,
             "difference": {"mean": -0.45, "low": -0.52, "high": -0.38, "n": 6}, "share_a_at_least_b": 0.0, "share_b_at_least_a": 1.0},
        ],
        "scenarios": [{"model": "xgboost", "held_out_scenario": "INTENT_OMEGA_PROBE", "n_seeds": 2, "held_out_fraud_rows": 30,
                       "held_out_recall": {"mean": 0.2, "low": 0.1, "high": 0.3, "n": 2},
                       "pr_auc": {"mean": 0.5, "low": 0.4, "high": 0.6, "n": 2}},
                      {"model": "xgboost", "held_out_scenario": "INTENT_OMEGA_HARVEST", "n_seeds": 2, "held_out_fraud_rows": 12,
                       "held_out_recall": {"mean": 0.7, "low": 0.6, "high": 0.8, "n": 2},
                       "pr_auc": {"mean": 0.58, "low": 0.5, "high": 0.66, "n": 2}}],
        "metrics": ["pr_auc"],
    })
    _write(harness, "scenario.json", {
        "settings": {}, "generator_revision": None, "timelines": [],
        "rows": [{"seed": seed, "fold": f"fold-3-without-{scenario}", "model": "xgboost",
                  "label_mode": "full", "held_out_scenario": scenario}
                 for seed in (42, 7) for scenario in ("INTENT_OMEGA_PROBE", "INTENT_OMEGA_HARVEST")],
    })
    _write(harness, "summary.md", "| Model | Labels |\n|---|---|\n| xgboost | full |\n")
    _write(harness, "scenario.md", "| Model | Held-out scenario |\n|---|---|\n| xgboost | INTENT_OMEGA_PROBE |\n")
    return harness


def test_the_harness_section_renders_from_its_artifacts(india_dir):
    _harness_dir(india_dir)
    text = render_results(tracks.INDIA, load_artifacts(tracks.INDIA, india_dir))
    assert "**Harness**" in text and "2 seeds" in text and "3 folds per seed" in text
    assert "0.61" in text and "0.55" in text and "0.42" in text and "0.35" in text, "the label-mode gap of the best model"
    assert text.index("Under operational labels") < text.index("with every label known"), "the operational regime leads"
    assert "0.45" in text and "100%" in text, "the pairwise difference under operational labels, from the best model's side"
    assert "4 of 12 operational runs" in text and "budget alone" in text
    assert "model's own single-row call" in text and "card histories" in text
    assert "INTENT_OMEGA_PROBE" in text and "0.200" in text, "the weakest novel attack"
    assert "never reported" in text and "461" in text and "1.3 days" in text
    assert "hypothes" in text and "150" in text and "60" in text
    assert "0123456" in text
    assert "| xgboost | INTENT_OMEGA_PROBE |" in text and "| xgboost | full |" in text, "the tables are embedded"


def test_the_harness_is_optional_and_its_absence_is_said(india_dir):
    text = render_results(tracks.INDIA, load_artifacts(tracks.INDIA, india_dir))
    assert "harness has not run" in text and "prism.harness.run" in text


def test_a_harness_whose_summary_disagrees_with_its_rows_is_refused(india_dir):
    harness = _harness_dir(india_dir)
    summary = json.loads((harness / "summary.json").read_text(encoding="utf-8"))
    summary["models"][0]["n_folds"] = 5
    _write(harness, "summary.json", summary)
    with pytest.raises(MixedRuns, match="harness"):
        load_artifacts(tracks.INDIA, india_dir)
    summary["models"][0]["n_folds"] = 6
    summary["scenarios"].pop()
    _write(harness, "summary.json", summary)
    with pytest.raises(MixedRuns, match="scenario"):
        load_artifacts(tracks.INDIA, india_dir)


def test_a_partial_harness_directory_is_refused(india_dir):
    harness = _harness_dir(india_dir)
    (harness / "scenario.md").unlink()
    with pytest.raises(MissingArtifact, match="harness"):
        load_artifacts(tracks.INDIA, india_dir)


def _live_dir(india_dir, complete: bool = True):
    live = india_dir / "live"
    live.mkdir()
    events = [
        {"as_of": 1_000.0, "stream_day": 60.0, "status": "skipped", "known_fraud": {"train": 3, "val": 0, "backtest": 0},
         "reason": "the validation window holds 0 known frauds among 4100 rows, fewer than min_known_fraud 10"},
        {"as_of": 2_000.0, "stream_day": 75.0, "status": "promoted", "reason": None,
         "challenger_bundle": "day-075-aaaaaa", "challenger_model": "xgboost", "challenger_pr_auc": 0.4312,
         "champion_bundle": "offline", "champion_model": "xgboost", "champion_pr_auc": 0.3377,
         "difference": {"low": 0.0123, "high": 0.0871}, "known_fraud": {"train": 120, "val": 31, "backtest": 52}},
        {"as_of": 3_000.0, "stream_day": 90.0, "status": "kept",
         "reason": "the paired interval of the PR-AUC difference [-0.0214, 0.0342] does not lie above zero",
         "challenger_bundle": "day-090-bbbbbb", "challenger_model": "xgboost", "challenger_pr_auc": 0.4402,
         "champion_bundle": "day-075-aaaaaa", "champion_model": "xgboost", "champion_pr_auc": 0.4351,
         "difference": {"low": -0.0214, "high": 0.0342}, "known_fraud": {"train": 160, "val": 40, "backtest": 61}},
    ]
    days = [
        {"day": d, "requests": 1500, "flagged": 7, "fraud": 8, "caught": 3, "failed": 0,
         "labels": {"CHARGEBACK_DISPUTE": 2}, "bundle_id": "offline" if d < 75 else "day-075-aaaaaa"}
        for d in range(120 if complete else 80)
    ]
    _write(live, "run.json", {
        "settings": {"seed": 101, "rows": 180000, "cards": 2000, "merchants": 300, "days": 120, "fraud_rate": 0.005,
                     "supervision": {"k_daily": 50, "seed": 0}, "retrain_every_days": 15, "first_retrain_day": 60,
                     "label_lag_days": 30, "backtest_days": 20, "val_days": 10, "min_known_fraud": 10,
                     "bootstrap_resamples": 2000},
        "stem": "data/raw/india/live/train_in", "generator_revision": "51e5c77", "origin": 0.0,
        "rows": 180000, "complete": complete, "days": days,
        "labels": {"posted": {"CHARGEBACK_DISPUTE": 412, "INVESTIGATOR_ALERT": 377, "UNLABELLED": 44213},
                   "refused": 0, "pending_at_end": 9187, "never_reported": 455},
        "failed_requests": 0,
        "retrains": events,
        "drift": [
            {"as_of": 1_000.0, "status": 409, "bundle_id": "offline", "detail": "The serving models have no drift reference"},
            {"as_of": 2_000.0, "status": 200, "bundle_id": "day-075-aaaaaa", "rows": 5000, "state": "warn",
             "score": {"psi": 0.0417, "state": "ok"},
             "features": [{"feature": "card_tx_1h", "psi": 0.1388, "state": "warn"}]},
            {"as_of": 3_000.0, "status": 200, "bundle_id": "day-075-aaaaaa", "rows": 5000, "state": "ok",
             "score": {"psi": 0.0123, "state": "ok"},
             "features": [{"feature": "amount", "psi": 0.0456, "state": "ok"}]},
        ],
        "windows": [
            {"window": k, "start_day": 15.0 * k, "end_day": 15.0 * (k + 1), "rows": 22500, "fraud": 120,
             "pr_auc": 0.3311 + 0.01 * k, "recall": 0.2917, "alerts": 105,
             "bundles": ["offline"] if k < 5 else ["day-075-aaaaaa"]}
            for k in range(8)
        ],
    })
    _write(live, "events.json", events)
    return live


def test_the_live_section_states_every_retrain_and_its_interval(india_dir):
    _live_dir(india_dir)
    text = render_results(tracks.INDIA, load_artifacts(tracks.INDIA, india_dir))
    assert "**Live loop**" in text
    assert "180,000" in text and "seed 101" in text and "51e5c77" in text and "50" in text
    assert "| 60 | skipped |" in text and "fewer than min_known_fraud 10" in text
    assert "| 75 | promoted |" in text and "0.4312" in text and "0.3377" in text and "[0.0123, 0.0871]" in text
    assert "| 90 | kept |" in text and "[-0.0214, 0.0342]" in text
    assert "1 of 3 retrains promoted" in text
    assert "412" in text and "377" in text and "455" in text, "labels by source and fraud never reported"
    assert "0.3311" in text and "day-075-aaaaaa" in text, "the served champion per window"
    assert "card_tx_1h" in text and "0.1388" in text and "0.0417" in text, "drift at each retrain"
    assert "no drift reference" in text


def test_a_partial_live_run_is_refused(india_dir):
    _live_dir(india_dir, complete=False)
    with pytest.raises(ArtifactError, match="live/run.json"):
        load_artifacts(tracks.INDIA, india_dir)


def test_the_live_loop_is_optional_and_its_absence_is_said(india_dir):
    text = render_results(tracks.INDIA, load_artifacts(tracks.INDIA, india_dir))
    assert "live loop has not run" in text and "prism.live.run" in text


def test_a_live_directory_missing_its_events_is_refused(india_dir):
    live = _live_dir(india_dir)
    (live / "events.json").unlink()
    with pytest.raises(MissingArtifact, match="live"):
        load_artifacts(tracks.INDIA, india_dir)


def test_a_failed_retrain_renders_with_its_reason(india_dir):
    live = _live_dir(india_dir)
    run = json.loads((live / "run.json").read_text(encoding="utf-8"))
    failed = {"as_of": 4_000.0, "stream_day": 105.0, "status": "failed", "reason": "MemoryError: out of memory"}
    run["retrains"].append(failed)
    run["drift"].append({"as_of": 4_000.0, "status": 409, "bundle_id": "day-075-aaaaaa", "detail": "short"})
    _write(live, "run.json", run)
    _write(live, "events.json", [*json.loads((live / "events.json").read_text(encoding="utf-8")), failed])
    text = render_results(tracks.INDIA, load_artifacts(tracks.INDIA, india_dir))
    assert "| 105 | failed |" in text and "MemoryError" in text


def test_the_live_section_sets_the_truth_beside_the_operational_interval(india_dir):
    live = _live_dir(india_dir)
    run = json.loads((live / "run.json").read_text(encoding="utf-8"))
    run["retrains"][1]["truth"] = {"challenger_pr_auc": 0.5123, "champion_pr_auc": 0.4877, "fraud": 64}
    _write(live, "run.json", run)
    text = render_results(tracks.INDIA, load_artifacts(tracks.INDIA, india_dir))
    assert "0.5123" in text and "0.4877" in text
    assert "incumbent" in text, "the bias of the operational backtest is stated"
