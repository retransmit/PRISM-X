"""Tests for the pinned evaluation protocol and grade-stability reporting (spec/18 §4).

Why these assertions exist
--------------------------
Measured 2026-10-10 over eleven runs of `benchmark --unified`, identical except for
`--seed`, the same command reported between 4 and 9 violations: several gates rest on
small denominators (2-8 streaming test days; pillar-2 attack simulations whose macro
mean moves in steps of ~0.06), so a single run's violation count was not a result.
`spec/18` pinned thresholds but not the run they were applied to.

Everything asserted here is the fix for that, in three parts:

1. The protocol is *read from spec/18* rather than restated in code, so the pin and the
   SSOT cannot drift apart.
2. A run that deviates from it is identified and reported as OFF-PROTOCOL instead of
   emitting a grade that looks comparable.
3. The stability summary counts *gates*, not strings -- violation texts embed their own
   observed values, so naive grouping would count every run as a different failure.
"""

from pathlib import Path

import pytest

from prism_generator.benchmark_reporter import (
    _gate_key,
    load_evaluation_protocol,
    protocol_mismatches,
    protocol_status_markdown,
    stability_markdown,
    summarize_stability,
)


@pytest.fixture(scope="module")
def protocol_and_source():
    return load_evaluation_protocol()


def test_protocol_is_pinned_by_spec_18(protocol_and_source):
    """The pin comes from spec/18 and names a complete, non-empty run."""
    protocol, source = protocol_and_source
    assert source.endswith("18_benchmark_reporting.yaml"), source
    assert protocol["n_transactions"] == 2000  # README's documented `-n 2000`
    assert protocol["seed"] == 42  # README's documented `--seed 42`
    assert protocol["region"] == "US"
    assert protocol["time_span_days"] == 12
    assert protocol["w_train_days"] == 3.0
    assert protocol["delta_delay_days"] == 1.5
    assert len(protocol["stability_seeds"]) >= 2, "stability needs at least two seeds"
    assert 42 in protocol["stability_seeds"], "the protocol seed must be part of its own range"


def test_a_run_matching_the_protocol_reports_no_deviation(protocol_and_source):
    protocol, _ = protocol_and_source
    assert protocol_mismatches(protocol, protocol) == []


def test_a_deviation_is_named_with_both_values(protocol_and_source):
    protocol, _ = protocol_and_source
    deviated = dict(protocol, n_transactions=500, region="IN", w_train_days=2.0)
    found = protocol_mismatches(protocol, deviated)
    assert len(found) == 3
    assert any("n_transactions" in line and "500" in line and "2000" in line for line in found)
    assert any(line.startswith("region:") for line in found)
    assert any(line.startswith("w_train_days:") for line in found)


def test_day_valued_windows_accept_equivalent_spellings(protocol_and_source):
    """`3` and `3.0` are the same window; a genuine difference is still caught."""
    protocol, _ = protocol_and_source
    assert protocol_mismatches(protocol, dict(protocol, w_train_days=3)) == []
    assert protocol_mismatches(protocol, dict(protocol, w_train_days=3.5)) != []


def test_gate_key_ignores_the_numbers_but_keeps_the_pillar():
    """Two runs failing one gate must count once, not once per observed value."""
    run_a = "Pillar 1 (Data Fidelity): MCC Jensen-Shannon divergence 0.1092 exceeds the 0.0500 maximum"
    run_b = "Pillar 1 (Data Fidelity): MCC Jensen-Shannon divergence 0.0527 exceeds the 0.0500 maximum"
    other = "Pillar 3 (Operational Streaming): prequential PR-AUC 0.0417 falls below the 0.1500 minimum"

    assert _gate_key(run_a) == _gate_key(run_b)
    assert _gate_key(run_a) != _gate_key(other)
    assert "Pillar 1 (Data Fidelity)" in _gate_key(run_a)
    assert "0.1092" not in _gate_key(run_a) and "0.0500" not in _gate_key(run_a)


def test_stability_summary_counts_gates_and_flags_the_ones_that_flip():
    solid = "Pillar 1 (Data Fidelity): inter-arrival Wasserstein 0.4246 exceeds the 0.1500 maximum"
    flipping = "Pillar 3 (Operational Streaming): prequential PR-AUC 0.0417 falls below the 0.1500 minimum"
    other_flip = "Pillar 1 (Data Fidelity): MCC Jensen-Shannon divergence 0.0586 exceeds the 0.0500 maximum"
    solid_key = _gate_key(solid)
    runs = [
        {"seed": 42, "grade": "NON_CERTIFIED_FAIL", "violations": [solid, flipping]},
        {"seed": 0, "grade": "NON_CERTIFIED_FAIL", "violations": [solid]},
        {"seed": 1, "grade": "NON_CERTIFIED_FAIL", "violations": [solid]},
        {"seed": 2, "grade": "NON_CERTIFIED_FAIL", "violations": [solid, other_flip]},
    ]

    summary = summarize_stability(runs, "spec/18_benchmark_reporting.yaml")

    assert summary.runs == 4
    assert summary.seeds == [42, 0, 1, 2]
    assert summary.grades == {"NON_CERTIFIED_FAIL": 4}
    assert summary.grade_stable is True
    assert summary.violation_counts == [2, 1, 1, 2]
    # inter-arrival failed all four runs; PR-AUC and MCC each failed one
    assert len(summary.gate_failures) == 3
    assert summary.gate_failures[solid_key] == 4
    solid_only = [g for g, c in summary.gate_failures.items() if c == summary.runs]
    assert solid_only == [solid_key]
    assert len(summary.unstable_gates) == 2
    assert any("PR-AUC" in g for g in summary.unstable_gates)
    assert all(g != solid_key for g in summary.unstable_gates)


def test_stability_summary_refuses_to_invent_an_answer_from_no_runs():
    with pytest.raises(ValueError):
        summarize_stability([], "spec/18_benchmark_reporting.yaml")


def test_markdown_says_whether_the_grade_is_comparable_and_where_it_flips():
    solid = "Pillar 1 (Data Fidelity): inter-arrival Wasserstein 0.4246 exceeds the 0.1500 maximum"
    flipping = "Pillar 3 (Operational Streaming): alert precision P@K 0.0000 falls below the 0.0500 minimum"

    off_protocol = protocol_status_markdown(["n_transactions: this run 500, spec/18 pins 2000"])
    assert "OFF-PROTOCOL" in off_protocol
    assert "not** comparable" in off_protocol
    assert "n_transactions" in off_protocol
    assert protocol_status_markdown([]).count("OFF-PROTOCOL") == 0

    summary = summarize_stability(
        [
            {"seed": 42, "grade": "NON_CERTIFIED_FAIL", "violations": [solid, flipping]},
            {"seed": 0, "grade": "NON_CERTIFIED_FAIL", "violations": [solid]},
        ],
        "spec/18_benchmark_reporting.yaml",
    )
    text = stability_markdown(summary, "spec/18_benchmark_reporting.yaml")
    assert "Grade Stability Across the Pinned Protocol Seeds" in text
    assert "violation counts 1-2" in text
    assert "solid failure" in text
    assert "FLIPS WITH SEED" in text
    assert "| gate | failed in | reading |" in text
    # markdown table cells must not be broken by the gate key separator
    assert text.count("| Pillar 3") >= 1
    # and the section must end with a blank line, since the template appends `##` next
    assert text.endswith("\n\n")


def test_runner_records_the_protocol_it_was_graded_under(tmp_path: Path):
    """An off-protocol run must say so in its own metadata and in its report."""
    from prism_generator.benchmark_reporter import (
        BenchmarkReportCompiler,
        UnifiedBenchmarkRunner,
    )

    # Deliberately deviating run: smaller sample, shorter stream.
    runner = UnifiedBenchmarkRunner(
        region="US", n_transactions=200, time_span_days=4.0, k_daily=10, seed=5
    )
    report = runner.run_benchmark()

    mismatches = report.metadata["protocol_mismatches"]
    assert mismatches, "a 200-transaction run cannot match a 2000-transaction protocol"
    assert any(m.startswith("n_transactions:") for m in mismatches)
    assert any(m.startswith("time_span_days:") for m in mismatches)
    assert report.metadata["pinned_protocol"]["n_transactions"] == 2000
    assert report.metadata["run_protocol"]["n_transactions"] == 200
    assert report.metadata["protocol_source"].endswith("18_benchmark_reporting.yaml")
    assert report.stability is None, "an off-protocol run must not publish a stability range"

    md_path = BenchmarkReportCompiler.compile_markdown(report, tmp_path / "REPORT.md")
    markdown = Path(md_path).read_text(encoding="utf-8")
    assert "OFF-PROTOCOL" in markdown
    assert "not** comparable" in markdown
    assert "Grade Stability Across the Pinned Protocol Seeds" not in markdown
