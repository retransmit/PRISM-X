"""Tests for the external-corpus Pillar 1 scorer (scripts/score_external_corpus.py).

The script exists to answer one question honestly -- *does real payment data pass
the thresholds our synthetic data fails?* -- which is only a meaningful question
if two properties hold, and both are asserted here:

1. It grades with the certification grader's own code and thresholds, so a
   difference between two corpora is a difference in data and not in arithmetic.
2. A corpus that does not carry a metric's inputs is not graded on that metric.
   The grader substitutes per-row defaults for missing keys (``mcc`` -> 5411,
   ``pos_entry_mode`` -> "01"), each of which makes its divergence trivially 0;
   a corpus missing those columns would otherwise be handed a pass it did not earn.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = REPO_ROOT / "scripts" / "score_external_corpus.py"


@pytest.fixture(scope="module")
def scorer():
    """Load scripts/score_external_corpus.py as a module (it is not a package)."""
    spec = importlib.util.spec_from_file_location("score_external_corpus", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_missing_columns_are_excluded_from_the_verdict(scorer):
    """A corpus without an MCC column is graded on 2 metrics, not 5."""
    records = [
        {"amount": 25.0, "tx_time_seconds": float(t)}
        for t in range(0, 100_000, 500)
    ]
    result = scorer.score("bare", records, {"n": len(records), "time_span_days": 100_000 / 86_400.0}, "US", 42)

    by_metric = {row["metric"]: row for row in result["metrics"]}
    assert by_metric["wasserstein_amount_log"]["comparable"] is True
    assert by_metric["wasserstein_arrival_log"]["comparable"] is True

    for metric in ("js_divergence_mcc", "js_divergence_channel", "spearman_frobenius_error"):
        assert by_metric[metric]["comparable"] is False, metric
        assert by_metric[metric]["pass"] is None, metric
        assert by_metric[metric]["missing_inputs"], metric

    assert result["comparable_metrics"] == 2
    assert result["verdict"] in {"PASS", "FAIL"}  # verdict covers the comparable subset only


def test_scorer_reproduces_the_graders_own_number(scorer):
    """Same seed, same batch: the script's figures equal the reporter's scorecard."""
    from prism_generator.benchmark_reporter import UnifiedBenchmarkRunner

    n, region, seed = 400, "US", 7
    records, summary = scorer.generate_synthetic(n, region, seed)

    runner = UnifiedBenchmarkRunner(
        region=region, n_transactions=n, time_span_days=summary["time_span_days"], seed=seed
    )
    reference = runner._evaluate_data_fidelity(records)
    result = scorer.score("synthetic", records, summary, region, seed)

    by_metric = {row["metric"]: row for row in result["metrics"]}
    assert by_metric["wasserstein_amount_log"]["value"] == pytest.approx(reference.wasserstein_amount_log, abs=1e-9)
    assert by_metric["wasserstein_arrival_log"]["value"] == pytest.approx(reference.wasserstein_arrival_log, abs=1e-9)
    assert by_metric["js_divergence_mcc"]["value"] == pytest.approx(reference.js_divergence_mcc, abs=1e-9)
    assert all(row["comparable"] for row in result["metrics"])  # synthetic carries every input
    assert result["threshold_source"].endswith("spec/18_benchmark_reporting.yaml")


def test_reference_control_fails_the_realism_gates(scorer):
    """A stream drawn from the hardcoded reference must now FAIL the amount gate.

    Up to 2026-10-15 this test asserted the opposite -- the control passed both
    continuous gates -- and that assertion *was* the bug the 2026-10-10 external
    scorecard documented: passing a reference-sampler while real traffic failed
    the same gates harder than the generator under test meant the gates ranked
    realism backwards. Since spec/18 `pillar_1_real_reference`, the continuous
    gates grade distance to measured real card traffic (IEEE-CIS, sha256-pinned)
    against a size-corrected limit. A sampler of the old hardcoded lognormal is
    there fore not real traffic and fails the amount bar, while real corpus
    windows pass it (see tests/test_real_reference_gate.py for the corpus
    self-consistency side of that claim).
    """
    from prism_generator.real_reference import load_real_reference

    if load_real_reference() is None:
        pytest.skip(
            "real-corpus reference unavailable; fallback reference passes control by construction"
        )

    records, summary = scorer.reference_control(4_000, "US", 42)
    result = scorer.score("control", records, summary, "US", 42)

    by_metric = {row["metric"]: row for row in result["metrics"]}
    # The inversion that makes the gate meaningful: the control is far from real
    # traffic, and the gate knows it.
    assert by_metric["wasserstein_amount_log"]["pass"] is False
    assert result["comparable_failures"] >= 1
    assert result["verdict"] == "FAIL"
    # The limits named in the result are the calibrated ones the grader used, not
    # the static 0.150 spec values.
    assert by_metric["wasserstein_amount_log"]["threshold"] != 0.150


def test_embedded_fallback_is_flagged_in_the_caveats(scorer, monkeypatch):
    """Regression (2026-10-15): a corpora-free environment must SAY so.

    The fallback's limit is the non-zero static spec value (0.15), so a
    `amount_limit == 0.0` test never detected it and the "reference did not load"
    caveat was dead code -- the exact silent degradation the design forbids. The
    verdict now branches on the explicit `reference_kind`.
    """
    import prism_generator.benchmark_reporter as reporter

    monkeypatch.setattr(reporter, "load_real_reference", lambda: None)
    records, summary = scorer.reference_control(2_000, "US", 42)
    result = scorer.score("control-fallback", records, summary, "US", 42)

    assert result["caveats"], "fallback must be disclosed in the caveats"
    assert any("did not load" in c for c in result["caveats"])
