"""Deterministic Verification Suite for Day 5 (Slice 18): Camera-Ready Reporting & Unified Benchmark CLI."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import pytest

from prism_generator.cli import cmd_generate
from prism_generator.benchmark_reporter import (
    AdversarialPrivacyScorecard,
    BenchmarkReportCompiler,
    CausalXAIScorecard,
    DataFidelityScorecard,
    OperationalStreamingScorecard,
    PublicationPlotter,
    UnifiedBenchmarkReportData,
    UnifiedBenchmarkRunner,
)
from prism_generator.evaluation import DailyStreamingMetrics


@pytest.fixture
def sample_report_data() -> UnifiedBenchmarkReportData:
    """Creates a deterministic reference benchmark report fixture."""
    return UnifiedBenchmarkReportData(
        schema_version="1.0.0",
        metadata={
            "timestamp_utc": "2026-09-24T12:00:00Z",
            "region": "US",
            "currency": "USD",
            "n_transactions": 2500,
            "seed": 42,
            "model_architecture": "HistGradientBoostingClassifier",
            "explainer_type": "Pearlian_SCM_vs_TreeSHAP",
        },
        system_provenance={
            "python_version": "3.13.0",
            "platform": "Windows-11",
            "cpu_count": 8,
            "runtime_seconds": 12.45,
        },
        fidelity=DataFidelityScorecard(
            wasserstein_amount_log=0.0821,
            wasserstein_arrival_log=0.0914,
            js_divergence_mcc=0.0312,
            js_divergence_channel=0.0185,
            spearman_frobenius_error=0.4210,
            passed=True,
        ),
        privacy=AdversarialPrivacyScorecard(
            dcr_5th_percentile=0.0652,
            dcr_reference_ratio=0.9210,
            nndr_mean=0.8415,
            mia_attack_roc_auc=0.5120,
            evasion_rate_macro_mean=0.1420,
            passed=True,
        ),
        streaming=OperationalStreamingScorecard(
            n_days_evaluated=4,
            w_train_days=3.0,
            w_test_days=1.0,
            delta_delay_days=1.5,
            mean_pr_auc=0.6840,
            mean_average_precision=0.6812,
            mean_p_at_k=0.7200,
            mean_cp_at_k=0.6500,
            mean_dollar_recall_at_k=0.8250,
            mean_dollar_precision_at_k=0.4120,
            total_cost_base=12540.50,
            total_cost_model=3420.25,
            overall_savings_ratio=0.7273,
            drift_detected_days_count=0,
            passed=True,
        ),
        xai=CausalXAIScorecard(
            mean_kendall_tau=0.7333,
            mean_spearman_rho=0.8857,
            mean_pearson_r=0.9240,
            mean_precision_at_3=1.0,
            mean_relative_attribution_error=0.1240,
            mean_causal_faithfulness=0.8800,
            passed=True,
        ),
        all_pillars_passed=True,
        certification_grade="TIER-1_GOLD",
        violations=[],
        daily_trajectory=[
            {"day_index": 0, "n_tx": 625, "n_fraud": 42, "pr_auc": 0.65, "p_at_k": 0.70, "cp_at_k": 0.60, "dollar_recall_at_k": 0.80, "savings_ratio": 0.71, "drift_detected": False},
            {"day_index": 1, "n_tx": 625, "n_fraud": 40, "pr_auc": 0.67, "p_at_k": 0.72, "cp_at_k": 0.64, "dollar_recall_at_k": 0.82, "savings_ratio": 0.72, "drift_detected": False},
            {"day_index": 2, "n_tx": 625, "n_fraud": 44, "pr_auc": 0.70, "p_at_k": 0.73, "cp_at_k": 0.67, "dollar_recall_at_k": 0.84, "savings_ratio": 0.74, "drift_detected": False},
            {"day_index": 3, "n_tx": 625, "n_fraud": 41, "pr_auc": 0.71, "p_at_k": 0.73, "cp_at_k": 0.69, "dollar_recall_at_k": 0.84, "savings_ratio": 0.74, "drift_detected": False},
        ],
        feature_attributions={
            "amount": 0.42,
            "haversine_velocity_kph": 0.28,
            "ip_distance_from_home_km": 0.16,
            "user_tx_count_1h": 0.08,
            "user_tx_count_24h": 0.04,
            "cvv_match_flag": 0.02,
        },
        ground_truth_phi={
            "amount": 0.45,
            "haversine_velocity_kph": 0.25,
            "ip_distance_from_home_km": 0.18,
            "user_tx_count_1h": 0.07,
            "user_tx_count_24h": 0.03,
            "cvv_match_flag": 0.02,
        },
    )


def test_camera_ready_figures_generation(tmp_path: Path):
    """Test 1: Verify all 5 publication figures are generated in PDF and PNG at 300 DPI."""
    plotter = PublicationPlotter(style="ieee")

    # Fig 1: Prequential Trajectory
    daily_objs = [
        DailyStreamingMetrics(
            day_index=i,
            t_start_seconds=i * 86400.0,
            t_end_seconds=(i + 1) * 86400.0,
            n_transactions=500,
            n_fraud_actual=35,
            pr_auc=0.65 + i * 0.02,
            average_precision=0.64 + i * 0.02,
            p_at_k=0.70,
            cp_at_k=0.60,
            dollar_recall_at_k=0.80,
            dollar_precision_at_k=0.05,
            cost_base=1000.0,
            cost_model=300.0,
            savings_ratio=0.70,
            n_train_admissible=200,
            drift_detected=False,
        )
        for i in range(4)
    ]
    p1 = plotter.plot_prequential_timeseries(daily_objs, output_path=tmp_path, formats=("png", "pdf"))
    assert p1["png"].exists() and p1["png"].stat().st_size > 1000
    assert p1["pdf"].exists() and p1["pdf"].stat().st_size > 1000

    # Fig 2: Operational Triage
    k_vals = [10, 25, 50, 100, 200]
    p_k = [0.85, 0.78, 0.70, 0.55, 0.38]
    cp_k = [0.75, 0.70, 0.65, 0.58, 0.45]
    p2 = plotter.plot_operational_triage_tradeoff(k_vals, p_k, cp_k, output_path=tmp_path, formats=("png", "pdf"))
    assert p2["png"].exists() and p2["png"].stat().st_size > 1000

    # Fig 3: Financial Savings
    s_ratios = [0.45, 0.68, 0.73, 0.70, 0.58]
    net_dollars = [4500.0, 6800.0, 7300.0, 7000.0, 5800.0]
    p3 = plotter.plot_financial_savings_utility(k_vals, s_ratios, net_dollars, output_path=tmp_path, formats=("png", "pdf"))
    assert p3["png"].exists() and p3["png"].stat().st_size > 1000

    # Fig 4: Streaming Drift Timeline
    days = [1, 2, 3, 4]
    ks_p = [0.65, 0.55, 0.42, 0.005]  # Day 4 has drift alarm
    psi_s = [0.02, 0.04, 0.08, 0.28]  # Day 4 has severe alarm
    p4 = plotter.plot_streaming_drift_timeline(days, ks_p, psi_s, output_path=tmp_path, formats=("png", "pdf"))
    assert p4["png"].exists() and p4["png"].stat().st_size > 1000

    # Fig 5: Ground-Truth XAI Comparison
    feats = ["amount", "velocity", "ip_dist", "tx_1h"]
    v_s = [0.45, 0.25, 0.20, 0.10]
    v_gt = [0.42, 0.28, 0.18, 0.12]
    p5 = plotter.plot_ground_truth_xai_comparison(feats, v_s, v_gt, pearson_r=0.98, spearman_rho=0.95, output_path=tmp_path, formats=("png", "pdf"))
    assert p5["png"].exists() and p5["pdf"].exists()


def test_benchmark_results_json_schema_validity(tmp_path: Path, sample_report_data: UnifiedBenchmarkReportData):
    """Test 2: Verify JSON serialization contains all required schema keys."""
    json_path = tmp_path / "benchmark_results.json"
    BenchmarkReportCompiler.compile_json(sample_report_data, json_path)

    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    # Invariant: Must contain all 4 evaluation pillars and metadata
    assert d["schema_version"] == "1.0.0"
    assert "metadata" in d and d["metadata"]["region"] == "US"
    assert "system_provenance" in d and d["system_provenance"]["runtime_seconds"] > 0
    assert "fidelity" in d and d["fidelity"]["wasserstein_amount_log"] == 0.0821
    assert "privacy" in d and d["privacy"]["dcr_5th_percentile"] == 0.0652
    assert d["privacy"]["dcr_reference_ratio"] == 0.9210
    assert "streaming" in d and d["streaming"]["overall_savings_ratio"] == 0.7273
    assert "xai" in d and d["xai"]["mean_spearman_rho"] == 0.8857
    assert d["all_pillars_passed"] is True
    assert d["certification_grade"] == "TIER-1_GOLD"


def test_four_pillar_scorecard_computation():
    """Test 3: Verify UnifiedBenchmarkRunner executes simulation and evaluates all 4 pillars."""
    runner = UnifiedBenchmarkRunner(
        region="IN",
        n_transactions=600,
        time_span_days=10.0,
        k_daily=10,
        w_train_days=2.0,
        w_test_days=1.0,
        delta_delay_days=1.0,
        seed=101,
    )
    report = runner.run_benchmark()

    assert report.metadata["region"] == "IN"
    assert report.metadata["currency"] == "INR"
    assert report.fidelity.wasserstein_amount_log >= 0.0
    assert 0.0 <= report.fidelity.js_divergence_mcc <= 1.0
    assert report.privacy.dcr_5th_percentile >= 0.0
    # Memorisation is the synthetic-to-real distance relative to the real-to-real one
    assert report.privacy.dcr_reference_ratio > 0.0
    assert 0.0 <= report.privacy.nndr_mean <= 1.0
    assert report.streaming.n_days_evaluated >= 1
    assert 0.0 <= report.streaming.mean_pr_auc <= 1.0
    assert report.streaming.overall_savings_ratio <= 1.0
    assert -1.0 <= report.xai.mean_spearman_rho <= 1.0
    assert report.certification_grade in ("TIER-1_GOLD", "TIER-2_SILVER", "NON_CERTIFIED_FAIL")


def test_offline_standalone_html_report_compilation(tmp_path: Path, sample_report_data: UnifiedBenchmarkReportData):
    """Test 4: Verify generated HTML report is 100% offline with zero external CDNs."""
    plotter = PublicationPlotter(style="ieee")
    fig_paths = {}

    daily_objs = [
        DailyStreamingMetrics(
            day_index=i, t_start_seconds=0.0, t_end_seconds=86400.0, n_transactions=100, n_fraud_actual=5,
            pr_auc=0.5, average_precision=0.5, p_at_k=0.5, cp_at_k=0.5, dollar_recall_at_k=0.5,
            dollar_precision_at_k=0.05, cost_base=100.0, cost_model=50.0, savings_ratio=0.5,
            n_train_admissible=50, drift_detected=False
        ) for i in range(2)
    ]
    fig_paths["fig1_prequential_timeseries"] = plotter.plot_prequential_timeseries(daily_objs, output_path=tmp_path)
    fig_paths["fig2_operational_triage_tradeoff"] = plotter.plot_operational_triage_tradeoff([10, 20], [0.8, 0.7], [0.6, 0.5], output_path=tmp_path)
    fig_paths["fig3_financial_savings_utility"] = plotter.plot_financial_savings_utility([10, 20], [0.5, 0.6], [500.0, 600.0], output_path=tmp_path)
    fig_paths["fig4_streaming_drift_timeline"] = plotter.plot_streaming_drift_timeline([1, 2], [0.5, 0.4], [0.02, 0.03], output_path=tmp_path)
    fig_paths["fig5_ground_truth_xai_comparison"] = plotter.plot_ground_truth_xai_comparison(
        ["amount", "velocity"], [0.6, 0.4], [0.55, 0.45], 0.95, 0.90, output_path=tmp_path
    )

    html_path = tmp_path / "benchmark_report.html"
    BenchmarkReportCompiler.compile_html(sample_report_data, fig_paths, html_path)

    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")

    # Invariants: Standalone self-contained HTML
    assert "data:image/png;base64," in content, "HTML report must embed figures as base64 images!"
    assert "http://" not in content, "Offline HTML must not reference external HTTP assets!"
    assert "https://" not in content, "Offline HTML must not reference external HTTPS assets!"
    assert "fonts.googleapis.com" not in content, "No external Google Fonts allowed in air-gapped forensic report!"


def test_cli_benchmark_run_end_to_end(tmp_path: Path):
    """Test 5: Verify CLI benchmark command with --unified and --camera-ready flags."""
    import argparse
    from prism_generator.cli import cmd_benchmark

    out_dir = tmp_path / "run_out"
    args = argparse.Namespace(
        command="benchmark",
        samples=400,
        region="US",
        fraud_rate=0.06,
        seed=77,
        unified=True,
        camera_ready=True,
        output_dir=str(out_dir),
        conference_style="ieee",
        w_train=2.0,
        w_test=1.0,
        delta_delay=1.0,
        k_daily=10,
        output_report=None,
        json=False,
    )

    cmd_benchmark(args)

    assert (out_dir / "benchmark_results.json").exists()
    assert (out_dir / "BENCHMARK_REPORT.md").exists()
    assert (out_dir / "benchmark_report.html").exists()
    assert (out_dir / "figures" / "fig1_prequential_timeseries.png").exists()
    assert (out_dir / "figures" / "fig5_ground_truth_xai_comparison.png").exists()


def test_cli_benchmark_report_end_to_end(tmp_path: Path, sample_report_data: UnifiedBenchmarkReportData):
    """Test 6: Verify CLI report compilation from an existing benchmark_results.json."""
    import argparse
    from prism_generator.cli import cmd_report

    # Save benchmark_results.json
    in_json = tmp_path / "benchmark_results.json"
    BenchmarkReportCompiler.compile_json(sample_report_data, in_json)

    out_dir = tmp_path / "compiled_reports"
    args = argparse.Namespace(
        command="report",
        input=str(in_json),
        output_dir=str(out_dir),
        format="json,md,html",
        conference_style="acm",
    )

    cmd_report(args)

    assert (out_dir / "benchmark_results.json").exists()
    assert (out_dir / "BENCHMARK_REPORT.md").exists()
    assert (out_dir / "benchmark_report.html").exists()
    assert (out_dir / "figures" / "fig2_operational_triage_tradeoff.png").exists()


def test_cli_generate_routing_without_arbitrary_threshold(tmp_path: Path, monkeypatch):
    """Test 7: Verify CLI generate command routes to discrete event engine for arbitrary N unless --parallel is passed."""
    import argparse
    from prism_generator.cli import cmd_generate

    mock_records = [
        {"transaction_id": "TX_001", "amount": 100.0, "is_fraud": 0, "card_id": "C_01"}
    ]
    captured = {}

    def mock_generate_batch(self, n_transactions, fraud_prevalence, time_span_days):
        captured["n"] = n_transactions
        captured["prevalence"] = fraud_prevalence
        return mock_records

    from prism_generator.engine import SimulationEngine
    monkeypatch.setattr(SimulationEngine, "generate_batch", mock_generate_batch)

    out_csv = tmp_path / "large_run.csv"
    args = argparse.Namespace(
        command="generate",
        n=150000,
        cards=100,
        merchants=20,
        region="US",
        fraud_rate=0.01,
        days=30,
        seed=42,
        output=str(out_csv),
        parallel=False,
        include_disputes=False,
        adversary_mode="intent",
        export_institutional_views=True,
        calibration=None,
    )

    cmd_generate(args)

    assert captured["n"] == 150000
    assert out_csv.exists()
    assert (tmp_path / "large_run_auth_stream.csv").exists()
    assert (tmp_path / "large_run_gateway_telemetry.csv").exists()


def _generate_args(tmp_path: Path, **overrides) -> argparse.Namespace:
    """A generate Namespace with every field cmd_generate reads, overridable per test."""
    base = dict(
        command="generate",
        n=1000,
        cards=100,
        merchants=20,
        region="US",
        fraud_rate=0.01,
        days=30,
        density=None,
        seed=42,
        output=str(tmp_path / "out.csv"),
        parallel=False,
        include_disputes=False,
        adversary_mode="intent",
        export_institutional_views=False,
        calibration=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_density_knob_sets_the_row_count(tmp_path: Path, monkeypatch):
    """`--density` must express tx/day directly, deriving n = density x days.

    Count and density are the same quantity written two ways: density = n / days. Pinning
    -n and --days together therefore fixes density as a side effect, which is why the total
    alone cannot answer "how many transactions happen in a day". --density exists so that
    can be chosen first, with the total following from it -- which is the relationship a
    user asking for "30 days of national transaction density" actually has in mind.
    """
    from prism_generator.engine import SimulationEngine

    captured = {}

    def mock_generate_batch(self, n_transactions, fraud_prevalence, time_span_days):
        captured["n"] = n_transactions
        captured["span"] = time_span_days
        return [{"transaction_id": "T1", "tx_time_seconds": 0.0, "is_fraud": 0, "card_id": "C1"}]

    monkeypatch.setattr(SimulationEngine, "generate_batch", mock_generate_batch)

    cmd_generate(_generate_args(tmp_path, n=7, density=120.0, days=30))

    assert captured["n"] == 120 * 30, (
        f"--density 120 x --days 30 should derive 3600 rows, got {captured['n']} "
        f"(the stale -n of {7} must not survive)"
    )


def test_density_ignores_no_density_flag(tmp_path: Path, monkeypatch):
    """Without --density, -n must still win exactly: the published 100k contract depends on it."""
    from prism_generator.engine import SimulationEngine

    captured = {}

    def mock_generate_batch(self, n_transactions, fraud_prevalence, time_span_days):
        captured["n"] = n_transactions
        return [{"transaction_id": "T1", "tx_time_seconds": 0.0, "is_fraud": 0, "card_id": "C1"}]

    monkeypatch.setattr(SimulationEngine, "generate_batch", mock_generate_batch)

    cmd_generate(_generate_args(tmp_path, n=100000, days=30, density=None))

    assert captured["n"] == 100000, "without --density the requested row count must be honoured exactly"


def test_density_report_is_printed_and_states_both_sides(tmp_path: Path, monkeypatch, capsys):
    """Generation must report requested vs realised span and density rather than stay silent.

    The realised span legitimately diverges from --days (N = density x days, and density
    itself is per-card rate times the active population, which fraud reduces), so hiding it
    is what made the divergence look like a defect. The report names both sides so the
    coupling is visible in the output itself.
    """
    from prism_generator.engine import SimulationEngine

    # 60 days of timestamps for a 30-day request: a 2x divergence the report must surface.
    def mock_generate_batch(self, n_transactions, fraud_prevalence, time_span_days):
        return [
            {"transaction_id": f"T{i}", "tx_time_seconds": float(i) * 86400.0, "is_fraud": 0, "card_id": "C1"}
            for i in range(60)
        ]

    monkeypatch.setattr(SimulationEngine, "generate_batch", mock_generate_batch)

    cmd_generate(_generate_args(tmp_path, n=60, days=30))
    err = capsys.readouterr().err

    assert "density report:" in err, "generation must print a density report"
    assert "requested 30 d" in err, f"the requested side must be stated:\n{err}"
    assert "realised" in err and "d @" in err, f"the realised side must be stated:\n{err}"


def test_benchmark_measurements_are_empirical_not_constants():
    """Verify that Frobenius error, MIA ROC-AUC, Causal Faithfulness, etc. are empirical measurements, not constants."""
    runner1 = UnifiedBenchmarkRunner(region="US", n_transactions=500, time_span_days=8.0, k_daily=10, seed=42)
    report1 = runner1.run_benchmark()

    runner2 = UnifiedBenchmarkRunner(region="US", n_transactions=500, time_span_days=8.0, k_daily=10, seed=99)
    report2 = runner2.run_benchmark()

    # 1. Frobenius error must be empirical, non-zero, and not a hardcoded 0.42
    assert report1.fidelity.spearman_frobenius_error != 0.42
    assert 0.0 <= report1.fidelity.spearman_frobenius_error <= 1.0

    # 2. MIA ROC-AUC must be empirical, computed from distance-based logistic attack, not a literal 0.518
    assert report1.privacy.mia_attack_roc_auc != 0.518
    assert 0.40 <= report1.privacy.mia_attack_roc_auc <= 0.65

    # 3. Causal faithfulness must be an empirical subset-ablation correlation, not a
    #    literal 0.88 and not the constant 1.0 a self-comparison would always yield.
    #    It is a Pearson correlation, so its valid range includes negative values:
    #    a negative score means the attribution is anti-correlated with the model's
    #    actual score drop, which the pillar-4 gate (< 0.500) reports as a failure.
    assert report1.xai.mean_causal_faithfulness != 0.88
    assert -1.0 <= report1.xai.mean_causal_faithfulness <= 1.0

    # 4. Multi-seed variation proves genuine data-dependent execution
    assert report1.fidelity.wasserstein_amount_log != report2.fidelity.wasserstein_amount_log
    # the distance to the closest record is rounded to four decimals and sits at 0.0004 for both seeds
    # at 500 rows; the nearest-neighbour distance ratio is the privacy figure that moves with the data
    assert report1.privacy.nndr_mean != report2.privacy.nndr_mean

    # 5. Triage curves and daily drift metrics must be populated from test stream
    assert "k_values" in report1.triage_curves and len(report1.triage_curves["k_values"]) == 5
    assert len(report1.triage_curves["p_at_k"]) == 5
    assert all("ks_drift_p_value" in d for d in report1.daily_trajectory)
    assert all("psi_score" in d for d in report1.daily_trajectory)


def test_html_report_dynamic_fail_pills(tmp_path: Path, sample_report_data: UnifiedBenchmarkReportData):
    """Verify HTML report renders pill-fail when metrics violate benchmark thresholds."""
    # Create failing report data
    failing_report = sample_report_data
    failing_report.fidelity.wasserstein_amount_log = 0.8888  # Violates <= 0.350
    failing_report.privacy.dcr_reference_ratio = 0.0001     # Violates >= 0.500
    failing_report.all_pillars_passed = False
    failing_report.certification_grade = "NON_CERTIFIED_FAIL"

    html_path = tmp_path / "failing_report.html"
    BenchmarkReportCompiler.compile_html(failing_report, {}, html_path)

    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")
    assert '<span class="pill pill-fail">FAIL</span>' in content, "Failing metrics must render pill-fail!"
    assert 'badge-danger' in content


def test_evasion_rate_counts_approved_fraud():
    """Evasion is the share of fraud the issuer approved. Records carry the answer in
    auth_response_code; reading a key no record has made the rate 0.0 on every run."""
    from prism_generator import DiscreteEventEngine

    records = DiscreteEventEngine(n_cards=60, n_merchants=20, region="US", seed=5).generate_batch(
        n_transactions=400, fraud_prevalence=0.1
    )
    fraud = [r for r in records if r["is_fraud"] == 1]
    approved = sum(1 for r in fraud if str(r["auth_response_code"]) == "00")
    assert 0 < approved < len(fraud), "the batch must hold approved and declined fraud or this proves nothing"
    runner = UnifiedBenchmarkRunner(region="US", n_transactions=400, time_span_days=4.0, k_daily=10, seed=5)
    scorecard = runner._evaluate_adversarial_privacy(records)
    # the scorecard stores four decimals
    assert scorecard.evasion_rate_macro_mean == pytest.approx(approved / len(fraud), abs=5e-5)


def test_privacy_gate_fails_a_generator_that_copies_records():
    """Negative control: a generator that replays real records verbatim must fail.

    Memorisation is measured as the synthetic-to-real 5th-percentile distance relative
    to the real-to-real one on the same points, so a copier drives the ratio to zero
    while a generator that models the distribution lands near 1.0. The gate this
    replaced was an absolute floor in normalised feature space, which a copier passes
    as long as it copies *common* records -- precisely the case the gate exists to
    catch, which is why the measured 0.0001 passed while meaning nothing.
    """
    from prism_generator import DiscreteEventEngine

    records = DiscreteEventEngine(n_cards=60, n_merchants=20, region="US", seed=5).generate_batch(
        n_transactions=400, fraud_prevalence=0.1
    )
    runner = UnifiedBenchmarkRunner(region="US", n_transactions=400, time_span_days=4.0, k_daily=10, seed=5)

    honest = runner._evaluate_adversarial_privacy(records)
    assert honest.dcr_reference_ratio > 0.5, (
        "a modelled stream should sit at roughly the real-to-real distance, "
        f"got {honest.dcr_reference_ratio}"
    )

    # The second half replays the first half: every synthetic record is a real record.
    cloner_records = [dict(r) for r in records]
    half = len(cloner_records) // 2
    cloner_records[half:] = [dict(r) for r in cloner_records[:half]]

    copied = runner._evaluate_adversarial_privacy(cloner_records)
    assert copied.dcr_reference_ratio < 0.5, "a verbatim copier must be caught by the DCR ratio"
    assert copied.passed is False


def test_causal_scorecard_without_ground_truth_does_not_pass():
    """With no fraud record carrying a ground-truth attribution there is nothing to
    compare, so the scorecard reports no measurement and does not pass. It used to
    report perfect agreement."""
    from prism_generator import DiscreteEventEngine

    records = DiscreteEventEngine(n_cards=60, n_merchants=20, region="US", seed=5).generate_batch(
        n_transactions=300, fraud_prevalence=0.1
    )
    stripped = [
        {k: v for k, v in r.items() if k not in ("analytical_shapley_probability", "counterfactual_twin")}
        for r in records
    ]
    runner = UnifiedBenchmarkRunner(region="US", n_transactions=300, time_span_days=4.0, k_daily=10, seed=5)
    scorecard, _, _ = runner._evaluate_causal_xai(stripped)
    assert scorecard.passed is False
    assert math.isnan(scorecard.mean_kendall_tau)
    assert math.isnan(scorecard.mean_causal_faithfulness)
