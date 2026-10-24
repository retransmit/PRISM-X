"""Deterministic verification suite for Slice 28: Multi-Seed Stability Matrix and Publication Leaderboard Release.

Validates:
1. Pinned evaluation protocol multi-tier loading (protocol_fast vs protocol_full).
2. UnifiedBenchmarkRunner protocol tier resolution, automated card population sizing, and badge stamping.
3. Multi-seed stability runner execution and StabilitySummary aggregation across pinned seeds.
4. Publication report compilation (Markdown & HTML) with grade stability matrices and badge pills.
5. 500k staging dataset card and cryptographic checksum integrity.
"""

from pathlib import Path
import subprocess
import sys

from prism_generator.benchmark_reporter import (
    BenchmarkReportCompiler,
    StabilitySummary,
    UnifiedBenchmarkRunner,
    load_evaluation_protocol,
)


def test_load_evaluation_protocol_explicit_tiers():
    """Invariant 1: load_evaluation_protocol cleanly distinguishes protocol_fast and protocol_full."""
    fast_proto, fast_src = load_evaluation_protocol("protocol_fast")
    assert fast_proto["n_transactions"] == 2000
    assert fast_proto["time_span_days"] == 12.0
    assert fast_proto["w_train_days"] == 3.0
    assert fast_proto["delta_delay_days"] == 1.5
    assert fast_proto["k_daily"] == 15
    assert fast_proto["protocol_tier"] == "protocol_fast"
    assert fast_proto["badge"] == "REGRESSION (PROTOCOL_FAST)"
    assert len(fast_proto["stability_seeds"]) >= 4

    full_proto, full_src = load_evaluation_protocol("protocol_full")
    assert full_proto["n_transactions"] == 100000
    assert full_proto["time_span_days"] == 75.0
    assert full_proto["w_train_days"] == 30.0
    assert full_proto["delta_delay_days"] == 14.0
    assert full_proto["k_daily"] == 50
    assert full_proto["protocol_tier"] == "protocol_full"
    assert full_proto["badge"] == "CERTIFIED (PROTOCOL_FULL)"
    assert len(full_proto["stability_seeds"]) >= 4

    # Default invocation without arguments matches protocol_fast
    default_proto, _ = load_evaluation_protocol()
    assert default_proto["n_transactions"] == 2000
    assert default_proto["protocol_tier"] == "protocol_fast"


def test_unified_benchmark_runner_tier_resolution():
    """Invariant 2: UnifiedBenchmarkRunner correctly resolves and infers protocol tiers."""
    # Explicit fast tier
    runner_fast = UnifiedBenchmarkRunner(n_transactions=500, protocol_tier="fast")
    assert runner_fast.protocol_tier == "protocol_fast"

    # Explicit full tier
    runner_full = UnifiedBenchmarkRunner(n_transactions=500, protocol_tier="full")
    assert runner_full.protocol_tier == "protocol_full"

    # Inferred fast tier
    runner_inferred_fast = UnifiedBenchmarkRunner(n_transactions=2000, time_span_days=12.0)
    assert runner_inferred_fast.protocol_tier == "protocol_fast"

    # Inferred full tier from transaction scale
    runner_inferred_full = UnifiedBenchmarkRunner(n_transactions=100000, time_span_days=75.0)
    assert runner_inferred_full.protocol_tier == "protocol_full"


def test_unified_benchmark_runner_run_stability():
    """Invariant 3: run_stability evaluates multiple seeds and attaches StabilitySummary."""
    runner = UnifiedBenchmarkRunner(
        region="US",
        n_transactions=400,
        time_span_days=12.0,
        k_daily=15,
        w_train_days=3.0,
        w_test_days=1.0,
        delta_delay_days=1.5,
        seed=42,
        protocol_tier="fast",
    )
    summary = runner.run_stability(seeds=[42, 0])
    assert isinstance(summary, StabilitySummary)
    assert summary.runs == 2
    assert summary.seeds == [42, 0]
    assert runner.stability is summary

    report = runner.run_benchmark()
    assert report.stability is summary
    assert report.metadata.get("protocol_tier") == "protocol_fast"
    assert "badge" in report.metadata


def test_report_compiler_renders_stability_and_badges(tmp_path: Path):
    """Invariant 4: BenchmarkReportCompiler correctly formats tier badges and multi-seed stability tables."""
    runner = UnifiedBenchmarkRunner(
        region="US",
        n_transactions=400,
        time_span_days=12.0,
        k_daily=15,
        w_train_days=3.0,
        w_test_days=1.0,
        delta_delay_days=1.5,
        seed=42,
        protocol_tier="fast",
    )
    runner.run_stability(seeds=[42, 0])
    report = runner.run_benchmark()

    # Markdown compilation
    md_path = tmp_path / "BENCHMARK_REPORT.md"
    BenchmarkReportCompiler.compile_markdown(report, md_path)
    md_text = md_path.read_text(encoding="utf-8")
    assert "Grade Stability Across the Pinned Protocol Seeds" in md_text
    assert "REGRESSION (PROTOCOL_FAST)" in md_text or "OFF-PROTOCOL" in md_text

    # HTML compilation
    html_path = tmp_path / "benchmark_report.html"
    BenchmarkReportCompiler.compile_html(report, {}, html_path)
    html_text = html_path.read_text(encoding="utf-8")
    assert "Grade stability across" in html_text
    assert "spec/18 evaluation_protocol" in html_text


def test_staging_500k_manifest_integrity():
    """Invariant 5: 500k US staging dataset card references existing SHA256SUMS verbatim."""
    staging_dir = Path(__file__).resolve().parent.parent / "staging_us_500k"
    readme_path = staging_dir / "README.md"
    sums_path = staging_dir / "SHA256SUMS"

    assert readme_path.exists(), "staging_us_500k/README.md does not exist"
    assert sums_path.exists(), "staging_us_500k/SHA256SUMS does not exist"

    readme_content = readme_path.read_text(encoding="utf-8")
    sums_content = sums_path.read_text(encoding="utf-8")

    for line in sums_content.strip().splitlines():
        if line.strip():
            sha, filename = line.split()
            assert sha in readme_content, f"SHA256 {sha} for {filename} missing from README.md"
            assert filename in readme_content, f"Filename {filename} missing from README.md"


def test_cli_protocol_argument_exposure():
    """Invariant 6: CLI benchmark subcommand exposes --protocol and --days flags."""
    cmd = [sys.executable, "-m", "prism_generator.cli", "benchmark", "--help"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "--protocol {fast,full}" in res.stdout
    assert "--days DAYS" in res.stdout
