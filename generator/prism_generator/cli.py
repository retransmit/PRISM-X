"""Command-line interface for Prism Generator."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .engine import SimulationEngine
from .spec_loader import load_all_specs


from .storage import AUTH_STREAM_COLUMNS, GATEWAY_TELEMETRY_COLUMNS  # spec/19

THREAT_INTEL_GRAPH_COLUMNS = [
    "transaction_id",
    "card_id",
    "merchant_id",
    "syndicate_id",
    "botnet_cluster_id",
    "mule_ring_id",
    "beneficiary_account_id",
    "ip_subnet_prefix",
    "device_fingerprint_id",
    "mule_bank_id",
    "layering_hop_latency_seconds",
]



CLEARING_SETTLEMENT_COLUMNS = [
    "transaction_id",
    "clearing_mti",
    "clearing_delay_hours",
    "amount",
    "amount_minor",
    "settled_amount",
    "settled_amount_minor",
    "currency",
    "interchange_fee_minor",
    "mcc",
    "merchant_id",
    "mid",
    "acquirer_bin",
]

DISPUTE_RECOVERY_COLUMNS = [
    "transaction_id",
    "card_id",
    "amount",
    "currency",
    "is_fraud",
    "scenario_tag",
    "dispute_status",
    "dispute_reason_code",
    "ce3_qualified",
    "arbitration_fee_usd",
    "rbi_liability_tier",
    "rbi_provisional_credit_mandate_days",
    "cfcfrms_1930_lien_status",
]

DISPUTE_EXCLUDE_COLUMNS = {
    "dispute_status",
    "dispute_reason_code",
    "ce3_qualified",
    "arbitration_fee_usd",
    "rbi_liability_tier",
    "rbi_provisional_credit_mandate_days",
    "cfcfrms_1930_lien_status",
}


def _export_csv_view(file_path: Path, records: list[dict], fieldnames: list[str]) -> None:
    """Helper to export a projected institutional view table."""
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def cmd_generate(args: argparse.Namespace) -> None:
    """Generates synthetic transactions via discrete event engine or parallel coordinator."""
    if getattr(args, "parallel", False):
        if getattr(args, "export_institutional_views", False):
            print(
                "Error: --export-institutional-views is currently supported in single-file mode (run without --parallel).",
                file=sys.stderr,
            )
            sys.exit(2)
        if getattr(args, "calibration", None):
            print(
                "Error: --calibration reporting is currently supported in single-file mode (run without --parallel).",
                file=sys.stderr,
            )
            sys.exit(2)

        if getattr(args, "annual", False):
            from .storage import EntityPartitionedParallelCoordinator

            coordinator = EntityPartitionedParallelCoordinator(
                total_cards=args.cards,
                time_span_days=args.days,
                num_workers=args.workers,
                region=args.region,
                output_dir=args.output,
                chunk_size=args.chunk_size,
                adversary_mode=args.adversary_mode,
                annual_seasonality=True,
                p2p_enabled=getattr(args, "p2p", True),
                base_seed=args.seed,
            )
            summary = coordinator.run()
            print(
                f"Annual parallel simulation completed: {summary['total_transactions']:,} transactions written to {args.output}",
                file=sys.stderr,
            )
            return

        from .storage import ParallelSimulationCoordinator

        coordinator = ParallelSimulationCoordinator(
            total_transactions=args.n,
            num_workers=args.workers,
            region=args.region,
            output_dir=args.output,
            chunk_size=args.chunk_size,
            adversary_mode=args.adversary_mode,
            base_seed=args.seed,
        )
        summary = coordinator.run()
        print(
            f"Parallel simulation completed: {summary['total_transactions']:,} transactions written to {args.output}",
            file=sys.stderr,
        )
        return

    profile = None
    if getattr(args, "calibration", None):
        profile = load_all_specs().calibration_profiles[args.calibration]
        if profile.region != args.region.upper():
            print(
                f"Calibration profile {args.calibration} is for region {profile.region}; run with --region {profile.region}.",
                file=sys.stderr,
            )
            sys.exit(2)
    # The registry rate when calibrated, the historical default otherwise; an explicit --fraud-rate wins and is reported as a boost.
    fraud_rate = args.fraud_rate if args.fraud_rate is not None else (profile.fraud_prevalence if profile else 0.02)

    # Density and total are the same quantity written two ways: density = n / days. Pinning both -n
    # and --days therefore fixes density too, and it is density -- not the total -- that describes
    # "how many transactions happen in a day". --density exists so that can be chosen directly.
    if getattr(args, "density", None):
        args.n = max(1, int(round(args.density * args.days)))
        print(
            f"density {args.density:g} tx/day x {args.days} days = {args.n} transactions",
            file=sys.stderr,
        )

    engine = SimulationEngine(
        n_cards=args.cards,
        n_merchants=args.merchants,
        region=args.region,
        adversary_mode=args.adversary_mode,
        seed=args.seed,
    )
    print(
        f"Synthesizing {args.n} transactions ({args.region}, adversary_mode={args.adversary_mode}, fraud_rate={fraud_rate:.3g})...",
        file=sys.stderr,
    )
    records = engine.generate_batch(
        n_transactions=args.n,
        fraud_prevalence=fraud_rate,
        time_span_days=args.days,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Master output formatting
    export_records = records
    if not args.include_disputes:
        export_records = [
            {k: v for k, v in r.items() if k not in DISPUTE_EXCLUDE_COLUMNS}
            for r in records
        ]

    if out_path.suffix == ".json":
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(export_records, f, indent=2)
    elif out_path.suffix == ".jsonl":
        with open(out_path, "w", encoding="utf-8") as f:
            for r in export_records:
                f.write(json.dumps(r) + "\n")
    else:  # CSV default
        if export_records:
            all_keys = list({k: None for r in export_records for k in r.keys()}.keys())
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(export_records)

    print(f"Successfully wrote {len(export_records)} transactions to {out_path}", file=sys.stderr)

    # Report what was actually produced. Count, density and duration are one equation
    # (N = density x days, with density itself = per-card rate x active cards), and fraud
    # freezes cards, so the realised span can diverge from --days even though the pacing was
    # right. Printing it keeps the coupling visible instead of silent.
    if records:
        _stamps = [float(r.get("tx_time_seconds", 0.0)) for r in records]
        _span = (max(_stamps) - min(_stamps)) / 86400.0 if len(_stamps) > 1 else 0.0
        if _span > 0.0:
            _req_density = args.n / max(args.days, 1)
            _real_density = len(records) / _span
            _frozen = sum(1 for c in getattr(engine, "cards", []) if getattr(c, "is_frozen", False))
            print(
                f"density report: requested {args.days} d @ {_req_density:.1f} tx/day "
                f"({args.n:,} rows) | realised {_span:.2f} d @ {_real_density:.1f} tx/day "
                f"({len(records):,} rows)",
                file=sys.stderr,
            )
            if abs(_span - args.days) / max(args.days, 1) > 0.05:
                _n_cards = len(getattr(engine, "cards", []))
                if _span > args.days:
                    # More time needed than asked: the population lost generating capacity.
                    print(
                        f"  span EXCEEDS --days because fraud froze {_frozen}/{_n_cards} cards; "
                        f"active cards carry the spec'd per-card rate, so fewer of them take longer "
                        f"to deliver {args.n:,} rows (N = density x days, density = rate x active cards). "
                        f"Use --density to set tx/day directly, or --cards to change the population.",
                        file=sys.stderr,
                    )
                else:
                    # Finished early: too few transactions per card for the per-card first arrival
                    # to amortise, so the stream ends before the horizon rather than after it.
                    print(
                        f"  span is SHORT of --days: at {args.n:,} rows over {_n_cards} cards that is "
                        f"{args.n / max(_n_cards, 1):.1f} tx/card, and each card's first arrival is "
                        f"drawn from its own inter-arrival distribution, so the batch ends once the last "
                        f"card has fired rather than at the horizon. Raise -n (or --density) to fill "
                        f"{args.days} days at the spec'd per-card rate.",
                        file=sys.stderr,
                    )

    # Export partitioned institutional views if requested
    if args.export_institutional_views and records:
        parent_dir = out_path.parent
        stem = out_path.stem

        views = [
            (parent_dir / f"{stem}_auth_stream.csv", AUTH_STREAM_COLUMNS, "Authorization Feed (ISO 8583)"),
            (parent_dir / f"{stem}_gateway_telemetry.csv", GATEWAY_TELEMETRY_COLUMNS, "Gateway Risk Telemetry"),
            (parent_dir / f"{stem}_clearing_settlement.csv", CLEARING_SETTLEMENT_COLUMNS, "Clearing & Settlement Presentment"),
            (parent_dir / f"{stem}_dispute_recovery.csv", DISPUTE_RECOVERY_COLUMNS, "Dispute & Chargeback Recovery"),
            (parent_dir / f"{stem}_threat_intel_graph.csv", THREAT_INTEL_GRAPH_COLUMNS, "Threat Intelligence Graph Enclave"),
        ]

        for path, cols, label in views:
            _export_csv_view(path, records, cols)
            print(f"Exported {label} -> {path}", file=sys.stderr)

    if profile is not None and records:
        from .calibration import calibration_report, format_report

        report = calibration_report(records, profile_id=profile.id, requested_fraud_prevalence=fraud_rate)
        report_path = out_path.parent / f"{out_path.stem}_calibration.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(format_report(report), file=sys.stderr)
        print(f"Calibration report -> {report_path}", file=sys.stderr)


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Executes empirical XAI, Tripartite industrial, or Unified Four-Pillar benchmark."""
    import json
    from dataclasses import asdict
    from pathlib import Path

    if getattr(args, "unified", False):
        from .benchmark_reporter import (
            BenchmarkReportCompiler,
            PublicationPlotter,
            UnifiedBenchmarkRunner,
            load_evaluation_protocol,
        )

        tier = getattr(args, "protocol", None)
        norm_tier = ("protocol_full" if "full" in tier.lower() else "protocol_fast") if tier else None

        # Resolve tier defaults if protocol tier is specified
        samples = args.samples
        time_span_days = getattr(args, "days", None)
        w_train = getattr(args, "w_train", 3.0)
        w_test = getattr(args, "w_test", 1.0)
        delta_delay = getattr(args, "delta_delay", 1.5)
        k_daily = getattr(args, "k_daily", 15)

        if norm_tier == "protocol_full":
            if args.samples == 2000:
                samples = 100000
            if time_span_days is None:
                time_span_days = 75.0
            if w_train == 3.0:
                w_train = 30.0
            if delta_delay == 1.5:
                delta_delay = 14.0
            if k_daily == 15:
                k_daily = 50
        elif norm_tier == "protocol_fast":
            if time_span_days is None:
                time_span_days = 12.0
        else:
            if time_span_days is None:
                time_span_days = 12.0

        runner = UnifiedBenchmarkRunner(
            region=args.region,
            n_transactions=samples,
            time_span_days=time_span_days,
            k_daily=k_daily,
            w_train_days=w_train,
            w_test_days=w_test,
            delta_delay_days=delta_delay,
            seed=args.seed,
            protocol_tier=norm_tier,
        )
        print(f"Executing Unified Four-Pillar Benchmark on {samples} transactions ({args.region}) under {runner.protocol_tier}...", file=sys.stderr)
        report_data = runner.run_benchmark()

        mismatches = report_data.metadata.get("protocol_mismatches") or []
        if mismatches:
            # A deviating run is still graded -- it is useful for exploration -- but its
            # number of violations is not comparable to anything, so say so where the
            # reader is about to copy the number from.
            print("OFF-PROTOCOL: this run deviates from spec/18 evaluation_protocol,", file=sys.stderr)
            print("so its violation count is NOT comparable to the pinned certification:", file=sys.stderr)
            for line in mismatches:
                print(f"  - {line}", file=sys.stderr)
        else:
            # On-protocol: one seed's violation count is not a result. Re-run the pinned
            # seeds and record how many runs each gate actually failed in.
            protocol, protocol_source = load_evaluation_protocol(tier=runner.protocol_tier)
            seeds = list(protocol["stability_seeds"])
            requested_seeds = getattr(args, "stability_seeds", None)
            if requested_seeds is not None:
                seeds = seeds[: max(1, int(requested_seeds))]
            if len(seeds) > 1:
                print(
                    f"Measuring grade stability across {len(seeds)} pinned seeds "
                    f"({', '.join(str(s) for s in seeds)})...",
                    file=sys.stderr,
                )
                report_data.stability = runner.run_stability(seeds=seeds)

        out_dir = Path(getattr(args, "output_dir", "reports/benchmark"))
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = BenchmarkReportCompiler.compile_json(report_data, out_dir / "benchmark_results.json")
        md_path = BenchmarkReportCompiler.compile_markdown(report_data, out_dir / "BENCHMARK_REPORT.md")

        if getattr(args, "camera_ready", False):
            plotter = PublicationPlotter(style=getattr(args, "conference_style", "ieee"))
            fig_dir = out_dir / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)
            figure_paths = {}

            if report_data.streaming.n_days_evaluated > 0:
                from .evaluation import DailyStreamingMetrics
                daily_objs = [
                    DailyStreamingMetrics(
                        day_index=d["day_index"],
                        t_start_seconds=0.0,
                        t_end_seconds=86400.0,
                        n_transactions=d["n_tx"],
                        n_fraud_actual=d["n_fraud"],
                        pr_auc=d["pr_auc"],
                        average_precision=d["pr_auc"],
                        p_at_k=d["p_at_k"],
                        cp_at_k=d["cp_at_k"],
                        dollar_recall_at_k=d["dollar_recall_at_k"],
                        dollar_precision_at_k=0.05,
                        cost_base=100.0,
                        cost_model=90.0,
                        savings_ratio=d["savings_ratio"],
                        n_train_admissible=20,
                        drift_detected=d["drift_detected"],
                    )
                    for d in report_data.daily_trajectory
                ]
                figure_paths["fig1_prequential_timeseries"] = plotter.plot_prequential_timeseries(
                    daily_objs, delta_delay_days=report_data.streaming.delta_delay_days, output_path=fig_dir
                )

            k_vals = (report_data.triage_curves.get("k_values") if report_data.triage_curves else None) or [10, 25, 50, 100, 200]
            p_k = (report_data.triage_curves.get("p_at_k") if report_data.triage_curves else None) or [
                report_data.streaming.mean_p_at_k for _ in k_vals
            ]
            cp_k = (report_data.triage_curves.get("cp_at_k") if report_data.triage_curves else None) or [
                report_data.streaming.mean_cp_at_k for _ in k_vals
            ]
            dr_k = (report_data.triage_curves.get("dollar_recall_at_k") if report_data.triage_curves else None) or [
                report_data.streaming.mean_dollar_recall_at_k for _ in k_vals
            ]
            figure_paths["fig2_operational_triage_tradeoff"] = plotter.plot_operational_triage_tradeoff(
                k_vals, p_k, cp_k, dr_k, output_path=fig_dir
            )

            s_ratios = (report_data.triage_curves.get("savings_ratios") if report_data.triage_curves else None) or [
                report_data.streaming.overall_savings_ratio for _ in k_vals
            ]
            net_dollars = (report_data.triage_curves.get("net_savings_nominal") if report_data.triage_curves else None) or [
                s * report_data.streaming.total_cost_base for s in s_ratios
            ]
            figure_paths["fig3_financial_savings_utility"] = plotter.plot_financial_savings_utility(
                k_vals, s_ratios, net_dollars, currency=report_data.metadata["currency"], output_path=fig_dir
            )

            days = [d["day_index"] + 1 for d in report_data.daily_trajectory] or [1, 2]
            ks_p = [float(d["ks_drift_p_value"]) if d.get("ks_drift_p_value") is not None else 0.50 for d in report_data.daily_trajectory] or [0.50, 0.50]
            psi_s = [float(d["psi_score"]) if d.get("psi_score") is not None else 0.02 for d in report_data.daily_trajectory] or [0.02, 0.02]
            figure_paths["fig4_streaming_drift_timeline"] = plotter.plot_streaming_drift_timeline(
                days, ks_p, psi_s, output_path=fig_dir
            )

            feats = list(report_data.feature_attributions.keys())
            v_shap = list(report_data.feature_attributions.values())
            v_gt = list(report_data.ground_truth_phi.values())
            figure_paths["fig5_ground_truth_xai_comparison"] = plotter.plot_ground_truth_xai_comparison(
                feats, v_shap, v_gt,
                pearson_r=report_data.xai.mean_pearson_r,
                spearman_rho=report_data.xai.mean_spearman_rho,
                output_path=fig_dir,
            )

            html_path = BenchmarkReportCompiler.compile_html(report_data, figure_paths, out_dir / "benchmark_report.html")
            print(f"Standalone HTML Report saved: {html_path}", file=sys.stderr)

        if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
            try:
                sys.stdout.reconfigure(encoding="utf-8")
            except Exception:
                pass

        if args.json:
            print(Path(json_path).read_text(encoding="utf-8", errors="replace"))
        else:
            print("\n" + Path(md_path).read_text(encoding="utf-8", errors="replace") + "\n")
        print(f"Benchmark results saved to {out_dir}", file=sys.stderr)
        return

    if getattr(args, "tripartite", False):
        from .benchmark import TripartiteBenchmarkHarness, generate_tripartite_markdown_report

        harness = TripartiteBenchmarkHarness(
            n_transactions=args.samples,
            fraud_prevalence=args.fraud_rate,
            region=args.region,
            seed=args.seed,
        )
        print(f"Executing Tripartite Industrial Benchmark on {args.samples} transactions ({args.region})...", file=sys.stderr)
        summary = harness.run_tripartite_benchmark()

        if args.json:
            print(json.dumps(asdict(summary), indent=2))
        else:
            report = generate_tripartite_markdown_report(summary)
            print("\n" + report + "\n")

        if getattr(args, "output_report", None):
            report_content = generate_tripartite_markdown_report(summary)
            Path(args.output_report).write_text(report_content, encoding="utf-8")
            print(f"Report saved to {args.output_report}", file=sys.stderr)

    else:
        from .benchmark import XAIBenchmarkHarness

        harness = XAIBenchmarkHarness(
            n_transactions=args.samples,
            fraud_prevalence=args.fraud_rate,
            region=args.region,
            seed=args.seed,
        )
        print(f"Executing XAI Benchmark on {args.samples} transactions ({args.model})...", file=sys.stderr)
        summary = harness.run_benchmark(model_type=args.model)

        if args.json:
            print(json.dumps(asdict(summary), indent=2))
        else:
            print("\n" + "=" * 65)
            print("  PRISM GENERATOR EMPIRICAL XAI BENCHMARK RESULTS")
            print("=" * 65)
            print(f"  Model Architecture:           {summary.model_name.upper()}")
            print(f"  Explainer Method:             {summary.explainer_name}")
            print(f"  Evaluated Fraud Samples:      {summary.n_evaluated_samples}")
            print(f"  Classifier ROC-AUC:           {summary.auc_roc:.4f}")
            print(f"  Classifier PR-AUC:            {summary.pr_auc:.4f}")
            print("-" * 65)
            print(f"  Ranking Concordance (Kendall Tau):      {summary.mean_kendall_tau:.4f}")
            print(f"  Rank Correlation (Spearman Rho):        {summary.mean_spearman_rho:.4f}")
            print(f"  Top-3 Support Recovery (Precision@3):   {summary.mean_precision_at_3:.4f}")
            print(f"  Intervention Precision (P@3):           {summary.mean_intervention_precision_at_3:.4f}")
            print(f"  Intervention Recall (R@3):              {summary.mean_intervention_recall_at_3:.4f}")
            print(f"  Relative Attribution Error (RAE):       {summary.mean_relative_attribution_error:.4f}")
            print("=" * 65 + "\n")


def cmd_visualize(args: argparse.Namespace) -> None:
    """Generates synthetic batch and compiles it into an interactive standalone HTML visualizer."""
    if getattr(args, "input", None):
        from .visualizer import generate_visualization_from_dir

        out_path = Path(args.output)
        print(
            f"Loading scale simulation from {args.input} (max_nodes={args.max_nodes})...",
            file=sys.stderr,
        )
        res_path = generate_visualization_from_dir(
            input_dir=args.input,
            output_path=str(out_path),
            max_nodes=args.max_nodes,
            open_browser=args.open,
        )
        print(f"Scale visualizer dashboard generated: {res_path.resolve()}", file=sys.stderr)
        return

    from .visualizer import generate_visualization_file

    out_path = Path(args.output)
    print(
        f"Synthesizing {args.n} transactions ({args.region}, adversary_mode={args.adversary_mode})...",
        file=sys.stderr,
    )
    res_path = generate_visualization_file(
        output_path=str(out_path),
        n_transactions=args.n,
        region=args.region,
        fraud_rate=args.fraud_rate,
        adversary_mode=args.adversary_mode,
        seed=args.seed,
        open_browser=args.open,
    )
    print(f"Interactive visualizer dashboard generated: {res_path.resolve()}", file=sys.stderr)


def cmd_report(args: argparse.Namespace) -> None:
    """Compiles publication reports from benchmark results JSON."""
    import json
    from pathlib import Path
    from .benchmark_reporter import (
        AdversarialPrivacyScorecard,
        BenchmarkReportCompiler,
        CausalXAIScorecard,
        DataFidelityScorecard,
        OperationalStreamingScorecard,
        PublicationPlotter,
        UnifiedBenchmarkReportData,
    )

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input benchmark results file not found: {in_path}")

    with open(in_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    # Reconstruct report object
    report_data = UnifiedBenchmarkReportData(
        schema_version=d.get("schema_version", "1.0.0"),
        metadata=d.get("metadata", {}),
        system_provenance=d.get("system_provenance", {}),
        fidelity=DataFidelityScorecard(**d["fidelity"]),
        privacy=AdversarialPrivacyScorecard(**d["privacy"]),
        streaming=OperationalStreamingScorecard(**d["streaming"]),
        xai=CausalXAIScorecard(**d["xai"]),
        all_pillars_passed=d.get("all_pillars_passed", False),
        certification_grade=d.get("certification_grade", "NON_CERTIFIED_FAIL"),
        violations=d.get("violations", []),
        daily_trajectory=d.get("daily_trajectory", []),
        feature_attributions=d.get("feature_attributions", {}),
        ground_truth_phi=d.get("ground_truth_phi", {}),
        triage_curves=d.get("triage_curves", {}),
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = [fmt.strip().lower() for fmt in args.format.split(",")]

    plotter = PublicationPlotter(style=args.conference_style)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = {}

    feats = list(report_data.feature_attributions.keys()) or ["amount", "velocity"]
    v_shap = list(report_data.feature_attributions.values()) or [0.6, 0.4]
    v_gt = list(report_data.ground_truth_phi.values()) or [0.55, 0.45]
    figure_paths["fig5_ground_truth_xai_comparison"] = plotter.plot_ground_truth_xai_comparison(
        feats, v_shap, v_gt,
        pearson_r=report_data.xai.mean_pearson_r,
        spearman_rho=report_data.xai.mean_spearman_rho,
        output_path=fig_dir,
    )

    k_vals = (report_data.triage_curves.get("k_values") if report_data.triage_curves else None) or [10, 25, 50, 100, 200]
    p_k = (report_data.triage_curves.get("p_at_k") if report_data.triage_curves else None) or [
        report_data.streaming.mean_p_at_k for _ in k_vals
    ]
    cp_k = (report_data.triage_curves.get("cp_at_k") if report_data.triage_curves else None) or [
        report_data.streaming.mean_cp_at_k for _ in k_vals
    ]
    dr_k = (report_data.triage_curves.get("dollar_recall_at_k") if report_data.triage_curves else None) or [
        report_data.streaming.mean_dollar_recall_at_k for _ in k_vals
    ]
    figure_paths["fig2_operational_triage_tradeoff"] = plotter.plot_operational_triage_tradeoff(
        k_vals, p_k, cp_k, dr_k, output_path=fig_dir
    )

    s_ratios = (report_data.triage_curves.get("savings_ratios") if report_data.triage_curves else None) or [
        report_data.streaming.overall_savings_ratio for _ in k_vals
    ]
    net_dollars = (report_data.triage_curves.get("net_savings_nominal") if report_data.triage_curves else None) or [
        s * report_data.streaming.total_cost_base for s in s_ratios
    ]
    figure_paths["fig3_financial_savings_utility"] = plotter.plot_financial_savings_utility(
        k_vals, s_ratios, net_dollars, currency=report_data.metadata.get("currency", "USD"), output_path=fig_dir
    )

    days = [d["day_index"] + 1 for d in report_data.daily_trajectory] or [1, 2]
    ks_p = [float(d["ks_drift_p_value"]) if d.get("ks_drift_p_value") is not None else 0.50 for d in report_data.daily_trajectory] or [0.50, 0.50]
    psi_s = [float(d["psi_score"]) if d.get("psi_score") is not None else 0.02 for d in report_data.daily_trajectory] or [0.02, 0.02]
    figure_paths["fig4_streaming_drift_timeline"] = plotter.plot_streaming_drift_timeline(
        days, ks_p, psi_s, output_path=fig_dir
    )

    if "json" in formats:
        BenchmarkReportCompiler.compile_json(report_data, out_dir / "benchmark_results.json")
    if "md" in formats:
        BenchmarkReportCompiler.compile_markdown(report_data, out_dir / "BENCHMARK_REPORT.md")
    if "html" in formats:
        BenchmarkReportCompiler.compile_html(report_data, figure_paths, out_dir / "benchmark_report.html")

    print(f"Publication reports compiled successfully to {out_dir}", file=sys.stderr)


def cmd_validate(args: argparse.Namespace) -> None:
    """Validates synthetic data distributional quality, diversity, and class overlap."""
    from .quality_report import QualityReport
    from .engine import DiscreteEventEngine

    if args.input:
        import csv
        with open(args.input, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            records = list(reader)
    else:
        print(f"Synthesizing {args.samples} transactions ({args.region}) for quality validation...", file=sys.stderr)
        engine = DiscreteEventEngine(
            n_cards=max(80, int(args.samples / 15)),
            n_merchants=max(25, int(args.samples / 50)),
            region=args.region,
            seed=args.seed,
        )
        records = engine.generate_batch(
            n_transactions=args.samples,
            fraud_prevalence=args.fraud_rate,
        )

    summary = QualityReport.generate_quality_summary(
        records=records,
        n_pairs=args.pairs,
        seed=args.seed,
    )

    if args.json:
        from dataclasses import asdict
        print(json.dumps(asdict(summary), indent=2))
    else:
        report_md = summary.format_markdown()
        print("\n" + report_md + "\n")

    if args.output_report:
        out_p = Path(args.output_report)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(summary.format_markdown(), encoding="utf-8")
        print(f"Quality report saved to {out_p}", file=sys.stderr)

    if not summary.overall_passed:
        sys.exit(1)


def cmd_gui(args: argparse.Namespace) -> None:
    """Launch the Prism Generator Studio interactive GUI web server."""
    try:
        import uvicorn
    except ImportError:
        print(
            "Error: GUI dependencies not installed. Install with:\n"
            "  pip install -e \".[gui]\"\n"
            "or:\n"
            "  pip install fastapi uvicorn",
            file=sys.stderr,
        )
        sys.exit(1)

    import threading
    import webbrowser

    url = f"http://{args.host}:{args.port}"
    print("\n=======================================================")
    print(f"  Starting Prism Generator Studio GUI at {url}")
    print(f"  API Docs available at {url}/docs")
    print("=======================================================\n")

    if args.open:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    uvicorn.run("gui.backend.app:app", host=args.host, port=args.port, reload=args.reload)


def main() -> None:
    parser = argparse.ArgumentParser(prog="prism-generate", description="Prism Generator CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # GUI subcommand
    p_gui = subparsers.add_parser("gui", help="Launch the Prism Generator Studio interactive web GUI")
    p_gui.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    p_gui.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    p_gui.add_argument("--reload", action="store_true", default=False, help="Enable auto-reload for development")
    p_gui.add_argument("--open", action="store_true", default=False, help="Open default web browser automatically")
    p_gui.set_defaults(func=cmd_gui)

    # Generate subcommand
    p_gen = subparsers.add_parser("generate", help="Generate batch transactions")
    p_gen.add_argument("-n", type=int, default=5000, help="Number of transactions to synthesize")
    p_gen.add_argument("-c", "--cards", type=int, default=1000, help="Number of simulated cardholders")
    p_gen.add_argument("-m", "--merchants", type=int, default=150, help="Number of simulated merchants")
    p_gen.add_argument("--region", type=str, choices=["US", "IN"], default="US", help="Geographic banking ecosystem: US (USD dual-message) or IN (INR RBI AFA/RuPay/CoFT)")
    p_gen.add_argument("--fraud-rate", type=float, default=None, help="Fraud prevalence ratio (default 0.02; with --calibration, the registry rate of the profile, and an explicit value is reported as a demo boost)")
    p_gen.add_argument("--days", type=int, default=30, help="Simulation horizon in days. Sets the PACING RATE alongside -n (density = n/days); the realised span can differ because fraud freezes cards -- see the density report printed after generation")
    p_gen.add_argument("--density", type=float, default=None, help="Transactions per day to simulate. Overrides -n (n = density x --days). This is the knob for 'how many happen in a day': -n alone cannot express density, only the total")
    p_gen.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    p_gen.add_argument("-o", "--output", type=str, default="synthetic_transactions.csv", help="Output file (.csv, .json, .jsonl) or directory for parallel runs")
    p_gen.add_argument("--parallel", action="store_true", default=False, help="Run multi-core parallel simulation pipeline")
    p_gen.add_argument("--annual", action="store_true", default=False, help="Enable 365-day annual seasonality, holiday calendars, and macro drift dynamics")
    p_gen.add_argument("--p2p", action="store_true", default=True, help="Include peer-to-peer transfers (UPI P2P, Zelle push) in transaction stream")
    p_gen.add_argument("--workers", type=int, default=None, help="Number of parallel worker processes (defaults to CPU count - 2)")
    p_gen.add_argument("--chunk-size", type=int, default=10000, help="Batch chunk size for streaming disk flushes")
    p_gen.add_argument("--include-disputes", action="store_true", default=False, help="Include post-authorization dispute and chargeback lifecycle columns in master output")
    p_gen.add_argument("--adversary-mode", type=str, choices=["intent", "playbook"], default="intent", help="Adversary decision architecture: 'intent' (first-principles POMDP/IDS) or 'playbook' (legacy static rules)")
    p_gen.add_argument("--export-institutional-views", action="store_true", default=False, help="Export partitioned institutional banking warehouse feeds")
    p_gen.add_argument("--calibration", type=str, choices=sorted(load_all_specs().calibration_profiles), default=None, help="Run at the fraud prevalence of a published calibration profile and write <output>_calibration.json comparing the batch with its targets (spec/08)")
    p_gen.set_defaults(func=cmd_generate)

    # Benchmark subcommand
    p_bench = subparsers.add_parser("benchmark", help="Run empirical XAI, Tripartite, or Unified benchmark")
    p_bench.add_argument("-n", "--samples", type=int, default=2000, help="Number of synthetic transactions")
    p_bench.add_argument("--model", type=str, choices=["lightgbm", "rf"], default="lightgbm", help="ML model architecture")
    p_bench.add_argument("--region", type=str, choices=["US", "IN"], default="US", help="Banking ecosystem region")
    p_bench.add_argument("--fraud-rate", type=float, default=0.05, help="Fraud prevalence ratio")
    p_bench.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    p_bench.add_argument("--tripartite", action="store_true", default=False, help="Run complete Tripartite Industrial Benchmark Suite")
    p_bench.add_argument("--unified", action="store_true", default=False, help="Run comprehensive Four-Pillar Unified Industrial Benchmark")
    p_bench.add_argument("--camera-ready", action="store_true", default=False, help="Generate camera-ready publication figures (PNG, PDF)")
    p_bench.add_argument("--output-dir", type=str, default="reports/benchmark", help="Directory for compiled benchmark artifacts and figures")
    p_bench.add_argument("--conference-style", type=str, choices=["ieee", "acm", "neurips"], default="ieee", help="Publication figure styling preset")
    p_bench.add_argument("--w-train", type=float, default=3.0, help="Training history window in days")
    p_bench.add_argument("--w-test", type=float, default=1.0, help="Streaming test window in days")
    p_bench.add_argument("--delta-delay", type=float, default=1.5, help="Chargeback delay blackout window in days")
    p_bench.add_argument("--k-daily", type=int, default=15, help="Daily analyst investigation capacity budget")
    p_bench.add_argument("--output-report", type=str, default=None, help="Path to write Markdown certification report")
    p_bench.add_argument("--json", action="store_true", help="Output benchmark metrics in JSON format")
    p_bench.add_argument(
        "--protocol",
        type=str,
        choices=["fast", "full"],
        default=None,
        help="Evaluation protocol tier: 'fast' (12d stream, CI/CD) or 'full' (75d stream, Dal Pozzolo 2018 camera-ready publications)",
    )
    p_bench.add_argument(
        "--days",
        type=float,
        default=None,
        help="Simulation stream horizon in days (defaults to protocol tier requirement: 12d for fast, 75d for full)",
    )
    p_bench.add_argument(
        "--stability-seeds",
        type=int,
        default=None,
        metavar="K",
        help="K of spec/18 evaluation_protocol.stability_seeds to repeat the run over for the "
        "grade-stability table (default: all of them; 1 reports a single run; ignored when "
        "the run is OFF-PROTOCOL)",
    )
    p_bench.set_defaults(func=cmd_benchmark)

    # Report subcommand
    p_rep = subparsers.add_parser("report", help="Compile publication reports from benchmark results JSON")
    p_rep.add_argument("-i", "--input", type=str, required=True, help="Input benchmark_results.json file")
    p_rep.add_argument("-o", "--output-dir", type=str, default="reports/compiled", help="Output directory")
    p_rep.add_argument("--format", type=str, default="json,md,html", help="Comma-separated report formats (json, md, html)")
    p_rep.add_argument("--conference-style", type=str, choices=["ieee", "acm", "neurips"], default="ieee", help="Publication figure styling preset")
    p_rep.set_defaults(func=cmd_report)

    # Visualize subcommand
    p_vis = subparsers.add_parser("visualize", help="Generate and render interactive Cybercrime Threat Graph and Switch Funnel dashboard")
    p_vis.add_argument("-n", type=int, default=1500, help="Number of transactions to synthesize")
    p_vis.add_argument("-i", "--input", type=str, default=None, help="Path to scale simulation directory to visualize")
    p_vis.add_argument("--max-nodes", type=int, default=5000, help="Maximum number of entity nodes in interactive threat graph")
    p_vis.add_argument("--region", type=str, choices=["US", "IN"], default="US", help="Banking ecosystem region")
    p_vis.add_argument("--fraud-rate", type=float, default=0.04, help="Fraud prevalence ratio")
    p_vis.add_argument("--adversary-mode", type=str, choices=["intent", "playbook"], default="intent", help="Adversary decision mode: 'intent' or 'playbook'")
    p_vis.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    p_vis.add_argument("-o", "--output", type=str, default="reports/prism_generator_visualizer.html", help="Output HTML file path")
    p_vis.add_argument("--open", action="store_true", default=False, help="Automatically open generated visualizer in default web browser")
    p_vis.set_defaults(func=cmd_visualize)

    # Validate subcommand
    p_val = subparsers.add_parser("validate", help="Validate synthetic data distributional quality, diversity, and class overlap")
    p_val.add_argument("-i", "--input", type=str, default=None, help="Path to input CSV to validate (if omitted, synthesizes a test batch)")
    p_val.add_argument("-n", "--samples", type=int, default=2000, help="Number of transactions to synthesize if no input file is provided")
    p_val.add_argument("--region", type=str, choices=["US", "IN"], default="US", help="Banking ecosystem region")
    p_val.add_argument("--fraud-rate", type=float, default=0.05, help="Fraud prevalence ratio")
    p_val.add_argument("--pairs", type=int, default=1000, help="Number of random pairs for internal diversity estimation")
    p_val.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    p_val.add_argument("--output-report", type=str, default=None, help="Optional path to write Markdown quality report")
    p_val.add_argument("--json", action="store_true", help="Output summary in JSON format")
    p_val.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
