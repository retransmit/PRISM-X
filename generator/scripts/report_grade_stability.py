#!/usr/bin/env python
"""Prints the per-run and per-gate view of stored benchmark_results.json files.

Why this exists
---------------
`spec/18` pins the evaluation protocol (slice-23), but a sweep run *outside* that
protocol -- varying `-n` or `--seed` -- produces violation counts that cannot be
compared with anything. This tool turns a directory of stored runs into the two
tables an honest claim needs:

1. one row per stored run (size, seed, violation count, streaming test days), so a
   quoted range such as "4 to 9 violations" is a computed min/max rather than a
   remembered one;
2. the per-gate fail counts across those runs, using the same gate grouping the
   on-protocol stability table uses (`benchmark_reporter.summarize_stability`), so
   "this gate always fails" and "this gate flips with the seed" are arithmetic.

Every number printed by `reports/certification_grade_stability.md` comes from here.

Usage
-----
    python scripts/report_grade_stability.py /tmp/grade-sweep/**/benchmark_results.json
    python scripts/report_grade_stability.py reports/benchmark/benchmark_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prism_generator.benchmark_reporter import summarize_stability  # noqa: E402


def _load_run(path: Path) -> Dict[str, Any]:
    """Reads one stored report and extracts only what the tables need."""
    with path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    metadata = report.get("metadata", {})
    run_protocol = metadata.get("run_protocol") or {}
    streaming = report.get("streaming", {})
    return {
        "path": str(path),
        "n_transactions": run_protocol.get("n_transactions") or metadata.get("n_transactions"),
        "seed": metadata.get("seed"),
        "grade": report.get("certification_grade", "UNKNOWN"),
        "violations": list(report.get("violations") or []),
        "n_days_evaluated": streaming.get("n_days_evaluated"),
        "mean_pr_auc": streaming.get("mean_pr_auc"),
    }


def run_table(runs: Sequence[Dict[str, Any]]) -> str:
    """One markdown row per stored run, in the order the files were given."""
    lines = [
        "| run | n | seed | violations | streaming test days | mean PR-AUC |",
        "|---|---|---|---|---|---|",
    ]
    for run in runs:
        days = "-" if run["n_days_evaluated"] is None else str(run["n_days_evaluated"])
        pr_auc = "-" if run["mean_pr_auc"] is None else f"{run['mean_pr_auc']:.4f}"
        lines.append(
            f"| `{run['path']}` "
            f"| {run['n_transactions']} "
            f"| {run['seed']} "
            f"| {len(run['violations'])} "
            f"| {days} "
            f"| {pr_auc} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "paths",
        nargs="+",
        metavar="BENCHMARK_RESULTS_JSON",
        help="stored benchmark_results.json files to summarise",
    )
    args = parser.parse_args(argv)

    missing = [p for p in args.paths if not Path(p).is_file()]
    if missing:
        print(f"not a file: {missing[0]}", file=sys.stderr)
        return 2

    runs = [_load_run(Path(p)) for p in args.paths]
    counts = sorted(len(r["violations"]) for r in runs)
    grades: Dict[str, int] = {}
    for run in runs:
        grades[run["grade"]] = grades.get(run["grade"], 0) + 1

    print(f"# {len(runs)} stored benchmark run(s)\n")
    print(f"violation counts: {counts[0]}-{counts[-1]} ({', '.join(str(c) for c in counts)})")
    print(f"grades: {', '.join(f'{g} x{c}' for g, c in sorted(grades.items()))}\n")
    print(run_table(runs))

    summary = summarize_stability(runs, "stored runs")
    print("\n## Per-gate failures across these runs\n")
    print("| gate | failed in | reading |")
    print("|---|---|---|")
    for gate, count in summary.gate_failures.items():
        reading = "always" if count == summary.runs else "FLIPS"
        print(f"| {gate} | {count}/{summary.runs} | {reading} |")
    print(
        f"\n{len(summary.unstable_gates)} of {len(summary.gate_failures)} failing gates "
        "flip between runs; quote them with the n and seed that produced them."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
