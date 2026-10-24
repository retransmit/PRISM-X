#!/usr/bin/env python3
"""Score an external transaction corpus against the certification grader's Pillar 1.

Why this exists
---------------
Pillar 1 grades our synthetic stream against reference distributions taken from
payment-rail sources (Federal Reserve Payments Study / DCPC -- see
``spec/09_us_calibration_targets.yaml``). A failing grade has two possible causes
and the report cannot tell them apart:

  (a) our generator is off, or
  (b) the reference distribution does not describe real card traffic either.

This script runs the *same function the grader runs*
(``UnifiedBenchmarkRunner._evaluate_data_fidelity``) on a real corpus, so both are
measured by identical code against identical thresholds loaded from
``spec/18_benchmark_reporting.yaml``. If real data passes a threshold we fail, the
generator is at fault; if real data fails it too, the reference or the threshold is.

Comparability
-------------
The grader reads a missing ``mcc`` as 5411 for every row, a missing
``pos_entry_mode`` as "01" for every row, and the five correlation columns as 0.0 --
each of which makes its metric trivially pass. A corpus that does not carry those
columns must therefore not be graded on them. Only metrics whose inputs the corpus
actually supplies enter the verdict; the rest are printed as ``n/a`` and named
explicitly in the output.

Usage
-----
    python scripts/score_external_corpus.py \
        --synthetic 1500 \
        --ulb /path/to/creditcard.arff \
        --csv paysim.csv --amount-col amount --time-col step --time-unit hours \
             --label-col isFraud \
        --json /path/out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import scipy.stats  # noqa: E402

from prism_generator.benchmark_reporter import (  # noqa: E402
    UnifiedBenchmarkRunner,
    load_certification_thresholds,
)

# The five columns Pillar 1 correlates. A corpus carrying all five is graded on
# spearman_frobenius_error; anything short of all five is not.
CORRELATION_COLUMNS: Tuple[str, ...] = (
    "amount",
    "tx_count_1h",
    "tx_count_24h",
    "haversine_velocity_kph",
    "ip_distance_from_home_km",
)

# metric name -> the record keys it needs to be graded honestly
METRIC_INPUTS: Dict[str, Tuple[str, ...]] = {
    "wasserstein_amount_log": ("amount",),
    "wasserstein_arrival_log": ("tx_time_seconds",),
    "js_divergence_mcc": ("mcc",),
    "js_divergence_channel": ("pos_entry_mode",),
    "spearman_frobenius_error": CORRELATION_COLUMNS,
}

METRIC_LABELS: Dict[str, str] = {
    "wasserstein_amount_log": "log10(1+amount) Wasserstein",
    "wasserstein_arrival_log": "log10(1+inter-arrival s) Wasserstein",
    "js_divergence_mcc": "MCC Jensen-Shannon",
    "js_divergence_channel": "channel Jensen-Shannon",
    "spearman_frobenius_error": "correlation Frobenius error",
}


# ------------------------------------------------------------------------------
# loaders
# ------------------------------------------------------------------------------
def _arrival_resolution(times: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    """Smallest and median positive inter-arrival, in seconds.

    Printed because the inter-arrival gate is meaningless without it: a corpus
    timestamped to the hour (BankSim's ``step``) has a floor of 3600 s and can
    never look like a sub-second arrival process, whatever its arrival shape.
    """
    t = np.sort(np.asarray(times, dtype=np.float64))
    deltas = np.diff(t)
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        return None, None
    return float(deltas.min()), float(np.median(deltas))


def load_ulb_arff(path: Path, legit_only: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """The ULB credit-card corpus (OpenML data/1597 -> file 1673544).

    Columns used: ``Time`` (seconds since the first transaction),
    ``Amount`` (EUR -- see the currency note printed with the result),
    ``Class`` (1 = fraud).

    ``legit_only`` drops the fraud rows: the amount reference describes ordinary
    card traffic, and ULB is fraud-enriched (0.17% against ~0.1% of real card
    volume), so scoring the two populations together confounds the comparison.
    """
    from scipy.io import arff

    data, meta = arff.loadarff(str(path))
    names = [str(n).lower() for n in meta.names()]
    if not {"time", "amount", "class"} <= set(names):
        raise ValueError(f"{path} is not the ULB layout: columns are {names[:6]} ...")

    times = np.asarray(data["Time"], dtype=np.float64)
    amounts = np.asarray(data["Amount"], dtype=np.float64)
    labels = np.asarray(data["Class"], dtype=np.float64)

    keep = labels == 0 if legit_only else np.ones_like(labels, dtype=bool)
    times, amounts, labels = times[keep], amounts[keep], labels[keep]

    records = [
        {"amount": float(a), "tx_time_seconds": float(t)}
        for a, t in zip(amounts, times)
    ]
    d_min, d_med = _arrival_resolution(times)
    summary = {
        "n": len(records),
        "fraud_rate": float(labels.mean()) if labels.size else None,
        "amount_unit": "EUR",
        "time_span_days": float((times.max() - times.min()) / 86400.0) if len(times) else 0.0,
        "amount_mean": float(amounts.mean()) if amounts.size else 0.0,
        "amount_median": float(np.median(amounts)) if amounts.size else 0.0,
        "delta_min_seconds": d_min,
        "delta_median_seconds": d_med,
    }
    return records, summary


def load_csv(
    path: Path,
    amount_col: str,
    time_col: str,
    time_unit: str = "seconds",
    label_col: Optional[str] = None,
    mcc_col: Optional[str] = None,
    channel_col: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """A generic corpus: the caller names the columns, nothing is guessed.

    ``time_unit`` converts ``time_col`` into the seconds the grader expects
    (PaySim and BankSim record ``step`` in hours).
    """
    import csv

    seconds_per_unit = {"seconds": 1.0, "minutes": 60.0, "hours": 3600.0}[time_unit]
    records: List[Dict[str, Any]] = []
    labels: List[float] = []
    times: List[float] = []
    amounts: List[float] = []

    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        needed = [amount_col, time_col] + [c for c in (label_col, mcc_col, channel_col) if c]
        missing = [c for c in needed if c not in header]
        if missing:
            raise ValueError(f"{path}: columns {missing} not in header {header[:12]} ...")
        for row in reader:
            t = float(row[time_col]) * seconds_per_unit
            a = float(row[amount_col])
            rec: Dict[str, Any] = {"amount": a, "tx_time_seconds": t}
            if mcc_col:
                rec["mcc"] = float(row[mcc_col])
            if channel_col:
                rec["pos_entry_mode"] = str(row[channel_col])
            records.append(rec)
            times.append(t)
            amounts.append(a)
            if label_col:
                labels.append(float(row[label_col]))

    t_arr = np.asarray(times, dtype=np.float64)
    a_arr = np.asarray(amounts, dtype=np.float64)
    d_min, d_med = _arrival_resolution(t_arr)
    summary = {
        "n": len(records),
        "fraud_rate": float(np.mean(labels)) if labels else None,
        "amount_unit": "unlabelled",
        "time_span_days": float((t_arr.max() - t_arr.min()) / 86400.0) if len(t_arr) else 0.0,
        "amount_mean": float(a_arr.mean()) if len(a_arr) else 0.0,
        "amount_median": float(np.median(a_arr)) if len(a_arr) else 0.0,
        "delta_min_seconds": d_min,
        "delta_median_seconds": d_med,
    }
    return records, summary


def reference_control(n: int, region: str, seed: int, span_days: float = 12.0) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Negative control: draw amounts and inter-arrivals straight from the reference.

    This is not a generator of anything. It exists to show what Pillar 1's two
    continuous gates actually measure -- distance to the reference distribution
    the reporter hardcodes. A stream that *is* the reference must score ~0 and
    pass, whatever real card traffic looks like.
    """
    rng = np.random.default_rng(seed)
    if region == "US":
        amounts = rng.lognormal(mean=3.75, sigma=0.85, size=n)
    else:
        is_cc = rng.random(n) < 0.45
        amounts = np.where(
            is_cc,
            rng.lognormal(mean=7.74, sigma=0.85, size=n),
            rng.lognormal(mean=6.45, sigma=0.95, size=n),
        )
    mean_delta = (span_days * 86400.0) / n
    deltas = rng.exponential(scale=mean_delta, size=n)
    times = np.cumsum(deltas)

    records = [
        {"amount": float(a), "tx_time_seconds": float(t)}
        for a, t in zip(amounts, times)
    ]
    d_min, d_med = _arrival_resolution(times)
    summary = {
        "n": n,
        "fraud_rate": None,
        "amount_unit": "USD" if region == "US" else "INR",
        "time_span_days": span_days,
        "amount_mean": float(amounts.mean()),
        "amount_median": float(np.median(amounts)),
        "delta_min_seconds": d_min,
        "delta_median_seconds": d_med,
    }
    return records, summary


def generate_synthetic(n: int, region: str, seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from prism_generator.engine import SimulationEngine

    engine = SimulationEngine(n_cards=80, n_merchants=25, region=region, seed=seed)
    records = engine.generate_batch(
        n_transactions=n, fraud_prevalence=0.07, time_span_days=12
    )
    times = np.array([float(r["tx_time_seconds"]) for r in records], dtype=np.float64)
    amounts = np.array([float(r["amount"]) for r in records], dtype=np.float64)
    labels = np.array([1.0 if r.get("is_fraud") else 0.0 for r in records])
    d_min, d_med = _arrival_resolution(times)
    summary = {
        "n": len(records),
        "fraud_rate": float(labels.mean()),
        "amount_unit": "USD",
        "time_span_days": float((times.max() - times.min()) / 86400.0),
        "amount_mean": float(amounts.mean()),
        "amount_median": float(np.median(amounts)),
        "delta_min_seconds": d_min,
        "delta_median_seconds": d_med,
    }
    return records, summary


# ------------------------------------------------------------------------------
# scoring -- identical code path to the certification grader
# ------------------------------------------------------------------------------
def score(
    label: str,
    records: List[Dict[str, Any]],
    summary: Dict[str, Any],
    region: str,
    seed: int,
) -> Dict[str, Any]:
    """Run the grader's own Pillar 1 function on ``records``.

    ``time_span_days`` is taken from the corpus itself, because the inter-arrival
    reference is an exponential whose mean is ``span / n`` -- using the default
    span would grade this corpus against another corpus's arrival rate.
    """
    thresholds, threshold_source = load_certification_thresholds()
    th = thresholds["pillar_1_data_fidelity"]

    runner = UnifiedBenchmarkRunner(
        region=region,
        n_transactions=len(records),
        time_span_days=max(summary.get("time_span_days") or 0.0, 1e-9),
        seed=seed,
    )
    # A corpus that does not carry all five correlation columns leaves constant
    # columns here, which scipy warns about; that metric is excluded from the
    # verdict below anyway, so the warning would only obscure the real output.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=scipy.stats.ConstantInputWarning)
        card = runner._evaluate_data_fidelity(records)

    values = {
        "wasserstein_amount_log": float(card.wasserstein_amount_log),
        "wasserstein_arrival_log": float(card.wasserstein_arrival_log),
        "js_divergence_mcc": float(card.js_divergence_mcc),
        "js_divergence_channel": float(card.js_divergence_channel),
        "spearman_frobenius_error": float(card.spearman_frobenius_error),
    }
    # The two continuous gates are graded against the limit the grader itself used
    # for THIS record set -- since 2026-10-15 (spec/18 `pillar_1_real_reference`)
    # that is the size-corrected real-to-real bootstrap quantile, not the static
    # spec value. The three remaining gates keep their static spec limits.
    limits = {
        "wasserstein_amount_log": ("max", float(card.amount_limit)),
        "wasserstein_arrival_log": ("max", float(card.arrival_limit)),
        "js_divergence_mcc": ("max", float(th["js_divergence_mcc_max"])),
        "js_divergence_channel": ("max", float(th["js_divergence_channel_max"])),
        "spearman_frobenius_error": ("max", float(th["spearman_frobenius_error_max"])),
    }

    rows: List[Dict[str, Any]] = []
    comparable_failed = 0
    for metric, (direction, limit) in limits.items():
        inputs = METRIC_INPUTS[metric]
        present = all(k in records[0] for k in inputs) if records else False
        value = values[metric]
        ok = (value <= limit) if direction == "max" else (value >= limit)
        if present and not ok:
            comparable_failed += 1
        rows.append(
            {
                "metric": metric,
                "label": METRIC_LABELS[metric],
                "value": round(value, 4),
                "threshold": limit,
                "direction": direction,
                "comparable": present,
                "missing_inputs": [] if present else [k for k in inputs if k not in (records[0] if records else {})],
                "pass": bool(ok) if present else None,
            }
        )

    # Where the bar comes from, spelled out: the grader resolves the reference
    # from spec/18 (`pillar_1_real_reference`) and hands back the exact note, so
    # a reported number always names its corpus, sample size and calibration.
    reference_note = card.reference_note

    caveats: List[str] = []
    # Branch on the explicit provenance flag, never the numeric limit: under the
    # fallback the limit is the non-zero static spec value (0.15), so a
    # `card.amount_limit == 0.0` test would never fire here.
    if region == "US" and card.reference_kind == "embedded_fallback":
        caveats.append(
            "the real-corpus reference did not load (missing file, spec section or "
            "sha256 mismatch), so the continuous gates fell back to the embedded "
            "hardcoded reference"
        )
    elif region == "US" and summary.get("amount_unit") not in (None, "USD"):
        caveats.append(
            f"corpus amounts are {summary['amount_unit']} while the active reference "
            "corpus is USD-denominated; parity is assumed, not verified"
        )

    return {
        "label": label,
        "region": region,
        "seed": seed,
        "threshold_source": threshold_source,
        "reference_note": reference_note,
        "caveats": caveats,
        "summary": summary,
        "metrics": rows,
        "comparable_metrics": sum(1 for r in rows if r["comparable"]),
        "comparable_failures": comparable_failed,
        "verdict": "PASS" if comparable_failed == 0 else "FAIL",
    }


# ------------------------------------------------------------------------------
# reporting
# ------------------------------------------------------------------------------
def render(results: Sequence[Dict[str, Any]]) -> str:
    if not results:
        return "no corpora scored"
    out: List[str] = []
    out.append("Pillar 1 (data fidelity) -- same function, same thresholds, different data")
    out.append(f"threshold source: {results[0]['threshold_source']}")
    out.append(f"reference model : {results[0]['reference_note']}")
    for res in results:
        for caveat in res["caveats"]:
            out.append(f"caveat ({res['label']}): {caveat}")
    out.append("")

    metric_order = list(METRIC_LABELS)
    header = f"{'metric':<34}{'limit':>9}" + "".join(f"{r['label'][:30]:>32}" for r in results)
    out.append(header)
    out.append("-" * len(header))
    for metric in metric_order:
        row0 = next(r for r in results[0]["metrics"] if r["metric"] == metric)
        line = f"{METRIC_LABELS[metric]:<34}{row0['threshold']:>9.3f}"
        for res in results:
            row = next(r for r in res["metrics"] if r["metric"] == metric)
            if not row["comparable"]:
                cell = "n/a"
            else:
                cell = f"{row['value']:.4f} {'PASS' if row['pass'] else 'FAIL'}"
            line += f"{cell:>32}"
        out.append(line)

    out.append("-" * len(header))
    for res in results:
        s = res["summary"]
        rate = f", fraud rate {s['fraud_rate']:.4%}" if s.get("fraud_rate") is not None else ""
        d_min = s.get("delta_min_seconds")
        d_med = s.get("delta_median_seconds")
        resolution = (
            f", dt_min={d_min:g} s, dt_med={d_med:g} s"
            if d_min is not None and d_med is not None
            else ", dt=n/a"
        )
        out.append(
            f"{res['label']:<34}{res['verdict']:>9}  n={s['n']:,}, span={s['time_span_days']:.2f} d, "
            f"median amount={s['amount_median']:.2f} {s['amount_unit']}{resolution}, "
            f"graded on {res['comparable_metrics']}/5 metrics{rate}"
        )
    out.append("")
    out.append(
        "n/a = the corpus does not carry that metric's inputs; it is excluded from the verdict "
        "rather than scored on the grader's per-row defaults."
    )
    out.append(
        "dt_min = smallest positive inter-arrival in the corpus. A dt_min of 3600 s means the "
        "corpus is timestamped to the hour, so its inter-arrival gate measures clock resolution "
        "as much as arrival shape."
    )
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synthetic", type=int, action="append", metavar="N",
                   help="generate N transactions from this repo's engine (repeatable: sample-size sensitivity)")
    p.add_argument("--ulb", type=Path, action="append", metavar="ARFF",
                   help="ULB credit-card corpus, OpenML file 1673544 (repeatable)")
    p.add_argument("--ulb-legit", type=Path, action="append", metavar="ARFF",
                   help="the same corpus with its fraud rows dropped (the amount reference describes normal traffic)")
    p.add_argument("--reference-control", type=int, action="append", metavar="N",
                   help="negative control: N draws straight from the reference (repeatable)")
    p.add_argument("--csv", type=Path, help="generic corpus")
    p.add_argument("--amount-col", help="CSV column holding the amount")
    p.add_argument("--time-col", help="CSV column holding the event time")
    p.add_argument("--time-unit", choices=["seconds", "minutes", "hours"], default="seconds")
    p.add_argument("--label-col", help="CSV column holding the fraud label")
    p.add_argument("--mcc-col", help="CSV column holding genuine MCC codes")
    p.add_argument("--channel-col", help="CSV column holding POS entry mode / channel")
    p.add_argument("--region", default="US", choices=["US", "IN"], help="reference model to grade against")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json", type=Path, help="write the scorecard here")
    args = p.parse_args(argv)

    results: List[Dict[str, Any]] = []

    if args.synthetic:
        for n in args.synthetic:
            records, summary = generate_synthetic(n, args.region, args.seed)
            results.append(score(f"Prism Generator synthetic n={n}", records, summary, args.region, args.seed))

    if args.ulb:
        for path in args.ulb:
            records, summary = load_ulb_arff(path)
            results.append(score("ULB real (all rows)", records, summary, args.region, args.seed))

    if args.ulb_legit:
        for path in args.ulb_legit:
            records, summary = load_ulb_arff(path, legit_only=True)
            results.append(score("ULB real (legit only)", records, summary, args.region, args.seed))

    for n in args.reference_control or []:
        records, summary = reference_control(n, args.region, args.seed)
        results.append(score(f"reference-sampler control n={n}", records, summary, args.region, args.seed))

    if args.csv:
        if not args.amount_col or not args.time_col:
            p.error("--csv requires --amount-col and --time-col")
        records, summary = load_csv(
            args.csv,
            amount_col=args.amount_col,
            time_col=args.time_col,
            time_unit=args.time_unit,
            label_col=args.label_col,
            mcc_col=args.mcc_col,
            channel_col=args.channel_col,
        )
        results.append(score(args.csv.stem, records, summary, args.region, args.seed))

    if not results:
        p.error("nothing to score: pass --synthetic, --ulb or --csv")

    print(render(results))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
        print(f"\njson: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
