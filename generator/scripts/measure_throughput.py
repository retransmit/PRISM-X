#!/usr/bin/env python3
"""Measures discrete-event engine throughput so documentation can cite a real number.

Counts every event pushed to and popped from the engine's priority queue while a
batch is generated, then reports events/second, transactions/second and peak
resident memory. Every figure printed here comes from this run; there is no
hard-coded baseline to agree with.

Usage:
    python scripts/measure_throughput.py --transactions 200000 --region US --seed 42
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import heapq  # noqa: E402

from prism_generator.engine import DiscreteEventEngine  # noqa: E402


class _EventCounter:
    """Counts priority-queue pushes and pops for the duration of a measurement."""

    def __init__(self) -> None:
        self.pushed = 0
        self.popped = 0
        self._original_push = heapq.heappush
        self._original_pop = heapq.heappop

    def __enter__(self) -> "_EventCounter":
        def counted_push(heap, item):  # type: ignore[no-untyped-def]
            self.pushed += 1
            return self._original_push(heap, item)

        def counted_pop(heap):  # type: ignore[no-untyped-def]
            self.popped += 1
            return self._original_pop(heap)

        heapq.heappush = counted_push  # type: ignore[assignment]
        heapq.heappop = counted_pop  # type: ignore[assignment]
        return self

    def __exit__(self, *exc: object) -> None:
        heapq.heappush = self._original_push  # type: ignore[assignment]
        heapq.heappop = self._original_pop  # type: ignore[assignment]


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transactions", type=int, default=200_000)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--region", choices=["US", "IN"], default="US")
    parser.add_argument("--fraud-rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only")
    args = parser.parse_args()

    engine = DiscreteEventEngine(n_cards=500, n_merchants=100, region=args.region, seed=args.seed)

    counter = _EventCounter()
    started = time.perf_counter()
    with counter:
        records = engine.generate_batch(
            n_transactions=args.transactions,
            fraud_prevalence=args.fraud_rate,
            time_span_days=args.days,
            enforce_invariants=True,
        )
    elapsed = time.perf_counter() - started

    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # Linux reports KiB
    events_per_sec = counter.pushed / elapsed if elapsed > 0 else 0.0
    tx_per_sec = len(records) / elapsed if elapsed > 0 else 0.0

    result = {
        "transactions": len(records),
        "days": args.days,
        "region": args.region,
        "seed": args.seed,
        "wall_seconds": round(elapsed, 3),
        "events_pushed": counter.pushed,
        "events_popped": counter.popped,
        "events_per_second": round(events_per_sec, 1),
        "transactions_per_second": round(tx_per_sec, 1),
        "events_per_transaction": round(counter.pushed / len(records), 2) if records else 0.0,
        "peak_rss_mb": round(peak_rss_mb, 1),
        "cpu": cpu_model(),
        "python": platform.python_version(),
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print("=" * 66)
    print("  PRISM GENERATORA ENGINE THROUGHPUT MEASUREMENT")
    print("=" * 66)
    print(f"  Workload:                 {result['transactions']:,} transactions over {args.days} days ({args.region}, seed {args.seed})")
    print(f"  Wall time:                {result['wall_seconds']} s")
    print(f"  DES events pushed:        {result['events_pushed']:,}")
    print(f"  Event throughput:         {result['events_per_second']:,.0f} events/sec")
    print(f"  Transaction throughput:   {result['transactions_per_second']:,.0f} tx/sec")
    print(f"  Events per transaction:   {result['events_per_transaction']}")
    print(f"  Peak resident memory:     {result['peak_rss_mb']} MB")
    print(f"  CPU:                      {result['cpu']}")
    print(f"  Python:                   {result['python']}")
    print("=" * 66)
    print("Reproduce: python scripts/measure_throughput.py --transactions "
          f"{args.transactions} --days {args.days} --region {args.region} --seed {args.seed} --json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
