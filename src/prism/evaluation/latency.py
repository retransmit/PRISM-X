"""Inference latency benchmark for the §9 "< 100 ms per transaction" claim.

    python -m prism.evaluation.latency

The budget is judged on **p99, not the mean**. A mean of 40 ms hiding a p99 of
400 ms fails the requirement in exactly the way that matters to a payment
processor: the slow tail is where transactions time out, and the tail is not
rare when you are scoring millions a day.

Warmup calls are discarded. The first few inferences pay for lazy allocation
inside XGBoost and torch, and including them would misrepresent steady state.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol, Sequence

import numpy as np


class _Scorer(Protocol):
    def score(self, features: dict[str, float]) -> dict: ...


@dataclass(frozen=True)
class LatencyReport:
    n: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    budget_ms: float
    within_budget: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        verdict = "PASS" if self.within_budget else "FAIL"
        return (
            f"latency over {self.n} requests: mean={self.mean_ms:.2f}ms  "
            f"p50={self.p50_ms:.2f}ms  p95={self.p95_ms:.2f}ms  "
            f"p99={self.p99_ms:.2f}ms  max={self.max_ms:.2f}ms  "
            f"budget={self.budget_ms:.0f}ms [{verdict}]"
        )


def _report(timings: Sequence[float], budget_ms: float) -> LatencyReport:
    samples = np.asarray(timings, dtype=float)
    p99 = float(np.percentile(samples, 99))
    return LatencyReport(
        n=len(samples),
        mean_ms=float(samples.mean()),
        p50_ms=float(np.percentile(samples, 50)),
        p95_ms=float(np.percentile(samples, 95)),
        p99_ms=p99,
        max_ms=float(samples.max()),
        budget_ms=budget_ms,
        within_budget=bool(p99 <= budget_ms),
    )


def measure_latency(
    engine: _Scorer,
    payloads: Sequence[dict[str, float]],
    *,
    warmup: int = 10,
    budget_ms: float = 100.0,
) -> LatencyReport:
    """Time one scoring call per payload and summarise the distribution."""
    if not payloads:
        raise ValueError("measure_latency needs at least one payload.")

    for index in range(warmup):
        engine.score(payloads[index % len(payloads)])

    timings: list[float] = []
    for payload in payloads:
        started = time.perf_counter()
        engine.score(payload)
        timings.append((time.perf_counter() - started) * 1000.0)

    return _report(timings, budget_ms)


def measure_explain_latency(
    engine, payloads: Sequence[dict], *, warmup: int = 10, budget_ms: float = 100.0
) -> LatencyReport:
    """Time the explanation alone: score first, then clock explain() on the frame.

    Explanation runs only for flagged rows, so it has its own budget and its
    own number; folding it into the scoring figure would hide it.
    """
    if not payloads:
        raise ValueError("measure_explain_latency needs at least one payload.")
    # The service explains the enriched dict the engine returns (inputs plus
    # what it derived), never the bare payload.
    for index in range(warmup):
        result = engine.score(payloads[index % len(payloads)])
        engine.explain(result["frame"], raw=result.get("features", payloads[index % len(payloads)]))
    timings: list[float] = []
    for payload in payloads:
        result = engine.score(payload)
        frame, raw = result["frame"], result.get("features", payload)
        started = time.perf_counter()
        engine.explain(frame, raw=raw)
        timings.append((time.perf_counter() - started) * 1000.0)
    return _report(timings, budget_ms)


def measure_http_latency(
    app,
    payloads: Sequence[dict],
    *,
    app_spec: str | None = None,
    env: dict[str, str] | None = None,
    budget_ms: float = 100.0,
    warmup: int = 10,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> dict[str, LatencyReport]:
    """Round-trip time of POST /api/v1/predict against a uvicorn server on the
    loopback interface, split by the answer's ``flagged``.

    ``app`` is served in a thread of this process (for tests); ``app_spec``
    ("module:attribute") is served by uvicorn in a process of its own, with
    ``env`` added to its environment, which is how the published figure is
    taken: a caller's clock against a service that does its own work. The
    clock stops when the answer arrives, so whatever the service does after
    answering is outside it. Payloads are sent in the order given, so a
    time-ordered sample builds the card history the velocity features read.
    """
    import os
    import subprocess
    import sys
    import threading

    import httpx

    base = f"http://{host}:{port}"
    timings: dict[str, list[float]] = {"unflagged": [], "flagged": []}
    process = None
    thread = None
    server = None
    if app_spec is not None:
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", app_spec, "--host", host, "--port", str(port), "--log-level", "warning"],
            env={**os.environ, **(env or {})},
        )
    else:
        import uvicorn

        server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
    try:
        with httpx.Client(timeout=10.0) as client:
            for _ in range(1200):
                if process is not None and process.poll() is not None:
                    raise RuntimeError(f"the service process exited with {process.returncode} before answering")
                try:
                    if client.get(f"{base}/api/v1/health").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.05)
            else:
                raise RuntimeError("the service did not come up for the latency run")
            for index in range(warmup):
                client.post(f"{base}/api/v1/predict", json=payloads[index % len(payloads)])
            for payload in payloads:
                started = time.perf_counter()
                response = client.post(f"{base}/api/v1/predict", json=payload)
                elapsed = (time.perf_counter() - started) * 1000.0
                response.raise_for_status()
                timings["flagged" if response.json()["flagged"] else "unflagged"].append(elapsed)
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                process.kill()
        if server is not None and thread is not None:
            server.should_exit = True
            thread.join(timeout=10.0)
    return {name: _report(samples, budget_ms) for name, samples in timings.items() if samples}


def main() -> None:
    import argparse

    import pandas as pd

    from prism import config, tracks
    from prism.api.inference import InferenceEngine
    from prism.data.frames import load_frame
    from prism.data.preprocess import prepare_splits

    parser = argparse.ArgumentParser(description="Time the scoring path on unseen rows.")
    tracks.add_track_argument(parser)
    parser.add_argument("--explain", action="store_true", help="also time the explanation path")
    parser.add_argument("--http", action="store_true", help="also time the HTTP round trip with card history")
    args = parser.parse_args()
    schema = tracks.activate_from_args(args)

    frame = load_frame(schema)
    splits = prepare_splits(frame, schema=schema)
    engine = InferenceEngine().load()

    # Score unseen rows only, in raw form; the service applies the encoder and
    # the scaler. Categoricals and the card id travel as text, as a stream sends them.
    raw_test = frame.loc[splits.X_test.index]
    sample = raw_test.sample(
        n=min(500, len(raw_test)), random_state=config.seed()
    )
    text_columns = set(schema.categorical_columns) | {schema.card_column}
    payloads = [
        {name: (str(row[name]) if name in text_columns else float(row[name])) for name in schema.input_columns}
        for _, row in sample.iterrows()
    ]

    report = measure_latency(engine, payloads, warmup=20, budget_ms=100.0)
    print(report)

    metrics_dir = schema.metrics_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "latency.json").write_text(
        json.dumps(report.as_dict(), indent=2), encoding="utf-8"
    )
    print(f"Written -> {metrics_dir / 'latency.json'}")

    if args.explain:
        explain_report = measure_explain_latency(engine, payloads, warmup=20, budget_ms=100.0)
        print(f"explanation {explain_report}")
        (metrics_dir / "latency_explain.json").write_text(
            json.dumps(explain_report.as_dict(), indent=2), encoding="utf-8"
        )
        print(f"Written -> {metrics_dir / 'latency_explain.json'}")

    if args.http:
        # Every fraud in the test window (up to 100) plus the random sample, in
        # time order, so the flagged figure rests on more than a handful of
        # rows and the card history builds as it would in service.
        frauds = raw_test[raw_test[schema.target_column] == 1].head(100)
        rows = pd.concat([frauds, sample]).drop_duplicates().sort_values(schema.time_column)
        ordered = [
            {name: (str(row[name]) if name in text_columns else float(row[name])) for name in schema.input_columns}
            for _, row in rows.iterrows()
        ]
        scratch = metrics_dir / "latency_http.db"
        if scratch.exists():
            scratch.unlink()
        reports = measure_http_latency(
            None,
            ordered,
            app_spec="prism.api.main:app",
            # The offline models are measured: a live loop's pointer is never followed here.
            env={"PRISM_TRACK": schema.name, "PRISM_DATABASE_URL": f"sqlite:///{scratch.as_posix()}", "PRISM_LIVE": "0"},
            warmup=20,
            budget_ms=100.0,
        )
        for name, report in reports.items():
            print(f"http {name} {report}")
        (metrics_dir / "latency_http.json").write_text(
            json.dumps({name: report.as_dict() for name, report in reports.items()}, indent=2), encoding="utf-8"
        )
        print(f"Written -> {metrics_dir / 'latency_http.json'}")
        # The server process may hold the scratch database a moment after it is told to stop.
        for _ in range(40):
            try:
                scratch.unlink(missing_ok=True)
                break
            except PermissionError:
                time.sleep(0.25)


if __name__ == "__main__":
    main()
