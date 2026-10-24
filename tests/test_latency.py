"""Tests for the §9 latency benchmark."""

from __future__ import annotations

import time

import pytest

from prism.evaluation.latency import LatencyReport, measure_latency


class FakeEngine:
    """Stand-in with a controllable cost; no trained model needed to test the maths."""

    def __init__(self, durations_ms: list[float]) -> None:
        self.durations_ms = durations_ms
        self.calls = 0

    def score(self, features: dict[str, float]) -> dict:
        index = min(self.calls, len(self.durations_ms) - 1)
        self.calls += 1
        # Busy-wait rather than sleep: Windows sleep resolution is ~15ms, the
        # same order as the thing being measured.
        target = self.durations_ms[index] / 1000.0
        start = time.perf_counter()
        while time.perf_counter() - start < target:
            pass
        return {"risk_score": 0.1, "flagged": False}


def test_percentiles_are_ordered():
    engine = FakeEngine([1.0] * 30 + [8.0])
    report = measure_latency(engine, [{}] * 31, warmup=0, budget_ms=100.0)

    assert report.p50_ms <= report.p95_ms <= report.p99_ms <= report.max_ms
    assert report.n == 31


def test_warmup_calls_are_excluded_from_the_measurement():
    engine = FakeEngine([1.0] * 20)
    report = measure_latency(engine, [{}] * 15, warmup=5, budget_ms=100.0)

    assert report.n == 15
    assert engine.calls == 20  # 5 warmup + 15 measured


def test_budget_is_judged_on_p99_not_the_mean():
    """A slow tail must not be hidden by a fast average.

    5% of calls at 150ms drags p99 past the budget while the mean stays under
    10ms. Judging on the mean would pass this engine; judging on p99 fails it,
    which is the behaviour a payment processor needs.
    """
    engine = FakeEngine([1.0] * 95 + [150.0] * 5)
    report = measure_latency(engine, [{}] * 100, warmup=0, budget_ms=100.0)

    assert report.mean_ms < 100.0, "the mean alone would have passed this engine"
    assert report.p99_ms > 100.0
    assert report.within_budget is False


def test_a_single_outlier_does_not_trip_the_budget():
    """p99 rather than max, deliberately.

    One 250ms spike in 100 requests is a GC pause, not a latency regression.
    Judging on max would fail an otherwise healthy service on noise; p99
    interpolates it away. This is the trade-off being made, so it is pinned.
    """
    engine = FakeEngine([1.0] * 99 + [250.0])
    report = measure_latency(engine, [{}] * 100, warmup=0, budget_ms=100.0)

    assert report.max_ms > 100.0
    assert report.within_budget is True


def test_a_fast_engine_passes_the_budget():
    engine = FakeEngine([1.0] * 50)
    report = measure_latency(engine, [{}] * 50, warmup=0, budget_ms=100.0)

    assert report.within_budget is True


def test_report_is_json_serialisable():
    import json

    engine = FakeEngine([1.0] * 10)
    report = measure_latency(engine, [{}] * 10, warmup=0, budget_ms=100.0)
    json.dumps(report.as_dict())
    assert isinstance(report, LatencyReport)


def test_empty_payload_list_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        measure_latency(FakeEngine([1.0]), [], warmup=0)


def test_explain_latency_times_only_the_explanation():
    from prism.evaluation.latency import measure_explain_latency

    class Engine:
        def __init__(self):
            self.scored = self.explained = 0
            self.raws = []

        def score(self, features):
            self.scored += 1
            # The service explains the enriched dict (inputs plus what it derived),
            # never the bare payload; the harness must do the same.
            return {"frame": features, "features": {**features, "derived": 1.0}}

        def explain(self, frame, raw=None):
            self.explained += 1
            self.raws.append(raw)
            return {"narrative": "x"}

    engine = Engine()
    report = measure_explain_latency(engine, [{"a": 1.0}] * 30, warmup=5, budget_ms=100.0)
    assert report.n == 30
    assert engine.explained == 35 and engine.scored == 35
    assert all(raw is not None and "derived" in raw for raw in engine.raws)
    assert report.within_budget


fastapi = pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
pytest.importorskip("httpx")

from fastapi import BackgroundTasks  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from prism.evaluation.latency import measure_http_latency  # noqa: E402


class Row(BaseModel):
    i: int


def _slow_after_answering() -> tuple[fastapi.FastAPI, list[int]]:
    """A service that answers at once and lingers 200 ms afterwards. Defined at
    module level: with postponed annotations, FastAPI resolves parameter types
    through the module's globals."""
    app = fastapi.FastAPI()
    finished: list[int] = []

    def linger(i: int) -> None:
        time.sleep(0.2)
        finished.append(i)

    @app.get("/api/v1/health")
    def health() -> dict:
        return {"status": "ready"}

    @app.post("/api/v1/predict")
    def predict(body: Row, background: BackgroundTasks) -> dict:
        background.add_task(linger, body.i)
        return {"flagged": body.i % 2 == 0}

    return app, finished


def test_http_measurement_splits_flagged_from_unflagged_and_stops_at_the_answer():
    app, _ = _slow_after_answering()
    reports = measure_http_latency(app, [{"i": i} for i in range(6)], budget_ms=100.0, warmup=1, port=8799)
    assert set(reports) == {"flagged", "unflagged"}
    assert reports["flagged"].n == 3 and reports["unflagged"].n == 3
    assert reports["flagged"].p99_ms < 150.0, "the clock stops at the answer, before the 200 ms task"



def test_http_measurement_can_drive_a_separate_server_process():
    """The published figure comes from a server in its own process, as a
    caller sees it; a server thread sharing the client's interpreter would
    charge the client's own work to the service."""
    reports = measure_http_latency(
        None, [{"i": i} for i in range(6)], app_spec="tests._fake_service:app", budget_ms=100.0, warmup=1, port=8798
    )
    assert reports["flagged"].n == 3 and reports["unflagged"].n == 3
    assert reports["flagged"].p99_ms < 150.0
