"""The dashboard needs curve geometry, not just PR-AUC scalars."""

from __future__ import annotations

import json

import numpy as np
import pytest

from prism.evaluation.threshold import downsample_curve, pr_curve_points


@pytest.fixture
def big_curve():
    rng = np.random.default_rng(4)
    y = np.zeros(5_000, dtype=int)
    y[:50] = 1
    scores = np.concatenate(
        [rng.uniform(0.5, 1.0, size=50), rng.uniform(0.0, 0.6, size=4_950)]
    )
    return pr_curve_points(y, scores)


def test_downsampling_caps_the_point_count(big_curve):
    assert len(big_curve["recall"]) > 200
    thinned = downsample_curve(big_curve, max_points=200)
    assert len(thinned["recall"]) <= 200
    assert len(thinned["precision"]) == len(thinned["recall"])


def test_downsampling_keeps_both_endpoints(big_curve):
    thinned = downsample_curve(big_curve, max_points=50)
    assert thinned["recall"][0] == pytest.approx(big_curve["recall"][0])
    assert thinned["recall"][-1] == pytest.approx(big_curve["recall"][-1])


def test_downsampling_preserves_recall_ordering(big_curve):
    """precision_recall_curve returns recall descending; thinning must not shuffle it."""
    thinned = downsample_curve(big_curve, max_points=100)
    recall = thinned["recall"]
    assert all(a >= b for a, b in zip(recall, recall[1:]))


def test_a_short_curve_is_returned_unchanged():
    curve = {"precision": [1.0, 0.5], "recall": [1.0, 0.0], "thresholds": [0.5]}
    thinned = downsample_curve(curve, max_points=200)
    assert thinned["precision"] == [1.0, 0.5]
    assert thinned["recall"] == [1.0, 0.0]


def test_thresholds_are_dropped_from_the_wire_format(big_curve):
    """Drawing a curve needs x and y only; thresholds are one element shorter
    and aligning them after thinning buys nothing the UI uses."""
    thinned = downsample_curve(big_curve, max_points=50)
    assert set(thinned) == {"precision", "recall"}


def test_metrics_endpoint_serves_curves_when_present(tmp_path, monkeypatch):
    """The endpoint reads the curves from the served track's metrics directory,
    which the engine names; a stub engine points it at a temporary one so the
    real reports/metrics/ulb/pr_curves.json is never touched."""
    pytest.importorskip("fastapi")
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from prism import config
    from prism.api import db
    from prism.api.main import app, get_engine

    curves = {"xgboost": {"precision": [1.0, 0.8], "recall": [0.0, 1.0]}}
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "pr_curves.json").write_text(json.dumps(curves), encoding="utf-8")

    url = f"sqlite:///{(tmp_path / 'm.db').as_posix()}"
    monkeypatch.setitem(config.load_config()["api"], "database_url", url)
    db.init_db(url)

    class StubEngine:
        schema = SimpleNamespace(name="ulb", metrics_dir=metrics_dir)
        models = {"xgboost": object()}
        primary = "xgboost"
        mean_latency_ms = 1.0

        def threshold_for(self, name: str) -> float:
            return 0.5

    app.dependency_overrides[get_engine] = lambda: StubEngine()
    try:
        with TestClient(app) as client:
            body = client.get("/api/v1/metrics").json()
    finally:
        app.dependency_overrides.clear()

    assert body["track"] == "ulb"
    assert body["pr_curves"] == curves
    assert body["benchmark"] is None


def test_a_perfect_ranking_is_drawn_at_full_precision():
    """95 frauds ranked above 56,000 legitimate rows: thinning by position kept
    one of the 95 points at precision 1.0 and drew the model at half its precision."""
    y = np.zeros(56_746, dtype=int)
    y[:95] = 1
    scores = np.linspace(1.0, 0.0, 56_746)
    thinned = downsample_curve(pr_curve_points(y, scores), max_points=200)
    at_half = [p for p, r in zip(thinned["precision"], thinned["recall"]) if abs(r - 0.5) < 0.02]
    assert at_half and min(at_half) == 1.0
    assert len(thinned["recall"]) <= 200
    at_one = [p for p, r in zip(thinned["precision"], thinned["recall"]) if r == 1.0]
    assert min(at_one) < 0.01 and max(at_one) == 1.0, "the vertical run at recall 1.0 is kept whole"
