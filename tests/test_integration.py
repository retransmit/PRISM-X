"""Phase 8: end-to-end proof that the tiers compose.

Raw unscaled row in; scored, explained and persisted audit record out, driving
the same code path the streaming simulator uses. Every other test file covers
one tier in isolation; this one is the only thing asserting they fit together.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def pinned_primary(monkeypatch):
    """The deployment here records no manifest, so there is no champion for auto to serve."""
    from prism import config

    monkeypatch.setitem(config.load_config()["api"], "primary_model", "xgboost")

pytest.importorskip("fastapi")
pytest.importorskip("xgboost")

from fastapi.testclient import TestClient  # noqa: E402

from prism import config  # noqa: E402
from prism.api import db  # noqa: E402
from prism.api.inference import FEATURE_ORDER, InferenceEngine  # noqa: E402
from prism.api.main import app, get_engine, get_engine_optional  # noqa: E402
from prism.data.preprocess import prepare_splits, save_scaler  # noqa: E402
from prism.evaluation.latency import measure_latency  # noqa: E402
from prism.evaluation.threshold import tune_threshold  # noqa: E402
from prism.models import build  # noqa: E402

PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]


@pytest.fixture(scope="module")
def deployment(tmp_path_factory):
    """A trained, persisted deployment plus the held-out rows to replay."""
    rng = np.random.default_rng(33)
    n, n_fraud = 8_000, 80

    frame = pd.DataFrame(
        {
            **{c: rng.normal(size=n) for c in PCA_COLUMNS},
            "Time": rng.uniform(0, 172_800, size=n),
            "Amount": rng.gamma(2.0, 50.0, size=n),
        }
    )
    labels = np.zeros(n, dtype=int)
    labels[rng.choice(n, size=n_fraud, replace=False)] = 1
    frame["Class"] = labels

    mask = frame["Class"] == 1
    for column in PCA_COLUMNS[:6]:
        frame.loc[mask, column] += rng.normal(6.0, 1.0, size=n_fraud)
    frame.loc[mask, "Amount"] *= 6.0
    frame = frame[["Time", *PCA_COLUMNS, "Amount", "Class"]]

    splits = prepare_splits(frame)
    model_dir = tmp_path_factory.mktemp("deployment")

    thresholds = {}
    for name in ("isolation_forest", "xgboost"):
        model = build(name, **({"n_estimators": 80} if name == "xgboost" else {}))
        model.fit(splits.X_train, splits.y_train)
        model.save(model_dir)
        thresholds[name] = tune_threshold(splits.y_val, model.risk_score(splits.X_val))

    save_scaler(splits.scaler, model_dir / "scaler.joblib")
    (model_dir / "thresholds.json").write_text(json.dumps(thresholds), encoding="utf-8")

    return model_dir, frame.loc[splits.X_test.index]


def test_a_stream_of_transactions_is_scored_explained_and_persisted(
    deployment, tmp_path, monkeypatch
):
    model_dir, raw_test = deployment

    url = f"sqlite:///{(tmp_path / 'e2e.db').as_posix()}"
    monkeypatch.setitem(config.load_config()["api"], "database_url", url)
    db.init_db(url)

    engine = InferenceEngine(model_dir).load()
    app.dependency_overrides[get_engine] = lambda: engine
    app.dependency_overrides[get_engine_optional] = lambda: engine

    # Deliberately mix known fraud in, as the simulator's fraud_boost does.
    frauds = raw_test[raw_test["Class"] == 1].head(5)
    normals = raw_test[raw_test["Class"] == 0].head(25)
    stream = pd.concat([normals, frauds]).sample(frac=1.0, random_state=1)

    try:
        with TestClient(app) as client:
            assert client.get("/api/v1/health").json()["status"] == "ready"

            for index, (_, row) in enumerate(stream.iterrows()):
                body = {name: float(row[name]) for name in FEATURE_ORDER}
                body["transaction_id"] = f"txn-{index:04d}"
                response = client.post("/api/v1/predict", json=body)
                assert response.status_code == 200, response.text

            rows = client.get("/api/v1/transactions?limit=100").json()
            metrics = client.get("/api/v1/metrics").json()
    finally:
        app.dependency_overrides.clear()

    assert len(rows) == len(stream), "every transaction must reach the audit log"
    assert metrics["processed"] == len(stream)

    flagged = [r for r in rows if r["flagged"]]
    assert flagged, "no fraud detected end-to-end; the pipeline is broken"
    assert metrics["flagged"] == len(flagged)

    # Every alert must carry its reasoning; that is the regulatory claim in §3.3.
    for record in flagged:
        explanation = record["explanation"]
        assert explanation is not None
        assert explanation["top_features"]
        assert explanation["narrative"]

    # And every record must carry both models' opinions, not just the primary's.
    for record in rows:
        assert {s["model_name"] for s in record["scores"]} == {
            "isolation_forest",
            "xgboost",
        }


def test_end_to_end_latency_meets_the_spec_budget(deployment):
    """§9 commits to <100ms per transaction. Judged on p99."""
    model_dir, raw_test = deployment
    engine = InferenceEngine(model_dir).load()

    payloads = [
        {name: float(row[name]) for name in FEATURE_ORDER}
        for _, row in raw_test.head(200).iterrows()
    ]
    report = measure_latency(engine, payloads, warmup=20, budget_ms=100.0)
    print(f"\n{report}")

    assert report.within_budget, f"latency budget exceeded: {report}"
