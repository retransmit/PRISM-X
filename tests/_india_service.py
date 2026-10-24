"""An Indian deployment and a service client over it, shared by the live-loop tests."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

from fastapi.testclient import TestClient

from prism import config, tracks
from prism.api import db, schemas
from prism.api.inference import InferenceEngine
from prism.api.main import app, get_engine, get_engine_optional
from prism.data import india
from prism.data.preprocess import prepare_splits, save_encoder, save_scaler
from prism.evaluation.threshold import tune_threshold
from prism.explain import artifacts
from prism.features.velocity import add_velocity_features
from prism.models import build

INDIA = tracks.INDIA
CREDENTIALS = {"analyst": "analyst-secret", "customer": "customer-secret", "label_feed": "label-secret"}


def as_role(role: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {CREDENTIALS[role]}"}


def payload_from(row, txn_id: str | None = None) -> dict:
    body = {}
    for column in INDIA.input_columns:
        value = row[column]
        body[column] = str(value) if (column in INDIA.categorical_columns or column == INDIA.card_column) else float(value)
    body.update({
        "transaction_id": txn_id or str(row["transaction_id"]), "merchant": str(row["merchant_id"]),
        "channel": str(row["channel_type"]), "currency": "INR",
    })
    return body


def build_deployment(stem: Path, model_dir: Path) -> Path:
    """Isolation Forest and a small XGBoost trained on the export, with the scaler, the
    encoder, the thresholds and the explanation artifacts the engine loads."""
    tracks.set_active("india")
    try:
        frame = add_velocity_features(india.load_india(stem))
        splits = prepare_splits(frame, schema=INDIA)
    finally:
        tracks.set_active("ulb")
    model_dir.mkdir(parents=True, exist_ok=True)
    thresholds = {}
    for name in ("isolation_forest", "xgboost"):
        overrides = {"categorical_columns": splits.categorical_columns}
        if name == "xgboost":
            overrides["n_estimators"] = 60
        model = build(name, **overrides).fit(splits.X_train, splits.y_train)
        model.save(model_dir)
        thresholds[name] = tune_threshold(
            splits.y_val, model.risk_score(splits.X_val), strategy="fpr_budget", fpr_budget_per_10k=50
        )
    save_scaler(splits.scaler, model_dir / "scaler.joblib")
    save_encoder(splits.encoder, model_dir / "encoder.joblib")
    (model_dir / "thresholds.json").write_text(json.dumps(thresholds), encoding="utf-8")
    reference, cases = artifacts.build(frame, splits, schema=INDIA)
    artifacts.save(reference, cases, model_dir)
    return model_dir


def use_temp_database(tmp_path: Path, monkeypatch) -> str:
    url = f"sqlite:///{(tmp_path / 'audit.db').as_posix()}"
    api = config.load_config()["api"]
    monkeypatch.setitem(api, "database_url", url)
    # A deployment built here records no manifest, so there is no champion for auto to serve.
    monkeypatch.setitem(api, "primary_model", "xgboost")
    monkeypatch.setitem(api, "credentials", dict(CREDENTIALS))
    db.init_db(url)
    return url


@contextmanager
def service(engine: InferenceEngine | None, *, override: bool = True):
    """A test client on the Indian track. With ``override`` the given engine is served;
    without it the service loads what its own startup finds."""
    if override:
        app.dependency_overrides[get_engine] = lambda: engine
        app.dependency_overrides[get_engine_optional] = lambda: engine
    try:
        with TestClient(app) as client:
            schemas.set_active_schema(INDIA)
            yield client
    finally:
        app.dependency_overrides.clear()
        schemas.set_active_schema(None)
