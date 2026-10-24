"""The training entry point, driven end to end on a small frame: a full run,
a one-model retrain that keeps the others, and the shortcut ceiling."""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from prism.models import MODEL_NAMES  # noqa: E402
pytest.importorskip("torch")

from prism import config, tracks  # noqa: E402
from prism.training import train  # noqa: E402

PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]


def _frame(n: int = 2_400, n_fraud: int = 48, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "Time": rng.uniform(0, 172_800, size=n),
        **{c: rng.normal(size=n) for c in PCA_COLUMNS},
        "Amount": rng.gamma(2.0, 50.0, size=n),
    })
    labels = np.zeros(n, dtype=int)
    labels[rng.choice(n, size=n_fraud, replace=False)] = 1
    frame["Class"] = labels
    frame.loc[frame["Class"] == 1, PCA_COLUMNS[:4]] += 5.0
    return frame


@pytest.fixture
def small_pipeline(tmp_path, monkeypatch):
    """train.main() on a small ULB-shaped frame, writing under tmp_path only."""
    frame = _frame()
    monkeypatch.setattr(train, "load_frame", lambda schema: frame)
    settings = config.load_config()
    monkeypatch.setitem(settings["api"], "model_dir", str(tmp_path / "models"))
    monkeypatch.setitem(settings["models"]["xgboost"], "n_estimators", 20)
    monkeypatch.setitem(settings["models"]["autoencoder"], "epochs", 2)
    monkeypatch.setattr(tracks.TrackSchema, "metrics_dir", property(lambda self: tmp_path / "metrics" / self.name))
    monkeypatch.setattr(tracks.TrackSchema, "figures_dir", property(lambda self: tmp_path / "figures" / self.name))

    def run(*args: str) -> None:
        monkeypatch.setattr(sys, "argv", ["train", "--track", "ulb", *args])
        train.main()

    return run, tmp_path / "models" / "ulb", tmp_path / "metrics" / "ulb"


def test_a_one_model_retrain_keeps_the_other_thresholds_and_rows(small_pipeline):
    run, model_dir, metrics_dir = small_pipeline
    run("--model", "all")
    before = json.loads((model_dir / "thresholds.json").read_text(encoding="utf-8"))
    assert set(before) == set(MODEL_NAMES)

    run("--model", "isolation_forest")

    after = json.loads((model_dir / "thresholds.json").read_text(encoding="utf-8"))
    assert after["xgboost"] == before["xgboost"] and after["autoencoder"] == before["autoencoder"]
    rows = json.loads((metrics_dir / "test_results.json").read_text(encoding="utf-8"))
    assert {row["model"] for row in rows} == set(MODEL_NAMES)
    record = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    assert set(record["models"]) == set(MODEL_NAMES)
    assert record["champion"]["model"] in MODEL_NAMES and record["primary_model"] == record["champion"]["model"]
    assert all("sha256" in entry for entry in record["models"].values())
    assert not (model_dir / "drift_reference.json").exists(), "the ULB track has no stream to drift"


def test_the_shortcut_ceiling_refuses_and_an_allowance_lets_it_through(small_pipeline, monkeypatch):
    run, _, metrics_dir = small_pipeline
    ulb = config.load_config()["tracks"]["ulb"]
    monkeypatch.setitem(ulb, "shortcut_auc_ceiling", 0.9)
    with pytest.raises(SystemExit, match="V1"):
        run("--model", "isolation_forest")
    assert (metrics_dir / "shortcut_audit.json").exists(), "the audit is written before the refusal"

    monkeypatch.setitem(ulb, "allowed_shortcuts", {"V1": "a stated reason", "V2": "x", "V3": "x", "V4": "x"})
    run("--model", "isolation_forest")
