"""Artifacts and reports go under the active track's directory, never the root."""

from __future__ import annotations

import pytest

from prism import config, tracks
from prism.data.preprocess import load_scaler, save_scaler


@pytest.fixture(autouse=True)
def restore_track():
    yield
    tracks.set_active(None)


def test_scaler_defaults_follow_the_active_track(tmp_path, monkeypatch):
    from sklearn.preprocessing import StandardScaler

    monkeypatch.setitem(config.load_config()["api"], "model_dir", str(tmp_path))
    tracks.set_active("india")
    scaler = StandardScaler().fit([[0.0], [2.0]])
    written = save_scaler(scaler)
    assert written == tmp_path / "india" / "scaler.joblib"
    assert load_scaler().mean_[0] == pytest.approx(1.0)


def test_benchmark_and_metrics_dirs_are_per_track():
    from prism.evaluation import benchmark

    tracks.set_active("india")
    assert benchmark.metrics_dir() == tracks.INDIA.metrics_dir
    assert benchmark.figures_dir() == tracks.INDIA.figures_dir
    tracks.set_active("ulb")
    assert benchmark.metrics_dir().name == "ulb"


def test_ulb_readers_use_the_track_keys():
    assert config.get("tracks.ulb.raw_csv") == "data/raw/creditcard.csv"
    with pytest.raises(KeyError):
        config.get("data.raw_csv")
