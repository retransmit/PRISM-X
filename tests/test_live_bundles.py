"""Model bundles, the pointer the service polls, and a swap that never leaves it without an engine."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("xgboost")

from prism import config, tracks  # noqa: E402
from prism.api.inference import InferenceEngine  # noqa: E402
from prism.data.frames import load_frame  # noqa: E402
from prism.live import bundles  # noqa: E402
from tests._india_service import build_deployment, payload_from, service, use_temp_database  # noqa: E402

INDIA = tracks.INDIA


@pytest.fixture(scope="module")
def deployment(tmp_path_factory, india_stem):
    return build_deployment(india_stem, tmp_path_factory.mktemp("bundle_models"))


@pytest.fixture(scope="module")
def frame(india_stem):
    return load_frame(INDIA, india_stem)


@pytest.fixture(autouse=True)
def models_root(tmp_path, monkeypatch, deployment):
    """api.model_dir points at a scratch root whose india directory is the offline deployment."""
    root = tmp_path / "models"
    shutil.copytree(deployment, root / "india")
    monkeypatch.setitem(config.load_config()["api"], "model_dir", str(root))
    monkeypatch.setitem(config.load_config()["live"], "model_dir", str(root / "live"))
    monkeypatch.setitem(config.load_config()["live"], "enabled", True)
    monkeypatch.setitem(config.load_config()["live"], "bundle_poll_seconds", 0)
    use_temp_database(tmp_path, monkeypatch)
    tracks.set_active("india")
    return root


def add_bundle(deployment, bundle_id: str):
    target = bundles.root(INDIA) / bundle_id
    shutil.copytree(deployment, target)
    return target


def test_without_a_pointer_the_offline_directory_serves(models_root):
    assert bundles.serving_dir(INDIA) == ("offline", models_root / "india")


def test_a_bundle_id_names_the_stream_day():
    bundle_id = bundles.new_id(1_704_067_200.0 + 61.5 * 86_400, origin=1_704_067_200.0)
    assert bundle_id.startswith("day-061-")
    assert len(bundle_id) == len("day-061-") + 6
    assert bundle_id != bundles.new_id(1_704_067_200.0 + 61.5 * 86_400, origin=1_704_067_200.0)


def test_promotion_rewrites_the_pointer_and_records_the_event(deployment):
    target = add_bundle(deployment, "day-060-abcdef")
    event = {"status": "promoted", "challenger_bundle": "day-060-abcdef"}
    bundles.promote(INDIA, "day-060-abcdef", event)
    assert bundles.serving_dir(INDIA) == ("day-060-abcdef", target)
    assert bundles.events(INDIA)[-1] == event
    pointer = json.loads((bundles.pointer_path(INDIA)).read_text(encoding="utf-8"))
    assert pointer["bundle_id"] == "day-060-abcdef"


def test_the_service_swaps_to_a_promoted_bundle(deployment, frame):
    with service(None, override=False) as client:
        assert client.get("/api/v1/health").json()["bundle_id"] == "offline"
        add_bundle(deployment, "day-075-123456")
        bundles.promote(INDIA, "day-075-123456", {"status": "promoted"})
        answer = client.post("/api/v1/predict", json=payload_from(frame.iloc[0]))
        assert answer.status_code == 200
        health = client.get("/api/v1/health").json()
        assert health["bundle_id"] == "day-075-123456"
        assert health["bundle_error"] is None
        from prism.api import main as running

        assert running.engine.model_dir == bundles.root(INDIA) / "day-075-123456"


def test_a_pointer_to_a_broken_bundle_keeps_the_serving_engine(frame):
    with service(None, override=False) as client:
        from prism.api import main as running

        before = running.engine
        broken = bundles.root(INDIA) / "day-090-badbad"
        broken.mkdir(parents=True)
        bundles.promote(INDIA, "day-090-badbad", {"status": "promoted"})
        answer = client.post("/api/v1/predict", json=payload_from(frame.iloc[0]))
        assert answer.status_code == 200
        assert running.engine is before
        health = client.get("/api/v1/health").json()
        assert health["bundle_id"] == "offline"
        assert "day-090-badbad" in health["bundle_error"]


def test_a_broken_pointer_at_startup_serves_the_offline_directory(frame):
    broken = bundles.root(INDIA) / "day-090-badbad"
    broken.mkdir(parents=True)
    bundles.promote(INDIA, "day-090-badbad", {"status": "promoted"})
    with service(None, override=False) as client:
        assert client.post("/api/v1/predict", json=payload_from(frame.iloc[0])).status_code == 200
        health = client.get("/api/v1/health").json()
        assert health["status"] == "ready"
        assert health["bundle_id"] == "offline"
        assert "day-090-badbad" in health["bundle_error"]


def test_the_champion_scorer_matches_the_engine(deployment, frame):
    rows = frame.iloc[:20]
    with service(None, override=False) as client:
        served = [client.post("/api/v1/predict", json=payload_from(row)).json()["risk_score"] for _, row in rows.iterrows()]
    name, scorer = bundles.champion_scorer(deployment, INDIA)
    assert name == InferenceEngine(deployment, track="india").load().primary
    np.testing.assert_allclose(scorer(rows), served, rtol=0, atol=1e-9)


def test_the_live_endpoint_names_the_bundle_the_labels_and_the_retrains(deployment, frame):
    add_bundle(deployment, "day-075-123456")
    with service(None, override=False) as client:
        body = payload_from(frame.iloc[0])
        client.post("/api/v1/predict", json=body)
        from prism.api import db

        db.record_label(transaction_id=body["transaction_id"], label=1, source="CHARGEBACK_DISPUTE", known_at=1.0)
        bundles.record_event(INDIA, {"status": "kept", "as_of": 1.0})
        bundles.promote(INDIA, "day-075-123456", {"status": "promoted", "as_of": 2.0})
        live = client.get("/api/v1/live").json()
    assert live["bundle_id"] == "day-075-123456"
    assert live["champion"] == "xgboost"
    assert live["labels"] == {"CHARGEBACK_DISPUTE": 1}
    assert [event["status"] for event in live["events"]] == ["kept", "promoted"]


def test_the_live_endpoint_answers_404_off_the_indian_track():
    from prism.api import schemas

    with service(None, override=False) as client:
        schemas.set_active_schema(tracks.ULB)
        response = client.get("/api/v1/live")
    assert response.status_code == 404
    assert "Indian track" in response.json()["detail"]


def test_bundles_and_the_pointer_live_under_the_live_model_root(models_root):
    assert bundles.root(INDIA) == models_root / "live" / "india" / "bundles"
    assert bundles.pointer_path(INDIA) == models_root / "live" / "india" / "current.json"


def test_without_the_live_switch_the_service_ignores_the_pointer(deployment, frame, monkeypatch):
    monkeypatch.setitem(config.load_config()["live"], "enabled", False)
    add_bundle(deployment, "day-075-123456")
    bundles.promote(INDIA, "day-075-123456", {"status": "promoted"})
    with service(None, override=False) as client:
        assert client.post("/api/v1/predict", json=payload_from(frame.iloc[0])).status_code == 200
        health = client.get("/api/v1/health").json()
        live = client.get("/api/v1/live")
    assert health["bundle_id"] == "offline" and health["bundle_error"] is None
    assert live.status_code == 404 and "PRISM_LIVE" in live.json()["detail"]


def test_the_live_switch_comes_from_the_environment(monkeypatch):
    monkeypatch.setitem(config.load_config()["live"], "enabled", False)
    monkeypatch.setenv("PRISM_LIVE", "1")
    assert bundles.live_enabled()
    monkeypatch.setenv("PRISM_LIVE", "")
    assert not bundles.live_enabled()


def test_a_malformed_pointer_at_startup_serves_the_offline_directory(frame):
    pointer = bundles.pointer_path(INDIA)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text("{", encoding="utf-8")
    with service(None, override=False) as client:
        assert client.post("/api/v1/predict", json=payload_from(frame.iloc[0])).status_code == 200
        health = client.get("/api/v1/health").json()
    assert health["status"] == "ready" and health["bundle_id"] == "offline"
    assert "current.json" in health["bundle_error"]


def test_a_pointer_broken_while_serving_keeps_the_engine_and_says_why(frame):
    with service(None, override=False) as client:
        pointer = bundles.pointer_path(INDIA)
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text('{"not_a_bundle": 1}', encoding="utf-8")
        assert client.post("/api/v1/predict", json=payload_from(frame.iloc[1])).status_code == 200
        health = client.get("/api/v1/health").json()
    assert health["bundle_id"] == "offline"
    assert "current.json" in health["bundle_error"]


def test_a_refused_bundle_is_retried_once_the_pointer_is_rewritten(deployment, frame):
    """Repairing a refused bundle and promoting it again under the same id swaps to it."""
    with service(None, override=False) as client:
        broken = bundles.root(INDIA) / "day-105-fixme1"
        broken.mkdir(parents=True)
        bundles.promote(INDIA, "day-105-fixme1", {"status": "promoted"})
        client.post("/api/v1/predict", json=payload_from(frame.iloc[0]))
        assert "day-105-fixme1" in client.get("/api/v1/health").json()["bundle_error"]
        shutil.rmtree(broken)
        add_bundle(deployment, "day-105-fixme1")
        bundles.promote(INDIA, "day-105-fixme1", {"status": "promoted"})
        assert client.post("/api/v1/predict", json=payload_from(frame.iloc[1])).status_code == 200
        health = client.get("/api/v1/health").json()
        assert health["bundle_id"] == "day-105-fixme1"
        assert health["bundle_error"] is None


def test_a_bundle_refused_at_startup_is_retried_once_the_pointer_is_rewritten(deployment, frame):
    broken = bundles.root(INDIA) / "day-120-fixme2"
    broken.mkdir(parents=True)
    bundles.promote(INDIA, "day-120-fixme2", {"status": "promoted"})
    with service(None, override=False) as client:
        assert client.get("/api/v1/health").json()["bundle_id"] == "offline"
        shutil.rmtree(broken)
        add_bundle(deployment, "day-120-fixme2")
        bundles.promote(INDIA, "day-120-fixme2", {"status": "promoted"})
        assert client.get("/api/v1/health").json()["bundle_id"] == "day-120-fixme2"
