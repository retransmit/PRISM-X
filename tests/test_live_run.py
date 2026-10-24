"""The live stream's runner: it plays the bank, releases labels as the bank learns them and
pauses the stream at each retrain."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("xgboost")
pytest.importorskip("prism_generator")

from prism import config, tracks  # noqa: E402
from prism.api.inference import InferenceEngine  # noqa: E402
from prism.live import run as runner  # noqa: E402
from tests._india_service import CREDENTIALS, build_deployment, service, use_temp_database  # noqa: E402
from tests.conftest import write_india_export  # noqa: E402

DAY = 86_400.0
TRUTH_FIELDS = {"is_fraud", "scenario_tag", "risk_score", "is_fraud_ground_truth"}


class Recorder:
    """The test client, with every request the runner sends written down in order."""

    def __init__(self, client) -> None:
        self.client = client
        self.sent: list[tuple[str, dict]] = []
        self.flagged: dict[str, bool] = {}

    def post(self, path, json=None, headers=None):
        self.sent.append((path, dict(json or {})))
        response = self.client.post(path, json=json, headers=headers)
        if path.endswith("/predict") and response.status_code == 200:
            self.flagged[json["transaction_id"]] = response.json()["flagged"]
        return response

    def get(self, path, headers=None):
        self.sent.append((path, {}))
        return self.client.get(path, headers=headers)

    def predicts(self) -> list[dict]:
        return [body for path, body in self.sent if path.endswith("/predict")]

    def labels(self) -> list[dict]:
        return [body for path, body in self.sent if path.endswith("/labels")]


@pytest.fixture(scope="module")
def deployment(tmp_path_factory, india_stem):
    return build_deployment(india_stem, tmp_path_factory.mktemp("run_models"))


@pytest.fixture(scope="module")
def stream(tmp_path_factory):
    return write_india_export(tmp_path_factory.mktemp("run_stream") / "live_in", n=600, cards=30, seed=5, days=4)


@pytest.fixture(scope="module")
def settings():
    s = dict(config.get("live"))
    s.update({
        "retrain_every_days": 2, "first_retrain_day": 2, "label_lag_days": 0.5, "backtest_days": 0.5, "val_days": 0.5,
        "min_known_fraud": 1, "supervision": {"k_daily": 50, "seed": 0}, "bundle_poll_seconds": 0, "swap_checks": 3,
    })
    return s


@pytest.fixture(scope="module")
def played(deployment, stream, settings, tmp_path_factory):
    """One run of the 600-row stream through the in-process service, retrains recorded;
    every test of the module reads the same run."""
    tmp_path = tmp_path_factory.mktemp("run_played")
    with pytest.MonkeyPatch.context() as monkeypatch:
        return _play(deployment, stream, settings, tmp_path, monkeypatch)


def _play(deployment, stream, settings, tmp_path, monkeypatch):
    use_temp_database(tmp_path, monkeypatch)
    monkeypatch.setitem(config.load_config()["live"], "model_dir", str(tmp_path / "live"))
    engine = InferenceEngine(deployment, track="india").load()
    retrains: list[dict] = []
    with service(engine) as client:
        recorder = Recorder(client)

        def fake_retrain(as_of: float) -> dict:
            sent_before = [body["tx_time_seconds"] for body in recorder.predicts()]
            retrains.append({"as_of": as_of, "sent": sent_before})
            return {"status": "skipped", "reason": "test", "as_of": as_of}

        out = tmp_path / "run.json"
        summary = runner.run(
            recorder, stream, settings, label_token=CREDENTIALS["label_feed"], retrain_fn=fake_retrain, out=out,
        )
    return recorder, retrains, summary, json.loads(out.read_text(encoding="utf-8"))


def test_the_runner_sends_every_row_and_releases_labels_in_discovery_order(played):
    recorder, _, _, log = played
    assert len(recorder.predicts()) == 600
    released = [label["discovery_time_seconds"] for label in recorder.labels()]
    assert released and released == sorted(released)
    # No label goes out before the stream has reached its discovery time.
    clock = None
    for path, body in recorder.sent:
        if path.endswith("/predict"):
            clock = body["tx_time_seconds"]
        elif path.endswith("/labels"):
            assert clock is not None
    for i, (path, body) in enumerate(recorder.sent):
        if path.endswith("/labels"):
            later = [b["tx_time_seconds"] for p, b in recorder.sent[i:] if p.endswith("/predict")]
            if later:
                assert body["discovery_time_seconds"] <= later[0]
    assert sum(day["requests"] for day in log["days"]) == 600
    assert log["labels"]["refused"] == 0


def test_the_investigator_queue_reviews_only_prisms_alerts(played):
    recorder, _, _, _ = played
    investigated = [label["transaction_id"] for label in recorder.labels() if label["label_source"] == "INVESTIGATOR_ALERT"]
    assert all(recorder.flagged[txn] for txn in investigated)
    assert any(recorder.flagged.values()), "the fixture service flags some rows"


def test_the_runner_pauses_for_each_retrain_and_records_it(played, stream):
    _, retrains, _, log = played
    origin = min(body["sent"][0] for body in retrains) if retrains and retrains[0]["sent"] else None
    assert len(retrains) >= 1
    first = retrains[0]
    assert origin is not None
    assert first["as_of"] == pytest.approx(origin + 2 * DAY)
    # Every row before the boundary had been answered, none after it had been sent.
    assert all(t < first["as_of"] for t in first["sent"])
    assert len(first["sent"]) == sum(1 for path, body in played[0].sent if path.endswith("/predict") and body["tx_time_seconds"] < first["as_of"])
    assert [event["as_of"] for event in log["retrains"]] == [r["as_of"] for r in retrains]
    assert len(log["drift"]) == len(retrains)


def test_the_run_log_never_gives_the_truth_to_the_service(played):
    recorder, _, _, _ = played
    for path, body in recorder.sent:
        assert not (TRUTH_FIELDS & set(body)), (path, sorted(TRUTH_FIELDS & set(body)))


def test_the_run_log_covers_every_day_and_scores_the_served_model_against_the_truth(played):
    _, _, summary, log = played
    assert [day["day"] for day in log["days"]] == list(range(len(log["days"])))
    assert sum(day["fraud"] for day in log["days"]) == summary["fraud"]
    assert all(day["caught"] <= day["fraud"] for day in log["days"])
    assert log["windows"] and all(window["bundles"] == ["offline"] for window in log["windows"])
    assert log["settings"]["retrain_every_days"] == 2
    assert log["complete"] is True and log["rows"] == 600 and log["limit"] == 0
    assert "generator_revision" in log


def test_the_live_stream_is_generated_with_the_live_settings_on_a_seed_of_its_own():
    from prism.harness import data as harness_data
    from prism.live import generate

    s = dict(config.get("live"))
    command = generate.command(s)
    assert command[1:4] == ["-m", "prism_generator.cli", "generate"]
    pairs = dict(zip(command[4::2], command[5::2]))
    assert pairs["-n"] == str(s["rows"]) and pairs["--days"] == str(s["days"]) and pairs["--seed"] == str(s["seed"])
    assert pairs["-o"] == str(config.get_path("live.stem")) + ".csv"
    assert int(s["seed"]) not in harness_data.seeds()
    assert int(s["seed"]) != config.seed()


def test_the_simulator_sends_the_payload_the_runner_sends():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "stream", Path(__file__).resolve().parents[1] / "simulator" / "stream.py"
    )
    simulator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(simulator)
    assert simulator.india_payload is runner.india_payload


def test_a_limited_run_is_never_marked_complete(deployment, stream, settings, tmp_path, monkeypatch):
    use_temp_database(tmp_path, monkeypatch)
    monkeypatch.setitem(config.load_config()["live"], "model_dir", str(tmp_path / "live"))
    engine = InferenceEngine(deployment, track="india").load()
    out = tmp_path / "limited.json"
    with service(engine) as client:
        runner.run(client, stream, settings, label_token=CREDENTIALS["label_feed"],
                   retrain_fn=lambda as_of: {"status": "skipped", "as_of": as_of}, out=out, limit=40)
    log = json.loads(out.read_text(encoding="utf-8"))
    assert log["limit"] == 40 and log["complete"] is False


def test_a_failed_retrain_is_recorded_with_its_stream_day(tmp_path, monkeypatch):
    import subprocess

    from prism.live import bundles

    monkeypatch.setitem(config.load_config()["live"], "model_dir", str(tmp_path / "live"))

    class Done:
        returncode = 1
        stdout = ""
        stderr = "Traceback\nMemoryError: out of memory"

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: Done())
    event = runner.subprocess_retrain("sqlite:///x.db", origin=1_000.0)(1_000.0 + 75 * DAY)
    assert event["status"] == "failed" and event["stream_day"] == 75.0
    assert "MemoryError" in event["reason"]
    assert bundles.events(tracks.INDIA)[-1] == event


def test_the_backtest_is_scored_against_the_truth_for_the_report(tmp_path):
    bundle = tmp_path / "day-075-abcdef"
    bundle.mkdir()
    (bundle / "backtest.json").write_text(json.dumps({
        "transaction_id": ["a", "b", "c", "d"], "challenger": [0.9, 0.2, 0.8, 0.1], "champion": [0.1, 0.2, 0.9, 0.8],
    }), encoding="utf-8")
    truth = {"a": 1, "b": 0, "c": 1, "d": 0}
    scored = runner.truth_backtest(bundle, truth)
    assert scored["fraud"] == 2
    assert scored["challenger_pr_auc"] == pytest.approx(1.0)
    assert scored["champion_pr_auc"] < 1.0


class Untouched:
    """A client the runner must not call."""

    def get(self, *args, **kwargs):
        raise AssertionError("the runner called the service")

    post = get


def test_a_schedule_whose_first_retrain_has_no_training_window_is_refused(stream, settings, tmp_path):
    s = dict(settings, first_retrain_day=1.5)
    with pytest.raises(ValueError, match="first_retrain_day 1.5 leaves no training window"):
        runner.run(Untouched(), stream, s, label_token="x", retrain_fn=lambda as_of: {}, out=tmp_path / "run.json")
    assert not (tmp_path / "run.json").exists()


def test_the_configured_schedule_leaves_a_training_window():
    runner.check_schedule(config.get("live"))


def _service_run(deployment, stream, s, tmp_path, monkeypatch, retrain_fn, wrap=None, before=None):
    use_temp_database(tmp_path, monkeypatch)
    monkeypatch.setitem(config.load_config()["live"], "model_dir", str(tmp_path / "live"))
    engine = InferenceEngine(deployment, track="india").load()
    out = tmp_path / "run.json"
    with service(engine) as client:
        if before:
            before(client)
        target = wrap(client) if wrap else client
        return runner.run(target, stream, s, label_token=CREDENTIALS["label_feed"], retrain_fn=retrain_fn, out=out), out


def test_a_live_root_holding_an_earlier_run_is_refused(deployment, stream, settings, tmp_path, monkeypatch):
    from prism.live import bundles

    def leftover(client):
        bundles.record_event(tracks.INDIA, {"status": "kept", "as_of": 1.0})

    with pytest.raises(RuntimeError, match="already holds a run"):
        _service_run(deployment, stream, settings, tmp_path, monkeypatch, lambda as_of: {}, before=leftover)


def test_a_service_whose_audit_log_holds_rows_is_refused(deployment, stream, settings, tmp_path, monkeypatch, india_stem):
    from prism.data.frames import load_frame
    from tests._india_service import payload_from

    def scored_one(client):
        row = load_frame(tracks.INDIA, india_stem).iloc[0]
        assert client.post("/api/v1/predict", json=payload_from(row)).status_code == 200

    with pytest.raises(RuntimeError, match="audit log already holds 1 payment"):
        _service_run(deployment, stream, settings, tmp_path, monkeypatch, lambda as_of: {}, before=scored_one)


class Flaky(Recorder):
    """The service answering one payment with a non-JSON 500 and timing out on another."""

    class Broken:
        status_code = 500
        text = "Internal Server Error"

        def json(self):
            raise ValueError("not JSON")

    def post(self, path, json=None, headers=None):
        if path.endswith("/predict"):
            n = len(self.predicts())
            if n == 2:
                self.sent.append((path, dict(json or {})))
                return self.Broken()
            if n == 4:
                import httpx

                self.sent.append((path, dict(json or {})))
                raise httpx.ReadTimeout("timed out")
        return super().post(path, json=json, headers=headers)


def test_a_non_json_error_or_a_timeout_counts_as_a_failed_request(deployment, stream, settings, tmp_path, monkeypatch):
    summary, out = _service_run(
        deployment, stream, settings, tmp_path, monkeypatch,
        lambda as_of: {"status": "skipped", "as_of": as_of}, wrap=Flaky,
    )
    assert summary["failed_requests"] == 2
    assert summary["requests"] == 598
    assert json.loads(out.read_text(encoding="utf-8"))["complete"] is True


def test_a_retrain_that_read_no_rows_stops_the_run(deployment, stream, settings, tmp_path, monkeypatch):
    def empty(as_of):
        return {"status": "skipped", "as_of": as_of, "rows": {"train": 0, "val": 0, "backtest": 0}}

    with pytest.raises(RuntimeError, match="does not name the service's audit log"):
        _service_run(deployment, stream, settings, tmp_path, monkeypatch, empty)
    log = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert log["complete"] is False and len(log["retrains"]) == 1


def test_a_promotion_the_service_never_serves_stops_the_run(deployment, stream, settings, tmp_path, monkeypatch):
    def promoted(as_of):
        return {"status": "promoted", "as_of": as_of, "challenger_bundle": "day-002-absent"}

    with pytest.raises(RuntimeError, match="did not swap to the promoted bundle day-002-absent after 3 health checks"):
        _service_run(deployment, stream, settings, tmp_path, monkeypatch, promoted)
