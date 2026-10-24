"""The retrain job: a challenger on the labels known so far, backtested against the champion."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("xgboost")

from prism import config, tracks  # noqa: E402
from prism.api import db  # noqa: E402
from prism.data.frames import load_frame  # noqa: E402
from prism.data.preprocess import load_encoder, load_scaler, transform_features  # noqa: E402
from prism.evaluation.threshold import tune_threshold  # noqa: E402
from prism.live import bundles, drift, retrain  # noqa: E402
from prism.models import get_model_class  # noqa: E402
from prism.training import manifest  # noqa: E402
from tests._india_service import build_deployment, payload_from  # noqa: E402
from tests.conftest import START  # noqa: E402

INDIA = tracks.INDIA
DAY = 86_400.0
SETTINGS = {
    "label_lag_days": 5, "backtest_days": 5, "val_days": 5, "min_known_fraud": 5,
    "bootstrap_resamples": 50, "drift_warn": 0.10, "drift_alert": 0.25, "drift_epsilon": 0.0001,
    "category_floor": 0.01, "drift_discrete_max": 10,
}
AS_OF = START + 30 * DAY


@pytest.fixture(scope="module")
def frame(india_stem):
    return load_frame(INDIA, india_stem)


@pytest.fixture(scope="module")
def deployment(tmp_path_factory, india_stem):
    return build_deployment(india_stem, tmp_path_factory.mktemp("retrain_offline"))


def fill_log(url: str, frame, *, labelled: bool) -> None:
    """The log as the service would have written it, without scoring: every row's inputs,
    and (``labelled``) every fraud reported a day after it happened."""
    db.init_db(url)
    for _, row in frame.iterrows():
        db.log_transaction(
            transaction_id=str(row["transaction_id"]), features={}, primary_model="xgboost", risk_score=0.0,
            flagged=False, latency_ms=0.0, scores=[], explanation=None, amount=float(row["amount"]),
            time_feature=float(row["tx_time_seconds"]), request=payload_from(row),
        )
        if labelled and int(row["is_fraud"]) == 1:
            db.record_label(
                transaction_id=str(row["transaction_id"]), label=1, source="CHARGEBACK_DISPUTE",
                known_at=float(row["tx_time_seconds"]) + DAY,
            )


@pytest.fixture(scope="module")
def logs(tmp_path_factory, frame):
    folder = tmp_path_factory.mktemp("retrain_logs")
    urls = {}
    for name, labelled in (("labelled", True), ("unlabelled", False)):
        urls[name] = f"sqlite:///{(folder / f'{name}.db').as_posix()}"
        fill_log(urls[name], frame, labelled=labelled)
    return urls


@pytest.fixture(autouse=True)
def small_models(monkeypatch):
    models = config.load_config()["models"]
    monkeypatch.setitem(models["xgboost"], "n_estimators", 40)
    monkeypatch.setitem(models["autoencoder"], "epochs", 3)
    monkeypatch.setitem(models["isolation_forest"], "n_estimators", 50)
    monkeypatch.setitem(config.load_config()["api"], "primary_model", "xgboost")
    # The fixture's fraud is separable on purpose; the audit's ceiling is tested on its own below.
    monkeypatch.setitem(config.load_config()["tracks"]["india"], "shortcut_auc_ceiling", 1.0)


@pytest.fixture
def models_root(tmp_path, monkeypatch, deployment):
    root = tmp_path / "models"
    shutil.copytree(deployment, root / "india")
    monkeypatch.setitem(config.load_config()["api"], "model_dir", str(root))
    monkeypatch.setitem(config.load_config()["live"], "model_dir", str(root / "live"))
    tracks.set_active("india")
    return root


@pytest.fixture
def labelled(logs, models_root):
    db.init_db(logs["labelled"])


def fixed_interval(monkeypatch, low: float) -> None:
    monkeypatch.setattr(
        retrain, "paired_bootstrap",
        lambda y, a, b, **kwargs: {"low": low, "high": low + 0.1, "share_b_at_least_a": 0.0},
    )


def test_windows_end_a_label_lag_before_the_clock():
    s = {"label_lag_days": 30, "backtest_days": 20, "val_days": 10}
    w = retrain.windows(100 * DAY, s, origin=0.0)
    assert w["backtest"] == (50 * DAY, 70 * DAY)
    assert w["val"] == (40 * DAY, 50 * DAY)
    assert w["train"] == (0.0, 40 * DAY)


def test_a_retrain_without_known_fraud_is_skipped_with_its_reason(logs, models_root):
    db.init_db(logs["unlabelled"])
    event = retrain.retrain(AS_OF, INDIA, SETTINGS)
    assert event["status"] == "skipped"
    assert "training window" in event["reason"] and "0 known" in event["reason"]
    assert [p for p in bundles.root(INDIA).glob("*") if p.is_dir()] == []
    assert not bundles.pointer_path(INDIA).exists()
    assert bundles.events(INDIA)[-1]["status"] == "skipped"


def test_a_challenger_that_does_not_win_by_the_interval_is_kept_out(labelled, monkeypatch):
    fixed_interval(monkeypatch, -0.01)
    event = retrain.retrain(AS_OF, INDIA, SETTINGS)
    assert event["status"] == "kept"
    assert event["difference"]["low"] == -0.01
    assert not bundles.pointer_path(INDIA).exists()
    assert (bundles.root(INDIA) / event["challenger_bundle"] / manifest.FILENAME).exists()
    assert event["champion_bundle"] == "offline"
    assert event["known_fraud"]["backtest"] >= SETTINGS["min_known_fraud"]
    assert bundles.events(INDIA)[-1] == json.loads(json.dumps(event))


def test_a_challenger_that_wins_is_promoted(labelled, monkeypatch):
    fixed_interval(monkeypatch, 0.02)
    event = retrain.retrain(AS_OF, INDIA, SETTINGS)
    assert event["status"] == "promoted"
    assert bundles.serving_dir(INDIA)[0] == event["challenger_bundle"]
    record = manifest.load(bundles.root(INDIA) / event["challenger_bundle"])
    assert record["val_end"] == event["windows"]["val"][1]
    assert record["train_end"] == event["windows"]["train"][1]
    assert record["as_of"] == AS_OF
    reference = drift.load(bundles.root(INDIA) / event["challenger_bundle"])
    assert reference is not None
    assert reference["rows"] == event["rows"]["train"] + event["rows"]["val"]


def test_a_backtest_window_inside_the_champions_own_windows_is_refused(labelled, monkeypatch):
    fixed_interval(monkeypatch, 0.02)
    first = retrain.retrain(AS_OF, INDIA, SETTINGS)
    assert first["status"] == "promoted"
    # A clock behind the serving bundle (a run replayed over an older log): the backtest window
    # starts before the serving bundle's validation window ends.
    second = retrain.retrain(AS_OF - DAY, INDIA, SETTINGS)
    assert second["status"] == "skipped"
    assert "overlaps" in second["reason"]
    assert bundles.serving_dir(INDIA)[0] == first["challenger_bundle"]


def test_the_challengers_threshold_comes_from_its_validation_window(labelled, monkeypatch, frame):
    fixed_interval(monkeypatch, -0.01)
    event = retrain.retrain(AS_OF, INDIA, SETTINGS)
    bundle = bundles.root(INDIA) / event["challenger_bundle"]
    start, end = event["windows"]["val"]
    val = frame[(frame["tx_time_seconds"] >= start) & (frame["tx_time_seconds"] < end)]
    X = transform_features(
        val, load_scaler(bundle / "scaler.joblib"), list(INDIA.feature_columns), list(INDIA.scale_columns),
        encoder=load_encoder(bundle / "encoder.joblib"), categorical_columns=INDIA.categorical_columns,
    )
    scores = get_model_class("xgboost").load(bundle).risk_score(X)
    y = val["is_fraud"].to_numpy()  # every fraud is reported a day later, inside the cut-off
    expected = tune_threshold(y, np.asarray(scores), strategy="fpr_budget")
    assert manifest.load(bundle)["models"]["xgboost"]["threshold"] == pytest.approx(expected, abs=1e-12)


def test_the_cli_prints_the_event_and_records_it(logs, models_root, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["retrain", "--as-of", str(AS_OF), "--database-url", logs["unlabelled"]])
    retrain.main()
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "skipped"
    assert bundles.events(INDIA)[-1] == printed


def test_a_retrain_whose_training_rows_give_the_label_away_is_skipped(labelled, monkeypatch):
    monkeypatch.setitem(config.load_config()["tracks"]["india"], "shortcut_auc_ceiling", 0.5)
    event = retrain.retrain(AS_OF, INDIA, SETTINGS)
    assert event["status"] == "skipped"
    assert "shortcut" in event["reason"] and "AUC" in event["reason"]
    assert [p for p in bundles.root(INDIA).glob("*") if p.is_dir()] == []


def test_the_bundle_keeps_its_shortcut_audit_and_its_backtest_scores(labelled, monkeypatch):
    fixed_interval(monkeypatch, -0.01)
    event = retrain.retrain(AS_OF, INDIA, SETTINGS)
    bundle = bundles.root(INDIA) / event["challenger_bundle"]
    audit = json.loads((bundle / "shortcut_audit.json").read_text(encoding="utf-8"))
    assert audit and "auc" in audit[0]
    backtest = json.loads((bundle / retrain.BACKTEST_FILE).read_text(encoding="utf-8"))
    assert len(backtest["transaction_id"]) == len(backtest["challenger"]) == len(backtest["champion"])
    assert len(backtest["transaction_id"]) == event["rows"]["backtest"]
    assert "label" not in backtest and "is_fraud" not in backtest
