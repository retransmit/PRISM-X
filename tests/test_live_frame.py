"""The live training frame: what the service scored, with the labels known at a cut-off."""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("xgboost")

from prism import config, tracks  # noqa: E402
from prism.api import db  # noqa: E402
from prism.api.inference import InferenceEngine  # noqa: E402
from prism.data.frames import load_frame  # noqa: E402
from prism.live.frame import labelled_frame  # noqa: E402
from tests._india_service import CREDENTIALS, build_deployment, payload_from, service  # noqa: E402
from tests.conftest import write_india_export  # noqa: E402

INDIA = tracks.INDIA


@pytest.fixture(scope="module")
def batch(tmp_path_factory):
    """A 300-row export, separate from the deployment's training export."""
    return write_india_export(tmp_path_factory.mktemp("live_batch") / "batch_in", n=300, cards=20, seed=3, days=10)


@pytest.fixture(scope="module")
def offline(batch):
    return load_frame(INDIA, batch)


@pytest.fixture(scope="module")
def logged(tmp_path_factory, india_stem, offline):
    """Every row of the batch scored through the service once, into one log for the module."""
    model_dir = build_deployment(india_stem, tmp_path_factory.mktemp("live_frame_models"))
    url = f"sqlite:///{(tmp_path_factory.mktemp('live_frame_db') / 'audit.db').as_posix()}"
    with pytest.MonkeyPatch.context() as patch:
        api = config.load_config()["api"]
        patch.setitem(api, "database_url", url)
        patch.setitem(api, "primary_model", "xgboost")
        patch.setitem(api, "credentials", dict(CREDENTIALS))
        db.init_db(url)
        engine = InferenceEngine(model_dir, track="india").load()
        with service(engine) as client:
            for _, row in offline.iterrows():
                assert client.post("/api/v1/predict", json=payload_from(row)).status_code == 200
    return url


@pytest.fixture(autouse=True)
def this_log(logged):
    db.init_db(logged)
    yield


def test_the_live_frame_has_the_offline_frames_columns(offline):
    live = labelled_frame(float(offline[INDIA.time_column].max()) + 1.0, INDIA)
    assert list(live.columns) == list(offline.columns)
    assert live.dtypes.to_dict() == offline.dtypes.to_dict()
    assert len(live) == len(offline)
    features = [c for c in offline.columns if c != INDIA.target_column]
    pd.testing.assert_frame_equal(live[features], offline[features], check_exact=False, rtol=1e-12)


def test_only_rows_before_the_cut_off_are_in_the_frame(offline):
    cut = float(offline[INDIA.time_column].iloc[150])
    live = labelled_frame(cut, INDIA)
    assert len(live) == int((offline[INDIA.time_column] < cut).sum())
    assert (live[INDIA.time_column] < cut).all()


def test_only_labels_known_at_the_cut_off_make_a_row_fraud(offline):
    fraud = offline[offline[INDIA.target_column] == 1]["transaction_id"].tolist()
    early, late = fraud[0], fraud[1]
    as_of = float(offline[INDIA.time_column].max()) + 1.0
    db.record_label(transaction_id=early, label=1, source="CHARGEBACK_DISPUTE", known_at=as_of - 10.0)
    db.record_label(transaction_id=late, label=1, source="CHARGEBACK_DISPUTE", known_at=as_of + 10.0)
    live = labelled_frame(as_of, INDIA)
    assert live.loc[live[INDIA.target_column] == 1, "transaction_id"].tolist() == [early]
    assert live[INDIA.target_column].dtype == offline[INDIA.target_column].dtype


def test_velocity_features_match_the_offline_derivation(offline):
    live = labelled_frame(float(offline[INDIA.time_column].max()) + 1.0, INDIA)
    for column in ("card_tx_24h", "seconds_since_last", "card_distinct_mcc_30m", "card_amount_24h"):
        assert live[column].tolist() == pytest.approx(offline[column].tolist(), rel=1e-12), column


def test_an_empty_log_gives_an_empty_frame_with_the_same_columns(offline, tmp_path):
    db.init_db(f"sqlite:///{(tmp_path / 'empty.db').as_posix()}")
    live = labelled_frame(1e12, INDIA)
    assert live.empty
    assert list(live.columns) == list(offline.columns)
