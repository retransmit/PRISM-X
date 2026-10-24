"""The manifest: every model's threshold and the preprocessing it was fitted under."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from prism.training import manifest

pytest.importorskip("xgboost")

from prism.api.inference import InferenceEngine  # noqa: E402
from prism.data.preprocess import prepare_splits, save_scaler  # noqa: E402
from prism.evaluation.threshold import tune_threshold  # noqa: E402
from prism.explain import artifacts  # noqa: E402
from prism.models import build  # noqa: E402

PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]


@pytest.fixture(autouse=True)
def pinned_primary(monkeypatch):
    """The deployment fixture records no manifest, so there is no champion for auto to serve;
    the auto tests set the key themselves."""
    from prism import config

    monkeypatch.setitem(config.load_config()["api"], "primary_model", "xgboost")


def _scaler(seed: int) -> StandardScaler:
    rng = np.random.default_rng(seed)
    return StandardScaler().fit(rng.normal(size=(50, 2)))


def _two_models(fingerprint: str) -> dict:
    return manifest.merge(None, track="ulb", fingerprint=fingerprint, trained={
        "isolation_forest": {"threshold": 0.9, "resample_applied": False},
        "xgboost": {"threshold": 0.6, "resample_applied": False},
    }, primary="xgboost")


def test_fingerprint_is_stable_for_the_same_fit_and_changes_with_it():
    assert manifest.preprocessing_fingerprint(_scaler(1), None) == manifest.preprocessing_fingerprint(_scaler(1), None)
    assert manifest.preprocessing_fingerprint(_scaler(1), None) != manifest.preprocessing_fingerprint(_scaler(2), None)
    encoder = OrdinalEncoder().fit(pd.DataFrame({"c": ["a", "b"]}))
    assert manifest.preprocessing_fingerprint(_scaler(1), encoder) != manifest.preprocessing_fingerprint(_scaler(1), None)


def test_a_partial_run_keeps_the_other_models_and_their_thresholds():
    second = manifest.merge(_two_models("f1"), track="ulb", fingerprint="f1", trained={
        "xgboost": {"threshold": 0.7, "resample_applied": True},
    }, primary="xgboost")
    assert manifest.thresholds(second) == {"isolation_forest": 0.9, "xgboost": 0.7}
    assert second["models"]["xgboost"]["resample_applied"] is True
    assert second["models"]["isolation_forest"]["fingerprint"] == "f1"


def test_a_partial_run_under_other_preprocessing_is_refused_before_writing():
    with pytest.raises(manifest.PreprocessingMismatch, match="isolation_forest"):
        manifest.check_compatible(_two_models("f1"), "f2", retraining=["xgboost"])
    manifest.check_compatible(_two_models("f1"), "f2", retraining=["isolation_forest", "xgboost"])
    manifest.check_compatible(None, "f2", retraining=["xgboost"])


def test_manifest_round_trips_through_the_model_directory(tmp_path):
    record = _two_models("f1")
    manifest.write(tmp_path, record)
    assert manifest.load(tmp_path) == json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")) == record
    assert manifest.load(tmp_path / "nowhere") is None


@pytest.fixture(scope="module")
def deployment(tmp_path_factory):
    """A small ULB-shaped deployment with two models and thresholds.json only."""
    rng = np.random.default_rng(5)
    n, n_fraud = 3_000, 40
    frame = pd.DataFrame({**{c: rng.normal(size=n) for c in PCA_COLUMNS}, "Time": rng.uniform(0, 172_800, size=n), "Amount": rng.gamma(2.0, 50.0, size=n)})
    labels = np.zeros(n, dtype=int)
    labels[rng.choice(n, size=n_fraud, replace=False)] = 1
    frame["Class"] = labels
    frame.loc[frame["Class"] == 1, PCA_COLUMNS[:4]] += 6.0
    splits = prepare_splits(frame)
    model_dir = tmp_path_factory.mktemp("manifest_models")
    thresholds = {}
    for name in ("isolation_forest", "xgboost"):
        model = build(name, **({"n_estimators": 30} if name == "xgboost" else {})).fit(splits.X_train, splits.y_train)
        model.save(model_dir)
        thresholds[name] = tune_threshold(splits.y_val, model.risk_score(splits.X_val))
    save_scaler(splits.scaler, model_dir / "scaler.joblib")
    (model_dir / "thresholds.json").write_text(json.dumps(thresholds), encoding="utf-8")
    reference, cases = artifacts.build(frame, splits)
    artifacts.save(reference, cases, model_dir)
    return model_dir


def test_the_engine_refuses_a_model_without_a_tuned_threshold(deployment, tmp_path):
    copy = tmp_path / "models"
    shutil.copytree(deployment, copy)
    (copy / "thresholds.json").write_text(json.dumps({"xgboost": 0.6}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="isolation_forest"):
        InferenceEngine(copy).load()


def _on_disk(model_dir) -> str:
    import joblib

    return manifest.preprocessing_fingerprint(joblib.load(model_dir / "scaler.joblib"), None)


def test_the_engine_refuses_a_model_the_manifest_does_not_vouch_for(deployment, tmp_path):
    copy = tmp_path / "models"
    shutil.copytree(deployment, copy)
    record = _two_models(_on_disk(copy))
    record["models"]["xgboost"]["fingerprint"] = "stale"
    manifest.write(copy, record)
    with pytest.raises(RuntimeError, match="xgboost"):
        InferenceEngine(copy).load()


def test_the_engine_serves_what_the_manifest_vouches_for(deployment, tmp_path):
    copy = tmp_path / "models"
    shutil.copytree(deployment, copy)
    manifest.write(copy, _two_models(_on_disk(copy)))
    engine = InferenceEngine(copy).load()
    assert engine.threshold_for("isolation_forest") == 0.9 and engine.threshold_for("xgboost") == 0.6


def test_a_partial_run_replaces_only_its_own_result_rows():
    """test_results.json is the record of every model's test metrics; a run
    that trains one model replaces that model's row and keeps the others."""
    existing = [{"model": "isolation_forest", "pr_auc": 0.2}, {"model": "xgboost", "pr_auc": 0.8}]
    fresh = [{"model": "xgboost", "pr_auc": 0.9}]
    merged = manifest.merge_rows(existing, fresh, replacing=["xgboost"])
    assert merged == [{"model": "isolation_forest", "pr_auc": 0.2}, {"model": "xgboost", "pr_auc": 0.9}]
    assert manifest.merge_rows(None, fresh, replacing=["xgboost"]) == fresh


def test_legacy_thresholds_are_adopted_only_under_identical_preprocessing(deployment):
    """A directory trained before the manifest existed carries thresholds.json
    alone. Its models are adopted when the scaler and encoder on disk are the
    ones this run fitted; otherwise they are recorded as legacy, which a
    partial run refuses and a full retrain replaces."""
    import joblib

    on_disk = manifest.preprocessing_fingerprint(joblib.load(deployment / "scaler.joblib"), None)
    adopted = manifest.adopt_legacy(deployment, on_disk, track="ulb", primary="xgboost")
    assert set(adopted["models"]) == {"isolation_forest", "xgboost"}
    assert all(entry["fingerprint"] == on_disk for entry in adopted["models"].values())
    manifest.check_compatible(adopted, on_disk, retraining=["xgboost"])

    legacy = manifest.adopt_legacy(deployment, "some-other-fit", track="ulb", primary="xgboost")
    assert all(entry["fingerprint"] == "legacy" for entry in legacy["models"].values())
    with pytest.raises(manifest.PreprocessingMismatch):
        manifest.check_compatible(legacy, "some-other-fit", retraining=["xgboost"])
    manifest.check_compatible(legacy, "some-other-fit", retraining=["isolation_forest", "xgboost"])
    assert manifest.adopt_legacy(deployment / "nowhere", on_disk, track="ulb", primary="xgboost") is None


def test_the_engine_refuses_preprocessing_that_differs_from_the_manifest(deployment, tmp_path):
    """The manifest records the fingerprint of the scaler and encoder the
    models were fitted under; the engine checks the files on disk against it,
    so a replaced scaler behind a stale manifest is refused."""
    import joblib

    copy = tmp_path / "models"
    shutil.copytree(deployment, copy)
    on_disk = manifest.preprocessing_fingerprint(joblib.load(copy / "scaler.joblib"), None)
    manifest.write(copy, manifest.adopt_legacy(copy, on_disk, track="ulb", primary="xgboost"))
    InferenceEngine(copy).load()

    joblib.dump(_scaler(9), copy / "scaler.joblib")
    with pytest.raises(RuntimeError, match="preprocessing"):
        InferenceEngine(copy).load()


def test_fingerprint_follows_the_transformation_not_the_pickle_bytes():
    """Two fits with the same means, scales and categories are the same
    preprocessing even when an incidental attribute's dtype differs, as it
    does between a saved scaler and a fresh one; the real Indian directory
    was refused for exactly that until the fingerprint hashed the transformation."""
    import copy

    a = _scaler(3)
    b = copy.deepcopy(a)
    b.n_samples_seen_ = np.int64(a.n_samples_seen_)
    assert manifest.preprocessing_fingerprint(a, None) == manifest.preprocessing_fingerprint(b, None)
    encoder = OrdinalEncoder().fit(pd.DataFrame({"c": ["a", "b"]}))
    twin = copy.deepcopy(encoder)
    twin.encoded_missing_value = float("nan")
    assert manifest.preprocessing_fingerprint(a, encoder) == manifest.preprocessing_fingerprint(a, twin)


def test_a_model_file_replaced_behind_the_manifest_is_refused(deployment, tmp_path):
    """The manifest records each model artifact's digest; a model saved by an
    interrupted run, or swapped in by hand, no longer matches its threshold
    and is refused."""
    copy = tmp_path / "models"
    shutil.copytree(deployment, copy)
    record = manifest.adopt_legacy(copy, _on_disk(copy), track="ulb", primary="xgboost")
    manifest.write(copy, record)
    InferenceEngine(copy).load()
    assert all("sha256" in entry and "artifact" in entry for entry in record["models"].values())

    artifact = copy / record["models"]["xgboost"]["artifact"]
    artifact.write_bytes(artifact.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="xgboost"):
        InferenceEngine(copy).load()


def test_a_manifest_listed_model_that_is_missing_is_refused_not_replaced(deployment, tmp_path):
    """With the champion's file gone the engine used to promote a challenger
    silently; a manifest that lists a model the directory cannot load is refused."""
    copy = tmp_path / "models"
    shutil.copytree(deployment, copy)
    manifest.write(copy, manifest.adopt_legacy(copy, _on_disk(copy), track="ulb", primary="xgboost"))
    (copy / "xgboost.json").unlink()
    with pytest.raises(RuntimeError, match="xgboost"):
        InferenceEngine(copy).load()


def test_merge_records_each_artifact_digest(deployment):
    digests = manifest.artifact_digests(deployment, ["isolation_forest", "xgboost"])
    assert set(digests) == {"isolation_forest", "xgboost"}
    assert digests["xgboost"]["artifact"] == "xgboost.json" and len(digests["xgboost"]["sha256"]) == 64


def test_the_champion_is_the_best_validation_pr_auc_and_auto_serves_it():
    trained = {
        "xgboost": {"threshold": 0.7, "resample_applied": False, "validation_pr_auc": 0.52},
        "autoencoder": {"threshold": 0.6, "resample_applied": False, "validation_pr_auc": 0.55},
        "isolation_forest": {"threshold": 0.5, "resample_applied": False, "validation_pr_auc": 0.08},
    }
    record = manifest.merge(None, track="india", fingerprint="f", trained=trained, primary="auto")
    assert record["champion"]["model"] == "autoencoder" and record["champion"]["value"] == 0.55
    assert record["champion"]["metric"] == "validation_pr_auc"
    assert record["champion"]["candidates"] == {"xgboost": 0.52, "autoencoder": 0.55, "isolation_forest": 0.08}
    assert record["primary_model"] == "autoencoder"
    pinned = manifest.merge(None, track="india", fingerprint="f", trained=trained, primary="xgboost")
    assert pinned["primary_model"] == "xgboost" and pinned["champion"]["model"] == "autoencoder"
    # a one-model retrain keeps the others' evidence and re-chooses over all of them
    again = manifest.merge(record, track="india", fingerprint="f", trained={"xgboost": {"threshold": 0.7, "resample_applied": False, "validation_pr_auc": 0.60}}, primary="auto")
    assert again["champion"]["model"] == "xgboost" and again["primary_model"] == "xgboost"
    assert manifest.merge(None, track="india", fingerprint="f", trained={"isolation_forest": {"threshold": 0.5}}, primary="auto")["champion"] is None


def test_auto_serves_the_champion_the_manifest_records_and_never_a_challenger(deployment, tmp_path, monkeypatch):
    from prism import config

    monkeypatch.setitem(config.load_config()["api"], "primary_model", "auto")
    copy = tmp_path / "models"
    shutil.copytree(deployment, copy)
    record = manifest.merge(None, track="ulb", fingerprint=_on_disk(copy), trained={
        "isolation_forest": {"threshold": 0.9, "resample_applied": False, "validation_pr_auc": 0.2},
        "xgboost": {"threshold": 0.6, "resample_applied": False, "validation_pr_auc": 0.8},
    }, primary="auto")
    manifest.write(copy, record)
    assert InferenceEngine(copy).load().primary == "xgboost"
    # the champion's file is gone: the engine refuses rather than promoting a challenger
    (copy / "xgboost.json").unlink()
    with pytest.raises(RuntimeError, match="xgboost"):
        InferenceEngine(copy).load()
    # a manifest from before champions existed resolves auto to the primary it recorded
    legacy = tmp_path / "legacy"
    shutil.copytree(deployment, legacy)
    manifest.write(legacy, _two_models(_on_disk(legacy)))
    assert InferenceEngine(legacy).load().primary == "xgboost"


def test_a_one_model_retrain_never_crowns_itself_over_models_without_evidence():
    legacy = manifest.merge(None, track="india", fingerprint="f", trained={
        "isolation_forest": {"threshold": 0.9, "resample_applied": False},
        "xgboost": {"threshold": 0.6, "resample_applied": False},
    }, primary="xgboost")
    assert legacy["champion"] is None
    retrained = manifest.merge(legacy, track="india", fingerprint="f", trained={
        "isolation_forest": {"threshold": 0.9, "resample_applied": False, "validation_pr_auc": 0.01},
    }, primary="auto")
    assert retrained["champion"] is None, "no champion without every model's evidence"
    assert retrained["primary_model"] == "xgboost", "auto keeps the primary the record had"
    complete = manifest.merge(retrained, track="india", fingerprint="f", trained={
        "xgboost": {"threshold": 0.6, "resample_applied": False, "validation_pr_auc": 0.5},
    }, primary="auto")
    assert complete["champion"]["model"] == "xgboost" and complete["primary_model"] == "xgboost"
    assert complete["champion"]["candidates"] == {"isolation_forest": 0.01, "xgboost": 0.5}


def test_auto_refuses_a_directory_without_a_manifest(deployment, tmp_path, monkeypatch):
    from prism import config

    monkeypatch.setitem(config.load_config()["api"], "primary_model", "auto")
    copy = tmp_path / "models"
    shutil.copytree(deployment, copy)
    with pytest.raises(RuntimeError, match="manifest"):
        InferenceEngine(copy).load()


def test_a_run_drops_entries_for_models_the_registry_no_longer_has():
    old = manifest.merge(None, track="ulb", fingerprint="f", trained={
        "xgboost": {"threshold": 0.7, "resample_applied": False, "validation_pr_auc": 0.5},
        "retired": {"threshold": 0.9, "resample_applied": False, "validation_pr_auc": 0.9},
    }, primary="auto")
    assert old["primary_model"] == "retired"
    fresh = manifest.merge(old, track="ulb", fingerprint="f", trained={
        "xgboost": {"threshold": 0.7, "resample_applied": False, "validation_pr_auc": 0.5},
    }, primary="auto", known=["xgboost", "autoencoder"])
    assert set(fresh["models"]) == {"xgboost"}
    assert fresh["champion"]["model"] == "xgboost" and fresh["primary_model"] == "xgboost"
    rows = manifest.merge_rows([{"model": "retired"}, {"model": "autoencoder"}], [{"model": "xgboost"}],
                               replacing=["xgboost"], known=["xgboost", "autoencoder"])
    assert rows == [{"model": "autoencoder"}, {"model": "xgboost"}]
