"""One run of the harness over a small stream: the invariants the design states, checked
on the rows, and the artifacts the driver writes."""

from __future__ import annotations

import json

import pytest

from prism import config, tracks
from prism.harness import data, folds, run, timeline

pytest.importorskip("xgboost")


@pytest.fixture
def quick(monkeypatch):
    harness = config.load_config()["harness"]
    monkeypatch.setitem(harness, "train_days", 12)
    monkeypatch.setitem(harness, "val_days", 4)
    monkeypatch.setitem(harness, "test_days", 6)
    monkeypatch.setitem(harness, "step_days", 8)
    monkeypatch.setitem(harness, "bootstrap_resamples", 50)
    monkeypatch.setitem(harness, "scoring_rows", 20)
    monkeypatch.setitem(config.load_config()["models"]["autoencoder"], "epochs", 1)
    monkeypatch.setitem(config.load_config()["models"]["xgboost"], "n_estimators", 10)
    monkeypatch.setitem(config.load_config()["models"]["isolation_forest"], "n_estimators", 20)
    tracks.set_active("india")
    yield
    tracks.set_active("ulb")


@pytest.fixture
def seed_zero(india_stem, monkeypatch):
    monkeypatch.setattr(data, "stem", lambda seed: india_stem)
    monkeypatch.setitem(config.load_config()["harness"], "seeds", [0])


def test_one_run_respects_the_windows_the_labels_and_the_fit_order(seed_zero, quick):
    frame = data.load_seed(0)
    tl = run.aligned_timeline(frame, timeline.load_or_build(0, frame))
    fold = folds.rolling_origin(frame["tx_time_seconds"].to_numpy(), **run.window_settings())[0]
    row = run.run_one(frame, tl, fold, "xgboost", "operational", tracks.INDIA, run.settings())
    assert row["fold"] == "fold-1" and row["model"] == "xgboost" and row["label_mode"] == "operational"
    # operational training labels: a fraud discovered after the cut-off trains as legitimate
    known = timeline.known_fraud_at(tl, fold.train_end).to_numpy()
    truth = frame["is_fraud"].astype(int).to_numpy()
    assert row["training_fraud_rows"] == int(known[fold.train].sum())
    assert 0 < row["training_fraud_rows"] < int(truth[fold.train].sum())
    # evaluation labels are the truth, and the test window is scored whole
    assert row["test_fraud_rows"] == int(truth[fold.test].sum())
    assert row["test_rows"] == len(fold.test)
    assert 0.0 <= row["pr_auc"] <= 1.0 and 0.0 <= row["threshold"] <= 1.0
    for key in (
        "recall", "precision", "fpr_per_10k", "adjusted_precision_at_prevalence", "share_of_fraud_value_caught",
        "alerts_per_day", "alerts_per_day_over_capacity", "cost_total_inr_per_10k", "cost_loss_inr_per_10k",
        "cost_review_inr_per_10k", "cost_friction_inr_per_10k", "alerts_per_10k", "missed_fraud_per_10k",
        "scoring_p99_ms", "fit_seconds",
    ):
        assert key in row, key
    assert row["scoring_p99_ms"] > 0.0
    assert "held_out_recall" not in row
    full = run.run_one(frame, tl, fold, "xgboost", "full", tracks.INDIA, run.settings())
    assert full["training_fraud_rows"] == int(truth[fold.train].sum())


def test_an_unknown_label_mode_is_refused(seed_zero, quick):
    frame = data.load_seed(0)
    tl = run.aligned_timeline(frame, timeline.load_or_build(0, frame))
    fold = folds.rolling_origin(frame["tx_time_seconds"].to_numpy(), **run.window_settings())[0]
    with pytest.raises(ValueError, match="label mode"):
        run.run_one(frame, tl, fold, "xgboost", "oracle", tracks.INDIA, run.settings())


def test_a_scenario_fold_measures_recall_on_the_held_out_rows_only(seed_zero, quick):
    frame = data.load_seed(0)
    tl = run.aligned_timeline(frame, timeline.load_or_build(0, frame))
    base = folds.rolling_origin(frame["tx_time_seconds"].to_numpy(), **run.window_settings())[-1]
    held = folds.leave_one_scenario_out(frame, base, "INTENT_OMEGA_HARVEST")
    row = run.run_one(frame, tl, held, "xgboost", "full", tracks.INDIA, run.settings())
    tags = frame.loc[base.test, "scenario_tag"].to_numpy()
    assert row["held_out_scenario"] == "INTENT_OMEGA_HARVEST"
    assert row["held_out_fraud_rows"] == int((tags == "INTENT_OMEGA_HARVEST").sum())
    assert 0.0 <= row["held_out_recall"] <= 1.0
    assert row["training_fraud_rows"] == int((frame.loc[held.train, "scenario_tag"] == "INTENT_OMEGA_PROBE").sum())


def test_the_driver_writes_results_for_every_seed_fold_model_and_mode_and_the_scenario_folds(seed_zero, quick, tmp_path):
    models = ["xgboost", "isolation_forest", "autoencoder"]
    out = run.run_all(seeds=[0], models=models, modes=["operational", "full"], out_dir=tmp_path)
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    n_folds = len(folds.rolling_origin(data.load_seed(0)["tx_time_seconds"].to_numpy(), **run.window_settings()))
    assert n_folds == 2
    assert len(results["rows"]) == n_folds * len(models) * 2
    assert {(r["seed"], r["fold"], r["model"], r["label_mode"]) for r in results["rows"]} == {
        (0, f"fold-{i}", m, mode) for i in (1, 2) for m in models for mode in ("operational", "full")
    }
    import math

    assert all(math.isfinite(r["explain_p99_ms"]) and r["explain_p99_ms"] > 0.0 for r in results["rows"]), "every model is explained and timed"
    assert results["settings"]["label_modes"] == ["operational", "full"]
    assert results["settings"]["models"] == models
    assert results["generator_revision"] is None or len(results["generator_revision"]) == 40
    assert results["timelines"][0]["seed"] == 0 and results["timelines"][0]["fraud_rows"] > 0
    scenario = json.loads((out / "scenario.json").read_text(encoding="utf-8"))
    assert all(r["label_mode"] == "full" and r["fold"].startswith("fold-2-without-") for r in scenario["rows"])
    assert all("held_out_recall" in r and "held_out_fraud_rows" in r for r in scenario["rows"])
    assert {r["held_out_scenario"] for r in scenario["rows"]} == set(folds.scenarios(data.load_seed(0)))
    assert not any(k.startswith("_") for r in results["rows"] + scenario["rows"] for k in r)


def test_a_validation_window_with_no_known_fraud_is_cut_at_the_budget_alone(seed_zero, quick):
    import numpy as np

    frame = data.load_seed(0)
    tl = run.aligned_timeline(frame, timeline.load_or_build(0, frame)).copy()
    fold = folds.rolling_origin(frame["tx_time_seconds"].to_numpy(), **run.window_settings())[0]
    # nothing in the validation window has been discovered by the time the model is deployed
    tl.loc[fold.val, "discovery_time_seconds"] = np.nan
    row = run.run_one(frame, tl, fold, "xgboost", "operational", tracks.INDIA, run.settings())
    assert row["validation_fraud_rows"] == 0
    assert row["threshold_from_budget_alone"] is True
    assert 0.0 < row["threshold"] < 1.0
    full = run.run_one(frame, tl, fold, "xgboost", "full", tracks.INDIA, run.settings())
    assert full["threshold_from_budget_alone"] is False


def test_fold_matrices_fit_the_scaler_on_the_training_rows_alone(seed_zero, quick):
    import numpy as np
    import pytest as _pytest

    frame = data.load_seed(0)
    fold = folds.rolling_origin(frame["tx_time_seconds"].to_numpy(), **run.window_settings())[0]
    frame = frame.copy()
    later = np.zeros(len(frame), dtype=bool)
    later[fold.val] = True
    later[fold.test] = True
    frame.loc[later, "amount"] = frame.loc[later, "amount"] * 1000.0  # a different scale after the cut-off
    X_train, X_val, X_test, encoder, scaler = run.fold_matrices(frame, fold, tracks.INDIA)
    column = list(tracks.INDIA.scale_columns).index("amount")
    assert scaler.mean_[column] == _pytest.approx(frame.loc[fold.train, "amount"].mean())
    assert len(X_train) == len(fold.train) and len(X_val) == len(fold.val) and len(X_test) == len(fold.test)
    assert abs(X_train["amount"].mean()) < 1e-9 and X_test["amount"].mean() > 100.0


def test_training_labels_on_hand_built_rows():
    import numpy as np
    import pandas as pd

    cutoff = 1000.0
    truth = np.array([1, 1, 1, 1, 0, 0])
    tl = pd.DataFrame({
        "transaction_id": [f"T{i}" for i in range(6)],
        "is_fraud": truth,
        # discovered before the cut-off, exactly at it, after it, never; a legitimate row
        # investigated before the cut-off; a legitimate row never labelled
        "discovery_time_seconds": [500.0, 1000.0, 1000.5, np.nan, 400.0, np.nan],
        "label_source": ["INVESTIGATOR_ALERT", "CHARGEBACK_DISPUTE", "CHARGEBACK_DISPUTE", "UNREPORTED_DARK_FRAUD", "INVESTIGATOR_ALERT", "UNLABELLED"],
    })
    assert run.training_labels(truth, tl, cutoff, "operational").tolist() == [1, 1, 0, 0, 0, 0]
    assert run.training_labels(truth, tl, cutoff, "full").tolist() == truth.tolist()
    assert timeline.known_fraud_at(tl, cutoff).tolist() == [True, True, False, False, False, False]


def test_results_are_checkpointed_after_every_seed_and_summarised_from_them(seed_zero, quick, tmp_path, monkeypatch):
    real = run.run_seed
    calls = []

    def flaky(seed, models, modes, schema, s):
        calls.append(seed)
        if len(calls) == 2:
            raise RuntimeError("the second seed dies")
        return real(0, models, modes, schema, s)

    monkeypatch.setattr(run, "run_seed", flaky)
    with pytest.raises(RuntimeError):
        run.run_all(seeds=[0, 1], models=["xgboost"], modes=["full"], out_dir=tmp_path)
    partial = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert {r["seed"] for r in partial["rows"]} == {0} and partial["settings"]["seeds"] == [0, 1]
    assert not (tmp_path / "summary.json").exists()
    run.summarize(tmp_path)
    written = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert written["models"][0]["n_folds"] == len(partial["rows"])
    assert (tmp_path / "summary.md").exists() and (tmp_path / "scenario.md").exists()
