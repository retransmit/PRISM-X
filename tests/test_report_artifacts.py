"""Every number the report quotes comes from a file a pipeline command wrote."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from prism import tracks
from prism.data import eda, india
from prism.data.preprocess import prepare_splits
from prism.features.velocity import add_velocity_features


def _ulb_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({name: rng.normal(size=n) for name in tracks.ULB.feature_columns})
    frame["Class"] = (np.arange(n) % 40 == 0).astype(int)
    return frame


def test_split_summary_is_available_as_a_dict():
    splits = prepare_splits(_ulb_frame(400, 0), schema=tracks.ULB)
    summary = splits.as_dict()
    assert summary["track"] == "ulb" and summary["rule"] == "stratified"
    assert summary["train"]["rows"] + summary["val"]["rows"] + summary["test"]["rows"] == 400
    assert summary["train"]["fraud"] + summary["val"]["fraud"] + summary["test"]["fraud"] == 10
    assert summary["duplicates_dropped"] == 0
    assert summary["n_features"] == len(tracks.ULB.feature_columns)
    assert summary["categorical_columns"] == []


def test_indian_split_summary_names_the_time_ordered_rule(india_stem):
    frame = add_velocity_features(india.load_india(india_stem))
    summary = prepare_splits(frame, schema=tracks.INDIA).as_dict()
    assert summary["rule"] == "time-ordered" and summary["track"] == "india"
    assert set(summary["categorical_columns"]) == set(tracks.INDIA.categorical_columns)


def test_export_facts_count_what_the_readme_quotes(india_stem):
    facts = india.export_facts(india_stem)
    frame = india.load_india(india_stem)
    assert facts["rows"] == len(frame)
    assert facts["fraud_rows"] == int(frame["is_fraud"].astype(int).sum())
    assert facts["cards"] == frame["card_id"].nunique()
    assert facts["merchants"] == frame["merchant_id"].nunique()
    span = (frame["tx_time_seconds"].astype(float).max() - frame["tx_time_seconds"].astype(float).min()) / 86_400.0
    assert facts["span_days"] == pytest.approx(span, abs=0.01)
    assert facts["calibration"]["profile"] == "rbi-psi-2026-07"
    assert facts["calibration"]["overall"] == "PASS"
    assert facts["calibration"]["generated"]["rows"] == len(frame)
    assert facts["calibration"]["gated"] == []


def test_export_facts_without_a_calibration_report(india_stem, tmp_path):
    for key, path in india.export_paths(india_stem).items():
        if key != "calibration":
            (tmp_path / path.name).write_bytes(path.read_bytes())
    facts = india.export_facts(tmp_path / india_stem.name)
    assert facts["calibration"] is None and facts["rows"] > 0


def test_training_writes_the_split_and_the_exports(india_stem, tmp_path, monkeypatch):
    """The two artifacts the results generator reads beside test_results.json."""
    from prism import config
    from prism.training import train

    settings = config.load_config()["tracks"]["india"]
    monkeypatch.setitem(settings, "export_stem", str(india_stem))
    monkeypatch.setitem(settings, "demo_stem", str(india_stem))
    frame = add_velocity_features(india.load_india(india_stem))
    splits = prepare_splits(frame, schema=tracks.INDIA)

    train.write_report_artifacts(tracks.INDIA, frame, splits, tmp_path)

    split = json.loads((tmp_path / "split.json").read_text(encoding="utf-8"))
    assert split["rule"] == "time-ordered"
    assert split["loaded"]["n_fraud"] == int(frame["is_fraud"].astype(int).sum())
    exports = json.loads((tmp_path / "exports.json").read_text(encoding="utf-8"))
    assert exports["training"]["rows"] == len(frame) and exports["demo"]["rows"] == len(frame)


def test_an_absent_demo_export_is_recorded_as_none(india_stem, tmp_path, monkeypatch):
    from prism import config
    from prism.training import train

    settings = config.load_config()["tracks"]["india"]
    monkeypatch.setitem(settings, "export_stem", str(india_stem))
    monkeypatch.setitem(settings, "demo_stem", str(tmp_path / "nowhere" / "demo_in"))
    frame = add_velocity_features(india.load_india(india_stem))
    splits = prepare_splits(frame, schema=tracks.INDIA)

    train.write_report_artifacts(tracks.INDIA, frame, splits, tmp_path)

    exports = json.loads((tmp_path / "exports.json").read_text(encoding="utf-8"))
    assert exports["training"]["rows"] == len(frame) and exports["demo"] is None


def test_a_track_without_exports_writes_the_split_only(tmp_path):
    from prism.training import train

    frame = _ulb_frame(300, 1)
    splits = prepare_splits(frame, schema=tracks.ULB)
    train.write_report_artifacts(tracks.ULB, frame, splits, tmp_path)
    assert (tmp_path / "split.json").exists() and not (tmp_path / "exports.json").exists()


def test_eda_writes_into_the_ulb_track_directories():
    metrics_path, figures_dir = eda.output_paths()
    assert metrics_path == tracks.ULB.metrics_dir / "eda.json"
    assert figures_dir == tracks.ULB.figures_dir


def test_split_artifact_records_the_settings_it_was_trained_under(india_stem, tmp_path, monkeypatch):
    """The shares and the budget the section quotes must come from the run, not
    from config.yaml at render time; a later config change must not reword an
    older run's numbers."""
    from prism import config
    from prism.training import train

    settings = config.load_config()["tracks"]["india"]
    monkeypatch.setitem(settings, "export_stem", str(india_stem))
    monkeypatch.setitem(settings, "demo_stem", str(india_stem))
    frame = add_velocity_features(india.load_india(india_stem))
    splits = prepare_splits(frame, schema=tracks.INDIA)

    train.write_report_artifacts(tracks.INDIA, frame, splits, tmp_path)

    recorded = json.loads((tmp_path / "split.json").read_text(encoding="utf-8"))["settings"]
    assert recorded["threshold_strategy"] == settings["threshold_strategy"]
    assert recorded["fpr_budget_per_10k"] == settings["fpr_budget_per_10k"]
    assert recorded["train_fraction"] == settings["train_fraction"]


def test_the_copied_calibration_report_is_sanitised(tmp_path):
    """The generator's report names the generator in its scope line; the copy
    beside the metrics keeps the measurements and not the name."""
    import base64
    import json

    from prism.training import train

    # The names the report carries, held encoded like the pattern that removes them.
    line, package = (base64.b64decode(s).decode() for s in ("RnJhdWR4QUk=", "ZnJhdWR4X3N5bnRoZXNpemVy"))
    report = tmp_path / "x_calibration.json"
    report.write_text(json.dumps({
        "profile": "rbi-psi-2026-07",
        "scope": f"{line} calibration of the IN card rail against RBI PSI",
        "generated": {"rows": 10, "fraud_rows": 1, "tool": f"{package} 0.3.0"},
        "targets": [{"name": "mean_credit_card_ticket_inr", "status": "PASS", "note": f"measured by {line}"}],
        "overall": "PASS",
    }), encoding="utf-8")
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()

    copied = train.copy_calibration_report(report, metrics_dir)

    text = copied.read_text(encoding="utf-8")
    assert not train.GENERATOR_NAMES.search(text)
    data = json.loads(text)
    assert data["generated"]["rows"] == 10 and data["overall"] == "PASS"
    assert data["targets"][0]["status"] == "PASS"


def test_export_facts_record_the_generator_revision_and_a_project_relative_stem(tmp_path, monkeypatch):
    """What the report quotes must say which generator made the export: the revision in
    generator/SOURCE when the generator ships in the repository, and a stem that does not
    carry one machine's drive letter."""
    from prism import config
    from tests.conftest import write_india_export

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    (tmp_path / "generator").mkdir()
    (tmp_path / "generator" / "SOURCE").write_text(
        "revision 0123456789abcdef0123456789abcdef01234567\ncommitted 2026-10-08T03:36:06+05:30\n", encoding="utf-8"
    )
    inside = tmp_path / "data" / "raw" / "india"
    inside.mkdir(parents=True)
    stem = write_india_export(inside / "train_in")
    facts = india.export_facts(stem)
    assert facts["generator_revision"] == "0123456789abcdef0123456789abcdef01234567"
    assert facts["stem"] == "data/raw/india/train_in"

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path / "elsewhere")
    facts = india.export_facts(stem)
    assert facts["generator_revision"] is None
    assert facts["stem"] == stem.as_posix()


def test_export_facts_record_the_generator_revision_and_a_project_relative_stem(tmp_path, monkeypatch):
    """What the report quotes must say which generator made the export: the revision in
    generator/SOURCE when the generator ships in the repository, and a stem that does not
    carry one machine's drive letter."""
    from prism import config
    from tests.conftest import write_india_export

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    (tmp_path / "generator").mkdir()
    (tmp_path / "generator" / "SOURCE").write_text(
        "revision 0123456789abcdef0123456789abcdef01234567\ncommitted 2026-10-08T03:36:06+05:30\n", encoding="utf-8"
    )
    inside = tmp_path / "data" / "raw" / "india"
    inside.mkdir(parents=True)
    stem = write_india_export(inside / "train_in")
    facts = india.export_facts(stem)
    assert facts["generator_revision"] == "0123456789abcdef0123456789abcdef01234567"
    assert facts["stem"] == "data/raw/india/train_in"

    # a SOURCE written by a Windows shell carries a byte-order mark
    (tmp_path / "generator" / "SOURCE").write_bytes(b"\xef\xbb\xbfrevision 89abcdef0123456789abcdef0123456789abcdef\r\n")
    assert india.export_facts(stem)["generator_revision"] == "89abcdef0123456789abcdef0123456789abcdef"

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path / "elsewhere")
    facts = india.export_facts(stem)
    assert facts["generator_revision"] is None
    assert facts["stem"] == stem.as_posix()
