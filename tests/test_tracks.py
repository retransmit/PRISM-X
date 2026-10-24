"""Two tracks, one pipeline (design D1): the registry every stage reads its columns from."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from prism import config, tracks

PCA = [f"V{i}" for i in range(1, 29)]


@pytest.fixture(autouse=True)
def restore_track():
    yield
    tracks.set_active(None)


def test_ulb_schema_is_the_dataset_layout():
    assert tracks.ULB.feature_columns == ["Time", *PCA, "Amount"]
    assert tracks.ULB.input_columns == ["Time", *PCA, "Amount"]
    assert tracks.ULB.scale_columns == ("Time", "Amount")
    assert tracks.ULB.target_column == "Class"
    assert tracks.ULB.is_interpretable("Amount") and not tracks.ULB.is_interpretable("V14")
    assert tracks.ULB.label("V14") == "V14"


def test_india_inputs_are_pre_authorisation_and_exclude_what_the_service_derives():
    india = tracks.INDIA
    for outcome in ("response_code", "auth_response_code", "auth_code", "trans_status_3ds"):
        assert outcome not in india.feature_columns
    for derived in india.derived_columns:
        assert derived in india.feature_columns
        assert derived not in india.input_columns
    assert india.time_column in india.input_columns and india.card_column in india.input_columns
    assert india.time_column not in india.feature_columns
    assert india.is_interpretable("vaai_score")
    assert india.label("vaai_score") == "network risk score"
    assert set(india.scale_columns) <= set(india.numeric_columns)


def test_active_track_comes_from_config_and_a_flag_overrides_it():
    tracks.set_active(None)
    assert tracks.active().name == config.get("track")
    tracks.set_active("india")
    assert tracks.active().name == "india"
    tracks.set_active(None)
    assert tracks.active().name == config.get("track")
    with pytest.raises(KeyError):
        tracks.get("kaggle")


def test_suite_is_pinned_to_ulb_whatever_the_deployment_track_says(monkeypatch):
    """A deployment's config.local.yaml says track: india; the tests build ULB frames."""
    monkeypatch.setitem(config.load_config(), "track", "india")
    assert tracks.active().name == "ulb"


def test_track_argument_activates_the_track():
    parser = argparse.ArgumentParser()
    tracks.add_track_argument(parser)
    args = parser.parse_args(["--track", "india"])
    tracks.activate_from_args(args)
    assert tracks.active().name == "india"


def test_artifact_directories_are_per_track():
    assert tracks.ULB.model_dir == config.get_path("api.model_dir") / "ulb"
    assert tracks.INDIA.metrics_dir == config.PROJECT_ROOT / "reports" / "metrics" / "india"
    assert tracks.INDIA.figures_dir.name == "india"


def test_local_overlay_merges_over_the_committed_config(tmp_path: Path):
    (tmp_path / "config.yaml").write_text("track: ulb\napi:\n  port: 8000\n  host: 127.0.0.1\n", encoding="utf-8")
    (tmp_path / "config.local.yaml").write_text("track: india\napi:\n  port: 9000\n", encoding="utf-8")
    config.load_config.cache_clear()
    try:
        merged = config.load_config(tmp_path / "config.yaml", tmp_path / "config.local.yaml")
    finally:
        config.load_config.cache_clear()
    assert merged["track"] == "india"
    assert merged["api"] == {"port": 9000, "host": "127.0.0.1"}


def test_missing_overlay_is_not_an_error(tmp_path: Path):
    (tmp_path / "config.yaml").write_text("track: ulb\n", encoding="utf-8")
    config.load_config.cache_clear()
    try:
        assert config.load_config(tmp_path / "config.yaml", tmp_path / "absent.yaml")["track"] == "ulb"
    finally:
        config.load_config.cache_clear()


def test_every_continuous_indian_numeric_is_scaled():
    """An unscaled hour of day (0 to 23) would dominate the case index and the
    autoencoder over columns standardised to unit variance (design D7)."""
    binary = {"cvv_match_flag", "is_cross_border", "first_mcc_30d", "first_cnp_30d"}
    continuous = {c for c in tracks.INDIA.numeric_columns if c not in binary}
    assert continuous <= set(tracks.INDIA.scale_columns), continuous - set(tracks.INDIA.scale_columns)
