"""Two settings a process may receive from its environment, so a server
started for a measurement or a deployment needs no file of its own."""

from __future__ import annotations

from prism import config, tracks


def test_the_environment_overrides_the_track_and_the_database_url(monkeypatch):
    monkeypatch.setenv("PRISM_TRACK", "india")
    monkeypatch.setenv("PRISM_DATABASE_URL", "sqlite:///from-the-environment.db")
    assert config.get("track") == "india"
    assert config.get("api.database_url") == "sqlite:///from-the-environment.db"
    tracks.set_active(None)
    assert tracks.active().name == "india"


def test_an_empty_variable_does_not_override(monkeypatch):
    monkeypatch.setenv("PRISM_DATABASE_URL", "")
    assert config.get("api.database_url") == config.load_config()["api"]["database_url"]



def test_the_suite_never_inherits_the_environment_overrides():
    """A developer who exported PRISM_DATABASE_URL to run the service must
    not have the tests write into that log; conftest clears both variables."""
    import os

    assert "PRISM_DATABASE_URL" not in os.environ and "PRISM_TRACK" not in os.environ
