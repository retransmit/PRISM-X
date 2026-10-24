"""Configuration loading.

Every module reads its settings from ``config.yaml`` at the project root rather
than hardcoding constants, so a hyperparameter changed for the report is changed
in exactly one place.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Sentinel distinguishing "no default supplied" from "default is None".
_MISSING = object()


LOCAL_CONFIG_PATH = PROJECT_ROOT / "config.local.yaml"


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Nested dicts merge key by key; anything else in the overlay replaces."""
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@functools.lru_cache(maxsize=1)
def load_config(path: Path | None = None, local_path: Path | None = None) -> dict[str, Any]:
    """Read config.yaml once and cache it, with config.local.yaml merged over it.

    The overlay is gitignored. It is how a deployment selects the Indian track
    or another port without editing a committed file, which would leave the
    checkout dirty and block the next pull.
    """
    config_file = path or CONFIG_PATH
    if not config_file.exists():
        raise FileNotFoundError(
            f"No config.yaml at {config_file}. Expected it at the project root."
        )
    with config_file.open("r", encoding="utf-8") as fh:
        settings = yaml.safe_load(fh)
    local_file = local_path or LOCAL_CONFIG_PATH
    if local_file.exists():
        with local_file.open("r", encoding="utf-8") as fh:
            settings = _deep_merge(settings, yaml.safe_load(fh) or {})
    return settings


# Settings a process may take from its environment, so a server started for
# a measurement or a deployment needs no configuration file of its own. An
# empty variable does not count.
ENV_OVERRIDES = {"track": "PRISM_TRACK", "api.database_url": "PRISM_DATABASE_URL", "live.enabled": "PRISM_LIVE"}


def get(dotted_key: str, default: Any = _MISSING) -> Any:
    """Look up a nested setting, e.g. ``get("models.xgboost.max_depth")``.

    Raises KeyError when the key is absent and no default was supplied. A
    silent ``None`` for a missing hyperparameter is far worse than a crash.
    """
    variable = ENV_OVERRIDES.get(dotted_key)
    if variable and os.environ.get(variable):
        return os.environ[variable]
    node: Any = load_config()
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            if default is _MISSING:
                raise KeyError(f"Missing config key: {dotted_key!r}")
            return default
        node = node[part]
    return node


def resolve_path(value: str | Path) -> Path:
    """Turn a config-relative path into an absolute one."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_path(dotted_key: str) -> Path:
    """Look up a setting and resolve it as a project-relative path."""
    return resolve_path(get(dotted_key))


def seed() -> int:
    return int(get("seed", 42))
