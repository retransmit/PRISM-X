"""The harness's data: one stream per generator seed, written by the shipped generator."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from prism import config
from prism.data import india
from prism.features.velocity import add_velocity_features

# Master-view columns the harness reads beside the features, to build folds, label
# timelines and evaluation masks. They are never features.
MASTER_COLUMNS = ("transaction_id", "is_fraud", "scenario_tag", "risk_score")


def settings() -> dict:
    return dict(config.get("harness"))


def seeds() -> list[int]:
    return [int(seed) for seed in settings()["seeds"]]


def stem(seed: int) -> Path:
    pattern = str(settings()["stem_pattern"])
    return config.PROJECT_ROOT / pattern.format(seed=seed)


def generator_command(s: dict, seed: int, out_stem: Path) -> list[str]:
    """The shipped generator's command for one Indian stream at the settings in ``s``, run
    through this interpreter so the host needs no activated environment."""
    return [
        sys.executable, "-m", "prism_generator.cli", "generate",
        "-n", str(int(s["rows"])), "-c", str(int(s["cards"])), "-m", str(int(s["merchants"])),
        "--region", "IN", "--calibration", str(s["calibration"]),
        "--fraud-rate", str(s["fraud_rate"]), "--days", str(int(s["days"])), "--seed", str(int(seed)),
        "-o", str(out_stem) + ".csv", "--export-institutional-views",
    ]


def generate_command(seed: int) -> list[str]:
    """The command for one harness seed, every setting from the config."""
    return generator_command(settings(), seed, stem(seed))


def load_seed(seed: int) -> pd.DataFrame:
    """The seed's frame as training sees it (features, velocity, label), with the master
    columns the harness needs for folds, label modes and the timeline, time-ordered."""
    path = stem(seed)
    frame = add_velocity_features(india.load_india(path))
    master = pd.read_csv(
        india.export_paths(path)["master"], dtype=str, keep_default_na=False, usecols=list(MASTER_COLUMNS)
    ).set_index("transaction_id")
    extra = master.loc[frame["transaction_id"].astype(str), ["scenario_tag", "risk_score"]]
    frame = frame.copy()
    frame["scenario_tag"] = extra["scenario_tag"].astype(str).to_numpy()
    frame["risk_score"] = pd.to_numeric(extra["risk_score"], errors="raise").astype(float).to_numpy()
    return frame.sort_values("tx_time_seconds", kind="stable").reset_index(drop=True)
