"""The one place an entry point asks for a track's rows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from prism import config, tracks
from prism.data.india import export_paths, load_india
from prism.data.loader import load_raw
from prism.features.velocity import add_velocity_features


def load_frame(schema: tracks.TrackSchema, source: Path | None = None) -> pd.DataFrame:
    """Raw rows for the track, with the columns the service derives already added."""
    if schema.name == tracks.INDIA.name:
        return add_velocity_features(load_india(source))
    return load_raw()


def calibration_report_path(schema: tracks.TrackSchema, source: Path | None = None) -> Path | None:
    """The generator's calibration report next to the export, if the track has one."""
    if schema.name != tracks.INDIA.name:
        return None
    stem = Path(source) if source is not None else config.get_path("tracks.india.export_stem")
    path = export_paths(stem)["calibration"]
    return path if path.exists() else None
