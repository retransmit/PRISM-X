"""load_frame hands every entry point the right source with the derived columns present."""

from __future__ import annotations

import pandas as pd

from prism import tracks
from prism.data import frames


def test_india_frame_carries_the_derived_columns(india_stem):
    frame = frames.load_frame(tracks.INDIA, source=india_stem)
    for column in tracks.INDIA.derived_columns:
        assert column in frame.columns
    assert frames.calibration_report_path(tracks.INDIA, source=india_stem).name == "fixture_in_calibration.json"


def test_ulb_frame_is_the_raw_dataset(monkeypatch):
    marker = pd.DataFrame({"Class": [0]})
    monkeypatch.setattr(frames, "load_raw", lambda: marker)
    assert frames.load_frame(tracks.ULB) is marker
    assert frames.calibration_report_path(tracks.ULB) is None
