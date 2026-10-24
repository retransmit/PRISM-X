"""Tests for the Phase 1 exploratory analysis artifacts."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from prism.data.eda import EDA_FIGURES, generate_eda_report

PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n, n_fraud = 5_000, 25

    data = pd.DataFrame(
        {
            **{c: rng.normal(size=n) for c in PCA_COLUMNS},
            "Time": rng.uniform(0, 172_800, size=n),
            "Amount": rng.gamma(2.0, 50.0, size=n),
        }
    )
    labels = np.zeros(n, dtype=int)
    labels[rng.choice(n, size=n_fraud, replace=False)] = 1
    data["Class"] = labels
    data.loc[data["Class"] == 1, "Amount"] *= 4.0
    return data[["Time", *PCA_COLUMNS, "Amount", "Class"]]


def test_writes_every_declared_figure(frame, tmp_path):
    generate_eda_report(frame, tmp_path)
    for name in EDA_FIGURES:
        assert (tmp_path / f"{name}.png").exists(), f"missing figure {name}"


def test_reports_the_headline_imbalance_numbers(frame, tmp_path):
    report = generate_eda_report(frame, tmp_path)
    assert report["n_total"] == 5_000
    assert report["n_fraud"] == 25
    assert report["fraud_rate"] == pytest.approx(0.005)
    assert report["imbalance_ratio"] == pytest.approx(4_975 / 25)


def test_contrasts_fraud_and_normal_amounts(frame, tmp_path):
    report = generate_eda_report(frame, tmp_path)
    assert report["median_amount_fraud"] > report["median_amount_normal"]


def test_report_is_json_serialisable(frame, tmp_path):
    """The dict is embedded in the report and served by /metrics, so no numpy types."""
    json.dumps(generate_eda_report(frame, tmp_path))


def test_amount_histogram_bins_over_the_range_the_figure_shows(frame):
    """Sixty bins over the full range (a maximum in the tens of thousands) put
    the whole legitimate distribution into one bar of the clipped figure; the
    bins must span the range the axis shows."""
    from prism.data.eda import amount_bins

    normal = frame[frame["Class"] == 0]["Amount"]
    edges = amount_bins(normal)
    assert len(edges) == 61
    assert edges[0] == 0.0
    assert edges[-1] == pytest.approx(float(normal.quantile(0.99)))
