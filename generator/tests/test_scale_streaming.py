"""Automated tests for streaming dataset writer, parallel coordinator, and scale visualizer."""

import json
import shutil
from pathlib import Path

import pytest
import polars as pl

from prism_generator.storage import (
    ParallelSimulationCoordinator,
    StreamingDatasetWriter,
)
from prism_generator.visualizer import (
    generate_visualization_from_dir,
)


@pytest.fixture
def tmp_dataset_dir(tmp_path: Path) -> Path:
    test_dir = tmp_path / "scale_dataset"
    test_dir.mkdir(parents=True, exist_ok=True)
    yield test_dir
    if test_dir.exists():
        shutil.rmtree(test_dir, ignore_errors=True)


def test_streaming_writer_parquet_export(tmp_dataset_dir: Path):
    """Verifies that StreamingDatasetWriter flushes chunks into partitioned Parquet tables."""
    writer = StreamingDatasetWriter(
        output_dir=tmp_dataset_dir,
        export_parquet=True,
        export_csv=False,
        chunk_size=100,
    )

    dummy_batch = [
        {
            "transaction_id": f"TX_{i:04d}",
            "card_id": f"CARD_{i % 10:04d}",
            "amount": 25.0 + i,
            "currency": "USD",
            "is_fraud": 1 if i % 5 == 0 else 0,
            "response_code": "00" if i % 4 != 0 else "51",
            "hop_origin": "ISSUER_HOST",
            "scenario_tag": "INTENT_OMEGA_HARVEST" if i % 5 == 0 else "ORGANIC_NORMAL",
            "hour_of_day": i % 24,
            "syndicate_id": "SYN_TEST",
            "botnet_cluster_id": "BOT_TEST",
            "merchant_id": "MER_TEST",
            "mule_ring_id": "MULE_TEST",
            "product_id": "US_PROD_REWARDS",
        }
        for i in range(250)
    ]

    writer.write_chunk(dummy_batch[:125])
    writer.write_chunk(dummy_batch[125:])
    meta = writer.finalize()

    assert meta["total_transactions"] == 250
    assert meta["total_fraud_count"] == 50
    assert (tmp_dataset_dir / "simulation_metadata.json").exists()
    assert (tmp_dataset_dir / "threat_graph_sample.json").exists()

    # Verify Parquet tables
    auth_pqs = list((tmp_dataset_dir / "auth_stream").glob("*.parquet"))
    assert len(auth_pqs) == 2
    df_auth = pl.read_parquet(tmp_dataset_dir / "auth_stream" / "*.parquet")
    assert df_auth.height == 250
    assert "transaction_id" in df_auth.columns
    assert "card_id" in df_auth.columns


def test_parallel_simulation_coordinator(tmp_dataset_dir: Path):
    """Verifies multi-process parallel simulation execution and shard consolidation."""
    coordinator = ParallelSimulationCoordinator(
        total_transactions=2000,
        num_workers=2,
        region="US",
        output_dir=tmp_dataset_dir,
        chunk_size=1000,
        adversary_mode="intent",
        base_seed=999,
    )

    summary = coordinator.run()
    assert summary["total_transactions"] == 2000
    assert summary["approved_count"] > 0
    assert summary["total_volume"] > 0.0
    assert "ISSUER_HOST" in summary["hop_drops"]
    assert len(summary["hourly_legitimate"]) == 24

    master_meta = tmp_dataset_dir / "master_simulation_metadata.json"
    assert master_meta.exists()
    loaded_meta = json.loads(master_meta.read_text(encoding="utf-8"))
    assert loaded_meta["total_transactions"] == 2000


def test_scale_visualizer_from_directory(tmp_dataset_dir: Path):
    """Verifies that the visualizer compiles an interactive HTML file from simulation metadata."""
    coordinator = ParallelSimulationCoordinator(
        total_transactions=1000,
        num_workers=2,
        region="US",
        output_dir=tmp_dataset_dir,
        chunk_size=500,
        adversary_mode="intent",
        base_seed=42,
    )
    coordinator.run()

    html_out = tmp_dataset_dir / "visualizer_scale_test.html"
    res_path = generate_visualization_from_dir(
        input_dir=tmp_dataset_dir,
        output_path=html_out,
        max_nodes=1000,
        open_browser=False,
    )

    assert res_path.exists()
    html_text = res_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html_text
    assert "Prism Generator Scale Visualizer" in html_text
    assert "initThreatGraph" in html_text
    assert "Hop 1: Gateway Edge Filter" in html_text
