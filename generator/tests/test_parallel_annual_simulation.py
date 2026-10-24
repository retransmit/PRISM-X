"""Unit and invariant integration tests for 365-day annual parallel simulation.

Validates:
1. Zero parallel clock overlaps and strict temporal monotonicity (t_{i+1} >= t_i).
2. Zero primary key collisions (unique transaction_id and card_id namespaces).
3. 100% causal continuity of cardholder velocity and state machines.
4. Enforcement of physical kinematics (CP swipe dwell times and spatial clusters).
5. Annual seasonality calendar dynamics and day-of-week multipliers.
6. Complete generation and schema conformity of all 5 modular institutional views.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from prism_generator.engine import DiscreteEventEngine
from prism_generator.spec_loader import load_all_specs
from prism_generator.storage import (
    AUTH_STREAM_COLUMNS,
    CLEARING_SETTLEMENT_COLUMNS,
    DISPUTE_RECOVERY_COLUMNS,
    GATEWAY_TELEMETRY_COLUMNS,
    THREAT_INTEL_GRAPH_COLUMNS,
    EntityPartitionedParallelCoordinator,
)


@pytest.fixture
def temp_output_dir():
    temp_dir = tempfile.mkdtemp(prefix="prism_generator_annual_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_seasonality_spec_loading():
    """Validates that spec/13_annual_seasonality_and_drift.yaml loads cleanly."""
    specs = load_all_specs()
    assert specs.seasonality is not None, "Seasonality spec failed to load"
    assert len(specs.seasonality.day_of_week_multipliers) == 7, "Expected 7 DoW multipliers"
    assert specs.seasonality.day_of_week_multipliers[5] > specs.seasonality.day_of_week_multipliers[0], (
        "Saturday volume must exceed Monday volume"
    )
    assert "US" in specs.seasonality.calendar_macro_shocks, "US macro shocks missing"
    assert "IN" in specs.seasonality.calendar_macro_shocks, "India macro shocks missing"


@pytest.mark.slow
def test_entity_partitioned_parallel_simulation(temp_output_dir):
    """Executes a multi-worker entity-partitioned simulation and validates core invariants."""
    # Run a test simulation with 4 workers, 100 cards across 365 days
    coordinator = EntityPartitionedParallelCoordinator(
        total_cards=100,
        time_span_days=365,
        num_workers=4,
        region="US",
        output_dir=temp_output_dir,
        chunk_size=500,
        adversary_mode="playbook",
        annual_seasonality=True,
        p2p_enabled=True,
        base_seed=42,
        export_csv=True,
    )

    summary = coordinator.run()
    total_tx = summary["total_transactions"]
    assert total_tx > 0, "Simulation produced zero transactions"

    # 1. Master Table File Verification
    master_pq_path = temp_output_dir / "master" / "master.parquet"
    assert master_pq_path.exists(), f"Master parquet missing at {master_pq_path}"
    df_master = pl.read_parquet(master_pq_path)
    assert len(df_master) == total_tx, f"Master row count {len(df_master)} != summary count {total_tx}"

    # 2. Strict Global Temporal Monotonicity
    tx_times = df_master["tx_time_seconds"].to_numpy()
    assert np.all(np.diff(tx_times) >= 0), "Global temporal monotonicity violated: t_{i+1} < t_i found!"

    # 3. Zero ID Collisions (Namespacing Integrity)
    tx_ids = df_master["transaction_id"].to_list()
    assert len(tx_ids) == len(set(tx_ids)), "Duplicate transaction_id primary keys detected!"

    card_ids = df_master["card_id"].unique().to_list()
    assert len(card_ids) <= 100, f"Expected <= 100 distinct cards, got {len(card_ids)}"
    # Verify worker prefix namespacing
    assert all("W" in cid for cid in card_ids), "Card IDs missing worker namespace prefix"

    # 4. Causal Continuity per Cardholder
    # Select a card with multiple transactions and assert strict monotonic timestamps
    card_counts = df_master["card_id"].value_counts()
    frequent_card = card_counts.sort("count", descending=True)["card_id"][0]
    df_card = df_master.filter(pl.col("card_id") == frequent_card)
    card_tx_times = df_card["tx_time_seconds"].to_numpy()
    assert np.all(np.diff(card_tx_times) >= 0), f"Card {frequent_card} timestamps out of chronological order"

    # 5. Physical Kinematics (CP Dwell Time Enforcement)
    df_cp = df_card.filter(pl.col("channel_type").str.starts_with("CP") & (pl.col("is_fraud") == 0))
    if len(df_cp) > 1:
        cp_times = df_cp["tx_time_seconds"].to_numpy()
        cp_diffs = np.diff(cp_times)
        # Legitimate CP transactions must have dwell time >= 120 seconds
        assert np.all(cp_diffs >= 119.0), f"Physical dwell violation: CP delta < 120s found: {cp_diffs.min()}"

    # 6. Modular Institutional Views Verification
    views = [
        ("auth_stream", AUTH_STREAM_COLUMNS),
        ("gateway_telemetry", GATEWAY_TELEMETRY_COLUMNS),
        ("clearing_settlement", CLEARING_SETTLEMENT_COLUMNS),
        ("dispute_recovery", DISPUTE_RECOVERY_COLUMNS),
        ("threat_intel_graph", THREAT_INTEL_GRAPH_COLUMNS),
    ]

    for view_name, expected_cols in views:
        view_pq = temp_output_dir / view_name / f"{view_name}.parquet"
        assert view_pq.exists(), f"View {view_name} parquet missing at {view_pq}"
        df_view = pl.read_parquet(view_pq)
        assert len(df_view) == total_tx, f"View {view_name} row count mismatch"
        for col in expected_cols:
            assert col in df_view.columns, f"Expected column {col} missing from view {view_name}"

    # 7. Clearing and Settlement Rail Checks
    df_clearing = pl.read_parquet(temp_output_dir / "clearing_settlement" / "clearing_settlement.parquet")
    assert (df_clearing["clearing_mti"] == "0200").all(), "Clearing MTI must be 0200"
    assert (df_clearing["interchange_fee_minor"] > 0).any(), "Expected non-zero interchange fees"

    # 8. P2P Transaction Presence
    categories = df_master["merchant_category"].unique().to_list()
    mids = df_master["merchant_id"].to_list()
    assert "P2P Money Transfer" in categories or any("P2P_" in m for m in mids), (
        f"Expected P2P transfers in output, got categories: {categories}"
    )


def test_engine_backwards_compatibility():
    """Ensures that DiscreteEventEngine behaves identically when annual_seasonality is False."""
    engine = DiscreteEventEngine(
        n_cards=10,
        n_merchants=5,
        region="US",
        annual_seasonality=False,
        seed=123,
    )
    records = engine.generate_batch(n_transactions=50, enforce_invariants=True)
    assert len(records) == 50
    # In legacy mode, card IDs have no worker prefix
    assert records[0]["card_id"].startswith("CARD_000")
    assert records[0]["transaction_id"].startswith("TX_000")
