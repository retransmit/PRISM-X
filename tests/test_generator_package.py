"""The transaction generator ships inside this repository under generator/, mirrored
from its own development line at a recorded revision and installed as prism_generator.

These tests pin what Prism relies on: the copy is traceable, importable, its export
contract is the one the Indian track reads, and a small batch generates end to end."""

from __future__ import annotations

import re

import pytest

from prism import config, tracks

GENERATOR = config.PROJECT_ROOT / "generator"

prism_generator = pytest.importorskip("prism_generator", reason="pip install -e generator/ to run the generator inside Prism")


def test_the_copy_records_the_revision_it_was_taken_from():
    source = (GENERATOR / "SOURCE").read_text(encoding="utf-8")
    assert re.search(r"^revision [0-9a-f]{40}$", source, re.MULTILINE), source
    assert re.search(r"^committed \d{4}-\d{2}-\d{2}T", source, re.MULTILINE), source


def test_the_copy_carries_its_specifications_and_tests_under_the_one_license():
    assert (GENERATOR / "spec" / "07_export_leakage_gate.yaml").exists()
    assert (GENERATOR / "spec" / "08_india_calibration_targets.yaml").exists()
    assert (GENERATOR / "tests" / "test_synthesizer" / "test_export_leakage_gate.py").exists()
    # one license covers the repository, the generator included
    assert not (GENERATOR / "LICENSE").exists()
    assert (config.PROJECT_ROOT / "LICENSE").read_text(encoding="utf-8").lstrip().startswith("Apache License")
    assert prism_generator.__file__ and "generator" in prism_generator.__file__


def test_the_export_contract_is_what_the_indian_track_reads():
    from prism_generator.storage import AUTH_STREAM_COLUMNS, GATEWAY_TELEMETRY_COLUMNS

    assert len(AUTH_STREAM_COLUMNS) == 25 and len(GATEWAY_TELEMETRY_COLUMNS) == 16
    contract = set(AUTH_STREAM_COLUMNS) | set(GATEWAY_TELEMETRY_COLUMNS)
    india = tracks.get("india")
    missing = [column for column in india.input_columns if column not in contract and column not in tracks.INDIA_DERIVED]
    assert missing == [], f"Indian inputs the contract does not carry: {missing}"


def test_a_small_indian_batch_generates_in_contract_form():
    from prism_generator import SimulationEngine
    from prism_generator.storage import AUTH_STREAM_COLUMNS, GATEWAY_TELEMETRY_COLUMNS

    engine = SimulationEngine(n_cards=40, n_merchants=15, region="IN", seed=3)
    records = engine.generate_batch(n_transactions=300, fraud_prevalence=0.05, time_span_days=7)
    assert len(records) == 300
    assert all(column in records[0] for column in AUTH_STREAM_COLUMNS + GATEWAY_TELEMETRY_COLUMNS)
    assert all(record["currency"] == "INR" for record in records)
    assert 0 < sum(record["is_fraud"] for record in records) < 300
