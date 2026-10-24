"""Calibrated India mode (spec/08_india_calibration_targets.yaml).

The profile carries the public RBI figures the Indian card stream is checked against, the
report compares a generated batch with them, and the CLI runs at the registry's own fraud
prevalence unless a demo rate is asked for explicitly, in which case the report says so.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from prism_generator import SimulationEngine
from prism_generator.calibration import calibration_report
from prism_generator.spec_loader import load_all_specs

PROFILE = "rbi-psi-2026-07"


def test_profile_carries_a_source_for_every_target():
    profile = load_all_specs().calibration_profiles[PROFILE]
    assert profile.region == "IN"
    assert abs(profile.fraud_prevalence * 80847.0 - 1.0) < 1e-3, "one fraud in every 80,847 transactions (PSI July 2026)"
    assert profile.targets, "a profile without targets checks nothing"
    for target in profile.targets:
        assert target.source.strip(), f"{target.id} has no source"
        assert target.target > 0.0
    assert profile.out_of_scope, "what is not synthesised must be written down"


def test_report_on_a_demo_batch_states_the_boost_and_checks_card_tickets():
    engine = SimulationEngine(n_cards=300, n_merchants=80, region="IN", seed=42)
    records = engine.generate_batch(n_transactions=5000, fraud_prevalence=0.03, time_span_days=30)

    report = calibration_report(records, profile_id=PROFILE, requested_fraud_prevalence=0.03)

    assert report["profile"] == PROFILE
    assert report["generated"]["rows"] == 5000
    assert 2000.0 < report["generated"]["boost_factor"] < 3000.0, "3 percent is about 2,400 times the registry rate"
    assert report["generated"]["expected_fraud_rows_at_target"] < 1.0

    by_id = {t["id"]: t for t in report["targets"]}
    for target in report["targets"]:
        assert target["observed"] is not None, f"{target['id']} was not measured"
        assert target["status"] in ("PASS", "FAIL", "REPORTED")
        assert target["source"]
    assert by_id["mean_credit_card_ticket_inr"]["status"] == "PASS"
    assert by_id["fraud_prevalence"]["status"] == "REPORTED"
    assert report["out_of_scope"]
    assert report["overall"] == "PASS", [t for t in report["targets"] if t["status"] == "FAIL" and t["gate"]]


def test_report_refuses_a_profile_of_another_region():
    engine = SimulationEngine(n_cards=50, n_merchants=20, region="US", seed=1)
    records = engine.generate_batch(n_transactions=200, fraud_prevalence=0.05)
    with pytest.raises(ValueError):
        calibration_report(records, profile_id=PROFILE, requested_fraud_prevalence=0.05)


def _run_cli(args, tmp_path: Path) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "prism_generator.cli", "generate", *args]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[2]))


def test_cli_calibrated_mode_runs_at_the_registry_rate_and_writes_the_report(tmp_path: Path):
    out = tmp_path / "calibrated.csv"
    result = _run_cli(["-n", "300", "--cards", "60", "--merchants", "20", "--region", "IN",
                       "--calibration", PROFILE, "--seed", "5", "-o", str(out)], tmp_path)
    assert result.returncode == 0, result.stderr

    report_path = tmp_path / "calibrated_calibration.json"
    assert report_path.exists(), "the calibration report must be written next to the output"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    profile = load_all_specs().calibration_profiles[PROFILE]
    assert report["generated"]["requested_fraud_prevalence"] == pytest.approx(profile.fraud_prevalence)
    assert report["generated"]["boost_factor"] == pytest.approx(1.0)
    assert report["generated"]["expected_fraud_rows_at_target"] < 0.01

    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 300
    assert sum(int(r["is_fraud"]) for r in rows) <= 1, "300 rows at one in 80,847 hold no fraud"


def test_cli_demo_rate_is_reported_as_a_boost(tmp_path: Path):
    out = tmp_path / "demo.csv"
    result = _run_cli(["-n", "300", "--cards", "60", "--merchants", "20", "--region", "IN",
                       "--calibration", PROFILE, "--fraud-rate", "0.05", "--seed", "5", "-o", str(out)], tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "demo_calibration.json").read_text(encoding="utf-8"))
    assert report["generated"]["requested_fraud_prevalence"] == pytest.approx(0.05)
    assert report["generated"]["boost_factor"] > 4000.0


def test_cli_rejects_the_profile_for_another_region(tmp_path: Path):
    result = _run_cli(["-n", "50", "--cards", "20", "--merchants", "10", "--region", "US",
                       "--calibration", PROFILE, "-o", str(tmp_path / "us.csv")], tmp_path)
    assert result.returncode != 0
    assert "IN" in result.stderr
