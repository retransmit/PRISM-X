"""Tests verifying production institutional banking schemas, ISO 8583 protocol conformance,
dual-region payment rails (US & India), and partitioned warehouse feeds.
"""

import csv
import ipaddress
import subprocess
import sys

import pytest
from prism_generator import SimulationEngine
from prism_generator.spec_loader import load_all_specs
from prism_generator.cli import (
    AUTH_STREAM_COLUMNS,
    CLEARING_SETTLEMENT_COLUMNS,
    DISPUTE_RECOVERY_COLUMNS,
    GATEWAY_TELEMETRY_COLUMNS,
)


def test_us_institutional_records():
    """Verifies US batch produces authentic ISO 8583, USD minor units, and clearing presentment."""
    engine = SimulationEngine(n_cards=100, n_merchants=50, region="US", seed=101)
    records = engine.generate_batch(n_transactions=500, fraud_prevalence=0.05)

    assert len(records) == 500

    for r in records:
        # Currency and minor units
        assert r["currency"] == "USD"
        assert isinstance(r["amount_minor"], int)
        assert r["amount_minor"] == int(round(r["amount"] * 100))

        # ISO 8583 Core Fields
        assert r["mti"] == "0100"
        assert len(r["stan"]) == 6
        assert r["stan"].isdigit()
        assert len(r["rrn"]) == 12
        assert r["response_code"] in ["00", "05", "10", "14", "51", "54", "57", "59", "63", "65", "82"]
        assert r["pos_entry_mode"] in ["051", "071", "901", "012", "102", "031", "812"]
        assert r["pos_condition_code"] in ["00", "59"]

        # Clearing & Settlement Presentment
        assert r["clearing_mti"] == "0200"
        assert 24 <= r["clearing_delay_hours"] <= 72
        assert isinstance(r["settled_amount_minor"], int)
        assert isinstance(r["interchange_fee_minor"], int)

        # Gateway Risk Telemetry
        assert "client_ip" in r
        assert r["asn_type"] in ["residential", "datacenter", "mobile"]
        assert len(r["device_canvas_hash"]) == (0 if r["channel_type"].startswith("CP") else 16)


def test_india_institutional_records():
    """Verifies India batch conforms to RBI AFA, INR paisa, RuPay UPI, and Indian card products."""
    engine = SimulationEngine(n_cards=100, n_merchants=50, region="IN", seed=202)
    records = engine.generate_batch(n_transactions=500, fraud_prevalence=0.06)

    assert len(records) == 500

    # Derived from spec/05 rather than a hand-copied list, so the invariant under test is
    # "every emitted Indian product is a declared Indian product" and stays true as
    # spec/05 gains products (IN_PROD_RETAIL_RUPAY_DEBIT was added 2026-10-12 and a
    # hard-coded list had to be edited, which is how such lists rot).
    indian_products = set(load_all_specs().indian_products)
    address_space = load_all_specs().telemetry.address_space
    indian_octets = {
        str(o) for o in address_space["residential_first_octets"]["IN"] + address_space["datacenter_first_octets"]["IN"]
    }

    for r in records:
        assert r["currency"] == "INR"
        assert r["product_id"] in indian_products
        assert isinstance(r["amount_minor"], int)
        assert r["amount_minor"] == int(round(r["amount"] * 100))

        # Indian consumer address space (spec/07 section 16) for a legitimate domestic online payment;
        # a fraudster connects from its own infrastructure, which sits inside the row's subnet.
        if not r["channel_type"].startswith("CP") and not r.get("is_cross_border", False):
            if r["is_fraud"] == 0:
                assert r["client_ip"].split(".")[0] in indian_octets, r["client_ip"]
            elif r.get("ip_subnet_prefix"):
                assert ipaddress.ip_address(r["client_ip"]) in ipaddress.ip_network(r["ip_subnet_prefix"], strict=False)

        # RuPay on UPI QR code channels use POS entry mode 031
        if r["channel_type"] == "UPI_QR_CREDIT":
            assert r["pos_entry_mode"] == "031"


def test_dual_message_settlement_adjustments():
    """Verifies dining tip adjustments and AFD gas pump reconciliation in dual-message settlement."""
    engine = SimulationEngine(n_cards=100, n_merchants=50, region="US", seed=303)
    records = engine.generate_batch(n_transactions=600, fraud_prevalence=0.02)

    # Dining transactions (MCC 5812) approved: settled amount must include 10-20% tip
    dining_approved = [
        r for r in records
        if r["mcc"] == 5812 and r["response_code"] == "00"
    ]
    for r in dining_approved:
        assert r["settled_amount"] >= r["amount"], "Dining settled amount must include tip adjustment"
        assert r["settled_amount_minor"] >= r["amount_minor"]

    # Fuel pump transactions (MCC 5542) approved: settled amount is actual pump volume
    fuel_approved = [
        r for r in records
        if r["mcc"] == 5542 and r["response_code"] == "00"
    ]
    for r in fuel_approved:
        assert r["settled_amount"] > 0.0
        assert r["settled_amount_minor"] > 0


def test_dispute_and_chargeback_lifecycles():
    """Verifies Visa CE 3.0 deflection in US and RBI Limited Liability Tiers & 1930 liens in India."""
    # US Disputes
    us_engine = SimulationEngine(n_cards=80, n_merchants=40, region="US", seed=404)
    us_records = us_engine.generate_batch(n_transactions=500, fraud_prevalence=0.12)
    us_fraud_approved = [r for r in us_records if r["is_fraud"] == 1 and r["response_code"] in ["00", "10"]]

    assert len(us_fraud_approved) > 0
    ce3_deflected = [r for r in us_fraud_approved if r["dispute_status"] == "DEFLECTED_PRE_DISPUTE_CE3"]
    first_chargebacks = [r for r in us_fraud_approved if r["dispute_status"] == "FIRST_CHARGEBACK"]

    assert len(ce3_deflected) + len(first_chargebacks) == len(us_fraud_approved)
    for r in first_chargebacks:
        assert r["dispute_reason_code"] in ["10.4", "10.5"]

    # India Disputes
    in_engine = SimulationEngine(n_cards=80, n_merchants=40, region="IN", seed=505)
    in_records = in_engine.generate_batch(n_transactions=500, fraud_prevalence=0.12)
    in_fraud_approved = [r for r in in_records if r["is_fraud"] == 1 and r["response_code"] in ["00", "10"]]

    assert len(in_fraud_approved) > 0
    valid_rbi_tiers = {
        "ZERO_LIABILITY",
        "BSBD_PMJDY_CAP_5000_INR",
        "STANDARD_CAP_10000_INR",
        "HNI_CAP_25000_INR",
        "BANK_BOARD_POLICY",
    }
    valid_lien_statuses = {
        "LIEN_FREEZE_SUCCESS",
        "LIEN_FREEZE_PARTIAL",
        "CRYPTO_SEVERED_UNHOSTED",
    }

    for r in in_fraud_approved:
        assert r["rbi_liability_tier"] in valid_rbi_tiers
        assert r["cfcfrms_1930_lien_status"] in valid_lien_statuses
        assert r["dispute_reason_code"] == "UNAUTHORIZED_ELECTRONIC_DEBIT"


def test_cli_export_institutional_views(tmp_path):
    """Verifies CLI generates master dataset and 4 partitioned warehouse CSV views."""
    out_file = tmp_path / "test_run.csv"

    cmd = [
        sys.executable,
        "-m",
        "prism_generator.cli",
        "generate",
        "-n",
        "100",
        "--cards",
        "30",
        "--merchants",
        "15",
        "--region",
        "IN",
        "--seed",
        "888",
        "-o",
        str(out_file),
        "--include-disputes",
        "--export-institutional-views",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    assert out_file.exists(), "Master output file was not created"

    # Verify 4 partitioned views exist
    auth_file = tmp_path / "test_run_auth_stream.csv"
    gateway_file = tmp_path / "test_run_gateway_telemetry.csv"
    clearing_file = tmp_path / "test_run_clearing_settlement.csv"
    dispute_file = tmp_path / "test_run_dispute_recovery.csv"

    for path, expected_cols in [
        (auth_file, AUTH_STREAM_COLUMNS),
        (gateway_file, GATEWAY_TELEMETRY_COLUMNS),
        (clearing_file, CLEARING_SETTLEMENT_COLUMNS),
        (dispute_file, DISPUTE_RECOVERY_COLUMNS),
    ]:
        assert path.exists(), f"Partitioned view {path.name} was not created"
        with open(path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == expected_cols, f"Columns mismatch in {path.name}"
            rows = list(reader)
            assert len(rows) == 100, f"Row count mismatch in {path.name}"


@pytest.mark.parametrize("region", ["IN", "US"])
def test_card_present_rows_never_carry_a_client_ip(region):
    """A card-present payment has no cardholder IP address: the terminal is on the
    acquirer's network. Every legitimate card-present row carries the 127.0.0.1 sentinel,
    so a fraud row carrying the fraudster's botnet address hands a detector the label."""
    engine = SimulationEngine(n_cards=100, n_merchants=50, region=region, seed=202)
    records = engine.generate_batch(n_transactions=500, fraud_prevalence=0.06)
    card_present = [r for r in records if r["channel_type"].startswith("CP")]
    assert any(r["is_fraud"] == 1 for r in card_present), "the batch must hold card-present fraud or this proves nothing"
    offenders = sorted({(r["is_fraud"], r["client_ip"]) for r in card_present if r["client_ip"] != "127.0.0.1"})
    assert offenders == []
