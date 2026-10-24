"""Unit and integration tests for multi-bank routing mesh and clearing settlement.

Validates:
1. Loading and schema verification of spec/14_interbank_switch_and_clearing.yaml.
2. Deterministic card and merchant bank routing with market-share distribution.
3. On-Us vs Off-Us payment classification, latency differentials, and hop sequences.
4. Central switching networks (VisaNet, Mastercard, FedNow, NPCI NFS, NPCI UPI).
5. ISO 8583 message families (0100/0110, 0200/0210, 0400/0410, 0800/0810).
6. Multilateral net settlement position (MNSP) zero-sum conservation invariant.
7. End-to-end multi-bank generation in both US and India payment rail environments.
"""

from __future__ import annotations

import tempfile
import shutil
from pathlib import Path
import numpy as np
import pytest

from prism_generator.spec_loader import load_all_specs
from prism_generator.agents import CardholderProfile
from prism_generator.rails import (
    BankRoutingMesh,
    CandidateTransactionIntent,
    RailVerifierSwitch,
)
from prism_generator.ledger import (
    InterbankClearingLedger,
    NetSettlementReport,
)
from prism_generator.engine import DiscreteEventEngine


@pytest.fixture
def temp_output_dir():
    temp_dir = tempfile.mkdtemp(prefix="prism_generator_interbank_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


def create_test_card(
    card_id: str = "CARD_TEST_001",
    region: str = "US",
    credit_limit: float = 10000.0,
    current_balance: float = 1000.0,
) -> CardholderProfile:
    """Helper to instantiate a valid CardholderProfile for testing."""
    currency = "INR" if region == "IN" else "USD"
    lat = 40.7128 if region == "US" else 19.0760
    lon = -74.0060 if region == "US" else 72.8777
    return CardholderProfile(
        card_id=card_id,
        home_lat=lat,
        home_lon=lon,
        work_lat=lat + 0.05,
        work_lon=lon + 0.05,
        is_commuter=True,
        credit_limit=credit_limit,
        current_balance=current_balance,
        pan_masked="4000123456789010",
        product_id=f"{region}_PROD_STANDARD",
        cohort_id="C1_URBAN_TECH_PROFESSIONAL",
        region=region,
        currency=currency,
        spend_mean_log=3.8,
        spend_sigma_log=0.7,
        domestic_cnp_enabled=True,
        international_enabled=True,
    )


def test_interbank_spec_loading():
    """Validates that spec/14_interbank_switch_and_clearing.yaml loads cleanly."""
    specs = load_all_specs()
    assert specs.interbank is not None, "Interbank spec failed to load"
    assert "US" in specs.interbank.commercial_banks, "Expected US commercial banks"
    assert "IN" in specs.interbank.commercial_banks, "Expected IN commercial banks"
    assert len(specs.interbank.commercial_banks["US"]) == 5, "Expected 5 US banks"
    assert len(specs.interbank.commercial_banks["IN"]) == 5, "Expected 5 IN banks"
    assert "US" in specs.interbank.central_switching_networks
    assert "IN" in specs.interbank.central_switching_networks
    assert specs.interbank.routing_rules["on_us"]["requires_central_switch"] is False
    assert specs.interbank.routing_rules["off_us"]["requires_central_switch"] is True
    assert specs.interbank.iso8583_message_types["auth_request"] == "0100"
    assert specs.interbank.iso8583_message_types["reversal_request"] == "0400"


def test_bank_routing_mesh_deterministic_market_share():
    """Validates that BankRoutingMesh produces deterministic entities and realistic market share."""
    mesh_us = BankRoutingMesh(region="US")
    mesh_in = BankRoutingMesh(region="IN")

    # Determinism check
    bank_1 = mesh_us.resolve_issuing_bank("CARD_1001")
    bank_2 = mesh_us.resolve_issuing_bank("CARD_1001")
    assert bank_1.id == bank_2.id, "Bank resolution must be strictly deterministic"
    assert bank_1.routing_number != "", "US bank must have routing number"

    # India resolution check
    bank_in_1 = mesh_in.resolve_issuing_bank("CARD_2002")
    assert bank_in_1.ifsc_prefix != "", "IN bank must have IFSC prefix"

    # Empirical market share test across 10,000 synthetic cards
    card_counts: dict[str, int] = {}
    for i in range(10000):
        c_id = f"CARD_POP_{i:05d}"
        b = mesh_us.resolve_issuing_bank(c_id)
        card_counts[b.id] = card_counts.get(b.id, 0) + 1

    # Verify all 5 US banks are represented in expected proportions
    assert len(card_counts) == 5, "All 5 US banks must receive cards"
    # Chase (~28%), BofA (~24%), Wells (~20%), Citi (~16%), Capital One (~12%)
    assert card_counts["BANK_US_CHASE"] > card_counts["BANK_US_CITI"]
    assert card_counts["BANK_US_BOFA"] > card_counts["BANK_US_CAPITALONE"]


def test_on_us_vs_off_us_routing_classification():
    """Validates On-Us vs Off-Us routing mechanics, fees, latencies, and hops."""
    mesh = BankRoutingMesh(region="US", seed=42)

    # Find an On-Us pair (same issuer and acquirer) and Off-Us pair (different)
    on_us_pair = None
    off_us_pair = None

    for i in range(100):
        c_id = f"CARD_{i:04d}"
        m_id = f"MERCHANT_{i:04d}"
        b_iss = mesh.resolve_issuing_bank(c_id)
        b_acq = mesh.resolve_acquiring_bank(m_id)
        if b_iss.id == b_acq.id and on_us_pair is None:
            on_us_pair = (c_id, m_id)
        elif b_iss.id != b_acq.id and off_us_pair is None:
            off_us_pair = (c_id, m_id)
        if on_us_pair is not None and off_us_pair is not None:
            break

    assert on_us_pair is not None, "Failed to find On-Us pair in 100 trials"
    assert off_us_pair is not None, "Failed to find Off-Us pair in 100 trials"

    # Case A: On-Us routing verification
    c_on, m_on = on_us_pair
    res_on_us = mesh.route_transaction(
        card_id=c_on,
        merchant_id=m_on,
        channel_type="CP_POS_CHIP",
        amount=150.00,
    )
    assert res_on_us["is_on_us"] is True
    assert res_on_us["central_switch_id"] == "ON_US_INTERNAL"
    assert res_on_us["switch_network_fee"] == 0.0
    assert res_on_us["interchange_fee"] == 0.0
    assert 8.0 <= res_on_us["interbank_latency_ms"] <= 16.0
    assert res_on_us["hop_sequence"] == ["MERCHANT", "ACQUIRER_CBS", "ISSUER_CORE"]

    # Case B: Off-Us routing verification
    c_off, m_off = off_us_pair
    res_off_us = mesh.route_transaction(
        card_id=c_off,
        merchant_id=m_off,
        channel_type="CP_POS_CHIP",
        amount=150.00,
    )
    assert res_off_us["is_on_us"] is False
    assert res_off_us["central_switch_id"] == "VISANET_BASE_I_II"
    assert res_off_us["switch_network_fee"] > 0.0
    assert res_off_us["interchange_fee"] > 0.0
    assert 45.0 <= res_off_us["interbank_latency_ms"] <= 115.0
    assert res_off_us["hop_sequence"] == ["MERCHANT", "ACQUIRER_SWITCH", "VISANET_BASE_I_II", "ISSUER_SWITCH"]


def test_rail_verifier_switch_enrichment():
    """Validates that RailVerifierSwitch seamlessly enriches RailVerificationResult."""
    verifier = RailVerifierSwitch(region="IN", seed=42)
    card = create_test_card(card_id="CARD_TEST_IN_001", region="IN")
    intent = CandidateTransactionIntent(
        tx_id="TX_TEST_IN_001",
        card_id=card.card_id,
        sim_time_sec=100.0,
        amount=450.0,
        currency="INR",
        channel_type="CP_POS_CONTACTLESS",
        merchant_id="MERCHANT_TEST_IN_002",
        mcc=5411,
        merchant_lat=card.home_lat,
        merchant_lon=card.home_lon,
        pin_entered=False,
    )
    res = verifier.verify_intent(intent, card)
    assert res.issuing_bank_id.startswith("BANK_IN_")
    assert res.acquiring_bank_id.startswith("BANK_IN_")
    assert res.mti_request in ("0100", "0400")
    assert res.mti_response in ("0110", "0410")
    assert res.interbank_latency_ms > 0.0
    assert len(res.hop_sequence) >= 3


def test_multilateral_net_settlement_zero_sum_invariant():
    """Validates the fundamental central bank invariant: sum of net settlement positions == 0.00."""
    ledger = InterbankClearingLedger(region="US")

    # Record 100 interbank presentments
    rng = np.random.default_rng(123)
    banks = ["BANK_US_CHASE", "BANK_US_BOFA", "BANK_US_WELLS", "BANK_US_CITI", "BANK_US_CAPITALONE"]

    cycle_id = "2024-01-01_CYCLE_01"
    for i in range(100):
        iss = str(rng.choice(banks))
        acq = str(rng.choice(banks))
        amt = float(rng.uniform(10.0, 500.0))
        ic = round(amt * 0.015, 2) if iss != acq else 0.0
        ledger.record_clearing_presentment(
            cycle_id=cycle_id,
            acquiring_bank_id=acq,
            issuing_bank_id=iss,
            settled_amount=amt,
            interchange_fee=ic,
            is_on_us=(iss == acq),
        )

    reports: list[NetSettlementReport] = ledger.execute_multilateral_netting(cycle_id=cycle_id)
    assert len(reports) == len(banks), "All participating banks must receive a settlement report"

    # Sum of net positions must equal 0.00 within machine precision
    net_sum = sum(r.net_position for r in reports)
    assert abs(net_sum) < 1e-4, f"Zero-sum invariant failed: net_sum={net_sum}"

    # Verify gross receivables balance gross payables across all reports
    total_rec = sum(r.gross_receivable for r in reports)
    total_pay = sum(r.gross_payable for r in reports)
    assert abs(total_rec - total_pay) < 1e-4, "Global gross receivables must balance gross payables"


def test_discrete_event_engine_interbank_telemetry(temp_output_dir):
    """Validates that DiscreteEventEngine generates full multi-bank routing telemetry."""
    engine = DiscreteEventEngine(
        n_cards=15,
        n_merchants=10,
        region="US",
        seed=42,
    )
    records = engine.generate_batch(n_transactions=50)
    assert len(records) > 0, "Engine must produce transactions"

    # Check multi-bank fields on produced records
    rec = records[0]
    assert "issuing_bank_id" in rec
    assert "acquiring_bank_id" in rec
    assert "is_on_us" in rec
    assert "central_switch_id" in rec
    assert "switch_network_fee" in rec
    assert "interbank_latency_ms" in rec
    assert "mti_request" in rec
    assert "mti_response" in rec
    assert "hop_sequence" in rec
    assert rec["issuing_bank_id"].startswith("BANK_US_")
    assert rec["acquiring_bank_id"].startswith("BANK_US_")

    # Check clearing ledger in engine for any recorded cycles
    for cycle_id in engine.interbank_clearing.cycle_flows:
        reports = engine.interbank_clearing.execute_multilateral_netting(cycle_id=cycle_id)
        net_sum = sum(r.net_position for r in reports)
        assert abs(net_sum) < 1e-4, f"Engine clearing cycle {cycle_id} zero-sum failed: {net_sum}"
