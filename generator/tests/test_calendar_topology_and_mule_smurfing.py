"""Tests for Calendar-Driven Topology Dynamics & Cross-Bank Money Mule Smurfing Rings.

Grounded in:
1. Federal Reserve Diary of Consumer Payment Choice (DCPC 2024).
2. Reserve Bank of India Annual Report 2023-24 (Table IX.1).
3. FinCEN Advisory FIN-2020-A003 on Money Mule Schemes & BSA 31 CFR 1010.311.
4. Indian Cyber Crime Coordination Centre (I4C) CFCFRMS 1930 Framework.
"""

from datetime import datetime, timezone
import numpy as np
import pytest

from prism_generator.engine import CalendarTopologyDynamics, DiscreteEventEngine
from prism_generator.rails import BankRoutingMesh
from prism_generator.syndicates import (
    CrossBankSmurfingSynthesizer,
    SmurfTransferLeg,
)


def test_calendar_topology_dynamics_exact_formula():
    """Validates exact closed-form logistic probability targets for US and IN."""
    # 1. United States Dynamics
    us_topo = CalendarTopologyDynamics(region="US")

    # Normal weekday mid-month (e.g. 10th of month, Wednesday)
    # Target: 0.0700 (7.0%)
    dt_us_normal = datetime(2024, 4, 10, 12, 0, tzinfo=timezone.utc)
    t_us_normal = dt_us_normal.timestamp()
    p_us_normal = us_topo.evaluate_p2p_probability(t_us_normal, is_holiday_sale=False)
    assert pytest.approx(0.0700, abs=0.005) == p_us_normal

    # Payday 1st of month (e.g. 1st of month, Monday)
    # Target: 0.1520 (15.2%)
    dt_us_payday_1 = datetime(2024, 4, 1, 12, 0, tzinfo=timezone.utc)
    p_us_payday_1 = us_topo.evaluate_p2p_probability(dt_us_payday_1.timestamp(), is_holiday_sale=False)
    assert pytest.approx(0.1520, abs=0.005) == p_us_payday_1

    # Payday 15th of month (e.g. 15th of month, Monday)
    # Target: 0.1520 (15.2%)
    dt_us_payday_15 = datetime(2024, 4, 15, 12, 0, tzinfo=timezone.utc)
    p_us_payday_15 = us_topo.evaluate_p2p_probability(dt_us_payday_15.timestamp(), is_holiday_sale=False)
    assert pytest.approx(0.1520, abs=0.005) == p_us_payday_15

    # Black Friday Megasale (e.g. 29th of November, Friday, is_holiday_sale=True)
    # Target: 0.025 to 0.030 (2.5% to 3.0%)
    dt_us_bf = datetime(2024, 11, 29, 14, 0, tzinfo=timezone.utc)
    p_us_bf = us_topo.evaluate_p2p_probability(dt_us_bf.timestamp(), is_holiday_sale=True)
    assert 0.020 <= p_us_bf <= 0.035

    # 2. India Dynamics
    in_topo = CalendarTopologyDynamics(region="IN")

    # Normal weekday mid-month (e.g. 15th of month, Wednesday)
    # Target: 0.2000 (20.0%)
    dt_in_normal = datetime(2024, 5, 15, 12, 0, tzinfo=timezone.utc)
    p_in_normal = in_topo.evaluate_p2p_probability(dt_in_normal.timestamp(), is_holiday_sale=False)
    assert pytest.approx(0.2000, abs=0.005) == p_in_normal

    # Month-end salary credit (e.g. 1st of month)
    # Target: 0.3400 (34.0%)
    dt_in_payday_1 = datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc)
    p_in_payday_1 = in_topo.evaluate_p2p_probability(dt_in_payday_1.timestamp(), is_holiday_sale=False)
    assert pytest.approx(0.3400, abs=0.005) == p_in_payday_1

    # Month-end day 31 smooth transition
    dt_in_payday_31 = datetime(2024, 5, 31, 12, 0, tzinfo=timezone.utc)
    p_in_payday_31 = in_topo.evaluate_p2p_probability(dt_in_payday_31.timestamp(), is_holiday_sale=False)
    assert 0.280 <= p_in_payday_31 <= 0.340

    # Diwali Megasale mid-month (is_holiday_sale=True, Oct 20)
    # Target: 0.0820 (8.2%)
    dt_in_diwali = datetime(2024, 10, 20, 14, 0, tzinfo=timezone.utc)
    p_in_diwali = in_topo.evaluate_p2p_probability(dt_in_diwali.timestamp(), is_holiday_sale=True)
    assert 0.075 <= p_in_diwali <= 0.095



def test_calendar_topology_kernel_continuity():
    """Asserts circular distance continuity at month boundaries (day 31 -> day 1)."""
    in_topo = CalendarTopologyDynamics(region="IN")

    # Circular distance from day 31 to day 1 is 1 day: min(|31 - 1|, 31 - |31 - 1|) = min(30, 1) = 1
    # Circular distance from day 2 to day 1 is 1 day: min(|2 - 1|, 31 - |2 - 1|) = min(1, 30) = 1
    k_31 = in_topo.compute_payday_kernel(31)
    k_2 = in_topo.compute_payday_kernel(2)
    assert pytest.approx(k_2, abs=1e-6) == k_31
    assert k_31 > 0.70  # Elevated kernel near payroll anchor

    # All days across month must yield valid probabilities in (0, 1)
    for d in range(1, 32):
        k = in_topo.compute_payday_kernel(d)
        assert 0.0 <= k <= 1.0


def test_p2p_interbank_routing_resolution():
    """Asserts that P2P transactions resolve acquiring bank to the recipient card issuing bank."""
    mesh = BankRoutingMesh(region="US", seed=42)

    sender_card = "CARD_000001"
    sender_bank = mesh.resolve_issuing_bank(sender_card)

    # Pick another card and verify routing
    recip_card = "CARD_000002"
    recip_bank = mesh.resolve_issuing_bank(recip_card)

    p2p_merchant_id = f"P2P_{recip_card}"
    route_info = mesh.route_transaction(
        card_id=sender_card,
        merchant_id=p2p_merchant_id,
        channel_type="P2P_PUSH",
        amount=50.0,
    )

    assert route_info["issuing_bank"].id == sender_bank.id
    assert route_info["acquiring_bank"].id == recip_bank.id

    if sender_bank.id == recip_bank.id:
        assert route_info["is_on_us"] is True
        assert route_info["central_switch_id"] == "ON_US_INTERNAL"
    else:
        assert route_info["is_on_us"] is False
        assert route_info["central_switch_id"] == "FEDNOW_SERVICE"


def test_cross_bank_smurfing_synthesizer_fan_out():
    """Validates categorical fan-out distribution K in {2, 3, 4, 5} and strict institutional diversity."""
    # 1. US Evaluation
    smurf_us = CrossBankSmurfingSynthesizer(region="US", seed=123)
    k_samples = [smurf_us.sample_fan_out_degree() for _ in range(2000)]
    mean_k = float(np.mean(k_samples))
    # Theoretical E[K] = 2(0.25) + 3(0.40) + 4(0.25) + 5(0.10) = 3.20
    assert 3.10 <= mean_k <= 3.30
    assert set(k_samples) == {2, 3, 4, 5}

    # Strict Institutional Diversity Invariant: Bank(M_k) != Bank(V_0) and Bank(M_k) != Bank(M_j)
    victim_bank = "BANK_US_CHASE"
    for _ in range(500):
        k = smurf_us.sample_fan_out_degree()
        dests = smurf_us.sample_destination_banks(victim_bank, k)
        expected_len = min(k, len(smurf_us.bank_ids) - 1)
        assert len(dests) == expected_len, "Destination bank list must contain min(K, N-1) distinct banks"
        assert len(set(dests)) == expected_len, "All destination banks must be distinct"
        assert victim_bank not in dests, "Destination banks must strictly exclude source victim bank"

    # 2. India Evaluation
    smurf_in = CrossBankSmurfingSynthesizer(region="IN", seed=456)
    victim_bank_in = "BANK_IN_SBI"
    for _ in range(500):
        k = smurf_in.sample_fan_out_degree()
        dests = smurf_in.sample_destination_banks(victim_bank_in, k)
        expected_len_in = min(k, len(smurf_in.bank_ids) - 1)
        assert len(dests) == expected_len_in
        assert len(set(dests)) == expected_len_in
        assert victim_bank_in not in dests



def test_cross_bank_smurfing_structuring_ticket_sizes():
    """Asserts that structuring amounts strictly comply with BSA ($10k) and PMLA (Rs 200k) ceilings."""
    # 1. US Tickets: Truncated LogNormal [$1,500, $8,500]
    smurf_us = CrossBankSmurfingSynthesizer(region="US", seed=789)
    us_tickets = [smurf_us.sample_structuring_ticket() for _ in range(5000)]
    assert all(1500.0 <= t <= 8500.0 for t in us_tickets), "US ticket size outside [1500, 8500]"
    assert all(t < 10000.0 for t in us_tickets), "US ticket size violated BSA $10,000 threshold"
    assert 3500.0 <= np.mean(us_tickets) <= 3900.0
    assert 3200.0 <= np.median(us_tickets) <= 3600.0

    # 2. IN Tickets: Truncated LogNormal [Rs 25,000, Rs 180,000]
    smurf_in = CrossBankSmurfingSynthesizer(region="IN", seed=987)
    in_tickets = [smurf_in.sample_structuring_ticket() for _ in range(5000)]
    assert all(25000.0 <= t <= 180000.0 for t in in_tickets), "IN ticket size outside [25000, 180000]"
    assert all(t < 200000.0 for t in in_tickets), "IN ticket size violated PMLA Rs 200,000 threshold"
    assert 70000.0 <= np.mean(in_tickets) <= 80000.0
    assert 65000.0 <= np.median(in_tickets) <= 75000.0


def test_cross_bank_smurfing_hop_latency():
    """Validates off-ramp liquidation latency distribution (median ~22.5 min)."""
    smurf = CrossBankSmurfingSynthesizer(region="US", seed=111)
    latencies = [smurf.sample_hop_latency_seconds() for _ in range(5000)]
    assert all(600.0 <= lat <= 3600.0 for lat in latencies), "Hop latency outside [600, 3600] seconds"
    median_lat = float(np.median(latencies))
    # Median should be close to 1350 seconds (22.5 minutes)
    assert 1250.0 <= median_lat <= 1450.0


def test_generate_smurfing_ring_operation_pipeline():
    """Validates complete 4-stage cross-bank smurfing operation output."""
    smurf_us = CrossBankSmurfingSynthesizer(region="US", seed=222)
    t0 = 1700000000.0
    legs = smurf_us.generate_smurfing_ring_operation(
        victim_id="CARD_VICTIM_001",
        victim_bank="BANK_US_CHASE",
        t_ingress_sec=t0,
    )

    assert 2 <= len(legs) <= 5
    dest_banks = [leg.mule_bank_id for leg in legs]
    assert len(dest_banks) == len(set(dest_banks)), "Every mule leg must target a distinct bank"
    assert "BANK_US_CHASE" not in dest_banks, "Mule legs must not target victim bank"

    for leg in legs:
        assert isinstance(leg, SmurfTransferLeg)
        assert leg.victim_account_id == "CARD_VICTIM_001"
        assert leg.victim_bank_id == "BANK_US_CHASE"
        assert leg.amount < 10000.0
        assert leg.dispatch_time_sec >= t0
        assert leg.liquidation_time_sec > leg.dispatch_time_sec
        assert 600.0 <= (leg.liquidation_time_sec - leg.dispatch_time_sec) <= 3600.0
        assert leg.offramp_channel in ("P2P_CRYPTO_USDT_ESCROW", "ATM_CLONED_CARD_CASHOUT")


def test_simulation_integration_p2p_and_mule_telemetry():
    """Executes a discrete-event batch and verifies P2P topology and cross-bank mule telemetry."""
    engine = DiscreteEventEngine(
        n_cards=100,
        n_merchants=20,
        region="US",
        seed=333,
        annual_seasonality=True,
        p2p_enabled=True,
    )

    records = engine.generate_batch(n_transactions=500, time_span_days=10)
    assert len(records) == 500


    # 1. Verify P2P transactions are generated
    p2p_txs = [r for r in records if str(r.get("channel_type", "")).startswith("P2P") or "P2P" in str(r.get("merchant_id", ""))]
    assert len(p2p_txs) > 0, "Simulation with p2p_enabled=True must generate P2P transactions"

    # 2. Verify all records carry mule_bank_id and layering_hop_latency_seconds
    for r in records:
        assert "mule_bank_id" in r
        assert "layering_hop_latency_seconds" in r

    # 3. Verify fraudulent records with mule assignments respect diversity
    fraud_mules = [r for r in records if r.get("is_fraud") == 1 and r.get("mule_bank_id")]
    for r in fraud_mules:
        card_id = str(r["card_id"])
        card_bank = engine.rail_switch.routing_mesh.resolve_issuing_bank(card_id).id
        mule_bank = str(r["mule_bank_id"])
        # If mule_bank is one of the commercial banks, assert diversity
        if mule_bank.startswith("BANK_US_"):
            assert mule_bank != card_bank, f"Mule bank {mule_bank} must differ from victim card bank {card_bank}"
