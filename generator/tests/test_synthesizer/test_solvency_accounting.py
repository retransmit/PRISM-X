"""Tests verifying core banking balance mechanics, pre-authorization holds,
dual-message settlement, per-target adversary adaptation, and cohort parameterization.
"""

import numpy as np
import pytest
from prism_generator import CardholderProfile
from prism_generator.agents import (
    AdaptiveFraudsterAgent,
    BankDecisionEngine,
    CardholderState,
    ISO8583Response,
    decay_after_insufficient_funds,
)


def test_swipe_over_fifty_percent_credit_line_approves():
    """Verifies that legitimate purchases > 50% of credit line approve (fixes Defect 1)."""
    bank = BankDecisionEngine()
    card = CardholderProfile(
        card_id="CARD_TEST_01",
        home_lat=40.7128,
        home_lon=-74.0060,
        work_lat=40.7589,
        work_lon=-73.9851,
        is_commuter=True,
        credit_limit=1000.0,
        posted_balance=100.0,
        pending_holds=0.0,
        is_frozen=False,
    )

    # Initial available credit: 1000 - 100 = 900.
    assert card.get_available_balance() == 900.0

    # Swipe for $750 (which is 83% of remaining $900 credit line, and 75% of total line)
    resp, trans_3ds, approved_amt = bank.evaluate_authorization(
        card=card,
        amount=750.0,
        channel="CP_POS_CHIP",
        sim_time=1704067200.0,
        velocity_kph=0.0,
        count_1h=1,
        cvv_valid=True,
    )

    # Under the defect, this would fail with ISO 51 Insufficient Funds.
    # In the fixed implementation, it MUST approve!
    assert resp == ISO8583Response.APPROVED_00
    assert approved_amt == 750.0


def test_declined_transaction_has_zero_balance_mutation():
    """Verifies that declined transactions do not mutate posted balance or pending holds."""
    bank = BankDecisionEngine()
    card = CardholderProfile(
        card_id="CARD_TEST_02",
        home_lat=40.7128,
        home_lon=-74.0060,
        work_lat=40.7589,
        work_lon=-73.9851,
        is_commuter=True,
        credit_limit=500.0,
        posted_balance=200.0,
        pending_holds=0.0,
        is_frozen=False,
    )

    initial_avail = card.get_available_balance()
    initial_posted = card.posted_balance
    initial_pending = card.pending_holds

    # Attempt a transaction exceeding available credit ($400 > $300 available)
    resp, _, _ = bank.evaluate_authorization(
        card=card,
        amount=400.0,
        channel="CP_POS_CHIP",
        sim_time=1704067200.0,
        velocity_kph=0.0,
        count_1h=1,
        cvv_valid=True,
    )

    assert resp == ISO8583Response.INSUFFICIENT_FUNDS_51
    assert card.get_available_balance() == initial_avail
    assert card.posted_balance == initial_posted
    assert card.pending_holds == initial_pending


def test_dual_message_clearing_hold_lifecycle():
    """Verifies pre-auth hold placement, clearing settlement, and balance transitions."""
    card = CardholderProfile(
        card_id="CARD_TEST_03",
        home_lat=40.7128,
        home_lon=-74.0060,
        work_lat=40.7589,
        work_lon=-73.9851,
        is_commuter=True,
        credit_limit=2000.0,
        posted_balance=0.0,
        pending_holds=0.0,
    )

    # 1. Place pre-auth hold for $100.00
    tx_id = "TX_TEST_1001"
    card.place_hold(tx_id=tx_id, hold_amount=100.0, expire_time_sec=1704153600.0)

    assert card.pending_holds == 100.0
    assert card.posted_balance == 0.0
    assert card.current_balance == 100.0
    assert card.get_available_balance() == 1900.0

    # 2. Clearing settlement presentment ($T+2) with dining tip ($100 auth -> $118 settled)
    card.settle_hold(tx_id=tx_id, settled_amount=118.0)

    assert card.pending_holds == 0.0
    assert card.posted_balance == 118.0
    assert card.current_balance == 118.0
    assert card.get_available_balance() == 1882.0


def test_per_target_adversary_memory_and_adaptation():
    """Verifies that adversary maintains distinct memory per target card and adapts."""
    rng = np.random.default_rng(42)
    fraudster = AdaptiveFraudsterAgent(rng=rng)
    card_a = CardholderProfile(
        card_id="CARD_VICTIM_A",
        home_lat=40.7128,
        home_lon=-74.0060,
        work_lat=40.7589,
        work_lon=-73.9851,
        is_commuter=True,
        credit_limit=5000.0,
    )
    card_b = CardholderProfile(
        card_id="CARD_VICTIM_B",
        home_lat=34.0522,
        home_lon=-118.2437,
        work_lat=34.0722,
        work_lon=-118.2637,
        is_commuter=True,
        credit_limit=5000.0,
    )

    # Attack Card A: probe
    params_a1 = fraudster.select_attack_playbook(card_a, 1704067200.0, 40.7128, -74.0060)
    amt_a1 = params_a1["amount"]

    # Feedback on Card A: ISO 51 (Insufficient Funds)
    fraudster.receive_feedback(
        response_code=ISO8583Response.INSUFFICIENT_FUNDS_51,
        trans_status_3ds="N",
        sim_time_seconds=1704067200.0,
        card_id=card_a.card_id,
    )

    # State for Card A has decayed probe amount
    target_a = fraudster.target_states[card_a.card_id]
    assert target_a.consecutive_declines == 1
    assert target_a.current_probe_amount < amt_a1
    assert target_a.current_probe_amount == decay_after_insufficient_funds(amt_a1)

    # Attack Card B: Should NOT be throttled or decayed by Card A's decline
    fraudster.select_attack_playbook(card_b, 1704067200.0, 34.0522, -118.2437)
    target_b = fraudster.target_states[card_b.card_id]
    assert target_b.consecutive_declines == 0


def test_commuter_anchor_location_schedule():
    """Verifies that commuter cards route to work during 09:00 - 17:00 on weekdays."""
    card = CardholderProfile(
        card_id="CARD_COMMUTER_01",
        home_lat=40.7128,
        home_lon=-74.0060,
        work_lat=40.7589,
        work_lon=-73.9851,
        is_commuter=True,
        credit_limit=3000.0,
    )

    # Monday 14:00 (work hours)
    # 1704117600 is 2024-01-01 14:00:00 UTC (Monday)
    work_t = 1704117600.0
    lat, lon, state = card.get_current_anchor_location(work_t)
    assert abs(lat - card.work_lat) < 1e-4
    assert abs(lon - card.work_lon) < 1e-4

    # Monday 22:00 (home hours)
    home_t = 1704146400.0
    lat, lon, state = card.get_current_anchor_location(home_t)
    assert state == CardholderState.HOMESTEAD
    assert abs(lat - card.home_lat) < 1e-4
    assert abs(lon - card.home_lon) < 1e-4


@pytest.mark.parametrize("previous", [1.95, 10.0, 24.99, 25.0, 25.01, 30.0, 100.0, 5000.0])
def test_insufficient_funds_never_raises_the_next_attempt(previous):
    """An ISO 51 decline means the amount was more than the card could pay, so the next
    attempt is lower, never higher. The floor only stops the decay of amounts above it:
    a 1.95 micro-probe followed by a 25.00 attempt would be an adversary learning backwards."""
    following = decay_after_insufficient_funds(previous)
    assert following < previous
    expected = max(round(previous * 0.70, 2), 25.0) if previous > 25.0 else round(previous * 0.70, 2)
    assert following == expected


def test_decline_feedback_lowers_the_target_and_the_agent_amount():
    """The per-card memory and the agent's own amount both follow the same rule."""
    fraudster = AdaptiveFraudsterAgent(rng=np.random.default_rng(7))
    card = CardholderProfile(
        card_id="CARD_VICTIM_C",
        home_lat=40.7128,
        home_lon=-74.0060,
        work_lat=40.7589,
        work_lon=-73.9851,
        is_commuter=False,
        credit_limit=5000.0,
    )
    fraudster.select_attack_playbook(card, 1704067200.0, 40.7128, -74.0060)
    target = fraudster.target_states[card.card_id]
    for previous in (1.95, 64.31):
        target.current_probe_amount = previous
        fraudster.current_amount = previous
        fraudster.receive_feedback(
            response_code=ISO8583Response.INSUFFICIENT_FUNDS_51,
            trans_status_3ds="N",
            sim_time_seconds=1704067300.0,
            card_id=card.card_id,
        )
        assert target.current_probe_amount < previous
        assert fraudster.current_amount < previous
