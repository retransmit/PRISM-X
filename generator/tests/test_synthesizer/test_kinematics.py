"""Tests verifying space-time kinematic velocity limits and state protection."""

import pytest
from prism_generator import SimulationEngine


def test_no_supersonic_velocities_on_legitimate_transactions():
    """Legitimate Card-Present transactions must never exceed commercial airliner speeds (< 900 km/h)."""
    engine = SimulationEngine(n_cards=300, n_merchants=80, seed=42)
    records = engine.generate_batch(n_transactions=3000, fraud_prevalence=0.03)

    # Filter for legitimate Card-Present transactions
    legit_cp = [r for r in records if r["is_fraud"] == 0 and r["channel_type"].startswith("CP")]
    assert len(legit_cp) > 500

    for r in legit_cp:
        v = float(r["haversine_velocity_kph"])
        assert v <= 900.0, (
            f"Legitimate Card-Present tx {r['transaction_id']} (Card {r['card_id']}) "
            f"registered supersonic velocity {v:.2f} km/h!"
        )


def test_foreign_attack_does_not_poison_cardholder_spatial_state():
    """Foreign attacker transaction must not poison cardholder's legitimate physical anchor."""
    engine = SimulationEngine(n_cards=10, n_merchants=20, seed=123)
    card = engine.cards[0]

    # Initial legitimate swipe at home
    engine.ledger.enrich_transaction(
        tx_id="TX_LEGIT_1",
        card=card,
        merchant_id="M_HOME",
        merchant_name="Home Grocery",
        mcc=5411,
        merchant_category="Grocery",
        merchant_lat=card.home_lat,
        merchant_lon=card.home_lon,
        amount=30.0,
        tx_time=1000.0,
        channel_type="CP_POS_CHIP",
        is_fraud=0,
        scenario_tag="ORGANIC_NORMAL",
    )
    assert card.last_physical_lat == card.home_lat

    # Attacker in Tokyo steals credentials and makes an IMPOSSIBLE_TRAVEL swipe 1 hour later
    tokyo_lat, tokyo_lon = 35.6762, 139.6503
    r2 = engine.ledger.enrich_transaction(
        tx_id="TX_FRAUD_TOKYO",
        card=card,
        merchant_id="M_TOKYO",
        merchant_name="Tokyo Luxury Goods",
        mcc=5094,
        merchant_category="Jewelry",
        merchant_lat=tokyo_lat,
        merchant_lon=tokyo_lon,
        amount=1200.0,
        tx_time=4600.0,  # 1 hour later
        channel_type="CP_POS_CHIP",
        is_fraud=1,
        scenario_tag="IMPOSSIBLE_TRAVEL",
    )
    # The attacker's transaction correctly registers impossible travel velocity
    assert r2["haversine_velocity_kph"] > 5000.0

    # CRITICAL INVARIANT: The cardholder's legitimate physical anchor MUST NOT be moved to Tokyo!
    assert card.last_physical_lat == pytest.approx(card.home_lat, abs=1e-5)
    assert card.last_physical_lon == pytest.approx(card.home_lon, abs=1e-5)

    # 3 hours later, legitimate cardholder buys lunch near home
    # This must NOT register Tokyo -> Home velocity (10,000 km/h)!
    r3 = engine.ledger.enrich_transaction(
        tx_id="TX_LEGIT_2",
        card=card,
        merchant_id="M_LUNCH",
        merchant_name="Midtown Diner",
        mcc=5812,
        merchant_category="Restaurant",
        merchant_lat=card.home_lat + 0.01,
        merchant_lon=card.home_lon + 0.01,
        amount=22.0,
        tx_time=15400.0,  # 3 hours later
        channel_type="CP_POS_CHIP",
        is_fraud=0,
        scenario_tag="ORGANIC_NORMAL",
    )
    # Velocity should be a gentle local driving/walking speed (< 50 km/h)
    assert r3["haversine_velocity_kph"] < 50.0, (
        f"Spatial state was poisoned by Tokyo attack! Registered velocity: {r3['haversine_velocity_kph']:.2f} km/h"
    )
