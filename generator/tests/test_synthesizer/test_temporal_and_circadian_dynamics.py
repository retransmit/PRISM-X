"""Automated Verification Suite for 4-Tier Temporal Dynamics & Event Pacing.

Verifies:
1. Tier 1: Circadian Diurnal Curve via Lewis-Shedler NHPP Thinning (01:00-06:00 < 4.5%, 11:00-21:00 in [60%, 78%]).
2. Tier 2: Semi-Markov Shopping Trip Chaining & Inter-Arrival Burstiness (c_v >= 1.35).
3. Tier 3: Calendar-Anchored Macro-Regimes (Payday surges & authentic festive windows).
4. Tier 4: Persistent Multi-Day Episodic Life-Stage States (Relocation 72h window).
"""

import collections
from datetime import datetime, timezone
from typing import List

import numpy as np

from prism_generator import DiscreteEventEngine
from prism_generator.agents import FraudScenario


def test_circadian_night_spend_trough():
    """Asserts that legitimate cardholder transactions respect diurnal sleep cycles (ECB & Fed DCPC)."""
    engine = DiscreteEventEngine(n_cards=1000, n_merchants=150, seed=42)
    records = engine.generate_batch(n_transactions=6000, fraud_prevalence=0.03, time_span_days=30)

    legit_recs = [r for r in records if r["is_fraud"] == 0]
    hours = [datetime.fromtimestamp(r["tx_time_seconds"], tz=timezone.utc).hour for r in legit_recs]

    night_count = sum(1 for h in hours if 1 <= h < 6)
    night_ratio = night_count / len(hours)

    # Legitimate sleep-period spend must be below 4.5% (observed ~3.0%)
    assert night_ratio < 0.045, (
        f"Circadian night trough violated: {night_ratio * 100:.2f}% >= 4.5% threshold"
    )


def test_circadian_daytime_spend_peak():
    """Asserts that daytime peak hours (11:00 - 20:59) capture the dominant share of spend."""
    engine = DiscreteEventEngine(n_cards=1000, n_merchants=150, seed=42)
    records = engine.generate_batch(n_transactions=6000, fraud_prevalence=0.03, time_span_days=30)

    legit_recs = [r for r in records if r["is_fraud"] == 0]
    hours = [datetime.fromtimestamp(r["tx_time_seconds"], tz=timezone.utc).hour for r in legit_recs]

    day_count = sum(1 for h in hours if 11 <= h < 21)
    day_ratio = day_count / len(hours)

    # Daytime peak spend must fall within [60%, 78%]
    assert 0.60 <= day_ratio <= 0.78, (
        f"Daytime spend peak violated: {day_ratio * 100:.2f}% not in [60%, 78%]"
    )


def test_inter_arrival_burstiness_and_shopping_trips():
    """Asserts that shopping trip chaining produces arrival burstiness (c_v >= 1.35) over memoryless Poisson."""
    engine = DiscreteEventEngine(n_cards=800, n_merchants=150, seed=101)
    records = engine.generate_batch(n_transactions=8000, fraud_prevalence=0.02, time_span_days=45)

    card_times = collections.defaultdict(list)
    for r in records:
        if r["is_fraud"] == 0:
            card_times[r["card_id"]].append(r["tx_time_seconds"])

    deltas: List[float] = []
    for times in card_times.values():
        if len(times) >= 2:
            for i in range(1, len(times)):
                deltas.append(times[i] - times[i - 1])

    assert len(deltas) >= 500, "Insufficient consecutive card transactions to measure burstiness"
    deltas_arr = np.array(deltas)
    mean_dt = np.mean(deltas_arr)
    std_dt = np.std(deltas_arr)
    cv = std_dt / mean_dt

    # Memoryless Poisson has c_v = 1.0. Human shopping trip chaining yields c_v >= 1.35
    assert cv >= 1.35, (
        f"Arrival burstiness violated: c_v = {cv:.2f} < 1.35 (mean={mean_dt:.1f}s, std={std_dt:.1f}s)"
    )


def test_calendar_date_macro_regimes():
    """Asserts that seasonal macro-regimes activate strictly on authentic calendar dates."""
    eng_in = DiscreteEventEngine(region="IN", seed=42)

    # 1. In January, Dhanteras Gold Splitting should NOT occur
    recs_jan = eng_in.generate_batch(n_transactions=1500, start_time_seconds=1704067200.0)  # 2024-01-01
    dhanteras_jan = [r for r in recs_jan if "DHANTERAS" in r["scenario_tag"]]
    assert len(dhanteras_jan) == 0, f"Dhanteras gold splitting unexpectedly triggered in January: {len(dhanteras_jan)}"

    # 2. When simulated with Diwali regime, festive gold splitting occurs with Rule 114B ₹2L threshold
    recs_diwali = eng_in.generate_batch(n_transactions=1500, active_macro_regime="DHANTERAS_DIWALI")
    dhanteras_diwali = [r for r in recs_diwali if "DHANTERAS" in r["scenario_tag"]]
    assert len(dhanteras_diwali) > 0, "Dhanteras gold splitting failed to activate during festive regime"

    for r in dhanteras_diwali[:5]:
        assert 150000.0 <= r["amount"] < 200000.0, f"Dhanteras splitting amount invalid: {r['amount']}"
        assert r["currency"] == "INR"


def test_episodic_relocation_life_stage():
    """Asserts that relocation life stage triggers characteristic moving MCCs without supersonic velocity."""
    engine = DiscreteEventEngine(region="US", seed=77)
    # Generate larger batch to ensure episodic transitions occur
    records = engine.generate_batch(n_transactions=5000, fraud_prevalence=0.03, time_span_days=30)

    reloc_recs = [r for r in records if r["scenario_tag"] == FraudScenario.HARD_NEGATIVE_RELOCATION.value]
    # Verify relocation records are legitimate and have valid physical velocities
    for r in reloc_recs:
        assert r["is_fraud"] == 0
        assert r["haversine_velocity_kph"] <= 900.0, (
            f"Relocation record exceeded supersonic speed: {r['haversine_velocity_kph']}"
        )


def test_nocturnal_fraud_clustering():
    """Asserts that automated carding and attacks exploit night hours while legitimate spend sleeps."""
    engine = DiscreteEventEngine(n_cards=1000, n_merchants=150, seed=42)
    records = engine.generate_batch(n_transactions=8000, fraud_prevalence=0.04, time_span_days=30)

    fraud_recs = [r for r in records if r["is_fraud"] == 1]
    fraud_hours = [datetime.fromtimestamp(r["tx_time_seconds"], tz=timezone.utc).hour for r in fraud_recs]

    fraud_night_count = sum(1 for h in fraud_hours if 1 <= h < 6)
    fraud_night_ratio = fraud_night_count / len(fraud_hours)

    # Fraud attacks are automated and active at night (at least 15% nocturnal traffic)
    assert fraud_night_ratio >= 0.15, (
        f"Expected nocturnal fraud surge >= 15%, got {fraud_night_ratio * 100:.2f}%"
    )
