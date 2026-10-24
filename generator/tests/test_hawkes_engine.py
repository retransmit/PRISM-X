"""Automated Verification Suite for Recursive Circadian Hawkes MTPP Engine.

Verifies:
1. Subcritical branching ratio (eta = alpha / beta < 1.0) and mathematical stability across all personas.
2. Exact O(1) recursive state accumulator identity: R_k = 1 + R_{k-1} * exp(-beta * dt) == sum_j exp(-beta * (t_k - t_j)).
3. Circadian 4-component periodic von Mises mixture and non-zero nocturnal floor.
4. Endogenous errand spree clustering and positive inter-arrival autocorrelation.
5. Closed-loop authorization feedback: approvals reinforce excitement; declines/freezes inhibit.
6. High-performance single-thread CPU throughput benchmark (>= 40,000 events/sec).
"""

import math
import time

import numpy as np
import pytest

from prism_generator.hawkes import (
    ADVERSARY_HAWKES_PROFILES,
    HawkesParameters,
    PERSONA_HAWKES_PROFILES,
    RecursiveCircadianHawkesEngine,
)


def test_subcritical_branching_ratio_and_stability():
    """Asserts that all calibrated human personas and adversarial profiles are subcritical (eta < 1.0)."""
    # 1. Verify all persona profiles
    for persona_id, params in PERSONA_HAWKES_PROFILES.items():
        assert params.branching_ratio < 1.0, (
            f"Persona {persona_id} has supercritical branching ratio: {params.branching_ratio:.4f} >= 1.0"
        )
        assert params.branching_ratio > 0.0
        assert params.expected_cluster_size > 1.0
        assert params.half_life_seconds > 0.0

    # 2. Verify all adversary profiles
    for adv_id, params in ADVERSARY_HAWKES_PROFILES.items():
        assert params.branching_ratio < 1.0, (
            f"Adversary {adv_id} has supercritical branching ratio: {params.branching_ratio:.4f} >= 1.0"
        )
        assert params.half_life_seconds > 0.0

    # 3. Supercritical parameters must raise ValueError on instantiation
    with pytest.raises(ValueError, match="Hawkes process is supercritical and unstable"):
        HawkesParameters(alpha=3.0e-3, beta=2.0e-3)  # eta = 1.5 >= 1.0


def test_recursive_accumulator_o1_identity():
    """Proves that the O(1) recursive update R_k == sum_{j=1}^k exp(-beta * (t_k - t_j)) exactly."""
    beta = 0.0025
    alpha = 0.0018
    params = HawkesParameters(mu_0=1.0e-5, alpha=alpha, beta=beta)
    engine = RecursiveCircadianHawkesEngine(default_params=params, seed=123)

    # Simulate 50 sequential arrival timestamps
    events = engine.simulate_stream(n_events=50, t_start_sec=1000.0, params=params)
    timestamps = [e["timestamp_sec"] for e in events]
    recursive_Rs = [e["hawkes_R"] for e in events]

    # Verify exact equivalence for every step k
    for k in range(len(timestamps)):
        t_k = timestamps[k]
        # Direct O(k) sum: sum_{j=0}^k exp(-beta * (t_k - t_j))
        direct_sum = sum(math.exp(-beta * (t_k - timestamps[j])) for j in range(k + 1))
        # Compare with recursive accumulator
        assert math.isclose(recursive_Rs[k], direct_sum, rel_tol=1e-9, abs_tol=1e-9), (
            f"Step {k}: Recursive R ({recursive_Rs[k]:.10f}) != Direct Sum ({direct_sum:.10f})"
        )


def test_circadian_nocturnal_floor_and_peak():
    """Asserts that circadian baseline density maintains non-zero nocturnal floor and daytime peak."""
    engine = RecursiveCircadianHawkesEngine(seed=42)

    # Midnight to 04:00 (overnight)
    night_phis = [engine.evaluate_circadian_phi(h * 3600.0) for h in np.linspace(2.0, 4.5, 30)]
    # Commute and lunch peaks (08:15 and 12:30)
    lunch_phi = engine.evaluate_circadian_phi(12.60 * 3600.0)
    evening_phi = engine.evaluate_circadian_phi(18.50 * 3600.0)

    # 1. Nocturnal floor must remain strictly positive (preventing unphysical zero-spend lockouts)
    assert min(night_phis) >= 0.024
    assert max(night_phis) <= 0.200

    # 2. Daytime peaks must significantly exceed night floor
    assert lunch_phi > 1.20
    assert evening_phi > 1.20
    assert lunch_phi > min(night_phis) * 10.0


def test_errand_spree_autocorrelation_clustering():
    """Asserts that Hawkes self-excitation creates endogenous errand sprees with positive autocorrelation."""
    # Suburban family profile (grocery -> pharmacy -> errands spree)
    params = PERSONA_HAWKES_PROFILES["C4_SUBURBAN_FAMILY"]
    engine = RecursiveCircadianHawkesEngine(default_params=params, seed=999)

    # Generate stream of events
    events = engine.simulate_stream(n_events=3000, t_start_sec=0.0, params=params)
    deltas = np.array([e["delta_t_sec"] for e in events])

    # Compute lag-1 autocorrelation of inter-arrival times
    dt_norm = deltas - np.mean(deltas)
    autocorr_lag1 = np.sum(dt_norm[:-1] * dt_norm[1:]) / np.sum(dt_norm ** 2)

    # Self-exciting clusters cause short intervals to follow short intervals (positive autocorrelation)
    # Memoryless Poisson has theoretical autocorrelation of 0.0
    assert autocorr_lag1 > 0.08, (
        f"Hawkes process failed to generate errand spree clustering: lag-1 autocorr = {autocorr_lag1:.4f} <= 0.08"
    )

    # Also verify that a non-trivial fraction of events arrive within a short cluster window (< 45 min = 2700s)
    short_gap_ratio = np.mean(deltas < 2700.0)
    assert short_gap_ratio >= 0.10, (
        f"Expected at least 10% clustered errand arrivals within 45 min, got {short_gap_ratio * 100:.2f}%"
    )


def test_closed_loop_feedback_adaptation():
    """Asserts that ISO 8583 authorization responses properly modulate Hawkes memory."""
    params = HawkesParameters(mu_0=1.0e-5, alpha=2.0e-3, beta=2.8e-3)
    engine = RecursiveCircadianHawkesEngine(default_params=params, seed=42)

    t_0 = 1000.0
    R_0 = 2.5

    # 1. Approved transaction (ISO 00) adds +1.0 to decayed memory
    t_1 = t_0 + 60.0  # 1 minute later
    R_approved = engine.update_feedback(
        R_current=R_0,
        last_tx_time_sec=t_0,
        event_time_sec=t_1,
        iso_response_code="00",
        params=params,
    )
    expected_decay = R_0 * math.exp(-params.beta * 60.0)
    assert math.isclose(R_approved, 1.0 + expected_decay, rel_tol=1e-6)
    assert R_approved > 1.0

    # 2. Insufficient funds (ISO 51) suppresses excitation by 50%
    R_insufficient = engine.update_feedback(
        R_current=R_0,
        last_tx_time_sec=t_0,
        event_time_sec=t_1,
        iso_response_code="51",
        params=params,
    )
    assert math.isclose(R_insufficient, expected_decay * 0.50, rel_tol=1e-6)

    # 3. Suspected fraud / velocity freeze (ISO 59 or 05) resets excitation to 0.0
    R_frozen = engine.update_feedback(
        R_current=R_0,
        last_tx_time_sec=t_0,
        event_time_sec=t_1,
        iso_response_code="59",
        params=params,
    )
    assert R_frozen == 0.0


def test_adversarial_bot_burst_dynamics():
    """Asserts that adversarial bot attack profile generates high-frequency sub-second bursts."""
    adv_params = ADVERSARY_HAWKES_PROFILES["ADV_MICRO_AUTH_PROBE"]
    engine = RecursiveCircadianHawkesEngine(default_params=adv_params, seed=777)

    # Adversarial micro-auth burst
    events = engine.simulate_stream(n_events=500, t_start_sec=100.0, params=adv_params)
    deltas = np.array([e["delta_t_sec"] for e in events])

    # Bot attacks have sub-second half-life (t_half ~ 0.35s)
    # Median delta should be very small during an active bot campaign
    median_dt = np.median(deltas)
    assert median_dt < 30.0, f"Adversarial bot probe failed to burst rapidly: median dt = {median_dt:.2f}s"
    # At least some events must arrive within 2 seconds
    sub_2s_count = np.sum(deltas < 2.0)
    assert sub_2s_count >= 10, f"Expected sub-second or rapid bursts, got {sub_2s_count} events under 2s"


def test_throughput_benchmark_cpu():
    """Asserts that O(1) Ogata thinning achieves >= 40,000 events/sec on single-thread CPU."""
    params = PERSONA_HAWKES_PROFILES["C3_YOUNG_ADULT_STUDENT"]
    engine = RecursiveCircadianHawkesEngine(default_params=params, seed=42)

    n_events = 20000
    t_start = time.perf_counter()
    events = engine.simulate_stream(n_events=n_events, t_start_sec=0.0, params=params)
    elapsed = time.perf_counter() - t_start

    throughput = n_events / elapsed
    assert len(events) == n_events
    # Must generate at least 40,000 events per second on CPU
    assert throughput >= 40000.0, (
        f"Hawkes engine throughput fell below performance threshold: {throughput:.1f} events/sec (elapsed={elapsed:.4f}s)"
    )
