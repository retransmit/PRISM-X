"""Recursive Circadian Marked Temporal Point Process (MTPP) Engine for Prism Generator.

Implements:
1. Continuous-time intensity lambda*(t | H_t) with periodic 24-hour von Mises mixture baseline.
2. Grounded non-zero nocturnal floor (beta_0 = 0.025) preventing unphysical zero-spend lockouts.
3. High-performance O(1) recursive state tracking (R_k = 1 + R_{k-1} * exp(-beta * dt))
   enabling exact Ogata thinning at >= 50,000 transactions/sec on single-thread CPU.
4. Endogenous legitimate shopping sprees (gas -> grocery -> pharmacy in 45-90 min)
   governed by branching ratio eta = alpha / beta < 1.0 (subcritical, mathematically stable).
5. High-frequency adversarial bot bursts (card testing micro-auth probes, t_half = 1.0s).
6. Closed-loop feedback: approvals excite memory; declines and freezes inhibit intensity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class HawkesParameters:
    """Calibrated physical and temporal parameters for Hawkes arrival processes."""
    mu_0: float = 1.0e-5           # Base unexcited arrival rate (~0.86 tx/day) in Hz (1/sec)
    alpha: float = 1.85e-3         # Hawkes excitation jump magnitude in Hz (1/sec)
    beta: float = 2.78e-3          # Exponential memory decay rate in Hz (1/sec), t_half ~ 250s
    beta_0_floor: float = 0.025    # Nocturnal floor ratio (2.5% baseline overnight)
    phi_max: float = 3.85          # Maximum relative density of 24-hour diurnal profile
    phase_offset_sec: float = 0.0  # Cardholder's local-clock offset into the diurnal cycle (2026-10-16)

    def __post_init__(self) -> None:
        branching_ratio = self.alpha / self.beta if self.beta > 0 else 0.0
        if branching_ratio >= 1.0:
            raise ValueError(
                f"Hawkes process is supercritical and unstable! "
                f"Branching ratio eta = {branching_ratio:.4f} >= 1.0 (alpha={self.alpha}, beta={self.beta})."
            )

    @property
    def branching_ratio(self) -> float:
        """Expected number of directly excited daughter events per parent event."""
        return self.alpha / self.beta

    @property
    def expected_cluster_size(self) -> float:
        """Expected total size of a self-excited cluster (including parent event)."""
        return 1.0 / (1.0 - self.branching_ratio)

    @property
    def half_life_seconds(self) -> float:
        """Half-life of behavioral excitation memory."""
        return math.log(2.0) / self.beta


# Default calibrated profiles grounded in spec/02_human_personas.yaml
PERSONA_HAWKES_PROFILES: Dict[str, HawkesParameters] = {
    "C1_HOURLY_GIG_WORKER": HawkesParameters(
        mu_0=9.13e-6,  # 1.85 tx/day target, eta = 0.650, E[cluster] ~ 2.857
        alpha=1.95e-3,
        beta=3.00e-3,
        beta_0_floor=0.035, # Higher nocturnal activity (rideshare/night shifts)
        phi_max=3.60,
    ),
    "C2_FIXED_INCOME_SENIOR": HawkesParameters(
        mu_0=9.14e-6,  # 1.35 tx/day target, eta = 0.520, E[cluster] ~ 2.083
        alpha=1.30e-3,
        beta=2.50e-3,
        beta_0_floor=0.010, # Very low nocturnal activity
        phi_max=4.20,
    ),
    "C3_YOUNG_ADULT_STUDENT": HawkesParameters(
        mu_0=8.88e-6,  # 2.10 tx/day target, eta = 0.700, E[cluster] ~ 3.333
        alpha=2.10e-3,
        beta=3.00e-3,
        beta_0_floor=0.045, # High nightlife / digital midnight browsing
        phi_max=3.50,
    ),
    "C4_SUBURBAN_FAMILY": HawkesParameters(
        mu_0=1.03e-5,  # 2.65 tx/day target, eta = 0.725, E[cluster] ~ 3.636
        alpha=2.25e-3,
        beta=3.10e-3,
        beta_0_floor=0.015,
        phi_max=3.95,
    ),
    "C5_TECH_PROFESSIONAL": HawkesParameters(
        mu_0=1.11e-5,  # 2.45 tx/day target, eta = 0.680, E[cluster] ~ 3.125
        alpha=1.90e-3,
        beta=2.80e-3,
        beta_0_floor=0.030,
        phi_max=3.75,
    ),
    "C5_URBAN_TECH_PROFESSIONAL": HawkesParameters(
        mu_0=1.11e-5,  # 2.45 tx/day target, eta = 0.680, E[cluster] ~ 3.125
        alpha=1.90e-3,
        beta=2.80e-3,
        beta_0_floor=0.030,
        phi_max=3.75,
    ),
    "C6_COMMERCIAL_SMALL_BIZ": HawkesParameters(
        mu_0=1.14e-5,  # 2.70 tx/day target, eta = 0.700, E[cluster] ~ 3.333
        alpha=2.10e-3,
        beta=3.00e-3,
        beta_0_floor=0.020,
        phi_max=4.10,
    ),
    "C6_SMALL_BUSINESS_OWNER": HawkesParameters(
        mu_0=1.14e-5,  # 2.70 tx/day target, eta = 0.700, E[cluster] ~ 3.333
        alpha=2.10e-3,
        beta=3.00e-3,
        beta_0_floor=0.020,
        phi_max=4.10,
    ),
    "C7_LUXURY_AFFLUENT": HawkesParameters(
        mu_0=9.53e-6,  # 2.60 tx/day target, eta = 0.740, E[cluster] ~ 3.846
        alpha=2.37e-3,
        beta=3.20e-3,
        beta_0_floor=0.030,
        phi_max=3.80,
    ),
}

# Calibrated Indian persona Hawkes profiles grounded in RBI Payment System Indicators & spec/05_india_payment_rails.yaml
# Derived via closed-form MTPP stationary expectation:
# mu_0 = (E[N(1 day)] * (1 - eta)) / 70917.12
# where daily frequency band reflects UPI micropayment substitution (cards reserved for high-value/planned purchases).
INDIAN_PERSONA_HAWKES_PROFILES: Dict[str, HawkesParameters] = {
    "IN_C1_HOURLY_GIG_WORKER": HawkesParameters(
        mu_0=2.855e-6,  # 0.45 tx/day target, eta = 0.550, E[cluster] ~ 2.222
        alpha=1.540e-3,
        beta=2.80e-3,
        beta_0_floor=0.035,
        phi_max=3.60,
    ),
    "IN_C2_FIXED_INCOME_SENIOR": HawkesParameters(
        mu_0=2.820e-6,  # 0.40 tx/day target, eta = 0.500, E[cluster] ~ 2.000
        alpha=1.250e-3,
        beta=2.50e-3,
        beta_0_floor=0.010,
        phi_max=4.20,
    ),
    "IN_C3_YOUNG_ADULT_STUDENT": HawkesParameters(
        mu_0=3.208e-6,  # 0.65 tx/day target, eta = 0.650, E[cluster] ~ 2.857
        alpha=1.950e-3,
        beta=3.00e-3,
        beta_0_floor=0.045,
        phi_max=3.50,
    ),
    "IN_C4_SUBURBAN_FAMILY": HawkesParameters(
        mu_0=3.610e-6,  # 0.80 tx/day target, eta = 0.680, E[cluster] ~ 3.125
        alpha=2.108e-3,
        beta=3.10e-3,
        beta_0_floor=0.015,
        phi_max=3.95,
    ),
    "IN_C5_TECH_PROFESSIONAL": HawkesParameters(
        mu_0=4.865e-6,  # 1.15 tx/day target, eta = 0.700, E[cluster] ~ 3.333
        alpha=1.960e-3,
        beta=2.80e-3,
        beta_0_floor=0.030,
        phi_max=3.75,
    ),
    "IN_C5_URBAN_TECH_PROFESSIONAL": HawkesParameters(
        mu_0=4.865e-6,  # 1.15 tx/day target, eta = 0.700, E[cluster] ~ 3.333
        alpha=1.960e-3,
        beta=2.80e-3,
        beta_0_floor=0.030,
        phi_max=3.75,
    ),
    "IN_C6_COMMERCIAL_SMALL_BIZ": HawkesParameters(
        mu_0=4.315e-6,  # 0.90 tx/day target, eta = 0.660, E[cluster] ~ 2.941
        alpha=1.980e-3,
        beta=3.00e-3,
        beta_0_floor=0.020,
        phi_max=4.10,
    ),
    "IN_C6_SMALL_BUSINESS_OWNER": HawkesParameters(
        mu_0=4.315e-6,  # 0.90 tx/day target, eta = 0.660, E[cluster] ~ 2.941
        alpha=1.980e-3,
        beta=3.00e-3,
        beta_0_floor=0.020,
        phi_max=4.10,
    ),
    "IN_C7_LUXURY_AFFLUENT": HawkesParameters(
        mu_0=4.738e-6,  # 1.20 tx/day target, eta = 0.720, E[cluster] ~ 3.571
        alpha=2.304e-3,
        beta=3.20e-3,
        beta_0_floor=0.030,
        phi_max=3.80,
    ),
    # Direct persona aliases for Indian simulation context
    "C1_HOURLY_GIG_WORKER": HawkesParameters(
        mu_0=2.855e-6,
        alpha=1.540e-3,
        beta=2.80e-3,
        beta_0_floor=0.035,
        phi_max=3.60,
    ),
    "C2_FIXED_INCOME_SENIOR": HawkesParameters(
        mu_0=2.820e-6,
        alpha=1.250e-3,
        beta=2.50e-3,
        beta_0_floor=0.010,
        phi_max=4.20,
    ),
    "C3_YOUNG_ADULT_STUDENT": HawkesParameters(
        mu_0=3.208e-6,
        alpha=1.950e-3,
        beta=3.00e-3,
        beta_0_floor=0.045,
        phi_max=3.50,
    ),
    "C4_SUBURBAN_FAMILY": HawkesParameters(
        mu_0=3.610e-6,
        alpha=2.108e-3,
        beta=3.10e-3,
        beta_0_floor=0.015,
        phi_max=3.95,
    ),
    "C5_TECH_PROFESSIONAL": HawkesParameters(
        mu_0=4.865e-6,
        alpha=1.960e-3,
        beta=2.80e-3,
        beta_0_floor=0.030,
        phi_max=3.75,
    ),
    "C5_URBAN_TECH_PROFESSIONAL": HawkesParameters(
        mu_0=4.865e-6,
        alpha=1.960e-3,
        beta=2.80e-3,
        beta_0_floor=0.030,
        phi_max=3.75,
    ),
    "C6_COMMERCIAL_SMALL_BIZ": HawkesParameters(
        mu_0=4.315e-6,
        alpha=1.980e-3,
        beta=3.00e-3,
        beta_0_floor=0.020,
        phi_max=4.10,
    ),
    "C6_SMALL_BUSINESS_OWNER": HawkesParameters(
        mu_0=4.315e-6,
        alpha=1.980e-3,
        beta=3.00e-3,
        beta_0_floor=0.020,
        phi_max=4.10,
    ),
    "C7_LUXURY_AFFLUENT": HawkesParameters(
        mu_0=4.738e-6,
        alpha=2.304e-3,
        beta=3.20e-3,
        beta_0_floor=0.030,
        phi_max=3.80,
    ),
}


# Adversarial botnet attack pacing profiles grounded in spec/04_adversarial_playbooks.yaml
ADVERSARY_HAWKES_PROFILES: Dict[str, HawkesParameters] = {
    "ADV_MICRO_AUTH_PROBE": HawkesParameters(
        mu_0=5.0e-4,
        alpha=1.50,      # Explosive sub-second surge
        beta=2.00,       # Sharp recovery (t_half ~ 0.35s), eta = 0.75
        beta_0_floor=0.45,
        phi_max=1.0,
    ),
    "ADV_CARDING_MICRO_PROBE": HawkesParameters(
        mu_0=5.0e-4,
        alpha=1.20,
        beta=1.80,       # eta = 0.667
        beta_0_floor=0.45,
        phi_max=1.0,
    ),
    "ADV_DISTRIBUTED_BIN_ENUMERATION": HawkesParameters(
        mu_0=1.0e-3,
        alpha=0.85,
        beta=1.25,       # eta = 0.68
        beta_0_floor=0.50,
        phi_max=1.0,
    ),
    "ADV_APPLE_PAY_YELLOW_PATH": HawkesParameters(
        mu_0=1.0e-4,
        alpha=0.015,     # Physical NFC tap burst
        beta=0.022,      # eta = 0.682, t_half ~ 31.5s
        beta_0_floor=0.10,
        phi_max=1.2,
    ),
    "ADV_NOCTURNAL_BURST": HawkesParameters(
        mu_0=2.0e-5,
        alpha=4.5e-3,
        beta=6.0e-3,      # eta = 0.750, concentrated nocturnal attack
        beta_0_floor=0.85, # Inverted diurnal: attacks occur overnight
        phi_max=2.5,
    ),
}


class RecursiveCircadianHawkesEngine:
    """High-performance vectorized O(1) Recursive Ogata Thinning Engine."""

    def __init__(self, default_params: Optional[HawkesParameters] = None, seed: int = 42):
        self.params = default_params or HawkesParameters()
        self.rng = np.random.default_rng(seed)
        # Vectorized candidate drawing buffers for high throughput
        self._block_size = 4096
        self._exp_buffer = np.empty(0, dtype=np.float64)
        self._u_buffer = np.empty(0, dtype=np.float64)
        self._buf_idx = self._block_size

    def _refill_buffers(self) -> None:
        """Refills vectorized random number arrays in large contiguous memory blocks."""
        self._exp_buffer = self.rng.exponential(scale=1.0, size=self._block_size)
        self._u_buffer = self.rng.uniform(low=0.0, high=1.0, size=self._block_size)
        self._buf_idx = 0

    @staticmethod
    def evaluate_circadian_phi(
        t_sec: float,
        beta_0: float = 0.025,
        phi_max: float = 3.85,
        phase_offset_sec: float = 0.0,
    ) -> float:
        """Evaluates 4-component periodic von Mises mixture density at local hour of day.

        `phase_offset_sec` is the cardholder's own local-clock offset (time zone plus
        chronotype): each person's day runs on their own clock, and the portfolio stream
        the corpus reference measures is the sum of those clocks, not one shared clock.

        Centers:
        - Commute: 08:15 (8.25h), kappa=4.2, weight=0.22
        - Lunch: 12:36 (12.60h), kappa=4.8, weight=0.32
        - Evening Errands: 18:30 (18.50h), kappa=3.6, weight=0.34
        - Night: 21:15 (21.25h), kappa=3.0, weight=0.12
        """
        hour = ((t_sec + phase_offset_sec) / 3600.0) % 24.0
        two_pi_24 = 2.0 * math.pi / 24.0

        # Periodic cos terms centered at each diurnal peak
        c1 = math.cos(two_pi_24 * (hour - 8.25))
        c2 = math.cos(two_pi_24 * (hour - 12.60))
        c3 = math.cos(two_pi_24 * (hour - 18.50))
        c4 = math.cos(two_pi_24 * (hour - 21.25))

        # Relative density mixture normalized in [0, 1]
        mix = (
            0.22 * math.exp(4.2 * (c1 - 1.0)) +
            0.32 * math.exp(4.8 * (c2 - 1.0)) +
            0.34 * math.exp(3.6 * (c3 - 1.0)) +
            0.12 * math.exp(3.0 * (c4 - 1.0))
        )
        # Scale between non-zero nocturnal floor and diurnal peak
        return beta_0 + (phi_max - beta_0) * mix

    def compute_intensity(
        self,
        t_sec: float,
        R_current: float,
        last_tx_time_sec: float,
        params: Optional[HawkesParameters] = None,
    ) -> float:
        """Calculates exact Hawkes conditional intensity lambda*(t | H_t) in O(1) constant time.
        
        lambda*(t) = mu(t) + alpha * R_k * exp(-beta * (t - t_last))
        """
        p = params or self.params
        phi = self.evaluate_circadian_phi(t_sec, p.beta_0_floor, p.phi_max, p.phase_offset_sec)
        mu_t = p.mu_0 * phi

        if R_current <= 0.0 or last_tx_time_sec < 0.0:
            return mu_t

        dt = max(0.0, t_sec - last_tx_time_sec)
        hawkes_excitation = p.alpha * R_current * math.exp(-p.beta * dt)
        return mu_t + hawkes_excitation

    def sample_next_arrival(
        self,
        current_time_sec: float,
        R_current: float,
        last_tx_time_sec: float,
        params: Optional[HawkesParameters] = None,
    ) -> Tuple[float, float]:
        """Samples next arrival timestamp using O(1) recursive Ogata thinning.
        
        Args:
            current_time_sec: Current simulation clock in seconds.
            R_current: Current recursive Hawkes memory accumulator.
            last_tx_time_sec: Timestamp of the most recent event.
            params: Optional persona-specific Hawkes parameters.

        Returns:
            Tuple of (next_tx_time_sec, R_new)
        """
        p = params or self.params
        t = current_time_sec
        R = R_current
        t_last = last_tx_time_sec if last_tx_time_sec >= 0.0 else current_time_sec

        while True:
            if self._buf_idx >= self._block_size:
                self._refill_buffers()

            # 1. Compute exact current Hawkes excitation at t+
            dt_from_last = max(0.0, t - t_last)
            decay = math.exp(-p.beta * dt_from_last) if dt_from_last > 0.0 else 1.0
            lambda_hawkes_now = p.alpha * R * decay

            # 2. Local supremum upper bound: peak circadian baseline + current excitation
            lambda_bar = (p.mu_0 * p.phi_max) + lambda_hawkes_now + 1e-9

            # 3. Draw candidate inter-arrival time
            dt_cand = self._exp_buffer[self._buf_idx] / lambda_bar
            u_check = self._u_buffer[self._buf_idx]
            self._buf_idx += 1

            t_cand = t + dt_cand

            # 4. Evaluate exact intensity at proposed candidate time t_cand
            dt_cand_last = max(0.0, t_cand - t_last)
            decay_cand = math.exp(-p.beta * dt_cand_last) if dt_cand_last > 0.0 else 1.0
            lambda_hawkes_cand = p.alpha * R * decay_cand
            phi_cand = self.evaluate_circadian_phi(t_cand, p.beta_0_floor, p.phi_max, p.phase_offset_sec)
            lambda_exact_cand = (p.mu_0 * phi_cand) + lambda_hawkes_cand

            # 5. Thinning acceptance test
            p_accept = min(1.0, lambda_exact_cand / lambda_bar)
            if u_check <= p_accept:
                # Candidate accepted! Update recursive accumulator:
                # R_new = 1.0 + R_old * exp(-beta * (t_cand - t_last))
                R_new = 1.0 + R * math.exp(-p.beta * (t_cand - t_last))
                return t_cand, R_new

            # Candidate rejected: advance clock to candidate time and continue
            t = t_cand

    def update_feedback(
        self,
        R_current: float,
        last_tx_time_sec: float,
        event_time_sec: float,
        iso_response_code: str,
        params: Optional[HawkesParameters] = None,
    ) -> float:
        """Closed-loop feedback updating Hawkes memory based on authorization outcomes.
        
        - ISO 00 (Approved): Strengthens excitement for legitimate shopping sprees.
        - ISO 51 (Insufficient Funds): Decays memory (throttles spend velocity).
        - ISO 05/59 (Suspected Fraud): Freezes account (clears excitation to 0.0).
        """
        p = params or self.params
        dt = max(0.0, event_time_sec - last_tx_time_sec) if last_tx_time_sec >= 0.0 else 0.0
        decayed_R = R_current * math.exp(-p.beta * dt)

        if iso_response_code == "00":
            # Successful authorization reinforces shopping cluster
            return 1.0 + decayed_R
        elif iso_response_code == "51":
            # Insufficient funds: suppress excitation by 50%
            return decayed_R * 0.50
        elif iso_response_code in ("05", "59", "65"):
            # Suspected fraud or velocity block: complete freeze
            return 0.0
        else:
            return decayed_R

    def simulate_stream(
        self,
        n_events: int,
        t_start_sec: float = 0.0,
        params: Optional[HawkesParameters] = None,
    ) -> List[Dict[str, float]]:
        """Generates a synthetic realization of n_events using recursive Ogata thinning.
        
        Returns a list of event dictionaries containing timestamps, inter-arrival times,
        and recursive state snapshots.
        """
        p = params or self.params
        events: List[Dict[str, float]] = []
        t_curr = t_start_sec
        t_last = -1.0
        R = 0.0

        for _ in range(n_events):
            t_next, R_next = self.sample_next_arrival(t_curr, R, t_last, p)
            dt = t_next - t_curr
            events.append({
                "timestamp_sec": t_next,
                "delta_t_sec": dt,
                "hour_of_day": (t_next / 3600.0) % 24.0,
                "hawkes_R": R_next,
            })
            t_curr = t_next
            t_last = t_next
            R = R_next

        return events
