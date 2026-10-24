"""Discrete-Event Multi-Agent Simulation Engine for Prism Generator.

Orchestrates:
1. 64-bit integer microsecond priority queue (heapq) guaranteeing strict global chronological monotonicity.
2. Per-cardholder physical lock-ahead (card_avail_time_us) enforcing realistic travel delays between physical transactions.
3. Closed-loop multi-agent feedback between Cardholder, AdaptiveFraudster, and BankDecisionEngine.
4. Institutional banking telemetry generation: ISO 8583 syntax, Gateway risk, Clearing & Settlement,
   Post-Authorization Dispute Lifecycles (Visa VCR, CE 3.0, RBI limited liability tiers, 1930 liens).
5. Dual-region execution: US Metro (USD, dual-message rails) and India Metro (INR, RBI AFA, RuPay, CoFT).
6. Bit-for-bit reproducible deterministic batch and streaming modes.
"""

from __future__ import annotations

import collections
import heapq
import itertools
import math
import zlib
from datetime import datetime, timezone
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

from .agents import (
    AdaptiveFraudsterAgent,
    CardholderProfile,
    CardholderState,
    FraudScenario,
    FraudsterState,
    ISO8583Response,
)
from .causal_scm import StructuralCausalEngine
from .hawkes import (
    HawkesParameters,
    INDIAN_PERSONA_HAWKES_PROFILES,
    PERSONA_HAWKES_PROFILES,
    RecursiveCircadianHawkesEngine,
)
from .invariants import verify_transaction_invariants
from .ledger import InterbankClearingLedger, StreamingLedger
from .syndicates import SyndicateRegistry
from .world import MCC_TAXONOMY, WorldEnvironment
from .rails import (
    CandidateTransactionIntent,
    RailVerifierSwitch,
)
from .spec_loader import load_all_specs

# spec/07 section 20: the ASN type of a proxied address is the network the proxy sits on
PROXY_NETWORK_ASN = {"DATACENTER_ROTATING": "datacenter", "MOBILE_4G_5G": "mobile", "RESIDENTIAL_STICKY": "residential"}


_NONE_TYPE = type(None)


def _plain(value: Any) -> Any:
    """A record's dictionary fields as plain Python values with floats rounded to six
    decimals, so the master export is the same bytes on every machine (spec/07 section 20).

    The exact-type shortcuts come first because they cover almost every leaf in a record;
    each still returns exactly what the original isinstance chain returned, and anything
    unusual (numpy scalars, dict/list subclasses) falls through to that chain.
    """
    t = type(value)
    if t is float:
        return round(value, 6) + 0.0
    if t is str or t is bool or t is int or t is _NONE_TYPE:
        return value
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        # plus 0.0 turns a negative zero, whose sign depends on the processor, into zero
        return round(float(value), 6) + 0.0
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    return value


# Event Type Enums
EVT_CARDHOLDER_TX = 1
EVT_FRAUD_ATTACK = 2
EVT_ALERT_FREEZE = 3
EVT_SETTLEMENT = 4
EVT_CLEARING = 5
EVT_BILLING_CYCLE_CLOSE = 6
EVT_STATEMENT_PAYMENT = 7
EVT_PAYROLL_DEPOSIT = 8

# Recurring-cadence constants. These are the ONLY places the periods are
# written down: the seeding loop and the reschedule handler both read them, so
# the two cannot drift apart. A mismatch between the seeded period and the
# rescheduled period is what previously produced a triangular pile-up of events.
#
# Both are grounded in spec/05_india_payment_rails.yaml
# `indian_consumer_behavioral_quirks.payday_liquidity_cycle`:
#   private_sector_salary_credit_days: [28, 29, 30, 31]  -> monthly, month-end
#   government_salary_credit_days:     [1, 2, 3]         -> monthly, month-start
# i.e. Indian salary credit is MONTHLY. A single global 14-day (biweekly) period
# is a US convention (26 credits/year) applied to Indian cardholders, which
# credited ~2.17x the intended monthly income and is a large part of why balances
# ran away. Hence the period is per-currency, not global.
PAYROLL_PERIOD_DAYS_US = 14.0   # US: biweekly direct deposit, 26 credits/year
PAYROLL_PERIOD_DAYS_IN = 30.0   # IN: monthly salary credit (spec/05, above)


def payroll_period_days(currency: str) -> float:
    """Days between salary credits for a card in this currency.

    Grounded per region rather than globally: Indian salary credit is monthly
    (spec/05 payday_liquidity_cycle), US direct deposit is biweekly.
    """
    return PAYROLL_PERIOD_DAYS_US if currency == "USD" else PAYROLL_PERIOD_DAYS_IN


# Statement cycle. Monthly, matching the 30-day close cadence the handler below
# already reschedules on (spec/03_payment_rail_gaps.yaml chargeback windows and
# spec/16_operational_supervision.yaml both assume a monthly statement).
BILLING_CYCLE_DAYS = 30.0

# Legacy flat payroll credit, in major units, used only as the fallback for
# cards that carry no income (annual_income == 0): bootstrap/test fixtures and
# backbone-replay cards. Live cards are grounded in cohort income -- see
# _initialize_cardholders (spec/02 annual_income_usd / annual_income_inr, with
# the FY2024-25 new-regime tax applied to Indian gross) and its consumption in
# the EVT_PAYROLL_DEPOSIT handler. Kept so that fallback stays one line.
PAYROLL_DEPOSIT_USD = 1500.0
PAYROLL_DEPOSIT_INR = 25000.0

# India FY2024-25 new tax regime (post 2024 Budget): standard deduction Rs 75,000,
# slabs 0-3L nil, 3-7L 5%, 7-10L 10%, 10-12L 15%, 12-15L 20%, >15L 30%.
# Used to convert a gross annual income into the net take-home that is actually
# credited to a salary account. Without this the deposit would overstate
# disposable cash and re-create the balance runaway the income grounding fixes.
def in_new_regime_tax(gross_annual_inr: float) -> float:
    taxable = max(0.0, gross_annual_inr - 75000.0)
    slabs = [(300000.0, 0.0), (400000.0, 0.05), (300000.0, 0.10),
             (200000.0, 0.15), (300000.0, 0.20), (float("inf"), 0.30)]
    tax = 0.0
    remaining = taxable
    for width, rate in slabs:
        if remaining <= 0:
            break
        chunk = min(remaining, width)
        tax += chunk * rate
        remaining -= chunk
    # 4% health & education cess
    tax *= 1.04
    return max(0.0, gross_annual_inr - tax)

# Calibrated Indian card product spend marginals (LogNormal parameters in INR nominal)
# Grounded dynamically in canonical spec/05_india_payment_rails.yaml for backwards compatibility
INDIAN_PRODUCT_SPEND_MARGINALS: Dict[str, Tuple[float, float]] = {
    p_id: (p.spend_mean_log, p.spend_sigma_log)
    for p_id, p in load_all_specs().indian_products.items()
}


def round_to_issuer_step(
    limit: float,
    steps: List[Tuple[Optional[float], float]],
    low: float = 0.0,
    high: Optional[float] = None,
) -> float:
    """A credit line on the issuer's assignment grid (spec/01 and spec/05 credit_limit_assignment):
    rounded to the step of its tier and kept inside the product's range."""
    if not steps:
        return float(limit)
    step = steps[-1][1]
    for up_to, tier_step in steps:
        if up_to is None or limit <= up_to:
            step = tier_step
            break
    rounded = max(step, round(limit / step) * step)
    floor = max(step, math.ceil((low - 1e-9) / step) * step) if low > 0.0 else step
    ceiling = math.floor((high + 1e-9) / step) * step if high else None
    if ceiling is not None and ceiling >= floor:
        rounded = min(max(rounded, floor), ceiling)
    return float(rounded)


class CalendarTopologyDynamics:
    """Calculates instantaneous P2P vs P2M routing probabilities across 365 days.

    Grounded in Federal Reserve DCPC 2024 and RBI Annual Report 2023-24 (Table IX.1).
    Models intra-month salary payroll surges and retail megasale denominator expansions.
    """

    def __init__(self, region: str = "US", specs: Optional[Any] = None):
        self.region = region.upper()
        self.specs = specs or load_all_specs()
        seasonality = getattr(self.specs, "seasonality", None)
        # Only entries whose value is a mapping are regions; `hypothesis` and
        # `replace_with` are section-level annotations and are not regions.
        all_cfg = seasonality.topology_seasonality if seasonality else {}
        supported = sorted(k for k, v in all_cfg.items() if isinstance(v, dict))
        if self.region not in supported:
            raise ValueError(
                f"spec/13 topology_seasonality has no entry for region {self.region!r}; "
                f"configured regions are {supported}. Refusing to borrow another "
                f"region's coefficients."
            )
        topo_cfg = all_cfg[self.region]
        if not topo_cfg:
            raise ValueError(
                f"spec/13 topology_seasonality.{self.region} is empty; the calendar "
                f"topology model needs beta_0, beta_payday, beta_sale, kernel_sigma, "
                f"payday_anchors and weekend_adjustment."
            )
        missing = [k for k in ("beta_0", "beta_payday", "beta_sale", "kernel_sigma",
                               "payday_anchors", "weekend_adjustment") if k not in topo_cfg]
        if missing:
            raise KeyError(
                f"spec/13 topology_seasonality.{self.region} is missing {missing}"
            )
        self.beta_0 = float(topo_cfg["beta_0"])
        self.beta_payday = float(topo_cfg["beta_payday"])
        self.beta_sale = float(topo_cfg["beta_sale"])
        self.kernel_sigma = float(topo_cfg["kernel_sigma"])
        self.payday_anchor_days = list(topo_cfg["payday_anchors"])
        self.weekend_adjustment = float(topo_cfg["weekend_adjustment"])

    def _circular_day_distance(self, day: int, target: int) -> float:
        diff = abs(day - target)
        return min(diff, 31 - diff)

    def compute_payday_kernel(self, day_of_month: int) -> float:
        """Circular Gaussian kernel evaluating proximity to payroll dates."""
        weights = []
        for anchor in self.payday_anchor_days:
            dist = self._circular_day_distance(day_of_month, anchor)
            weights.append(math.exp(-(dist ** 2) / (2.0 * (self.kernel_sigma ** 2))))
        return max(weights) if weights else 0.0

    def evaluate_p2p_probability(
        self,
        t_sec: float,
        is_holiday_sale: bool = False,
    ) -> float:
        """Evaluates exact closed-form logistic probability: P(is_p2p | t, day, sale, region)."""
        dt = datetime.fromtimestamp(t_sec, tz=timezone.utc)
        k_payday = self.compute_payday_kernel(dt.day)
        i_sale = 1.0 if is_holiday_sale else 0.0
        dow = dt.weekday()
        if self.region == "US":
            beta_dow = self.weekend_adjustment if dow in (4, 5, 6) else 0.0
        else:
            beta_dow = self.weekend_adjustment if dow in (5, 6) else 0.0

        z = self.beta_0 + (self.beta_payday * k_payday) - (self.beta_sale * i_sale) + beta_dow
        return 1.0 / (1.0 + math.exp(-z))


class DiscreteEventEngine:

    """High-performance discrete-event multi-agent payment fraud simulation engine."""

    def __init__(
        self,
        n_cards: int = 1000,
        n_merchants: int = 150,
        region: str = "US",
        center_lat: Optional[float] = None,
        center_lon: Optional[float] = None,
        radius_km: float = 35.0,
        adversary_mimicry: float = 0.55,
        adversary_mode: str = "playbook",
        seed: int = 42,
        annual_seasonality: bool = False,
        p2p_enabled: bool = False,
        id_prefix: Optional[str] = None,
        world: Optional[WorldEnvironment] = None,
        syndicates: Optional[SyndicateRegistry] = None,
    ):
        self.region = region.upper()
        self.seed = seed
        self.adversary_mimicry = adversary_mimicry
        self.adversary_mode = str(adversary_mode).lower()
        self.annual_seasonality = annual_seasonality
        self.p2p_enabled = p2p_enabled
        self.id_prefix = id_prefix
        self.rng = np.random.default_rng(seed)
        if world is not None:
            self.world = world
        else:
            self.world = WorldEnvironment(
                n_merchants=n_merchants,
                region=self.region,
                center_lat=center_lat,
                center_lon=center_lon,
                radius_km=radius_km,
                seed=seed,
            )
        self.causal_engine = StructuralCausalEngine(base_prevalence=0.0020)
        # Batch generation skips the per-record flow round-trip self-check (see
        # HeuristicBankScorer.verify_flow_consistency); the attributions and the
        # counterfactual twins themselves are still computed for every record.
        self.causal_engine.verify_flow_consistency = False
        self.fraudster = AdaptiveFraudsterAgent(self.rng, adversary_mimicry=adversary_mimicry)
        if syndicates is not None:
            self.syndicate_registry = syndicates
        else:
            self.syndicate_registry = SyndicateRegistry(region=self.region, seed=seed)
        self.ledger = StreamingLedger(seed=seed)
        self.rail_switch = RailVerifierSwitch(region=self.region, seed=seed)
        self.double_entry_ledger = self.ledger.double_entry
        self.interbank_clearing = InterbankClearingLedger(region=self.region)

        self.specs = load_all_specs()
        self.topology_dynamics = CalendarTopologyDynamics(region=self.region, specs=self.specs)
        self.cards: List[CardholderProfile] = []

        self.card_map: Dict[str, CardholderProfile] = {}
        self._initialize_cardholders(n_cards)

        # Diurnal 24-hour circular probability density
        self.diurnal_hourly_probs = self._compute_circular_diurnal_distribution()

        # Recursive Circadian Hawkes MTPP Engine
        self.hawkes_engine = RecursiveCircadianHawkesEngine(seed=seed)
        self.card_hawkes_R: Dict[str, float] = {c.card_id: 0.0 for c in self.cards}
        self.card_last_hawkes_time: Dict[str, float] = {c.card_id: -1.0 for c in self.cards}

        # Discrete-event priority queue state
        self.event_queue: List[Tuple[int, int, int, str, int, Dict[str, Any]]] = []
        self.seq_counter = itertools.count()
        self.card_avail_time_us: Dict[str, int] = {c.card_id: 0 for c in self.cards}
        self.card_generation: Dict[str, int] = {c.card_id: 0 for c in self.cards}

    def _persona_volume_scale_mean(self) -> float:
        """Card-weighted mean of each persona's volume scale, computed once and cached.

        `generate_batch` budgets `target_n / n_cards` transactions per card, then scales each
        card by its persona's relative volume. For that budget to be conserved, the scales must
        average 1.0 over the population that is actually being simulated -- otherwise the
        aggregate rate is the budget times the mean scale and `time_span_days` is not honoured.

        Divisors here are measured, not assumed: 55.0 and 0.684 were hard-coded constants that
        happened to equal one persona's volume (C1_HOURLY_GIG_WORKER) and a rounded macro mean.
        Against spec/02's own population_weight the US mean is 65.50, and against the realised
        card population 64.86, so every US batch ran ~19% hot; the IN branch ran ~5.8% hot.
        """
        cache = getattr(self, "_persona_scale_mean_cache", None)
        if cache is not None:
            return cache

        scales: List[float] = []
        for c in self.cards:
            if c.region == "IN":
                p = self._get_card_hawkes_params(c, mean_inter_arrival_sec=None)
                eta_p = min(0.95, p.alpha / p.beta)
                scales.append((86400.0 * p.mu_0 * 0.8208) / max(0.05, (1.0 - eta_p)))
            else:
                cohort_spec = self.specs.cohorts.get(c.cohort_id)
                scales.append(cohort_spec.monthly_tx_volume_mean if cohort_spec else 55.0)
        mean_scale = float(np.mean(scales)) if scales else 1.0
        self._persona_scale_mean_cache = mean_scale if mean_scale > 0 else 1.0
        return self._persona_scale_mean_cache

    def _get_card_hawkes_params(self, card: CardholderProfile, mean_inter_arrival_sec: Optional[float] = None) -> HawkesParameters:
        """Constructs calibrated persona HawkesParameters, optionally scaled to simulation time span."""
        cohort_spec = self.specs.cohorts.get(card.cohort_id)
        if card.region == "IN":
            in_key = f"IN_{card.cohort_id}" if not card.cohort_id.startswith("IN_") else card.cohort_id
            if in_key in INDIAN_PERSONA_HAWKES_PROFILES:
                profile = INDIAN_PERSONA_HAWKES_PROFILES[in_key]
            elif card.cohort_id in INDIAN_PERSONA_HAWKES_PROFILES:
                profile = INDIAN_PERSONA_HAWKES_PROFILES[card.cohort_id]
            else:
                profile = INDIAN_PERSONA_HAWKES_PROFILES.get("IN_C4_SUBURBAN_FAMILY", HawkesParameters(
                    mu_0=3.610e-6,
                    alpha=2.108e-3,
                    beta=3.10e-3,
                    beta_0_floor=0.015,
                    phi_max=3.95,
                ))
            base_mu = profile.mu_0
            alpha = profile.alpha
            beta = profile.beta
            beta_0 = profile.beta_0_floor
        elif cohort_spec and cohort_spec.hawkes_dynamics:
            hd = cohort_spec.hawkes_dynamics
            base_mu = float(hd.get("mu_0_hz", 1.0e-5))
            alpha = float(hd.get("alpha_excitation_hz", 1.85e-3))
            beta = float(hd.get("beta_decay_hz", 2.78e-3))
            beta_0 = float(hd.get("nocturnal_floor_beta_0", 0.025))
        elif card.cohort_id in PERSONA_HAWKES_PROFILES:
            profile = PERSONA_HAWKES_PROFILES[card.cohort_id]
            base_mu = profile.mu_0
            alpha = profile.alpha
            beta = profile.beta
            beta_0 = profile.beta_0_floor
        else:
            base_mu = 1.85e-5
            alpha = 1.80e-3
            beta = 2.70e-3
            beta_0 = 0.025

        if mean_inter_arrival_sec is not None and mean_inter_arrival_sec > 0:
            if card.region == "IN":
                # Grounded in RBI PSI: compute persona's calibrated expected daily frequency relative to macro mean 0.684 tx/day
                # The divisor is the population mean of that same quantity (measured, cached), so the
                # persona scales average 1.0 and the pacing budget is conserved -- see _persona_volume_scale_mean.
                eta_p = min(0.95, alpha / beta)
                persona_expected_daily = (86400.0 * base_mu * 0.8208) / max(0.05, (1.0 - eta_p))
                persona_vol_scale = persona_expected_daily / self._persona_volume_scale_mean()
            else:
                # Scaled by the population mean, not by a fixed constant: see _persona_volume_scale_mean.
                vol_mean = cohort_spec.monthly_tx_volume_mean if cohort_spec else 55.0
                persona_vol_scale = vol_mean / self._persona_volume_scale_mean()
            eta = min(0.95, alpha / beta)
            target_lambda = (1.0 / mean_inter_arrival_sec) * persona_vol_scale
            mu_0 = (target_lambda * (1.0 - eta)) / 0.8208
        else:
            mu_0 = base_mu

        return HawkesParameters(
            mu_0=mu_0,
            alpha=alpha,
            beta=beta,
            beta_0_floor=beta_0,
            phi_max=3.85,
            phase_offset_sec=float(getattr(card, "phase_offset_sec", 0.0)),
        )

    def _compute_circular_diurnal_distribution(self) -> np.ndarray:
        """Computes continuous 24-hour periodic diurnal arrival intensity from canonical circadian spec."""
        if not (hasattr(self, "specs") and self.specs and self.specs.circadian):
            raise RuntimeError(
                "DiscreteEventEngine requires valid circadian specification in self.specs.circadian. "
                "Silent trigonometric fallbacks violate AGENTS.md Anti-Astronaut Grounding."
            )
        hours = np.arange(24, dtype=np.float64)
        intensity = np.array([self.specs.circadian.evaluate_density(h, is_weekend=False) for h in hours])
        return intensity / intensity.sum()

    def _compute_macro_rate_multiplier(self, t_sec: float, active_regime: Optional[str] = None) -> float:
        """Computes arrival rate multiplier based on calendar day-of-month and macro regimes."""
        dt = datetime.fromtimestamp(t_sec, tz=timezone.utc)
        multiplier = 1.0

        if not self.annual_seasonality:
            # Backwards-compatible legacy logic
            if dt.day in (1, 2, 15, 16):
                multiplier *= 1.35
            if active_regime == "HOLIDAY_SURGE" or (self.region == "US" and dt.month == 11 and 22 <= dt.day <= 30):
                multiplier *= 2.50
            elif active_regime == "DHANTERAS_DIWALI" or (self.region == "IN" and ((dt.month == 10 and dt.day >= 25) or (dt.month == 11 and dt.day <= 5))):
                multiplier *= 1.80
            return multiplier

        # Full 365-day annual seasonality from spec/13
        seasonality = self.specs.seasonality
        if seasonality:
            # 1. Day of Week multiplier (Mon 0.88x to Sat 1.34x)
            dow = dt.weekday()
            multiplier *= seasonality.day_of_week_multipliers.get(dow, 1.0)

            # 2. Regional Payday Surges
            payday_cfg = seasonality.payday_cycles.get(self.region, {})
            paydays = payday_cfg.get("primary_days_of_month", [1, 15])
            if dt.day in paydays:
                multiplier *= float(payday_cfg.get("surge_multiplier", 1.35))

            # 3. Calendar Macro Shocks (Holidays, Black Friday, Diwali, etc.)
            day_of_year = dt.timetuple().tm_yday
            shocks = seasonality.calendar_macro_shocks.get(self.region, [])
            for shock in shocks:
                if "day_of_year" in shock and shock["day_of_year"] == day_of_year:
                    multiplier *= float(shock.get("volume_multiplier", 1.0))
                    break
                elif "start_day" in shock and "end_day" in shock:
                    if shock["start_day"] <= day_of_year <= shock["end_day"]:
                        multiplier *= float(shock.get("volume_multiplier", 1.0))
                        break

            # 4. Explicit active_regime overrides
            if active_regime == "HOLIDAY_SURGE":
                multiplier *= 2.50
            elif active_regime == "DHANTERAS_DIWALI":
                multiplier *= 2.65

        return multiplier

    def _compute_dynamic_p2p_probability(self, t_sec: float, is_holiday_sale: bool = False) -> float:
        """Computes instantaneous P2P vs P2M routing probability across 365 days.

        Grounded in Federal Reserve DCPC 2024 and RBI Annual Report 2023-24 (Table IX.1).
        Models salary direct deposit liquidity surges and retail megasale compression.
        """
        return self.topology_dynamics.evaluate_p2p_probability(t_sec, is_holiday_sale=is_holiday_sale)

    def _us_timezone_offset_hours(self, card_index: int) -> float:
        """US local-clock offset for a cardholder, in hours relative to the Eastern zone.

        The portfolio's arrival stream is the sum of individual local clocks, so a card
        whose zone runs behind the reference zone transacts later in absolute time.
        Population shares by zone are approximately ET 47%, CT 29%, MT 7%, PT 17%
        (Census state populations mapped to zones; AK/HI folded into PT so the four
        weights sum to 1.0). The draw uses a card-derived stream so the engine's main
        sampling stream is not shifted by this addition (2026-10-16, arrival-shape fix).
        """
        rng = np.random.default_rng([int(self.seed), 1977, int(card_index)])
        return float(rng.choice([0.0, -1.0, -2.0, -3.0], p=[0.47, 0.29, 0.07, 0.17]))

    def _initialize_cardholders(self, n_cards: int) -> None:

        """Generates realistic cardholder population calibrated to regional banking rails and YAML specs."""
        specs = load_all_specs()

        angles = self.rng.uniform(0.0, 2.0 * math.pi, size=n_cards)
        radii = self.world.radius_km * np.sqrt(self.rng.uniform(0.0, 1.0, size=n_cards))

        d_lat = (radii * np.sin(angles)) / 111.139
        mean_cos = math.cos(math.radians(self.world.center_lat))
        d_lon = (radii * np.cos(angles)) / (111.139 * mean_cos)

        home_lats = self.world.center_lat + d_lat
        home_lons = self.world.center_lon + d_lon

        cohort_specs = list(specs.cohorts.values())
        cohort_weights = [c.population_weight for c in cohort_specs]
        cohort_weights_norm = np.array(cohort_weights) / sum(cohort_weights)

        for i in range(n_cards):
            card_id = f"CARD_{self.id_prefix}_{i:06d}" if self.id_prefix else f"CARD_{i:06d}"
            cohort_spec = self.rng.choice(cohort_specs, p=cohort_weights_norm)
            cohort_id = cohort_spec.id

            if self.region == "IN":
                currency = "INR"
                if cohort_id == "C7_LUXURY_AFFLUENT":
                    product_id = "IN_PROD_SUPER_PREMIUM_HNI"
                elif cohort_id in ("C6_SMALL_BUSINESS_OWNER", "C6_COMMERCIAL_SMALL_BIZ"):
                    product_id = str(self.rng.choice(["IN_PROD_KISAN_CREDIT_CARD", "IN_PROD_SALARIED_PRIME_REWARDS"], p=[0.70, 0.30]))
                elif cohort_id == "C1_HOURLY_GIG_WORKER":
                    product_id = str(self.rng.choice(["IN_PROD_PMJDY_RUPAY_DEBIT", "IN_PROD_RETAIL_RUPAY_DEBIT", "IN_PROD_ENTRY_FD_BACKED"], p=[0.45, 0.35, 0.20]))
                elif cohort_id == "C2_FIXED_INCOME_SENIOR":
                    product_id = str(self.rng.choice(["IN_PROD_RETAIL_RUPAY_DEBIT", "IN_PROD_ENTRY_FD_BACKED", "IN_PROD_SALARIED_PRIME_REWARDS"], p=[0.40, 0.30, 0.30]))
                else:
                    # Salaried, professional and affluent cohorts overwhelmingly hold an
                    # ordinary retail debit card alongside any credit product; until
                    # 2026-10-12 they were modelled as credit-only, which is why the debit
                    # slice of the engine was PMJDY-only and could not represent the RBI
                    # debit population.
                    product_id = str(self.rng.choice(["IN_PROD_SALARIED_PRIME_REWARDS", "IN_PROD_RETAIL_RUPAY_DEBIT"], p=[0.60, 0.40]))

                prod_spec = specs.indian_products.get(product_id)
                if prod_spec and prod_spec.credit_limit_median_inr > 0:
                    credit_limit = float(self.rng.triangular(
                        prod_spec.credit_limit_min_inr,
                        prod_spec.credit_limit_median_inr,
                        max(prod_spec.credit_limit_min_inr + 1000.0, prod_spec.credit_limit_max_inr)
                    ))
                    credit_limit = round_to_issuer_step(
                        credit_limit, self.specs.credit_limit_steps.get("IN", []),
                        low=prod_spec.credit_limit_min_inr, high=prod_spec.credit_limit_max_inr,
                    )
                else:
                    credit_limit = 50000.0

                cnp_enabled = bool(self.rng.random() < 0.75)
                intl_enabled = bool(self.rng.random() < 0.12)
                contactless_enabled = bool(self.rng.random() < 0.80)
                pan_masked = f"607152******{self.rng.integers(1000, 9999)}"

                if product_id in self.specs.indian_products:
                    in_prod = self.specs.indian_products[product_id]
                    spend_mu_log = in_prod.spend_mean_log
                    spend_sigma_log = in_prod.spend_sigma_log
                elif product_id in INDIAN_PRODUCT_SPEND_MARGINALS:
                    prod_marginal = INDIAN_PRODUCT_SPEND_MARGINALS[product_id]
                    spend_mu_log = prod_marginal[0]
                    spend_sigma_log = prod_marginal[1]
                else:
                    raise KeyError(f"Product ID '{product_id}' not found in canonical Indian card products specification.")
                is_spliced = False
                u_val = 500.0
                xi_val = 0.20
                beta_val = 150.0
                tail_p = 0.02
                if product_id == "IN_PROD_PMJDY_RUPAY_DEBIT":
                    overdraft_limit = 10000.0
                elif "DEBIT" in product_id:
                    overdraft_limit = 5000.0
                else:
                    overdraft_limit = 0.0
            else:
                currency = "USD"
                product_id = str(self.rng.choice(cohort_spec.default_assigned_products))
                prod_spec = specs.products.get(product_id)
                if prod_spec and prod_spec.credit_limit_median_usd > 0:
                    credit_limit = float(self.rng.triangular(
                        prod_spec.credit_limit_min_usd,
                        prod_spec.credit_limit_median_usd,
                        max(prod_spec.credit_limit_min_usd + 100.0, prod_spec.credit_limit_max_usd)
                    ))
                    credit_limit = round_to_issuer_step(
                        credit_limit, self.specs.credit_limit_steps.get("US", []),
                        low=prod_spec.credit_limit_min_usd, high=prod_spec.credit_limit_max_usd,
                    )
                else:
                    credit_limit = 5000.0

                cnp_enabled = True
                intl_enabled = bool(self.rng.random() < 0.35)
                contactless_enabled = True
                pan_masked = f"414720******{self.rng.integers(1000, 9999)}"

                sp_spec = cohort_spec.spend_distribution
                spend_mu_log = sp_spec.mu_log
                spend_sigma_log = sp_spec.sigma_log
                is_spliced = (sp_spec.model == "Spliced_LogNormal_GPD")
                u_val = (sp_spec.threshold_u_cents / 100.0) if sp_spec.threshold_u_cents else 250.0
                xi_val = sp_spec.gpd_xi if sp_spec.gpd_xi is not None else 0.22
                beta_val = (sp_spec.gpd_beta / 100.0) if (sp_spec.gpd_beta is not None and sp_spec.gpd_beta > 500.0) else (sp_spec.gpd_beta or 95.0)
                tail_p = sp_spec.tail_prob if sp_spec.tail_prob is not None else 0.03
                overdraft_limit = 350.0 if "DEBIT" in product_id else 0.0

            if self.region == "IN":
                # Grounded in RBI Payment System Indicators (PSI 2024-2026): 57.6% CNP vs 42.4% CP
                ch_probs = {
                    "CP_POS_CHIP": 0.27,
                    "CP_POS_CONTACTLESS": 0.16,
                    "CNP_WEB": 0.28,
                    "CNP_MOBILE": 0.29,
                }
            else:
                # Grounded in Federal Reserve Payments Study (2022-2025): ~58% CP vs ~42% CNP
                ch_probs = {
                    "CP_POS_CHIP": 0.36,
                    "CP_POS_CONTACTLESS": 0.22,
                    "CNP_WEB": 0.24,
                    "CNP_MOBILE": 0.18,
                }
            total_ch = sum(ch_probs.values())
            ch_probs = {k: v / total_ch for k, v in ch_probs.items()}

            if cohort_id in ("C4_SUBURBAN_FAMILY", "C5_URBAN_TECH_PROFESSIONAL", "C7_LUXURY_AFFLUENT"):
                repay_cohort = str(self.rng.choice(["TRANSACTOR", "REVOLVER"], p=[0.80, 0.20]))
            elif cohort_id == "C1_HOURLY_GIG_WORKER" or product_id == "PROD_SUBPRIME_SECURED":
                repay_cohort = str(self.rng.choice(["REVOLVER", "TRANSACTOR", "DISTRESSED"], p=[0.55, 0.25, 0.20]))
            else:
                repay_cohort = str(self.rng.choice(["TRANSACTOR", "REVOLVER", "DISTRESSED"], p=[0.50, 0.40, 0.10]))

            billing_cycle_day = int(self.rng.choice([1, 5, 10, 15, 20, 25, 28]))
            if "DEBIT" in product_id or any(k in product_id for k in ("PREPAID", "EBT", "HSA")):
                initial_balance = float(self.rng.uniform(0.30, 0.70) * credit_limit)
            else:
                initial_balance = float(self.rng.uniform(0.05, 0.20) * credit_limit)

            # Ground the recurring payroll deposit in cohort income rather than a flat
            # constant: spec/02 annual_income_usd (US) or annual_income_inr (IN, PLFS/
            # Economic Survey 2024-25). For IN, apply the FY2024-25 new-regime tax so
            # the credited amount is net take-home, matching what a salary account
            # actually receives.
            if self.region == "IN":
                _income_range = cohort_spec.annual_income_inr
            else:
                _income_range = cohort_spec.annual_income_usd
            if _income_range:
                _gross = float(self.rng.uniform(float(_income_range[0]), float(_income_range[1])))
                if self.region == "IN":
                    annual_income = in_new_regime_tax(_gross)
                else:
                    annual_income = _gross
            else:
                annual_income = 0.0

            card = CardholderProfile(
                card_id=card_id,
                home_lat=float(home_lats[i]),
                home_lon=float(home_lons[i]),
                work_lat=float(home_lats[i] + self.rng.uniform(-0.04, 0.04)),
                work_lon=float(home_lons[i] + self.rng.uniform(-0.04, 0.04)),
                is_commuter=bool(self.rng.random() < 0.65),
                credit_limit=credit_limit,
                current_balance=initial_balance,
                posted_balance=initial_balance,
                pending_holds=0.0,
                overdraft_limit=overdraft_limit,
                spend_mean_log=spend_mu_log,
                spend_sigma_log=spend_sigma_log,
                preferred_channels=list(ch_probs.keys()),
                circadian_peak_hour=(_peak_hour := float(self.rng.uniform(11.5, 15.5))),
                phase_offset_sec=(
                    # spec/02 population_phase_dispersion is scoped to applies_to_regions:
                    # ["US"] (chronotype range + US time-zone shares). A non-US card still
                    # draws its chronotype peak (the draw order must not change) but carries
                    # no arrival phase; a leaked non-US phase pushed the IN leakage sweep's
                    # eci to 0.9501 over the 0.95 ceiling at playbook/7d, and restoring the
                    # declared scope puts the sweep's top back at eci 0.9419 (playbook/21d).
                    (_peak_hour - 13.5 + self._us_timezone_offset_hours(i)) * 3600.0
                    if self.region == "US"
                    else 0.0
                ),
                last_physical_lat=float(home_lats[i]),
                last_physical_lon=float(home_lons[i]),
                last_physical_time=-1.0,
                last_merchant_id=None,
                product_id=product_id,
                cohort_id=cohort_id,
                region=self.region,
                currency=currency,
                annual_income=annual_income,
                pan_masked=pan_masked,
                domestic_cnp_enabled=cnp_enabled,
                international_enabled=intl_enabled,
                contactless_enabled=contactless_enabled,
                billing_cycle_day=billing_cycle_day,
                repayment_cohort=repay_cohort,
                dominant_mccs=cohort_spec.dominant_mccs,
                channel_probabilities=ch_probs,
                vigilance_weights=cohort_spec.vigilance_weights.as_simplex_array(),
                is_spliced_gpd=is_spliced,
                gpd_threshold_u=u_val,
                gpd_xi=xi_val,
                gpd_beta=beta_val,
                gpd_tail_prob=tail_p,
            )
            self.cards.append(card)
            self.card_map[card_id] = card

        if self.annual_seasonality:
            wallet_tiers = ["PRIMARY", "SECONDARY", "BACKUP"]
            wallet_probs = [0.68, 0.24, 0.08]
            assigned_tiers = self.rng.choice(wallet_tiers, size=len(self.cards), p=wallet_probs)
            for idx, c in enumerate(self.cards):
                setattr(c, "wallet_tier", str(assigned_tiers[idx]))

    def _schedule_event(
        self,
        time_us: int,
        event_type: int,
        card_id: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Pushes event into 64-bit priority queue with stable sequence tie-breaker."""
        gen = self.card_generation.get(card_id, 0)
        heapq.heappush(
            self.event_queue,
            (time_us, next(self.seq_counter), event_type, card_id, gen, payload or {}),
        )

    def _legit_asn_type(self, channel: str) -> str:
        """The ASN mix of legitimate traffic on a channel; a mimicking attacker draws from the same mix."""
        if channel.startswith("CP"):
            return str(self.rng.choice(["residential", "mobile"], p=[0.75, 0.25]))
        if channel == "CNP_IN_APP":
            return str(self.rng.choice(["mobile", "residential", "datacenter"], p=[0.72, 0.24, 0.04]))
        if self.region == "IN":
            return str(self.rng.choice(["mobile", "residential", "datacenter"], p=[0.60, 0.35, 0.05]))
        return str(self.rng.choice(["residential", "mobile", "datacenter"], p=[0.55, 0.39, 0.06]))

    def _validated_cards(self) -> List[Any]:
        """Cards the adversary has already validated and not burned (spec/07 section 10)."""
        validated = []
        for c in self.cards:
            if c.is_frozen or self.fraudster.is_card_burned(c.card_id):
                continue
            belief = self.fraudster.belief_states.get(c.card_id)
            target = self.fraudster.target_states.get(c.card_id)
            if (belief is not None and belief.p_valid >= 1.0) or (
                target is not None and target.fsm_state in (FraudsterState.ACTIVE_CASHOUT, FraudsterState.SUCCESS_HARVEST, FraudsterState.AMOUNT_ADAPTATION)
            ):
                validated.append(c)
        return validated

    def _legit_ip_distance_km(self, channel: str) -> float:
        """spec/07 section 12: geolocation error of a legitimate cardholder's IP, long right tail."""
        params = self.specs.telemetry.legit_ip_distance_lognormal.get(channel)
        if channel.startswith("CP") or params is None:
            return float(self.rng.uniform(1.8, 22.0))
        median_km, sigma_log = params
        return float(self.rng.lognormal(mean=math.log(median_km), sigma=sigma_log))

    def _sample_network_risk_score(self, outcome: str) -> int:
        """Draws a network risk score (0 to 99) from the spec/07 Beta shape for the outcome."""
        alpha, beta = self.specs.telemetry.network_risk_score_shapes[outcome]
        return int(min(99, math.floor(self.rng.beta(alpha, beta) * 100.0)))

    def _micro_probe_amount(self, card: CardholderProfile) -> float:
        """A card-testing probe in the card's currency, drawn from the spec/04 ticket range (spec/07 section 3)."""
        low, high = self.specs.telemetry.micro_probe_ticket_range_usd
        return round(float(self.rng.uniform(low, high)) * self.specs.telemetry.usd_rate(card.currency), 2)

    def _probe_routes_cross_border(self, card: CardholderProfile, has_live_otp: bool) -> bool:
        """spec/07 section 3: an Indian card without a stolen OTP is probed at a foreign non-3DS merchant
        when international usage is enabled; domestic probes would meet mandatory AFA."""
        return bool(
            card.region == "IN"
            and self.specs.telemetry.route_indian_probes_cross_border
            and card.international_enabled
            and not has_live_otp
        )

    def _sample_channel_for_mcc(self, card: CardholderProfile, mcc: int) -> str:
        """Samples channel conditioned on MCC business model (grounded in spec/05).
        Digital/e-commerce/transit merchants are heavily CNP; physical retail/fuel are CP."""
        # 1. Digital & E-Commerce MCCs: 90%+ CNP
        if mcc in (5311, 5999, 5815, 4899, 4900, 7372, 8398, 4112, 4121, 5732, 5968):
            return str(self.rng.choice(
                ["CNP_WEB", "CNP_MOBILE", "CP_POS_CHIP", "CP_POS_CONTACTLESS"],
                p=[0.48, 0.44, 0.05, 0.03],
            ))
        # 2. In-Person Retail, Grocery & Fuel MCCs: 85%+ CP
        elif mcc in (5411, 5499, 5541, 5542, 5912, 5200, 7513, 5094, 5944):
            return str(self.rng.choice(
                ["CP_POS_CHIP", "CP_POS_CONTACTLESS", "CNP_WEB", "CNP_MOBILE"],
                p=[0.55, 0.35, 0.06, 0.04],
            ))
        # 3. Dining & Fast Food: Split between in-restaurant POS and delivery apps
        elif mcc in (5812, 5814):
            return str(self.rng.choice(
                ["CP_POS_CONTACTLESS", "CP_POS_CHIP", "CNP_MOBILE", "CNP_WEB"],
                p=[0.35, 0.25, 0.30, 0.10],
            ))
        # 4. Fallback to cardholder cohort distribution
        return card.sample_channel(self.rng)

    def generate_batch(
        self,
        n_transactions: Optional[int] = 5000,
        fraud_prevalence: float = 0.02,
        time_span_days: int = 30,
        start_time_seconds: float = 1704067200.0,  # 2024-01-01 00:00:00 UTC
        enforce_invariants: bool = True,
        active_macro_regime: Optional[str] = None,
        chunk_callback: Optional[Any] = None,
        chunk_size: int = 10000,
        pace_to_sample_budget: Optional[bool] = None,
        progress_callback: Optional[Callable[[int, int, Optional[Dict[str, Any]]], None]] = None,
        progress_interval: int = 250,
        simulation_mode: str = "transactions",
        max_transactions: int = 100000,
    ) -> List[Dict[str, Any]]:
        """Generates transactions via discrete-event priority queue with strict monotonicity."""
        records: List[Dict[str, Any]] = []
        self.event_queue.clear()
        self.seq_counter = itertools.count()
        stan_counter = itertools.count(1)

        day_seconds = 86400.0
        start_time_us = int(start_time_seconds * 1_000_000)

        # Clear leftover priority queue events to eliminate memory bloat across batches
        self.event_queue.clear()

        # Reset states
        for c in self.cards:
            self.card_avail_time_us[c.card_id] = start_time_us
            self.card_generation[c.card_id] = 0
            self.card_hawkes_R[c.card_id] = 0.0
            self.card_last_hawkes_time[c.card_id] = -1.0
            c.is_frozen = False
            c.unauthorized_alert_time = -1.0
            c.state = CardholderState.HOMESTEAD
            c.consecutive_pinless_contactless_count = 0
            c.cumulative_pinless_contactless_amount = 0.0
            c.active_trip_remaining_stops = 0
            c.active_trip_cluster_lat = 0.0
            c.active_trip_cluster_lon = 0.0
            c.relocation_until = -1.0
            c.last_physical_time = -1.0
            c.last_merchant_id = None
            c.last_physical_lat = c.home_lat
            c.last_physical_lon = c.home_lon
            c.active_travel_until = -1.0
            c.travel_transit_until = -1.0

        # 1. Compute aggregate arrival rates and pacing
        n_cards = len(self.cards)
        if pace_to_sample_budget is None:
            effective_pace = (simulation_mode == "transactions")
        else:
            effective_pace = bool(pace_to_sample_budget)

        if effective_pace:
            target_n = n_transactions if n_transactions is not None else 5000
            mean_tx_per_card = target_n / max(n_cards, 1)
            mean_inter_arrival_sec = (time_span_days * day_seconds) / max(mean_tx_per_card, 1.0)
            agg_routine_rate = n_cards / mean_inter_arrival_sec
        else:
            mean_inter_arrival_sec = None
            agg_routine_rate = sum(
                self._get_card_hawkes_params(c).mu_0 * 0.8208 / max(0.05, (1.0 - min(0.95, self._get_card_hawkes_params(c).branching_ratio)))
                for c in self.cards
            )

        if fraud_prevalence > 0.0:
            agg_fraud_rate = agg_routine_rate * (fraud_prevalence / (1.0 - fraud_prevalence))
            t_fraud_mean = 1.0 / agg_fraud_rate
        else:
            t_fraud_mean = None

        # Schedule initial cardholder routine arrivals using Recursive Circadian Hawkes MTPP
        macro_mult_0 = self._compute_macro_rate_multiplier(start_time_seconds, active_macro_regime)
        for card in self.cards:
            hawkes_p = self._get_card_hawkes_params(card, mean_inter_arrival_sec=mean_inter_arrival_sec)
            if macro_mult_0 != 1.0:
                hawkes_p = HawkesParameters(
                    mu_0=hawkes_p.mu_0 * macro_mult_0,
                    alpha=hawkes_p.alpha,
                    beta=hawkes_p.beta,
                    beta_0_floor=hawkes_p.beta_0_floor,
                    phi_max=hawkes_p.phi_max,
                )
            initial_t_sec, _ = self.hawkes_engine.sample_next_arrival(
                current_time_sec=start_time_seconds,
                R_current=0.0,
                last_tx_time_sec=-1.0,
                params=hawkes_p,
            )
            self._schedule_event(
                time_us=int(initial_t_sec * 1_000_000),
                event_type=EVT_CARDHOLDER_TX,
                card_id=card.card_id,
            )

        # Schedule initial fraud attack in the competing Poisson stream
        if t_fraud_mean is not None:
            initial_f_sec = start_time_seconds + float(self.rng.exponential(scale=t_fraud_mean))
            target_card = self.cards[int(self.rng.integers(0, n_cards))]
            self._schedule_event(
                time_us=int(initial_f_sec * 1_000_000),
                event_type=EVT_FRAUD_ATTACK,
                card_id=target_card.card_id,
            )

        # Schedule staggered billing statement closing and payroll deposits per cardholder
        for c in self.cards:
            if "PREPAID" not in c.product_id and "EBT" not in c.product_id:
                if "DEBIT" in c.product_id:
                    # Seed ONLY the first payday; the handler at EVT_PAYROLL_DEPOSIT
                    # reschedules the next one every PAYROLL_PERIOD_DAYS, so the chain owns
                    # the cadence. Seeding the whole span here as well double-booked every
                    # payday: boundary k then held k events, giving a triangular 1+2+...+n
                    # deposits per card instead of n (measured 406 vs 28 over 500 days).
                    # Because posted_balance only ever rises from these credits, the runaway
                    # disabled the ISO 51 solvency gate -- 3.1% of declines against the
                    # spec/03 target of [0.40, 0.50].
                    stagger_day = (zlib.crc32(c.card_id.encode("utf-8")) % int(payroll_period_days(c.currency))) + 1  # process-independent (spec/07 section 15)
                    p_us = int((start_time_seconds + stagger_day * day_seconds) * 1_000_000)
                    self._schedule_event(time_us=p_us, event_type=EVT_PAYROLL_DEPOSIT, card_id=c.card_id)
                else:
                    # Schedule repayment for prior month's statement balance carried into simulation
                    if c.posted_balance > 0.0:
                        c.statement_balance = c.posted_balance
                        prior_pay_day = float((zlib.crc32(c.card_id.encode("utf-8")) % 12) + 3)
                        p_us = int((start_time_seconds + prior_pay_day * day_seconds) * 1_000_000)
                        self._schedule_event(time_us=p_us, event_type=EVT_STATEMENT_PAYMENT, card_id=c.card_id)

                    stmt_day = c.billing_cycle_day
                    # Seed ONLY the first cycle. The EVT_BILLING_CYCLE_CLOSE handler
                    # reschedules the next close every BILLING_CYCLE_DAYS, so seeding
                    # every month here as well double-booked each close exactly as the
                    # payroll chain did (measured 271,582 closes over a 500-day run).
                    if stmt_day > 0:
                        c_us = int((start_time_seconds + stmt_day * day_seconds) * 1_000_000)
                        self._schedule_event(time_us=c_us, event_type=EVT_BILLING_CYCLE_CLOSE, card_id=c.card_id)

        card_last_tx: Dict[str, Dict[str, Any]] = {}
        last_global_tx_time = -1.0
        tx_counter = 0
        fraud_counter = 0
        pacing_warm_up = int(self.specs.telemetry.prevalence_warm_up_rows)
        # the legitimate rows of the last daily cycle: their rate follows freezes and burns over the
        # run without letting a night-time attack set a night-time rate for the next day
        pacing_window_sec = float(self.specs.telemetry.prevalence_window_hours) * 3600.0
        legit_times: collections.deque = collections.deque()
        loop_iterations = 0
        if simulation_mode == "days":
            end_time_seconds = start_time_seconds + time_span_days * day_seconds
            effective_limit = max_transactions
            if max_transactions == 100000 and time_span_days > 30:
                effective_limit = max(100000, int(len(self.cards) * time_span_days * 2.5))
            tx_target_limit = effective_limit
            target_total = max(50, int(len(self.cards) * time_span_days * 2.13))
            max_loop_iterations = max(50000, tx_target_limit * 25)
        else:
            end_time_seconds = float("inf")
            target_n = n_transactions if n_transactions is not None else 5000
            tx_target_limit = target_n
            target_total = target_n
            max_loop_iterations = max(50000, target_n * 25)

        while self.event_queue and tx_counter < tx_target_limit:
            loop_iterations += 1
            if loop_iterations >= max_loop_iterations:
                break

            t_us, _, evt_type, card_id, gen, payload = heapq.heappop(self.event_queue)
            tx_time_sec = t_us / 1_000_000.0
            if simulation_mode == "days" and tx_time_sec >= end_time_seconds:
                break
            is_fraud_evt = (evt_type == EVT_FRAUD_ATTACK)

            # Dynamically schedule next fraud attack in competing Poisson stream
            if is_fraud_evt and t_fraud_mean is not None:
                # spec/07 section 19: the fraud rate follows the legitimate rate realised over the
                # last rows, so the share delivered is the share requested whatever the batch size
                while legit_times and legit_times[0] < tx_time_sec - pacing_window_sec:
                    legit_times.popleft()
                elapsed_sec = tx_time_sec - start_time_seconds
                legit_so_far = tx_counter - fraud_counter
                realised_legit_rate = 0.0
                if len(legit_times) >= pacing_warm_up and elapsed_sec > 0.0:
                    realised_legit_rate = len(legit_times) / min(pacing_window_sec, elapsed_sec)
                elif legit_so_far >= pacing_warm_up and elapsed_sec > 0.0:
                    # too few rows in a day to measure: the whole run so far, so that a tiny batch
                    # whose cards are being frozen is not attacked at the planned rate for ever
                    realised_legit_rate = legit_so_far / elapsed_sec
                if realised_legit_rate > 0.0 and 0.0 < fraud_prevalence < 1.0:
                    t_fraud_mean = (1.0 - fraud_prevalence) / (fraud_prevalence * realised_legit_rate)
                next_f_delta_sec = float(self.rng.exponential(scale=t_fraud_mean))
                next_f_t_us = t_us + int(next_f_delta_sec * 1_000_000)
                active_cards = [c for c in self.cards if not c.is_frozen and not self.fraudster.is_card_burned(c.card_id)]
                if not active_cards:
                    active_cards = [c for c in self.cards if not c.is_frozen]
                if active_cards:
                    validated = self._validated_cards()
                    if validated and self.rng.random() < self.specs.telemetry.validated_card_follow_up_probability:
                        target_card_next = self.rng.choice(validated)
                    else:
                        # Activity-weighted victim selection, added 2026-10-12.
                        #
                        # Uniform selection over unfrozen cards made a card's recent
                        # activity ANTI-predictive of being attacked, which is backwards.
                        # Legitimate volume concentrates on frequently-used cards, so those
                        # cards have short gaps between transactions; a uniform count-weighted
                        # draw instead spreads attacks evenly across frequent and dormant
                        # cards, leaving attacked cards with long gaps.
                        #
                        # Measured on the published 100k batch before this change: the median
                        # gap before a first attack was 11.40 h against 0.28 h for legitimate
                        # traffic, and 69.0 percent of first attacks landed on cards dormant
                        # more than 6 h versus 24.6 percent of legitimate rows. On IEEE-CIS
                        # (Kaggle ieee-fraud-detection, 590,540 rows) the same measurement
                        # gives 1.95 h for fraud against 1.68 h for legitimate, and 33.3 versus
                        # 33.9 percent dormant -- real fraud shows NO preference by card
                        # activity, because stolen card numbers are attacked on cards that
                        # are actually in use.
                        #
                        # Weighting by observed transaction count makes the victim pool track
                        # legitimate volume rather than the card roster, which is what the
                        # real corpus shows. Verified after the change with
                        # scripts/transfer_test_real_data.py.
                        # Weighting by cumulative transaction count was tried first and did
                        # almost nothing (median history on attacked cards 73 against 80 for
                        # the roster, and the pre-attack gap barely moved). The signal that
                        # actually distinguishes a legitimate row from an attack row in this
                        # corpus is RECENCY, not lifetime volume: legitimate rows are
                        # volume-weighted toward cards that transact often and therefore have
                        # short gaps, so a count-weighted draw still landed on dormant cards.
                        #
                        # exp(-gap / tau) puts the victim pool on the same recency footing as
                        # legitimate traffic. tau is set so a card idle for ~6 h carries about
                        # 1/e of the weight of one that just transacted.
                        now_sec = t_us / 1_000_000.0
                        gaps = np.array(
                            [
                                max(0.0, now_sec - (c.last_tx_time_sec if c.last_tx_time_sec >= 0 else now_sec))
                                for c in active_cards
                            ],
                            dtype=np.float64,
                        )
                        weights = np.exp(-gaps / (6.0 * 3600.0))
                        total_w = weights.sum()
                        if total_w <= 0.0:
                            target_card_next = self.rng.choice(active_cards)
                        else:
                            cumulative = np.cumsum(weights)
                            pick = self.rng.random() * total_w
                            idx = int(np.searchsorted(cumulative, pick, side="right"))
                            target_card_next = active_cards[min(idx, len(active_cards) - 1)]
                    self._schedule_event(
                        time_us=next_f_t_us,
                        event_type=EVT_FRAUD_ATTACK,
                        card_id=target_card_next.card_id,
                    )

            # Legacy settlement event (no-op)
            if evt_type == EVT_SETTLEMENT:
                continue

            card = self.card_map.get(card_id)
            if card is None:
                continue

            # Lazy cancellation check (card frozen or reissued for cardholder purchases)
            if not is_fraud_evt and gen != self.card_generation[card_id]:
                continue

            # Clearing Presentment Event ($T+1$ to $T+3$)
            if evt_type == EVT_CLEARING:
                card.settle_hold(payload["tx_id"], payload["settled_amount"])
                self.double_entry_ledger.settle_hold(
                    tx_id=payload["tx_id"],
                    card_id=card.card_id,
                    merchant_id=payload.get("merchant_id", "M_DEFAULT"),
                    settled_amount=payload["settled_amount"],
                    sim_time_sec=tx_time_sec,
                )
                cycle_num = ((int(tx_time_sec) % 86400) // 21600) + 1
                clearing_cycle_id = f"CYCLE_{int(tx_time_sec // 86400):04d}_C{cycle_num}"
                self.interbank_clearing.record_clearing_presentment(
                    cycle_id=clearing_cycle_id,
                    acquiring_bank_id=payload.get("acquiring_bank_id", "BANK_DEFAULT"),
                    issuing_bank_id=payload.get("issuing_bank_id", "BANK_DEFAULT"),
                    settled_amount=payload["settled_amount"],
                    interchange_fee=payload.get("interchange_fee", 0.0),
                    is_on_us=payload.get("is_on_us", False),
                )
                continue

            # Monthly Billing Cycle Close Event
            if evt_type == EVT_BILLING_CYCLE_CLOSE:
                expired_ids = card.prune_expired_holds(tx_time_sec)
                for exp_tx_id in expired_ids:
                    self.double_entry_ledger.release_hold(exp_tx_id, card.card_id, sim_time_sec=tx_time_sec)
                stmt_bal, min_due, due_time = card.close_billing_statement(tx_time_sec)
                if stmt_bal > 0.0:
                    if card.repayment_cohort == "TRANSACTOR":
                        # Autopay clears balance 3-5 days after statement closing
                        pay_delay_days = float(self.rng.uniform(3.0, 5.0))
                    elif card.repayment_cohort == "REVOLVER":
                        # Revolvers make payment mid-grace period (10-15 days)
                        pay_delay_days = float(self.rng.uniform(10.0, 15.0))
                    else:
                        pay_delay_days = 25.0
                    due_us = int((tx_time_sec + pay_delay_days * 86400.0) * 1_000_000)
                    self._schedule_event(
                        time_us=due_us,
                        event_type=EVT_STATEMENT_PAYMENT,
                        card_id=card.card_id,
                    )
                # Reschedule next monthly cycle (single owner of the cadence -- the
                # seeding loop schedules only the FIRST cycle, see BILLING_CYCLE_DAYS)
                next_stmt_us = t_us + int(BILLING_CYCLE_DAYS * 86400 * 1_000_000)
                self._schedule_event(
                    time_us=next_stmt_us,
                    event_type=EVT_BILLING_CYCLE_CLOSE,
                    card_id=card.card_id,
                )
                continue

            # Scheduled Repayment Event
            if evt_type == EVT_STATEMENT_PAYMENT:
                card.execute_cohort_payment(tx_time_sec, self.rng)
                continue

            # Direct Deposit Payroll Replenishment
            if evt_type == EVT_PAYROLL_DEPOSIT:
                if getattr(card, "annual_income", 0.0) and card.annual_income > 0.0:
                    # Income-grounded recurring credit: per-payday slice of annual
                    # take-home, per the region's pay cadence (biweekly US / monthly
                    # IN). Falls back to the flat constant only for legacy/test cards
                    # that carry no income (annual_income == 0).
                    periods_per_year = 365.25 / payroll_period_days(card.currency)
                    dep_amt = card.annual_income / periods_per_year
                else:
                    dep_amt = PAYROLL_DEPOSIT_USD if card.currency == "USD" else PAYROLL_DEPOSIT_INR
                card.apply_payroll_deposit(dep_amt)
                # Reschedule the next credit (single owner of the cadence -- the seeding
                # loop schedules only the FIRST credit, see payroll_period_days)
                next_pay_us = t_us + int(payroll_period_days(card.currency) * 86400 * 1_000_000)
                self._schedule_event(
                    time_us=next_pay_us,
                    event_type=EVT_PAYROLL_DEPOSIT,
                    card_id=card.card_id,
                )
                continue

            # Alert freeze event: cardholder froze the card after discovering fraud
            if evt_type == EVT_ALERT_FREEZE:
                card.is_frozen = True
                card.state = CardholderState.FROZEN
                self.card_generation[card_id] += 1
                continue

            # Check if card was previously frozen (legitimate purchases skipped; fraud enters bank evaluation)
            card_is_frozen = card.check_freeze_status(tx_time_sec)
            if not is_fraud_evt and card_is_frozen:
                continue

            # Enforce lock-ahead for Card-Present transactions to prevent impossible travel velocities
            if not is_fraud_evt and t_us < self.card_avail_time_us[card_id]:
                rescheduled_t_us = self.card_avail_time_us[card_id] + int(self.rng.uniform(30.0, 180.0) * 1_000_000)
                self._schedule_event(
                    time_us=rescheduled_t_us,
                    event_type=evt_type,
                    card_id=card_id,
                    payload=payload,
                )
                continue

            hour_of_day = int(((tx_time_sec - start_time_seconds) % day_seconds) // 3600)

            # Determine transaction parameters
            otp_provided = True
            vaai_score = self._sample_network_risk_score("legitimate")
            override_avs = None
            override_cvv = None
            override_lat = None
            override_lon = None
            is_p2p = False
            recipient = None
            asn_type = "residential"

            factual_tx = None
            if is_fraud_evt:
                factual_mcc = card.dominant_mccs[0] if card.dominant_mccs else 5411
                factual_channel = self._sample_channel_for_mcc(card, factual_mcc)
                factual_amount = card.sample_spend_amount(self.rng)
                factual_ip_dist = float(self.rng.uniform(0.5, 18.0))
                factual_tx = {
                    "amount": factual_amount,
                    "channel_type": factual_channel,
                    "ip_distance_from_home_km": factual_ip_dist,
                    "is_cross_border": False,
                    "mcc": factual_mcc,
                    "avs_match_code": "Y",
                    "cvv_match_flag": 1,
                    "billing_shipping_match": 1,
                    "is_fraud": 0,
                    "scenario_tag": "ORGANIC_NORMAL",
                    "haversine_velocity_kph": 0.0,
                    "hour_of_day": hour_of_day,
                }

                if self.adversary_mode == "intent":
                    action_found = False
                    for _ in range(5):
                        attack_params = self.fraudster.select_intent_action(
                            card=card,
                            sim_time_seconds=tx_time_sec,
                            world_center_lat=self.world.center_lat,
                            world_center_lon=self.world.center_lon,
                        )
                        macro_opt = attack_params.get("macro_option")
                        amt = float(attack_params.get("amount", 0.0))
                        ch = str(attack_params.get("channel_type", "NONE"))

                        if macro_opt == "OMEGA_PURGE" or amt <= 0.0 or ch == "NONE":
                            # Target is burned or pruned. Adversary rotates to another unburned card.
                            unburned = [c for c in self.cards if not self.fraudster.is_card_burned(c.card_id) and c.card_id != card.card_id]
                            if unburned:
                                card = self.rng.choice(unburned)
                                continue
                            else:
                                break
                        elif macro_opt == "OMEGA_INCUBATE":
                            # Adversary delays attack until nocturnal window (02:00 local time)
                            nocturnal_sec = tx_time_sec + float(attack_params.get("incubation_seconds", 3600.0))
                            attack_params = self.fraudster.select_intent_action(
                                card=card,
                                sim_time_seconds=nocturnal_sec,
                                world_center_lat=self.world.center_lat,
                                world_center_lon=self.world.center_lon,
                            )
                            amt = float(attack_params.get("amount", 0.0))
                            ch = str(attack_params.get("channel_type", "NONE"))
                            if amt > 0.0 and ch != "NONE":
                                action_found = True
                                break
                            else:
                                unburned = [c for c in self.cards if not self.fraudster.is_card_burned(c.card_id) and c.card_id != card.card_id]
                                if unburned:
                                    card = self.rng.choice(unburned)
                                    continue
                                else:
                                    break
                        else:
                            action_found = True
                            break

                    if not action_found or float(attack_params.get("amount", 0.0)) <= 0.0:
                        fresh_cards = [c for c in self.cards if not self.fraudster.is_card_burned(c.card_id)]
                        card = fresh_cards[0] if fresh_cards else card
                        belief = self.fraudster.belief_states.get(card.card_id)
                        if belief:
                            belief.is_burned = False
                            belief.p_valid = 0.50
                            belief.consecutive_declines = 0
                        fallback_dossier = self.fraudster.dossiers.get(card.card_id)
                        attack_params = {
                            "amount": self._micro_probe_amount(card),
                            "channel_type": "CNP_WEB",
                            "preferred_mcc": 5815,
                            "is_fraud": 1,
                            "scenario_tag": "INTENT_OMEGA_PROBE",
                            "macro_option": "OMEGA_PROBE",
                            "ip_distance_km": float(self.rng.uniform(15.0, 85.0)),
                            "incubation_seconds": 0.0,
                            "is_cross_border": self._probe_routes_cross_border(
                                card, bool(fallback_dossier and fallback_dossier.has_live_otp)
                            ),
                        }

                    # Telemetry attributes for intent-driven attacks
                    dossier = self.fraudster.dossiers.get(card.card_id)
                    mimicking = bool(self.rng.random() < self.fraudster.adversary_mimicry)
                    if attack_params.get("macro_option") == "OMEGA_PROBE":
                        attack_params["asn_type"] = str(self.rng.choice(["datacenter", "residential", "mobile"], p=[0.45, 0.35, 0.20]))
                        attack_params["vaai_score"] = self._sample_network_risk_score("card_testing_probe")
                        attack_params["otp_submitted"] = bool(dossier and dossier.has_live_otp)
                    elif attack_params.get("macro_option") == "OMEGA_HARVEST":
                        attack_params["asn_type"] = str(self.rng.choice(["residential", "mobile", "datacenter"], p=[0.55, 0.35, 0.10]))
                        attack_params["vaai_score"] = self._sample_network_risk_score("cash_out_harvest")
                        attack_params["otp_submitted"] = bool(dossier and dossier.has_live_otp)
                    elif attack_params.get("macro_option") == "OMEGA_BISECT_DRAIN":
                        attack_params["asn_type"] = str(self.rng.choice(["residential", "mobile", "datacenter"], p=[0.60, 0.30, 0.10]))
                        attack_params["vaai_score"] = self._sample_network_risk_score("balance_bisection")
                        attack_params["otp_submitted"] = bool(dossier and dossier.has_live_otp)
                    if mimicking:
                        # a geo-matched residential proxy (the playbook adversary's stealth band) on the legitimate ASN mix
                        attack_params["ip_distance_km"] = float(self.rng.uniform(2.5, 32.0))
                        attack_params["asn_type"] = self._legit_asn_type(str(attack_params.get("channel_type", "CNP_WEB")))
                        attack_params["vaai_score"] = self._sample_network_risk_score("legitimate")
                        if attack_params.get("macro_option") in ("OMEGA_HARVEST", "OMEGA_BISECT_DRAIN"):
                            # spec/07 section 14: a mimicking cash-out buys where the cardholder normally does
                            attack_params["preferred_mcc"] = card.sample_preferred_mcc(self.rng)
                else:
                    attack_params = self.fraudster.select_attack_playbook(
                        card=card,
                        sim_time_seconds=tx_time_sec,
                        world_center_lat=self.world.center_lat,
                        world_center_lon=self.world.center_lon,
                    )
                    attack_params = self.fraudster.finish_playbook_params(attack_params)
                playbook_name = str(attack_params.get("playbook_name", attack_params.get("scenario_tag", "")))
                macro_opt = str(attack_params.get("macro_option", ""))
                cand_amount = float(attack_params.get("amount", 0.0))
                cand_mcc = int(attack_params.get("preferred_mcc", 0) or 0)
                card_dossier = self.fraudster.dossiers.get(card.card_id) if hasattr(self.fraudster, "dossiers") else None
                cand_tier = card_dossier.tier.value if (card_dossier and hasattr(card_dossier, "tier")) else ""
                
                if hasattr(self.syndicate_registry, "get_syndicate_for_intent"):
                    syn = self.syndicate_registry.get_syndicate_for_intent(
                        macro_option=macro_opt,
                        mcc=cand_mcc,
                        amount=cand_amount,
                        credential_tier=cand_tier,
                        playbook_name=playbook_name,
                    )
                else:
                    syn = self.syndicate_registry.get_syndicate_for_playbook(playbook_name)
                victim_bank = self.rail_switch.routing_mesh.resolve_issuing_bank(card.card_id).id
                syn_telemetry = syn.sample_telemetry(self.rng, victim_bank_id=victim_bank) if syn else {}

                amount = float(attack_params["amount"])
                channel = str(attack_params["channel_type"])
                # spec/07 section 17: an attack run from the victim's own device uses the victim's
                # connection; the syndicate and its mule ring stay, the botnet address and device go
                victim_device = bool(attack_params.get("victim_device", False)) and channel.startswith("CNP")
                if victim_device:
                    for marker in ("botnet_cluster_id", "ip_subnet_prefix", "client_ip", "device_fingerprint_id", "asn", "isp", "ja4_signature"):
                        if marker in syn_telemetry:
                            syn_telemetry[marker] = ""
                    attack_params["asn_type"] = self._legit_asn_type(channel)
                    attack_params["ip_distance_km"] = self._legit_ip_distance_km(channel)
                scenario_tag = str(attack_params["scenario_tag"])
                is_fraud = int(attack_params["is_fraud"])
                preferred_mcc = attack_params.get("preferred_mcc")
                is_cross_border = bool(attack_params.get("is_cross_border", False))
                ip_distance = float(attack_params.get("ip_distance_km", self.rng.uniform(15.0, 50.0)))
                override_lat = attack_params.get("override_lat")
                override_lon = attack_params.get("override_lon")
                override_avs = attack_params.get("avs_code")
                override_cvv = attack_params.get("cvv_match_flag")
                # spec/07 section 20: a row that carries the proxy's address carries the proxy's network
                proxy_type = str(syn_telemetry.get("proxy_type", ""))
                if syn_telemetry.get("client_ip") and proxy_type in PROXY_NETWORK_ASN:
                    attack_params["asn_type"] = PROXY_NETWORK_ASN[proxy_type]
                asn_type = str(attack_params.get("asn_type", "residential"))
                otp_provided = bool(attack_params.get("otp_submitted", False))
                if "vaai_score" in attack_params:
                    vaai_score = int(attack_params["vaai_score"])
                # spec/07 section 11: what the attacker holds decides how AVS, CVV and billing come out
                if self.adversary_mode == "intent":
                    intent_dossier = self.fraudster.dossiers.get(card.card_id)
                    credentials_complete = bool(intent_dossier and intent_dossier.tier.value in self.specs.telemetry.full_record_dossier_tiers)
                else:
                    credentials_complete = scenario_tag in self.specs.telemetry.full_record_playbooks

                syndicate_id = syn_telemetry.get("syndicate_id", "")
                botnet_cluster_id = syn_telemetry.get("botnet_cluster_id", "")
                mule_ring_id = syn_telemetry.get("mule_ring_id", "")
                beneficiary_account_id = syn_telemetry.get("beneficiary_account_id", "")
                ip_subnet_prefix = syn_telemetry.get("ip_subnet_prefix", "")
                device_fingerprint_id = syn_telemetry.get("device_fingerprint_id", "")
                override_client_ip = syn_telemetry.get("client_ip") or None
                mule_bank_id = syn_telemetry.get("mule_bank_id", "")
                layering_hop_latency_seconds = float(syn_telemetry.get("layering_hop_latency_seconds", 0.0))
            else:
                is_fraud = 0
                victim_device = False
                credentials_complete = False
                is_cross_border = False
                override_lat = None
                override_lon = None
                preferred_mcc = None
                syndicate_id = ""
                botnet_cluster_id = ""
                mule_ring_id = ""
                beneficiary_account_id = ""
                ip_subnet_prefix = ""
                device_fingerprint_id = ""
                override_client_ip = None
                mule_bank_id = ""
                layering_hop_latency_seconds = 0.0


                dt_cal = datetime.fromtimestamp(tx_time_sec, tz=timezone.utc)
                hour_now = (tx_time_sec / 3600.0) % 24.0

                # Check if current time falls in festive Diwali / Dhanteras calendar window
                is_diwali_calendar_window = (
                    active_macro_regime == "DHANTERAS_DIWALI"
                    or (self.region == "IN" and ((dt_cal.month == 10 and dt_cal.day >= 25) or (dt_cal.month == 11 and dt_cal.day <= 5)))
                )

                # Check if current time falls in US Holiday Surge / Black Friday calendar window
                is_us_holiday_window = (
                    active_macro_regime == "HOLIDAY_SURGE"
                    or (self.region == "US" and dt_cal.month == 11 and 22 <= dt_cal.day <= 30)
                )

                # 1. Active Multi-Day Relocation Episode (HN_HOME_RELOCATION, 72h window)
                if card.state == CardholderState.RELOCATING:
                    scenario_tag = FraudScenario.HARD_NEGATIVE_RELOCATION.value
                    mult = float(self.rng.uniform(1.8, 3.2)) if self.region == "US" else float(self.rng.uniform(1.6, 2.6))
                    amount = max(1.0 if self.region == "US" else 50.0, round(card.sample_spend_amount(self.rng) * mult, 2))
                    channel = "CP_POS_CHIP"
                    preferred_mcc = int(self.rng.choice([7513, 4225, 5200, 5712, 4900]))
                    ip_distance = float(self.rng.uniform(15.0, 65.0))

                # 2. Active Multi-Day Travel Window (HN_CROSS_BORDER_TRAVEL)
                elif card.state in (CardholderState.INTL_TRAVEL, CardholderState.DOMESTIC_TRAVEL):
                    scenario_tag = FraudScenario.HARD_NEGATIVE_TRAVEL.value
                    mult = float(self.rng.uniform(1.5, 2.5)) if self.region == "US" else float(self.rng.uniform(1.4, 2.2))
                    amount = max(1.0 if self.region == "US" else 50.0, round(card.sample_spend_amount(self.rng) * mult, 2))
                    channel = "CP_POS_CHIP"
                    preferred_mcc = int(self.rng.choice([5309, 5812, 7011, 4121]))
                    is_cross_border = bool(card.state == CardholderState.INTL_TRAVEL and card.international_enabled)
                    ip_distance = float(self.rng.uniform(500.0, 3000.0))

                # 3. Authentic Calendar-Anchored Festive Gold Splitting (Rule 114B ₹2L threshold)
                elif is_diwali_calendar_window and card.get_available_balance() >= 175000.0 and self.rng.random() < 0.25:
                    scenario_tag = FraudScenario.HARD_NEGATIVE_DHANTERAS_GOLD.value
                    avail = card.get_available_balance()
                    amount = float(self.rng.uniform(150000.0, min(avail * 0.98, 195000.0)))
                    channel = "CP_POS_CHIP"
                    preferred_mcc = 5944  # Jewelry Gold
                    ip_distance = float(self.rng.uniform(2.5, 18.0))

                # 4. Spontaneous legitimate anomalies / rare life events
                else:
                    rand_hard_neg = self.rng.random() if fraud_prevalence > 0.0 else 1.0
                    if rand_hard_neg < 0.005 or (hour_now < 5.0 and rand_hard_neg < 0.03):
                        # Emergency medical outlier (e.g. overnight pharmacy / medical care)
                        scenario_tag = FraudScenario.HARD_NEGATIVE_EMERGENCY.value
                        amount = float(self.rng.uniform(350.0, 850.0)) if self.region == "US" else float(self.rng.uniform(3500.0, 9500.0))
                        channel = "CP_POS_CHIP"
                        preferred_mcc = 5912  # Pharmacy / Medical
                        ip_distance = float(self.rng.uniform(2.0, 15.0))
                    elif rand_hard_neg < 0.012:
                        # Spontaneous travel burst
                        scenario_tag = FraudScenario.HARD_NEGATIVE_TRAVEL.value
                        amount = float(self.rng.uniform(180.0, 550.0)) if self.region == "US" else float(self.rng.uniform(4500.0, 12000.0))
                        channel = "CP_POS_CHIP"
                        preferred_mcc = 5812  # Dining / Travel
                        is_cross_border = bool(card.international_enabled and self.rng.random() < 0.50)
                        ip_distance = float(self.rng.uniform(500.0, 3000.0))
                        if card.state == CardholderState.HOMESTEAD:
                            dest_lat = float(card.home_lat + self.rng.uniform(2.0, 15.0))
                            dest_lon = float(card.home_lon + self.rng.uniform(2.0, 15.0))
                            card.initiate_travel(tx_time_sec, duration_days=float(self.rng.uniform(2.0, 7.0)), dest_lat=dest_lat, dest_lon=dest_lon, is_international=is_cross_border)
                    elif rand_hard_neg < 0.018 and card.state == CardholderState.HOMESTEAD:
                        # Transition cardholder to 72-hour relocation episode
                        card.initiate_relocation_episode(tx_time_sec)
                        scenario_tag = FraudScenario.HARD_NEGATIVE_RELOCATION.value
                        amount = float(self.rng.uniform(250.0, 850.0)) if self.region == "US" else float(self.rng.uniform(5000.0, 14000.0))
                        channel = "CP_POS_CHIP"
                        preferred_mcc = 5200  # Hardware Store
                        ip_distance = float(self.rng.uniform(15.0, 45.0))
                    else:
                        # Routine organic transaction with macro-regime channel & spend dynamics
                        scenario_tag = FraudScenario.ORGANIC_NORMAL.value
                        is_holiday_sale = bool(is_us_holiday_window or is_diwali_calendar_window)
                        p2p_prob = self._compute_dynamic_p2p_probability(tx_time_sec, is_holiday_sale=is_holiday_sale)
                        if (self.p2p_enabled or self.annual_seasonality) and len(self.cards) > 1 and self.rng.random() < p2p_prob:
                            is_p2p = True
                            recip_idx = int(self.rng.integers(0, len(self.cards)))
                            if len(self.cards) > 1 and self.cards[recip_idx].card_id == card.card_id:
                                recip_idx = (recip_idx + 1) % len(self.cards)
                            recipient = self.cards[recip_idx]
                            channel = "CNP_MOBILE"
                            preferred_mcc = 4829 if self.region == "IN" else 6538
                            p2p_mu = 7.1 if self.region == "IN" else 3.6
                            amount = max(1.0, round(float(self.rng.lognormal(mean=p2p_mu, sigma=0.6)), 2))
                            override_lat = card.home_lat
                            override_lon = card.home_lon

                        elif is_us_holiday_window and self.rng.random() < 0.40:
                            # Elevated online shopping during Black Friday / Cyber Week
                            channel = "CNP_WEB"
                            preferred_mcc = int(self.rng.choice([5732, 5311, 5651]))  # Electronics / Department Stores
                            amount = card.sample_spend_amount(self.rng) * 1.50
                        else:
                            amount = card.sample_spend_amount(self.rng)
                            preferred_mcc = card.sample_preferred_mcc(self.rng)
                            # spec/07 sections 8 and 9: traffic at MCCs no cohort prefers, and micro-tickets
                            telemetry = self.specs.telemetry
                            draw = self.rng.random()
                            cumulative = 0.0
                            for uncovered_mcc, shares in telemetry.legit_uncovered_mcc_share.items():
                                cumulative += shares.get(self.region, 0.0)
                                if draw < cumulative:
                                    preferred_mcc = int(uncovered_mcc)
                                    break
                            channel = self._sample_channel_for_mcc(card, preferred_mcc)
                            if channel.startswith("CP") and self.rng.random() < telemetry.legit_micro_ticket_share.get(self.region, 0.0):
                                low, high = telemetry.legit_micro_ticket_range.get(card.currency, (0.50, 4.99))
                                amount = round(float(self.rng.uniform(low, high)), 2)
                        ip_distance = self._legit_ip_distance_km(channel)

                asn_type = self._legit_asn_type(channel)

                # spec/07 sections 4 and 5: cardholders fail AFA sometimes, and those with international
                # usage enabled buy from foreign merchants online from home.
                if channel.startswith("CNP"):
                    telemetry = self.specs.telemetry
                    if self.rng.random() < telemetry.legit_afa_failure_share.get(self.region, 0.0):
                        otp_provided = False
                    if (
                        card.international_enabled
                        and not is_cross_border
                        and self.rng.random() < telemetry.legit_cross_border_cnp_share.get(self.region, 0.0)
                    ):
                        is_cross_border = True

            # Route merchant
            if is_p2p and recipient is not None:
                merchant_lat = float(override_lat)
                merchant_lon = float(override_lon)
                merchant_id = f"P2P_{recipient.card_id}"
                merchant_name = f"P2P Transfer ({recipient.card_id})"
                mcc = preferred_mcc or (4829 if self.region == "IN" else 6538)
                category = "P2P Money Transfer"
                mid = f"MID_P2P_{recipient.card_id}"
                tid = "TID_P2P"
                acquirer_bin = "400099"
                gateway_provider = "NPCI_UPI" if self.region == "IN" else "ZELLE_NETWORK"
                country_code = self.region
                postal_code = "400001" if self.region == "IN" else "10001"
            elif override_lat is not None and override_lon is not None:
                merchant_lat = float(override_lat)
                merchant_lon = float(override_lon)
                merchant_id = f"M_EXT_{tx_counter:06d}"
                merchant_name = "Remote / External Merchant"
                mcc = preferred_mcc or 5732
                category = MCC_TAXONOMY.get(mcc, {}).get("category", "High-Risk Retail")
                mid = f"MID_EXT_{tx_counter:06d}"
                tid = f"TID_{tx_counter:04d}"
                acquirer_bin = "400012"
                gateway_provider = "STRIPE"
                country_code = "US" if self.region == "US" else "IN"
                postal_code = "10001" if self.region == "US" else "400001"
            else:
                curr_lat, curr_lon, anchor_state = card.get_current_anchor_location(tx_time_sec)
                if anchor_state in (CardholderState.INTL_TRAVEL, CardholderState.DOMESTIC_TRAVEL):
                    is_cross_border = (anchor_state == CardholderState.INTL_TRAVEL)
                    ip_distance = float(self.rng.uniform(500.0, 3000.0))

                if channel.startswith("CP") and card.last_physical_time >= 0.0:
                    dt_sec = max(0.0, tx_time_sec - card.last_physical_time)
                    last_mid = card.last_merchant_id
                    last_plat = card.last_physical_lat
                    last_plon = card.last_physical_lon
                else:
                    dt_sec = None
                    last_mid = None
                    last_plat = None
                    last_plon = None

                merchant = self.world.route_merchant_by_gravity(
                    agent_lat=curr_lat,
                    agent_lon=curr_lon,
                    hour_of_day=hour_of_day,
                    channel_type=channel,
                    preferred_mcc=preferred_mcc,
                    delta_t_sec=dt_sec,
                    last_merchant_id=last_mid,
                    last_lat=last_plat,
                    last_lon=last_plon,
                )
                merchant_id = merchant.merchant_id
                merchant_name = merchant.name
                mcc = merchant.mcc
                category = merchant.category
                merchant_lat = merchant.lat
                merchant_lon = merchant.lon
                mid = merchant.mid
                tid = merchant.tid
                acquirer_bin = merchant.acquirer_bin
                gateway_provider = merchant.gateway_provider
                country_code = merchant.country_code
                postal_code = merchant.postal_code

            tx_id = f"TX_{self.id_prefix}_{tx_counter:08d}" if self.id_prefix else f"TX_{tx_counter:08d}"
            record = self.ledger.enrich_transaction(
                tx_id=tx_id,
                card=card,
                merchant_id=merchant_id,
                merchant_name=merchant_name,
                mcc=mcc,
                merchant_category=category,
                merchant_lat=merchant_lat,
                merchant_lon=merchant_lon,
                amount=amount,
                tx_time=tx_time_sec,
                channel_type=channel,
                is_fraud=is_fraud,
                scenario_tag=scenario_tag,
                mid=mid,
                tid=tid,
                acquirer_bin=acquirer_bin,
                gateway_provider=gateway_provider,
                merchant_country=country_code,
                postal_code=postal_code,
                is_cross_border=is_cross_border,
                ip_distance_km=ip_distance,
                asn_type=asn_type,
                override_cvv_match=override_cvv,
                override_avs_code=override_avs,
                override_client_ip=override_client_ip,
                credentials_complete=credentials_complete,
                syndicate_id=syndicate_id,
                botnet_cluster_id=botnet_cluster_id,
                mule_ring_id=mule_ring_id,
                beneficiary_account_id=beneficiary_account_id,
                ip_subnet_prefix=ip_subnet_prefix,
                device_fingerprint_id=device_fingerprint_id,
                victim_device=victim_device,
            )

            if is_fraud:
                record["asn"] = syn_telemetry.get("asn", "")
                record["isp"] = syn_telemetry.get("isp", "")
                record["ja4_signature"] = syn_telemetry.get("ja4_signature", "")
                record["mule_tier"] = syn_telemetry.get("mule_tier", "")
                record["liquidation_channel"] = syn_telemetry.get("liquidation_channel", "")

            record["mule_bank_id"] = mule_bank_id
            record["layering_hop_latency_seconds"] = layering_hop_latency_seconds


            # Boundary Layer 1: Decoupled Payment Rail Verifier Switch
            velocity_kph = float(record.get("haversine_velocity_kph", 0.0))
            count_1h = int(record.get("tx_count_1h", 0))
            cvv_match = (int(record.get("cvv_match_flag", 1)) == 1)

            # 1a. Pre-Authentication Risk Assessment.
            # The terminal/ACS authenticates before the issuer authorizes, so the 3DS/EMV
            # outcome is decided from the risk assessment that exists *before* authentication:
            # the same structural model, evaluated with the five authentication deviations
            # held at zero. Feeding the published score here instead would make the
            # authentication outcome a function of the causal ground truth that consumes it.
            pre_auth_score = self.causal_engine.pre_authentication_score(record, scenario_tag=scenario_tag)

            intent = CandidateTransactionIntent(
                tx_id=tx_id,
                card_id=card.card_id,
                sim_time_sec=tx_time_sec,
                amount=amount,
                currency=card.currency,
                channel_type=channel,
                merchant_id=merchant_id,
                mcc=mcc,
                merchant_lat=merchant_lat,
                merchant_lon=merchant_lon,
                is_cross_border=is_cross_border,
                is_fraud=is_fraud,
                scenario_tag=scenario_tag,
                otp_submitted=otp_provided,
                cvv_provided=cvv_match,
                avs_code=override_avs or record.get("avs_match_code", "Y"),
                billing_shipping_match=record.get("billing_shipping_match", 1),
                pin_entered=False,
                emv_chip_present=(channel == "CP_POS_CHIP"),
                emv_cryptogram_valid=True,
                three_ds_requested=channel.startswith("CNP"),
                risk_score=pre_auth_score,
                vaai_score=vaai_score,
                haversine_velocity_kph=velocity_kph,
                tx_count_1h=count_1h,
                tx_count_24h=int(record.get("tx_count_24h", 0)),
                tx_attempts_1h=int(record.get("tx_attempts_1h", 0)),
                distinct_mids_30m=int(record.get("distinct_mids_30m", 1)),
                subminute_attempts_60s=int(record.get("subminute_attempts_60s", 0)),
                ip_distance_km=ip_distance,
                asn_type=asn_type,
            )

            # 1b. Authentication outcome, decided once and carried on both sides:
            # the record (spec/21's explained_feature_contract reads these five fields) and
            # the intent (the authorization rules must not re-derive them).
            authentication = self.rail_switch.authenticate(intent, card)
            intent.authentication = authentication
            record["emv_arqc_verified"] = bool(authentication.arqc_verified)
            record["emv_pin_verified"] = bool(authentication.pin_verified)
            record["emv_tvr_clean"] = bool(authentication.tvr_clean)
            record["three_ds_authenticated"] = bool(authentication.three_ds_authenticated)
            record["three_ds_attempted"] = bool(authentication.three_ds_attempted)

            # 2. Real-Time Risk Scoring & Causal Evaluation (Pre-Authorization),
            # now including the authentication evidence the rails established above.
            causal_gt = self.causal_engine.evaluate(
                record=record,
                scenario_tag=scenario_tag,
                factual_counterfactual=factual_tx,
            )
            ml_risk_score = causal_gt.risk_score
            record["risk_score"] = ml_risk_score
            record["base_risk"] = causal_gt.base_risk
            record["dominant_causal_driver"] = causal_gt.dominant_causal_driver
            record["analytical_shapley_probability"] = causal_gt.analytical_shapley_probability
            record["analytical_shapley_log_odds"] = causal_gt.analytical_shapley_log_odds
            record["counterfactual_input_deltas"] = _plain(causal_gt.counterfactual_input_deltas)
            record["counterfactual_mode"] = causal_gt.counterfactual_mode
            record["counterfactual_twin"] = _plain(causal_gt.counterfactual_twin)
            record["normative_baseline"] = _plain(causal_gt.normative_baseline)
            record["explanation_narrative"] = causal_gt.explanation_narrative

            # 3. Authorization: the post-authentication risk score drives the issuer's
            # decision gate, and the rail reuses the authentication outcome decided above.
            intent.risk_score = ml_risk_score
            rail_result = self.rail_switch.verify_intent(intent, card)

            # Record authorization outcome in streaming ledger
            self.ledger.record_authorization_outcome(
                card=card,
                amount=amount,
                merchant_id=merchant_id,
                tx_time=tx_time_sec,
                is_approved=rail_result.approved,
                response_code=rail_result.iso_response_code,
            )

            try:
                auth_response = ISO8583Response(rail_result.iso_response_code)
            except ValueError:
                auth_response = ISO8583Response.DO_NOT_HONOR_05

            trans_status_3ds = rail_result.trans_status_3ds
            approved_amt = rail_result.approved_amount
            pos_entry = rail_result.pos_entry_mode
            pos_condition = rail_result.pos_condition_code
            eci = rail_result.eci
            auth_code = f"A{self.rng.integers(10000, 99999)}" if auth_response in (ISO8583Response.APPROVED_00, ISO8583Response.PARTIAL_APPROVAL_10) else ""

            # Generate Core ISO 8583 Protocol Fields
            stan_val = next(stan_counter) % 1000000
            stan_str = f"{stan_val:06d}"
            rrn_str = f"{int(tx_time_sec) % 10000000000:010d}{stan_val % 100:02d}"

            record["mti"] = "0100"
            record["stan"] = stan_str
            record["rrn"] = rrn_str
            record["auth_code"] = auth_code
            record["response_code"] = auth_response.value
            record["auth_response_code"] = auth_response.value
            record["pos_entry_mode"] = pos_entry
            record["pos_condition_code"] = pos_condition
            record["eci"] = eci
            record["trans_status_3ds"] = trans_status_3ds or "Y"
            record["vaai_score"] = vaai_score
            record["hop_origin"] = rail_result.hop_origin
            record["issuing_bank_id"] = rail_result.issuing_bank_id
            record["acquiring_bank_id"] = rail_result.acquiring_bank_id
            record["is_on_us"] = int(rail_result.is_on_us)
            record["central_switch_id"] = rail_result.central_switch_id
            record["switch_network_fee"] = rail_result.switch_network_fee
            record["interbank_latency_ms"] = rail_result.interbank_latency_ms
            record["mti_request"] = rail_result.mti_request
            record["mti_response"] = rail_result.mti_response
            record["hop_sequence"] = ";".join(rail_result.hop_sequence)

            # Dual-Message Clearing & Settlement ($T+1$ to $T+3$)
            record["clearing_mti"] = "0200"
            clearing_delay_hours = int(self.rng.integers(24, 72))
            record["clearing_delay_hours"] = clearing_delay_hours

            # Fuel Pump & Restaurant Settlement Adjustments
            if mcc == 5542 and auth_response == ISO8583Response.APPROVED_00:
                settled_amount = round(float(self.rng.uniform(25.0, 65.0)), 2) if self.region == "US" else round(float(self.rng.uniform(1500.0, 3500.0)), 2)
            elif mcc == 5812 and auth_response == ISO8583Response.APPROVED_00:
                settled_amount = round(amount * float(self.rng.uniform(1.10, 1.20)), 2)  # 10-20% dining tip
            else:
                settled_amount = approved_amt if approved_amt is not None and approved_amt > 0 else amount

            record["settled_amount"] = settled_amount
            record["settled_amount_minor"] = int(round(settled_amount * 100))
            if rail_result.is_on_us:
                interchange_fee = 0.0
            elif hasattr(rail_result, "interchange_fee") and rail_result.interchange_fee > 0.0:
                interchange_fee = rail_result.interchange_fee
            else:
                interchange_fee = round(settled_amount * 0.0175, 2)
            record["interchange_fee_minor"] = int(round(interchange_fee * 100))

            # Post-Auth Hold and Settlement Scheduling
            if auth_response in (ISO8583Response.APPROVED_00, ISO8583Response.PARTIAL_APPROVAL_10):
                hold_amount = approved_amt if approved_amt is not None and approved_amt > 0 else amount
                expire_time_sec = tx_time_sec + 7.0 * 86400.0
                card.place_hold(tx_id=tx_id, hold_amount=hold_amount, expire_time_sec=expire_time_sec)
                self.double_entry_ledger.place_pre_auth_hold(
                    tx_id=tx_id,
                    card_id=card.card_id,
                    hold_amount=hold_amount,
                    sim_time_sec=tx_time_sec,
                )
                card.total_tx_count += 1
                card.last_tx_time_sec = tx_time_sec
                card.consecutive_declines = 0

                # Schedule dual-message clearing presentment ($T+1$ to $T+3$)
                clearing_t_us = t_us + clearing_delay_hours * 3600 * 1_000_000
                self._schedule_event(
                    time_us=clearing_t_us,
                    event_type=EVT_CLEARING,
                    card_id=card.card_id,
                    payload={
                        "tx_id": tx_id,
                        "settled_amount": settled_amount,
                        "merchant_id": merchant_id,
                        "acquiring_bank_id": rail_result.acquiring_bank_id,
                        "issuing_bank_id": rail_result.issuing_bank_id,
                        "interchange_fee": interchange_fee,
                        "is_on_us": rail_result.is_on_us,
                    },
                )
            else:
                card.consecutive_declines += 1

            # Update Hawkes excitation memory based on authorization feedback
            hawkes_p = self._get_card_hawkes_params(card, mean_inter_arrival_sec=mean_inter_arrival_sec)
            prev_R = self.card_hawkes_R.get(card.card_id, 0.0)
            last_t = self.card_last_hawkes_time.get(card.card_id, -1.0)
            new_R = self.hawkes_engine.update_feedback(
                R_current=prev_R,
                last_tx_time_sec=last_t,
                event_time_sec=tx_time_sec,
                iso_response_code=auth_response.value,
                params=hawkes_p,
            )
            self.card_hawkes_R[card.card_id] = new_R
            self.card_last_hawkes_time[card.card_id] = tx_time_sec
            record["hawkes_intensity_R"] = round(new_R, 4)

            # Post-Authorization Dispute & Chargeback Lifecycle
            if is_fraud == 1 and auth_response in (ISO8583Response.APPROVED_00, ISO8583Response.PARTIAL_APPROVAL_10):
                if self.region == "US":
                    # Visa CE 3.0 deflection check: 2 prior undisputed tx in 120-365d matching IP/Device
                    if card.total_tx_count >= 3 and self.rng.random() < 0.28:
                        record["ce3_qualified"] = True
                        record["dispute_status"] = "DEFLECTED_PRE_DISPUTE_CE3"
                        record["dispute_reason_code"] = "10.4"
                        record["arbitration_fee_usd"] = 0.0
                    else:
                        record["ce3_qualified"] = False
                        record["dispute_status"] = "FIRST_CHARGEBACK"
                        record["dispute_reason_code"] = "10.4" if channel.startswith("CNP") else "10.5"
                        record["arbitration_fee_usd"] = 500.0 if amount >= 1000.0 else 0.0
                    record["rbi_liability_tier"] = ""
                    record["rbi_provisional_credit_mandate_days"] = 0
                    record["cfcfrms_1930_lien_status"] = "NO_LIEN"
                else:  # India
                    record["ce3_qualified"] = False
                    record["dispute_status"] = "FIRST_CHARGEBACK"
                    record["dispute_reason_code"] = "UNAUTHORIZED_ELECTRONIC_DEBIT"
                    record["arbitration_fee_usd"] = 0.0

                    # RBI Limited Liability Tiers (RBI/2017-18/15)
                    # Simulated reporting delay in days
                    report_days = float(self.rng.uniform(1.0, 8.0))
                    if report_days <= 3.0:
                        record["rbi_liability_tier"] = "ZERO_LIABILITY"
                    elif report_days <= 7.0:
                        if card.product_id == "IN_PROD_PMJDY_RUPAY_DEBIT":
                            record["rbi_liability_tier"] = "BSBD_PMJDY_CAP_5000_INR"
                        elif "SUPER_PREMIUM" in card.product_id:
                            record["rbi_liability_tier"] = "HNI_CAP_25000_INR"
                        else:
                            record["rbi_liability_tier"] = "STANDARD_CAP_10000_INR"
                    else:
                        record["rbi_liability_tier"] = "BANK_BOARD_POLICY"

                    record["rbi_provisional_credit_mandate_days"] = 10

                    # CFCFRMS / 1930 Helpline Golden Hour race condition
                    report_minutes = report_days * 1440.0
                    if report_minutes <= 15.0:
                        record["cfcfrms_1930_lien_status"] = "LIEN_FREEZE_SUCCESS"
                    elif report_minutes <= 60.0:
                        record["cfcfrms_1930_lien_status"] = "LIEN_FREEZE_PARTIAL"
                    else:
                        record["cfcfrms_1930_lien_status"] = "CRYPTO_SEVERED_UNHOSTED"
            else:
                record["ce3_qualified"] = False
                record["dispute_status"] = "NONE"
                record["dispute_reason_code"] = ""
                record["arbitration_fee_usd"] = 0.0
                record["rbi_liability_tier"] = ""
                record["rbi_provisional_credit_mandate_days"] = 0
                record["cfcfrms_1930_lien_status"] = "NO_LIEN"

            # Update contactless trackers
            if channel == "CP_POS_CONTACTLESS" and auth_response in (ISO8583Response.APPROVED_00, ISO8583Response.PARTIAL_APPROVAL_10):
                card.consecutive_pinless_contactless_count += 1
                card.cumulative_pinless_contactless_amount += amount
            elif channel == "CP_POS_CHIP":
                card.consecutive_pinless_contactless_count = 0
                card.cumulative_pinless_contactless_amount = 0.0

            # Adversary adapts based on bank response
            if is_fraud_evt:
                self.fraudster.receive_feedback(
                    response_code=auth_response,
                    trans_status_3ds=trans_status_3ds,
                    sim_time_seconds=tx_time_sec,
                    card_id=card.card_id,
                )
                if auth_response in (ISO8583Response.APPROVED_00, ISO8583Response.PARTIAL_APPROVAL_10):
                    card.trigger_unauthorized_alert(sim_time=tx_time_sec, rng=self.rng)
                    if card.unauthorized_alert_time > tx_time_sec:
                        alert_us = int(card.unauthorized_alert_time * 1_000_000)
                        self._schedule_event(
                            time_us=alert_us,
                            event_type=EVT_ALERT_FREEZE,
                            card_id=card.card_id,
                        )


            # Update lock-ahead timestamp for cardholder (enforce 120s dwell on legitimate CP swipes)
            if channel.startswith("CP") and is_fraud == 0:
                lock_us = t_us + 120_000_000
            else:
                lock_us = t_us + 10_000_000
            self.card_avail_time_us[card.card_id] = max(self.card_avail_time_us[card.card_id], lock_us)

            # Schedule cardholder's next routine event
            if not is_fraud_evt:
                if card.state == CardholderState.ACTIVE_SHOPPING_TRIP and card.active_trip_remaining_stops > 0:
                    # Next stop in local cluster (10 to 25 minutes)
                    trip_dt_sec = float(self.rng.lognormal(mean=6.9, sigma=0.4))
                    trip_dt_sec = max(300.0, min(1800.0, trip_dt_sec))  # 5 to 30 min
                    next_t_us = t_us + int(trip_dt_sec * 1_000_000)
                    card.advance_shopping_trip()
                    self._schedule_event(
                        time_us=next_t_us,
                        event_type=EVT_CARDHOLDER_TX,
                        card_id=card.card_id,
                    )
                else:
                    # Check if daytime CP transaction initiates a new shopping trip
                    hour_now = (tx_time_sec / 3600.0) % 24.0
                    if (
                        channel.startswith("CP")
                        and 8.5 <= hour_now <= 20.5
                        and card.state not in (CardholderState.DOMESTIC_TRAVEL, CardholderState.INTL_TRAVEL, CardholderState.RELOCATING)
                        and self.rng.random() < 0.35
                    ):
                        # Initiate multi-stop shopping trip: 1 to 3 stops
                        n_stops = int(self.rng.choice([1, 2, 3], p=[0.50, 0.35, 0.15]))
                        card.initiate_shopping_trip(cluster_lat=merchant_lat, cluster_lon=merchant_lon, num_stops=n_stops)
                        trip_dt_sec = float(self.rng.lognormal(mean=6.9, sigma=0.4))
                        trip_dt_sec = max(300.0, min(1800.0, trip_dt_sec))
                        next_t_us = t_us + int(trip_dt_sec * 1_000_000)
                        self._schedule_event(
                            time_us=next_t_us,
                            event_type=EVT_CARDHOLDER_TX,
                            card_id=card.card_id,
                        )
                    else:
                        # Draw next routine event using Recursive Circadian Hawkes MTPP
                        macro_mult = self._compute_macro_rate_multiplier(tx_time_sec, active_macro_regime)
                        card_hawkes = self._get_card_hawkes_params(card, mean_inter_arrival_sec=mean_inter_arrival_sec)
                        if macro_mult != 1.0:
                            card_p = HawkesParameters(
                                mu_0=card_hawkes.mu_0 * macro_mult,
                                alpha=card_hawkes.alpha,
                                beta=card_hawkes.beta,
                                beta_0_floor=card_hawkes.beta_0_floor,
                                phi_max=card_hawkes.phi_max,
                            )
                        else:
                            card_p = card_hawkes

                        next_arrival_sec, _ = self.hawkes_engine.sample_next_arrival(
                            current_time_sec=tx_time_sec,
                            R_current=self.card_hawkes_R[card.card_id],
                            last_tx_time_sec=self.card_last_hawkes_time[card.card_id],
                            params=card_p,
                        )
                        next_t_us = int(next_arrival_sec * 1_000_000)
                        self._schedule_event(
                            time_us=next_t_us,
                            event_type=EVT_CARDHOLDER_TX,
                            card_id=card.card_id,
                        )

            # Invariant Verification
            if enforce_invariants:
                if last_global_tx_time >= 0.0 and tx_time_sec < last_global_tx_time:
                    raise ValueError(
                        f"Global temporal monotonicity violated: {tx_time_sec:.2f} < {last_global_tx_time:.2f}"
                    )
                prev_tx = card_last_tx.get(card.card_id)
                violations = verify_transaction_invariants(record, prev_tx=prev_tx)
                if violations:
                    raise ValueError(
                        f"Invariant verification failed on transaction {tx_id}: {'; '.join(violations)}"
                    )

            card_last_tx[card.card_id] = record
            last_global_tx_time = tx_time_sec
            records.append(record)
            tx_counter += 1
            if is_fraud == 1:
                fraud_counter += 1
            else:
                legit_times.append(tx_time_sec)

            if progress_callback is not None and (tx_counter % progress_interval == 0 or tx_counter == tx_target_limit):
                try:
                    progress_callback(tx_counter, target_total, record, records)
                except TypeError:
                    progress_callback(tx_counter, target_total, record)

            if chunk_callback is not None and len(records) >= chunk_size:
                chunk_callback(records)
                records = []

        if progress_callback is not None and (tx_counter % progress_interval != 0 or simulation_mode == "days"):
            try:
                progress_callback(tx_counter, tx_counter if simulation_mode == "days" else target_total, records[-1] if records else None, records)
            except TypeError:
                progress_callback(tx_counter, tx_counter if simulation_mode == "days" else target_total, records[-1] if records else None)

        if chunk_callback is not None and records:
            chunk_callback(records)
            records = []

        return records

    def stream_continuous(
        self,
        duration_seconds: float = 60.0,
        target_tps: float = 10.0,
        fraud_prevalence: float = 0.02,
        start_time_seconds: float = 1704067200.0,
        enforce_invariants: bool = True,
        active_macro_regime: Optional[str] = None,
        real_time: bool = False,
    ) -> Iterator[Dict[str, Any]]:
        """Yields streaming transactions with strictly monotonic Poisson arrivals.

        Args:
            duration_seconds: Simulated duration in seconds.
            target_tps: Target transactions per second.
            fraud_prevalence: Ratio of adversarial fraud transactions.
            start_time_seconds: Epoch timestamp start.
            enforce_invariants: Whether to enforce physical/rail invariants.
            active_macro_regime: Optional macroeconomic regime.
            real_time: If True, paces emission using wall-clock time.sleep() matching
                       the simulated inter-arrival delta (dt). If False (default), yields
                       transactions immediately for high-throughput batch replay and testing.
        """
        batch_size = max(50, int(duration_seconds * target_tps))
        records = self.generate_batch(
            n_transactions=batch_size,
            fraud_prevalence=fraud_prevalence,
            time_span_days=max(1, int(duration_seconds / 86400.0) + 1),
            start_time_seconds=start_time_seconds,
            enforce_invariants=enforce_invariants,
            active_macro_regime=active_macro_regime,
        )
        sleep_interval = 1.0 / max(target_tps, 0.1) if real_time else 0.0
        for i, r in enumerate(records):
            if real_time and i > 0:
                time.sleep(sleep_interval)
            yield r


# Backward-compatible class alias
SimulationEngine = DiscreteEventEngine
