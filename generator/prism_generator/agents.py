"""Behavioral models for cardholders, merchants, adversarial fraudsters, and banks.

Implements closed-loop behavioral dynamics:
1. CardholderProfile: 11 Global (Visa/MC) and 5 Indian (RBI/RuPay) card products,
   7 Fed DCPC cohorts, diurnal Poisson arrivals, and authentic hard negatives.
2. AdaptiveFraudsterAgent: 13 grounded cybercrime playbooks (8 Global & 5 India),
   including distributed additive BIN attacks, triangulation fraud, and silent baking.
3. BankDecisionEngine: Multi-tier authorization switch evaluating ISO 8583 syntax,
   RBI AFA/Card Controls, PSD2 SCA exemptions, STIP floor limits, and Visa VAAI scoring.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .intent import (
    AnalyticalBeliefState,
    CredentialDossier,
    CredentialTier,
    InformationDirectedOptimizer,
    MacroOptionType,
)
from .spec_loader import load_all_specs

# Fallback parity scale for synthetic cardholders instantiated under USD default
# log-parameters (mu < 5.0) but configured for INR rails, ensuring INR ticket sizes
# align with empirical RBI baselines (spec/05 & spec/08).
INR_LEGACY_USD_SCALE_ADJUSTMENT: float = 40.0


class ISO8583Response(str, enum.Enum):
    APPROVED_00 = "00"
    DO_NOT_HONOR_05 = "05"
    PARTIAL_APPROVAL_10 = "10"
    INVALID_TRANSACTION_12 = "12"
    INVALID_CARD_14 = "14"
    INSUFFICIENT_FUNDS_51 = "51"
    EXPIRED_CARD_54 = "54"
    NOT_PERMITTED_57 = "57"
    SUSPECTED_FRAUD_59 = "59"
    SECURITY_VIOLATION_63 = "63"
    ACTIVITY_COUNT_EXCEEDED_65 = "65"
    ACTIVITY_LIMIT_EXCEEDED_65 = "65"  # Legacy alias for ACTIVITY_COUNT_EXCEEDED_65 per ISO 8583
    PIN_TRIES_EXCEEDED_75 = "75"
    INVALID_CVV_82 = "82"
    ISSUER_TIMEOUT_STIP_91 = "91"


class FraudScenario(str, enum.Enum):
    ORGANIC_NORMAL = "ORGANIC_NORMAL"
    # Global Playbooks
    ADV_MICRO_AUTH_PROBE = "ADV_MICRO_AUTH_PROBE"
    ADV_ATO_SILENT_BAKING = "ADV_ATO_SILENT_BAKING"
    ADV_SLEEPER_BUST_OUT = "ADV_SLEEPER_BUST_OUT"
    ADV_APPLE_PAY_YELLOW_PATH = "ADV_APPLE_PAY_YELLOW_PATH"
    ADV_NOCTURNAL_BURST = "ADV_NOCTURNAL_BURST"
    ADV_DISTRIBUTED_BIN_ENUMERATION = "ADV_DISTRIBUTED_BIN_ENUMERATION"
    ADV_TRIANGULATION_FRAUD = "ADV_TRIANGULATION_FRAUD"
    ADV_COLLUSIVE_BUST_OUT_MID = "ADV_COLLUSIVE_BUST_OUT_MID"
    # India Playbooks
    IN_ADV_REVERSE_PROXY_VISHING = "IN_ADV_REVERSE_PROXY_VISHING"
    IN_ADV_APK_SMS_STEALER = "IN_ADV_APK_SMS_STEALER"
    IN_ADV_INTL_NON_3DS_BYPASS = "IN_ADV_INTL_NON_3DS_BYPASS"
    IN_ADV_RENT_PORTAL_CASHOUT = "IN_ADV_RENT_PORTAL_CASHOUT"
    IN_ADV_SIM_SWAP_ESIM_HIJACK = "IN_ADV_SIM_SWAP_ESIM_HIJACK"
    # Authentic Hard Negatives
    HARD_NEGATIVE_TRAVEL = "HARD_NEGATIVE_TRAVEL"
    HARD_NEGATIVE_RELOCATION = "HARD_NEGATIVE_RELOCATION"
    HARD_NEGATIVE_DHANTERAS_GOLD = "HARD_NEGATIVE_DHANTERAS_GOLD"
    HARD_NEGATIVE_EMERGENCY = "HARD_NEGATIVE_EMERGENCY"
    # Legacy aliases
    CARD_TESTING_BURST = "ADV_MICRO_AUTH_PROBE"
    VELOCITY_BLITZ = "ADV_MICRO_AUTH_PROBE"
    ACCOUNT_TAKEOVER = "ADV_ATO_SILENT_BAKING"
    SLEEPER_BUST_OUT = "ADV_SLEEPER_BUST_OUT"
    CNP_FULLZ_EXPLOIT = "ADV_ATO_SILENT_BAKING"
    COUNTERFEIT_CLONE = "ADV_APPLE_PAY_YELLOW_PATH"
    IMPOSSIBLE_TRAVEL = "HARD_NEGATIVE_TRAVEL"
    HARD_NEGATIVE = "HARD_NEGATIVE_TRAVEL"


class CardholderState(str, enum.Enum):
    HOMESTEAD = "HOMESTEAD"
    COMMUTING = "COMMUTING"
    SHOPPING_LOCAL = "SHOPPING_LOCAL"
    ACTIVE_SHOPPING_TRIP = "ACTIVE_SHOPPING_TRIP"
    RELOCATING = "RELOCATING"
    DOMESTIC_TRAVEL = "DOMESTIC_TRAVEL"
    INTL_TRAVEL = "INTL_TRAVEL"
    ALERTED = "ALERTED"
    FROZEN = "FROZEN"


class FraudsterState(str, enum.Enum):
    DUMP_INGESTION = "DUMP_INGESTION"
    MICRO_PROBING = "MICRO_PROBING"
    ATO_INFILTRATION = "ATO_INFILTRATION"
    SILENT_BAKING = "SILENT_BAKING"
    ACTIVE_CASHOUT = "ACTIVE_CASHOUT"
    AMOUNT_ADAPTATION = "AMOUNT_ADAPTATION"
    GATEWAY_HOP = "GATEWAY_HOP"
    VELOCITY_BACKOFF = "VELOCITY_BACKOFF"
    CARD_PURGE = "CARD_PURGE"
    BIN_ROTATION = "BIN_ROTATION"
    SUCCESS_HARVEST = "SUCCESS_HARVEST"


class ChannelType(str, enum.Enum):
    CP_POS_CHIP = "CP_POS_CHIP"
    CP_POS_CONTACTLESS = "CP_POS_CONTACTLESS"
    CP_POS_MAGSTRIPE = "CP_POS_MAGSTRIPE"
    CNP_WEB = "CNP_WEB"
    CNP_MOBILE = "CNP_MOBILE"
    CNP_API = "CNP_API"
    UPI_QR_CREDIT = "UPI_QR_CREDIT"


def sample_spliced_lognormal_gpd(
    rng: np.random.Generator,
    mu: float,
    sigma: float,
    u: float = 250.0,
    pi_u: float = 0.03,
) -> float:
    """Samples ticket spend from a composite LogNormal distribution with Generalized Pareto (GPD) heavy tail."""
    z_u = (math.log(max(u, 1.0)) - mu) / max(sigma, 0.05)
    phi_z = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * z_u * z_u)
    f_ln_u = phi_z / (u * max(sigma, 0.05))

    beta = max(10.0, pi_u / max(f_ln_u, 1e-6))
    xi = max(0.05, min(0.40, (pi_u / (u * max(f_ln_u, 1e-6))) * (1.0 + (math.log(u) - mu) / (sigma ** 2)) - 1.0))

    draw = rng.uniform(0.0, 1.0)
    if draw <= (1.0 - pi_u):
        val = rng.lognormal(mean=mu, sigma=sigma)
        return float(np.clip(val, 1.50, u))
    else:
        uniform_tail = rng.uniform(0.0, 0.92)
        gpd_excess = (beta / xi) * (math.pow(1.0 - uniform_tail, -xi) - 1.0)
        return float(np.clip(u + gpd_excess, 1.50, u * 2.5))


@dataclass
class CardholderProfile:
    """Cardholder behavioral profile with discrete state tracking, product limits, and regulatory controls.
    
    Fully grounded in canonical specifications (spec/01 - spec/05).
    """
    card_id: str
    home_lat: float
    home_lon: float
    work_lat: float
    work_lon: float
    is_commuter: bool
    credit_limit: float
    current_balance: float = 0.0
    spend_mean_log: float = 3.5
    spend_sigma_log: float = 0.65
    preferred_channels: List[str] = field(default_factory=lambda: ["CP_POS_CHIP", "CP_POS_CONTACTLESS", "CNP_WEB"])
    circadian_peak_hour: float = 14.0
    phase_offset_sec: float = 0.0
    """Local-clock offset into the diurnal cycle: time zone (US population spread) plus
    the chronotype peak drawn at initialization (2026-10-16; see engine._initialize_cardholders)."""

    # Kinematics & Physical anchor tracking
    last_physical_lat: float = 0.0
    last_physical_lon: float = 0.0
    last_physical_time: float = -1.0
    last_merchant_id: Optional[str] = None

    # Behavioral state machine
    state: CardholderState = CardholderState.HOMESTEAD
    is_frozen: bool = False
    active_travel_until: float = -1.0
    travel_transit_until: float = -1.0
    travel_lat: float = 0.0
    travel_lon: float = 0.0
    unauthorized_alert_time: float = -1.0
    total_tx_count: int = 0
    # Simulation time of this card's most recent emitted transaction. Used to weight
    # attack targeting by recency; see the note at the victim-selection site in engine.py.
    last_tx_time_sec: float = -1.0
    consecutive_declines: int = 0

    # Shopping trip & Life-stage episode tracking
    active_trip_remaining_stops: int = 0
    active_trip_cluster_lat: float = 0.0
    active_trip_cluster_lon: float = 0.0
    relocation_until: float = -1.0

    # Institutional Banking Attributes
    product_id: str = "PROD_CLASSIC_REVOLVING"
    cohort_id: str = "C4_SUBURBAN_FAMILY"
    region: str = "US"
    currency: str = "USD"
    annual_income: float = 0.0  # annual, major units; grounds the recurring payroll deposit
    pan_masked: str = "414720******1234"
    domestic_cnp_enabled: bool = True
    international_enabled: bool = False
    contactless_enabled: bool = True
    consecutive_pinless_contactless_count: int = 0
    cumulative_pinless_contactless_amount: float = 0.0

    # Grounded Solvency & Billing Cycle Attributes
    posted_balance: float = 0.0
    pending_holds: float = 0.0
    overdraft_limit: float = 0.0
    active_holds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    billing_cycle_day: int = 15
    repayment_cohort: str = "TRANSACTOR"
    apr: float = 0.2199
    statement_balance: float = 0.0
    minimum_payment_due: float = 0.0
    payment_due_time: float = -1.0
    days_past_due: int = 0
    is_delinquent: bool = False

    # Grounded Spec Behavioral Elements
    dominant_mccs: List[int] = field(default_factory=list)
    channel_probabilities: Dict[str, float] = field(default_factory=dict)
    vigilance_weights: List[float] = field(default_factory=lambda: [0.35, 0.45, 0.12, 0.08])
    circadian_hourly_probs: np.ndarray = field(default_factory=lambda: np.ones(24) / 24.0)
    is_spliced_gpd: bool = False
    gpd_threshold_u: float = 250.0
    gpd_xi: float = 0.22
    gpd_beta: float = 95.0
    gpd_tail_prob: float = 0.03

    def __post_init__(self) -> None:
        if self.posted_balance == 0.0 and self.current_balance > 0.0:
            self.posted_balance = self.current_balance
        self.current_balance = self.posted_balance + self.pending_holds

    def get_theoretical_mean_spend(self) -> float:
        """Computes theoretical expected spend."""
        if not self.is_spliced_gpd:
            mean_val = math.exp(self.spend_mean_log + 0.5 * (self.spend_sigma_log ** 2))
        else:
            ln_mean = math.exp(self.spend_mean_log + 0.5 * (self.spend_sigma_log ** 2))
            gpd_mean = self.gpd_threshold_u + (self.gpd_beta / max(0.01, 1.0 - self.gpd_xi))
            mean_val = (1.0 - self.gpd_tail_prob) * ln_mean + self.gpd_tail_prob * gpd_mean

        if self.currency == "INR" and self.spend_mean_log < 5.0:
            mean_val *= INR_LEGACY_USD_SCALE_ADJUSTMENT
        return mean_val

    def get_available_balance(self) -> float:
        """Computes real-time solvency balance based on instrument category."""
        if "DEBIT" in self.product_id:
            return max(0.0, (self.posted_balance + self.overdraft_limit) - self.pending_holds)
        elif any(k in self.product_id for k in ("PREPAID", "EBT", "HSA")):
            return max(0.0, self.posted_balance - self.pending_holds)
        elif "BNPL" in self.product_id:
            effective_limit = self.credit_limit * 1.15
            return max(0.0, effective_limit - (self.posted_balance + self.pending_holds))
        else:
            # Standard revolving credit
            return max(0.0, self.credit_limit - (self.posted_balance + self.pending_holds))

    def place_hold(self, tx_id: str, hold_amount: float, expire_time_sec: float) -> bool:
        """Places pre-authorization hold upon approval. ZERO effect on posted_balance."""
        self.pending_holds += hold_amount
        self.active_holds[tx_id] = (hold_amount, expire_time_sec)
        self.current_balance = self.posted_balance + self.pending_holds
        return True

    def release_hold(self, tx_id: str) -> float:
        """Releases hold upon decline, reversal, or expiry."""
        if tx_id in self.active_holds:
            hold_amt, _ = self.active_holds.pop(tx_id)
            self.pending_holds = max(0.0, self.pending_holds - hold_amt)
            self.current_balance = self.posted_balance + self.pending_holds
            return hold_amt
        return 0.0

    def settle_hold(self, tx_id: str, settled_amount: float) -> None:
        """Transitions pending hold into posted balance upon clearing presentment."""
        if tx_id in self.active_holds:
            hold_amt, _ = self.active_holds.pop(tx_id)
            self.pending_holds = max(0.0, self.pending_holds - hold_amt)
        if "DEBIT" in self.product_id or any(k in self.product_id for k in ("PREPAID", "EBT", "HSA")):
            self.posted_balance = max(0.0, self.posted_balance - settled_amount)
        else:
            self.posted_balance += settled_amount
        self.current_balance = self.posted_balance + self.pending_holds

    def prune_expired_holds(self, current_time_sec: float) -> List[str]:
        """Drops expired pre-auth holds that received no clearing presentment."""
        expired = [tx_id for tx_id, (_, exp_t) in self.active_holds.items() if current_time_sec >= exp_t]
        for tx_id in expired:
            self.release_hold(tx_id)
        return expired

    def close_billing_statement(self, current_time_sec: float) -> Tuple[float, float, float]:
        """Closes 30-day billing cycle, computes interest, and sets payment due date."""
        self.prune_expired_holds(current_time_sec)
        stmt_bal = self.posted_balance
        if stmt_bal > 0.0 and self.repayment_cohort == "REVOLVER":
            interest = round(stmt_bal * (self.apr / 365.0) * 30.0, 2)
            self.posted_balance += interest
            stmt_bal += interest

        if self.region == "IN":
            min_due = min(stmt_bal, max(500.0, round(0.05 * stmt_bal, 2)))
        else:
            min_due = min(stmt_bal, max(25.0, round(0.02 * stmt_bal, 2)))

        self.statement_balance = stmt_bal
        self.minimum_payment_due = min_due
        self.payment_due_time = current_time_sec + 25.0 * 86400.0  # Statutory 25-day grace period
        self.current_balance = self.posted_balance + self.pending_holds
        return stmt_bal, min_due, self.payment_due_time

    def execute_cohort_payment(self, sim_time: float, rng: np.random.Generator) -> float:
        """Executes repayment based on consumer financial cohort."""
        if self.posted_balance <= 0.0:
            return 0.0

        if self.repayment_cohort == "TRANSACTOR":
            payment = min(self.posted_balance, self.statement_balance)
            self.posted_balance = max(0.0, self.posted_balance - payment)
            self.days_past_due = 0
            self.is_delinquent = False

        elif self.repayment_cohort == "REVOLVER":
            if self.posted_balance <= self.minimum_payment_due:
                payment = self.posted_balance
            else:
                zeta = float(rng.beta(1.5, 4.5))
                discretionary = zeta * max(0.0, self.statement_balance - self.minimum_payment_due)
                payment = min(self.posted_balance, round(self.minimum_payment_due + discretionary, 2))
            self.posted_balance = max(0.0, self.posted_balance - payment)
            self.days_past_due = 0
            self.is_delinquent = False

        elif self.repayment_cohort == "DISTRESSED":
            if rng.random() < 0.50:
                self.days_past_due += 30
                self.is_delinquent = True
                late_fee = 750.0 if self.region == "IN" else 35.0
                self.posted_balance += late_fee
                payment = 0.0
                if self.days_past_due >= 60:
                    self.is_frozen = True
            else:
                payment = min(self.posted_balance, self.minimum_payment_due)
                self.posted_balance = max(0.0, self.posted_balance - payment)
        else:
            payment = min(self.posted_balance, self.statement_balance)
            self.posted_balance = max(0.0, self.posted_balance - payment)

        self.current_balance = self.posted_balance + self.pending_holds
        return payment

    def apply_payroll_deposit(self, deposit_amount: float) -> None:
        """Credits direct deposit payroll to checking deposit balance."""
        self.posted_balance += deposit_amount
        self.current_balance = self.posted_balance + self.pending_holds

    def apply_monthly_payment(self, payment_ratio: float = 0.90) -> None:
        """Simulates periodic credit balance settlement (backwards compatible)."""
        self.posted_balance = max(0.0, self.posted_balance * (1.0 - payment_ratio))
        self.current_balance = self.posted_balance + self.pending_holds

    def sample_spend_amount(self, rng: np.random.Generator) -> float:
        """Samples ticket amount according to cohort specification (LogNormal vs Spliced GPD)."""
        # A debit instrument has no credit line, so capping its ticket by
        # `credit_limit * 0.50` is meaningless -- and for a PMJDY account (Rs 6,000
        # limit) that cap sat below the mean debit ticket the RBI reports, making the
        # reference unreachable by construction. Non-credit instruments get a
        # ceiling expressed in rupees instead.
        is_credit_instrument = not any(
            marker in self.product_id
            for marker in ("DEBIT", "PREPAID", "EBT", "HSA", "PNJ")
        )
        if is_credit_instrument and self.credit_limit > 0:
            max_spend = self.credit_limit * 0.50
        else:
            max_spend = 250000.0  # Rs 2,50,000 ceiling for non-credit instruments

        if not self.is_spliced_gpd:
            val = float(rng.lognormal(mean=self.spend_mean_log, sigma=self.spend_sigma_log))
        else:
            draw = float(rng.uniform(0.0, 1.0))
            if draw <= (1.0 - self.gpd_tail_prob):
                val = float(rng.lognormal(mean=self.spend_mean_log, sigma=self.spend_sigma_log))
                val = min(val, self.gpd_threshold_u)
            else:
                u_draw = float(rng.uniform(0.0, 0.95))
                xi = max(0.05, min(0.45, self.gpd_xi))
                gpd_excess = (self.gpd_beta / xi) * (math.pow(1.0 - u_draw, -xi) - 1.0)
                val = self.gpd_threshold_u + gpd_excess

        if self.currency == "INR":
            return float(np.clip(val, 10.0, max_spend))
        else:
            return float(np.clip(val, 1.50, max_spend))

    def sample_preferred_mcc(self, rng: np.random.Generator) -> int:
        """Samples preferred MCC from cohort-specific probability distribution."""
        if not self.dominant_mccs or rng.random() > 0.85:
            return int(rng.choice([5411, 5812, 5814, 5541, 5912, 5311, 5732, 5999]))
        return int(rng.choice(self.dominant_mccs))

    def sample_channel(self, rng: np.random.Generator) -> str:
        """Samples transaction channel using cohort channel distribution."""
        if not self.channel_probabilities:
            return "CP_POS_CHIP"
        channels = list(self.channel_probabilities.keys())
        weights = list(self.channel_probabilities.values())
        return str(rng.choice(channels, p=weights))

    def get_current_anchor_location(self, sim_time: float) -> Tuple[float, float, CardholderState]:
        """Calculates dynamic physical coordinates based on commute schedule, shopping trips, and travel state."""
        # 1. Travel State (Domestic or International)
        if self.active_travel_until > 0.0 and sim_time < self.active_travel_until:
            if self.travel_transit_until > 0.0 and sim_time < self.travel_transit_until:
                # Still in transit / flight: at departure airport or en route
                return self.home_lat, self.home_lon, CardholderState.COMMUTING
            trav_state = CardholderState.INTL_TRAVEL if self.international_enabled else CardholderState.DOMESTIC_TRAVEL
            return self.travel_lat, self.travel_lon, trav_state

        if self.active_travel_until > 0.0 and sim_time >= self.active_travel_until:
            self.active_travel_until = -1.0
            self.travel_transit_until = -1.0
            self.state = CardholderState.HOMESTEAD

        # 2. Relocation Episode (72-hour moving window)
        if self.relocation_until > 0.0 and sim_time < self.relocation_until:
            return self.home_lat, self.home_lon, CardholderState.RELOCATING

        if self.relocation_until > 0.0 and sim_time >= self.relocation_until:
            self.relocation_until = -1.0
            self.state = CardholderState.HOMESTEAD

        # 3. Active Shopping Trip Chaining
        if self.active_trip_remaining_stops > 0:
            return self.active_trip_cluster_lat, self.active_trip_cluster_lon, CardholderState.ACTIVE_SHOPPING_TRIP

        # 4. Commuter / Homestead Dynamics
        day_of_week = int(sim_time // 86400) % 7
        if not self.is_commuter or day_of_week >= 5:
            return self.home_lat, self.home_lon, CardholderState.HOMESTEAD

        hour_of_day = ((sim_time % 86400.0) / 3600.0)

        # 07:30 - 09:00 Inbound Commute
        if 7.5 <= hour_of_day < 9.0:
            alpha = (hour_of_day - 7.5) / 1.5
            lat = (1.0 - alpha) * self.home_lat + alpha * self.work_lat
            lon = (1.0 - alpha) * self.home_lon + alpha * self.work_lon
            return lat, lon, CardholderState.COMMUTING

        # 09:00 - 17:00 Work Office
        elif 9.0 <= hour_of_day < 17.0:
            return self.work_lat, self.work_lon, CardholderState.SHOPPING_LOCAL

        # 17:00 - 18:30 Outbound Commute
        elif 17.0 <= hour_of_day < 18.5:
            alpha = (hour_of_day - 17.0) / 1.5
            lat = (1.0 - alpha) * self.work_lat + alpha * self.home_lat
            lon = (1.0 - alpha) * self.work_lon + alpha * self.home_lon
            return lat, lon, CardholderState.COMMUTING

        # 18:30 - 07:30 Homestead / Residential
        else:
            return self.home_lat, self.home_lon, CardholderState.HOMESTEAD

    def initiate_shopping_trip(self, cluster_lat: float, cluster_lon: float, num_stops: int) -> None:
        """Transitions cardholder into an active multi-stop shopping trip anchored to a commercial hub."""
        self.state = CardholderState.ACTIVE_SHOPPING_TRIP
        self.active_trip_remaining_stops = max(1, num_stops)
        self.active_trip_cluster_lat = cluster_lat
        self.active_trip_cluster_lon = cluster_lon

    def advance_shopping_trip(self) -> bool:
        """Advances shopping trip by one stop. Returns True if stops remain, False if trip concluded."""
        self.active_trip_remaining_stops -= 1
        if self.active_trip_remaining_stops <= 0:
            self.state = CardholderState.HOMESTEAD
            self.active_trip_remaining_stops = 0
            return False
        return True

    def initiate_relocation_episode(self, sim_time_sec: float, duration_sec: float = 259200.0) -> None:
        """Initiates 72-hour persistent home relocation episode."""
        self.state = CardholderState.RELOCATING
        self.relocation_until = sim_time_sec + duration_sec

    def initiate_travel(
        self,
        sim_time: float,
        duration_days: float,
        dest_lat: float,
        dest_lon: float,
        is_international: bool = False,
    ) -> None:
        """Transitions cardholder into multi-day domestic or international travel state."""
        self.active_travel_until = sim_time + duration_days * 86400.0
        self.travel_lat = dest_lat
        self.travel_lon = dest_lon
        self.state = CardholderState.INTL_TRAVEL if is_international else CardholderState.DOMESTIC_TRAVEL
        # Calculate flight transit holding buffer (commercial airline speed ~850 km/h + airport buffer)
        d_lat = math.radians(dest_lat - self.home_lat)
        d_lon = math.radians(dest_lon - self.home_lon)
        a = math.sin(d_lat / 2.0) ** 2 + math.cos(math.radians(self.home_lat)) * math.cos(math.radians(dest_lat)) * math.sin(d_lon / 2.0) ** 2
        c = 2.0 * math.asin(math.sqrt(min(1.0, a)))
        d_km = 6371.0 * c
        transit_hours = (d_km / 850.0) + 1.5 if d_km > 200.0 else 0.5
        self.travel_transit_until = sim_time + transit_hours * 3600.0

    def trigger_unauthorized_alert(self, sim_time: float, rng: np.random.Generator) -> None:
        """Schedules cardholder fraud discovery based on calibrated vigilance survival distributions."""
        if self.is_frozen or self.unauthorized_alert_time >= 0.0:
            return

        tier_choice = rng.choice([1, 2, 3, 4], p=self.vigilance_weights)
        if tier_choice == 1:
            delay_sec = float(rng.lognormal(mean=5.2, sigma=0.85))
        elif tier_choice == 2:
            scale_sec = 36.0 * 3600.0
            delay_sec = float(scale_sec * ((-math.log(max(1e-6, rng.uniform(0.0, 1.0)))) ** (1.0 / 1.8)))
        elif tier_choice == 3:
            delay_sec = float(rng.uniform(3.0 * 86400.0, 45.0 * 86400.0))
        else:
            delay_sec = float(1e9)

        self.unauthorized_alert_time = sim_time + delay_sec
        self.state = CardholderState.ALERTED

    def check_freeze_status(self, sim_time: float) -> bool:
        """Checks whether notification delay elapsed and card should be permanently frozen."""
        if self.is_frozen:
            return True
        if self.unauthorized_alert_time >= 0.0 and sim_time >= self.unauthorized_alert_time:
            self.is_frozen = True
            self.state = CardholderState.FROZEN
            return True
        return False


# After an ISO 51 (insufficient funds) decline the adversary retries at 70 percent of the
# last amount. The floor stops that decay from reaching absurdly small amounts, but only for
# amounts that were above it: a micro-probe below the floor keeps decaying, so a decline never
# raises the next attempt. The floor is in the card's own currency, as the amounts are.
INSUFFICIENT_FUNDS_DECAY = 0.70
INSUFFICIENT_FUNDS_FLOOR = 25.0


def decay_after_insufficient_funds(previous: float, floor: float = INSUFFICIENT_FUNDS_FLOOR) -> float:
    """The next attempt's amount after an ISO 51 decline of ``previous``."""
    decayed = round(previous * INSUFFICIENT_FUNDS_DECAY, 2)
    return max(decayed, floor) if previous > floor else decayed


@dataclass
class TargetCardAdversaryState:
    """Per-target adversary memory tracking discovered state and campaign lifecycle."""
    card_id: str
    fsm_state: FraudsterState = FraudsterState.DUMP_INGESTION
    current_probe_amount: float = 450.0
    consecutive_declines: int = 0
    last_response_code: Optional[ISO8583Response] = None
    target_mcc: int = 5732
    active_merchant_tier: str = "TIER_A"
    cooldown_until_sec: float = 0.0
    playbook_step: int = 0
    is_compromised: bool = True
    is_burned: bool = False
    is_dormant_until: float = 0.0
    discovered_limits: Dict[str, float] = field(default_factory=dict)
    usd_rate: float = 1.0  # card-currency units per US dollar; the intent optimiser reasons in USD


# The optimiser's lattice names two channels the engine does not: a tap with a provisioned token
# is a contactless payment, and a wallet provisioning runs in the wallet app.
ENGINE_CHANNEL_NAMES = {"CP_CONTACTLESS_NFC": "CP_POS_CONTACTLESS", "PROVISION_DIGITAL_WALLET": "CNP_MOBILE"}


class AdaptiveFraudsterAgent:
    """Stateful adversarial agent executing closed-loop adaptive cybercrime playbooks with per-target memory."""

    def __init__(self, rng: np.random.Generator, adversary_mimicry: float = 0.55, optimizer_seed: int = 42):
        self.rng = rng
        self.adversary_mimicry = adversary_mimicry
        self.target_states: Dict[str, TargetCardAdversaryState] = {}
        # Slice 11: First-Principles Intent Optimization Engine
        self.intent_optimizer = InformationDirectedOptimizer(seed=optimizer_seed)
        self.belief_states: Dict[str, AnalyticalBeliefState] = {}
        self.dossiers: Dict[str, CredentialDossier] = {}
        self.telemetry = load_all_specs().telemetry
        # Global fallback trackers
        self.state = FraudsterState.DUMP_INGESTION
        self.current_amount = 450.0
        self.attack_interval_sec = 2.0
        self.consecutive_declines = 0
        self.active_merchant_tier = "TIER_A"
        self.cooldown_until_sec = 0.0

    def get_or_create_target(self, card: CardholderProfile) -> TargetCardAdversaryState:
        if card.card_id not in self.target_states:
            initial_amt = (
                float(self.rng.uniform(250.0, 650.0))
                if card.currency == "USD"
                else float(self.rng.uniform(20000.0, 50000.0))
            )
            self.target_states[card.card_id] = TargetCardAdversaryState(
                card_id=card.card_id,
                fsm_state=FraudsterState.DUMP_INGESTION,
                current_probe_amount=initial_amt,
                target_mcc=5732,
                usd_rate=self.telemetry.usd_rate(card.currency),
            )
        # Slice 11: Initialize Analytical Belief State and Dossier (belief amounts in USD, spec/07 section 3)
        if card.card_id not in self.belief_states:
            rate = self.telemetry.usd_rate(card.currency)
            avail = max(500.0, float(card.credit_limit - card.current_balance))
            self.belief_states[card.card_id] = AnalyticalBeliefState(
                card_id=card.card_id,
                p_valid=0.50,
                mu_balance=avail / rate,
                sigma_balance=max(100.0, float(card.credit_limit * 0.2)) / rate,
                min_balance=0.0,
                max_balance=float(card.credit_limit) / rate,
                acquisition_cost_usd=15.0,
            )
            # Empirical compromise mix from threat intelligence (e-commerce CNP fullz, mobile wallet tokens, track 2)
            rand_tier = float(self.rng.random())
            has_live_otp = bool(card.region == "IN" and self.rng.random() < self.telemetry.indian_intent_live_otp_probability)

            if rand_tier < 0.15:
                # Tokenized mobile wallet (Apple Pay / Google Pay yellow-path provisioned token)
                tier = CredentialTier.TIER_APP_DEVICE_TOKEN
                has_bound = True
                has_chip = False
            elif rand_tier < 0.28:
                # Track 2 magnetic stripe / physical counterfeit fallback clone
                tier = CredentialTier.TIER_TRACK_2_DUMP
                has_bound = False
                has_chip = True
            elif has_live_otp:
                tier = CredentialTier.TIER_PHISHED_OTP_STREAM
                has_bound = False
                has_chip = False
            else:
                tier = CredentialTier.TIER_CNP_FULLZ
                has_bound = False
                has_chip = False

            self.dossiers[card.card_id] = CredentialDossier(
                tier=tier,
                pan=card.card_id,
                expiry_month=12,
                expiry_year=2028,
                cvv2="123",
                billing_zip=getattr(card, "zip_code", "94105"),
                live_otp="000000" if has_live_otp else None,
                has_live_otp=has_live_otp,
                has_bound_token=has_bound,
                has_chip_cryptogram=has_chip,
            )
        return self.target_states[card.card_id]

    def _attack_channel_preference(self, key: str, table: str = "by_dossier_tier") -> Optional[str]:
        """spec/07 section 18: the channel an attacker draws for this attempt from its tier's or playbook's mix."""
        mix = self.telemetry.attack_channel_mix.get(table, {}).get(key)
        if not mix:
            return None
        names = list(mix.keys())
        weights = np.array([mix[n] for n in names], dtype=float)
        return str(self.rng.choice(names, p=weights / weights.sum()))

    def _uses_victim_device(self, *, tier: Optional[str] = None, playbook: Optional[str] = None) -> bool:
        """spec/07 section 17: whether this attempt runs on the cardholder's own device and connection."""
        spec = self.telemetry.device_sharing.get("victim_device_fraud_share", {})
        share = None
        if tier is not None:
            share = spec.get("by_dossier_tier", {}).get(tier)
        elif playbook is not None:
            share = spec.get("by_playbook", {}).get(playbook)
        if share is None:
            share = spec.get("default", 0.0)
        return bool(self.rng.random() < float(share))

    def finish_playbook_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Playbook mode: a playbook written for the web draws its card-not-present channel from the
        spec/07 section 18 mix, and decides whether it runs on the victim's device (section 17)."""
        name = str(params.get("playbook_name", params.get("scenario_tag", "")))
        if params.get("channel_type") == "CNP_WEB":
            preferred = self._attack_channel_preference(name, "by_playbook") or self._attack_channel_preference("default_cnp", "by_playbook")
            if preferred:
                params["channel_type"] = preferred
        params["victim_device"] = self._uses_victim_device(playbook=name)
        return params

    def _otp_theft_cashout_mcc(self) -> int:
        """spec/07 section 13: where an attacker holding a live OTP converts the card to cash."""
        mix = self.telemetry.otp_theft_cashout_mccs
        if not mix:
            return 6051
        mccs = list(mix.keys())
        weights = np.array([mix[m] for m in mccs], dtype=float)
        return int(self.rng.choice(mccs, p=weights / weights.sum()))

    def _network_risk_score(self, outcome: str) -> int:
        """Draws a network risk score (0 to 99) from the spec/07 Beta shape for the outcome."""
        alpha, beta = self.telemetry.network_risk_score_shapes[outcome]
        return int(min(99, math.floor(self.rng.beta(alpha, beta) * 100.0)))

    def is_card_burned(self, card_id: str) -> bool:
        """Returns True if the card is burned and discarded by adversary."""
        target = self.target_states.get(card_id)
        return target.is_burned if target else False

    def select_attack_playbook(
        self,
        card: CardholderProfile,
        sim_time_seconds: float,
        world_center_lat: float,
        world_center_lon: float,
        scenario_override: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Generates attack transaction parameters guided by per-target memory and stateful adaptation."""
        target = self.get_or_create_target(card)
        is_mimicry = bool(self.rng.random() < self.adversary_mimicry)
        stealth_ip = float(self.rng.uniform(2.5, 32.0))

        if scenario_override is not None:
            chosen_scenario = scenario_override
        else:
            if target.fsm_state == FraudsterState.MICRO_PROBING:
                chosen_scenario = FraudScenario.ADV_MICRO_AUTH_PROBE.value
            elif target.fsm_state == FraudsterState.ACTIVE_CASHOUT:
                chosen_scenario = FraudScenario.ADV_ATO_SILENT_BAKING.value
            elif target.fsm_state == FraudsterState.GATEWAY_HOP:
                chosen_scenario = FraudScenario.ADV_MICRO_AUTH_PROBE.value
            elif target.fsm_state == FraudsterState.AMOUNT_ADAPTATION:
                chosen_scenario = FraudScenario.ADV_ATO_SILENT_BAKING.value
            else:
                if card.region == "IN":
                    playbook_choices = [
                        FraudScenario.ADV_MICRO_AUTH_PROBE.value,
                        FraudScenario.IN_ADV_REVERSE_PROXY_VISHING.value,
                        FraudScenario.IN_ADV_APK_SMS_STEALER.value,
                        FraudScenario.IN_ADV_INTL_NON_3DS_BYPASS.value,
                        FraudScenario.IN_ADV_RENT_PORTAL_CASHOUT.value,
                        FraudScenario.ADV_APPLE_PAY_YELLOW_PATH.value,
                        FraudScenario.ADV_NOCTURNAL_BURST.value,
                    ]
                    playbook_weights = [0.25, 0.20, 0.15, 0.15, 0.10, 0.08, 0.07]
                else:
                    playbook_choices = [
                        FraudScenario.ADV_MICRO_AUTH_PROBE.value,
                        FraudScenario.ADV_ATO_SILENT_BAKING.value,
                        FraudScenario.ADV_SLEEPER_BUST_OUT.value,
                        FraudScenario.ADV_APPLE_PAY_YELLOW_PATH.value,
                        FraudScenario.ADV_NOCTURNAL_BURST.value,
                        FraudScenario.ADV_DISTRIBUTED_BIN_ENUMERATION.value,
                        FraudScenario.ADV_TRIANGULATION_FRAUD.value,
                    ]
                    playbook_weights = [0.35, 0.25, 0.05, 0.15, 0.08, 0.06, 0.06]
                chosen_scenario = str(self.rng.choice(playbook_choices, p=playbook_weights))

        scenario_val = chosen_scenario.value if hasattr(chosen_scenario, "value") else str(chosen_scenario)

        # 1. Micro-Auth Probe (Global)
        if scenario_val in (FraudScenario.ADV_MICRO_AUTH_PROBE.value, "VELOCITY_BLITZ", "CARD_TESTING_BURST"):
            target.fsm_state = FraudsterState.MICRO_PROBING
            amount = round(float(self.rng.uniform(3.50, 18.50) if is_mimicry else self.rng.uniform(0.50, 1.99)) * target.usd_rate, 2)
            target.current_probe_amount = amount
            target.target_mcc = 8398
            ip_dist = stealth_ip if is_mimicry else float(self.rng.uniform(250.0, 1500.0))
            is_cross_border = False
            if card.region == "IN":
                # spec/07 section 3: Indian cards tested at foreign merchants if international enabled;
                # else domestic probe attempted without OTP fails AFA with ISO 63
                is_cross_border = bool(card.international_enabled)
            return {
                "amount": amount,
                "channel_type": "CNP_WEB",
                "is_fraud": 1,
                "scenario_tag": FraudScenario.ADV_MICRO_AUTH_PROBE.value,
                "ip_distance_km": ip_dist,
                "preferred_mcc": 8398,
                "is_cross_border": is_cross_border,
                "avs_code": "Z",
                "asn_type": "residential",
            }

        # 2. Account Takeover Silent Baking (Global)
        elif scenario_val in (FraudScenario.ADV_ATO_SILENT_BAKING.value, "ACCOUNT_TAKEOVER", "CNP_FULLZ_EXPLOIT"):
            if target.fsm_state == FraudsterState.AMOUNT_ADAPTATION:
                amount = round(target.current_probe_amount, 2)
            else:
                if is_mimicry:
                    amount = (
                        round(float(self.rng.uniform(45.0, 320.0)), 2)
                        if card.currency == "USD"
                        else round(float(self.rng.uniform(2500.0, 18000.0)), 2)
                    )
                else:
                    if card.currency == "INR":
                        low_b = min(card.credit_limit * 0.15, 3000.0)
                        high_b = max(low_b + 500.0, min(card.credit_limit * 0.45, 25000.0))
                    else:
                        low_b = min(card.credit_limit * 0.15, 85.0)
                        high_b = max(low_b + 25.0, min(card.credit_limit * 0.45, 650.0))
                    amount = round(float(self.rng.uniform(low_b, high_b)), 2)
                target.current_probe_amount = amount
            target.target_mcc = card.sample_preferred_mcc(self.rng) if is_mimicry else 5732  # spec/07 section 14
            return {
                "amount": amount,
                "channel_type": "CNP_WEB",
                "is_fraud": 1,
                "scenario_tag": FraudScenario.ADV_ATO_SILENT_BAKING.value,
                "ip_distance_km": float(self.rng.uniform(5.0, 25.0)),
                "preferred_mcc": target.target_mcc,
                "is_cross_border": False,
                "asn_type": "residential",
            }

        # 3. Sleeper Bust-Out (Global & India)
        elif scenario_val in (FraudScenario.ADV_SLEEPER_BUST_OUT.value, "SLEEPER_BUST_OUT"):
            remaining = max(100.0, card.credit_limit - card.current_balance)
            if is_mimicry:
                amount = (
                    round(float(self.rng.uniform(120.0, 480.0)), 2)
                    if card.currency == "USD"
                    else round(float(self.rng.uniform(8500.0, 35000.0)), 2)
                )
            else:
                is_collusive_terminal = bool(self.rng.random() < 0.40)
                threshold = 2000000.0 if card.currency == "INR" else 25000.0
                if is_collusive_terminal or remaining <= threshold:
                    amount = round(float(self.rng.uniform(0.90, 0.98) * remaining), 2)
                else:
                    if card.currency == "INR":
                        split_target = min(remaining * 0.95, float(self.rng.uniform(1000000.0, 2000000.0)))
                    else:
                        split_target = min(remaining * 0.95, float(self.rng.uniform(12000.0, 24500.0)))
                    amount = round(split_target, 2)
            target.current_probe_amount = amount
            target.target_mcc = 5094
            ip_dist = float(self.rng.uniform(4.0, 28.0)) if is_mimicry else float(self.rng.uniform(8.0, 48.0))
            return {
                "amount": amount,
                "channel_type": "CNP_WEB",
                "is_fraud": 1,
                "scenario_tag": FraudScenario.ADV_SLEEPER_BUST_OUT.value,
                "ip_distance_km": ip_dist,
                "preferred_mcc": 5094,
                "is_cross_border": False,
                "asn_type": "residential",
            }

        # 4. Apple Pay Yellow Path (Global)
        elif scenario_val in (FraudScenario.ADV_APPLE_PAY_YELLOW_PATH.value, "COUNTERFEIT_CLONE"):
            amount = round(float(self.rng.uniform(45.0, 250.0) if is_mimicry else self.rng.uniform(500.0, 2500.0)) * target.usd_rate, 2)
            target.current_probe_amount = amount
            target.target_mcc = 5732
            ip_dist = float(self.rng.uniform(3.5, 25.0)) if is_mimicry else float(self.rng.uniform(6.0, 35.0))
            return {
                "amount": amount,
                "channel_type": "CP_POS_CONTACTLESS",
                "is_fraud": 1,
                "scenario_tag": FraudScenario.ADV_APPLE_PAY_YELLOW_PATH.value,
                "ip_distance_km": ip_dist,
                "preferred_mcc": 5732,
                "is_cross_border": False,
                "override_lat": card.home_lat + float(self.rng.uniform(-0.05, 0.05)),
                "override_lon": card.home_lon + float(self.rng.uniform(-0.05, 0.05)),
                "asn_type": "residential",
            }

        # 5. Nocturnal Carding Burst (Global & India)
        elif scenario_val == FraudScenario.ADV_NOCTURNAL_BURST.value:
            amount = round(float(self.rng.uniform(150.0, 650.0)) * target.usd_rate, 2)
            target.current_probe_amount = amount
            target.target_mcc = 5311
            if is_mimicry:
                ip_dist = float(self.rng.uniform(12.0, 55.0))
                is_cb = False
                asn = str(self.rng.choice(["residential", "mobile"], p=[0.70, 0.30]))
            else:
                ip_dist = float(self.rng.uniform(4500.0, 11500.0))
                is_cb = True
                asn = str(self.rng.choice(["datacenter", "residential", "mobile"], p=[0.40, 0.40, 0.20]))
            return {
                "amount": amount,
                "channel_type": "CNP_WEB",
                "is_fraud": 1,
                "scenario_tag": FraudScenario.ADV_NOCTURNAL_BURST.value,
                "ip_distance_km": ip_dist,
                "preferred_mcc": 5311,
                "is_cross_border": is_cb,
                "asn_type": asn,
            }

        # 6. Distributed BIN Enumeration (PEA Additive Probing)
        elif scenario_val == FraudScenario.ADV_DISTRIBUTED_BIN_ENUMERATION.value:
            amount = round(float(self.rng.uniform(4.50, 24.50) if is_mimicry else self.rng.uniform(1.00, 3.50)) * target.usd_rate, 2)
            target.current_probe_amount = amount
            target.target_mcc = 8398
            ip_dist = stealth_ip if is_mimicry else float(self.rng.uniform(180.0, 2800.0))
            return {
                "amount": amount,
                "channel_type": "CNP_WEB",
                "is_fraud": 1,
                "scenario_tag": FraudScenario.ADV_DISTRIBUTED_BIN_ENUMERATION.value,
                "ip_distance_km": ip_dist,
                "preferred_mcc": 8398,
                "is_cross_border": False,
                "asn_type": "residential",
                "vaai_score": self._network_risk_score("bin_enumeration"),
            }

        # 7. Triangulation Fraud
        elif scenario_val == FraudScenario.ADV_TRIANGULATION_FRAUD.value:
            amount = round(float(self.rng.uniform(35.0, 195.0)), 2) if is_mimicry else round(float(self.rng.uniform(250.0, 650.0)), 2)
            target.current_probe_amount = amount
            target.target_mcc = 5732
            ip_dist = stealth_ip if is_mimicry else float(self.rng.uniform(250.0, 3500.0))
            return {
                "amount": amount,
                "channel_type": "CNP_WEB",
                "is_fraud": 1,
                "scenario_tag": FraudScenario.ADV_TRIANGULATION_FRAUD.value,
                "ip_distance_km": ip_dist,
                "preferred_mcc": 5732,
                "is_cross_border": False,
                "asn_type": "residential",
            }

        # 8. India: Reverse-Proxy Vishing & Digital Arrest
        elif scenario_val == FraudScenario.IN_ADV_REVERSE_PROXY_VISHING.value:
            avail = max(1500.0, card.get_available_balance())
            if is_mimicry:
                probe_ratio = float(self.rng.choice([0.04, 0.08, 0.15]))
                amount = round(float(np.clip(probe_ratio * avail, 1200.0, 15000.0)), 2)
                ip_dist = stealth_ip
            else:
                probe_ratio = float(self.rng.choice([0.06, 0.15, 0.30, 0.65]))
                amount = round(float(np.clip(probe_ratio * avail, 1500.0, 35000.0)), 2)
                ip_dist = float(self.rng.uniform(120.0, 1850.0))
            target.current_probe_amount = amount
            target.target_mcc = card.sample_preferred_mcc(self.rng) if is_mimicry else self._otp_theft_cashout_mcc()
            return {
                "amount": amount,
                "channel_type": "CNP_WEB",
                "is_fraud": 1,
                "scenario_tag": FraudScenario.IN_ADV_REVERSE_PROXY_VISHING.value,
                "ip_distance_km": ip_dist,
                "preferred_mcc": target.target_mcc,
                "is_cross_border": False,
                "otp_submitted": True,
                "asn_type": "residential",
                "cvv_match_flag": 1,
            }

        # 9. India: Malicious APK SMS Stealer
        elif scenario_val == FraudScenario.IN_ADV_APK_SMS_STEALER.value:
            avail = max(1200.0, card.get_available_balance())
            if is_mimicry:
                probe_ratio = float(self.rng.choice([0.03, 0.07, 0.12]))
                amount = round(float(np.clip(probe_ratio * avail, 800.0, 10000.0)), 2)
                ip_dist = stealth_ip
            else:
                probe_ratio = float(self.rng.choice([0.05, 0.12, 0.25, 0.50]))
                amount = round(float(np.clip(probe_ratio * avail, 1000.0, 30000.0)), 2)
                ip_dist = float(self.rng.uniform(45.0, 950.0))
            target.current_probe_amount = amount
            target.target_mcc = card.sample_preferred_mcc(self.rng) if is_mimicry else self._otp_theft_cashout_mcc()
            return {
                "amount": amount,
                "channel_type": "CNP_MOBILE",
                "is_fraud": 1,
                "scenario_tag": FraudScenario.IN_ADV_APK_SMS_STEALER.value,
                "ip_distance_km": ip_dist,
                "preferred_mcc": target.target_mcc,
                "is_cross_border": False,
                "otp_submitted": True,
                "asn_type": "mobile",
                "cvv_match_flag": 1,
            }

        # 10. India: International Non-3DS Bypass
        elif scenario_val == FraudScenario.IN_ADV_INTL_NON_3DS_BYPASS.value:
            if is_mimicry:
                amount_usd = round(float(self.rng.uniform(15.0, 120.0)), 2)
                ip_dist = float(self.rng.uniform(25.0, 150.0))
                asn = str(self.rng.choice(["residential", "mobile"], p=[0.60, 0.40]))
            else:
                amount_usd = round(float(self.rng.uniform(25.0, 350.0)), 2)
                ip_dist = float(self.rng.uniform(5500.0, 12000.0))
                asn = str(self.rng.choice(["datacenter", "residential", "mobile"], p=[0.35, 0.45, 0.20]))
            amount_inr = round(amount_usd * 83.5, 2)
            target.current_probe_amount = amount_inr
            target.target_mcc = 5732
            return {
                "amount": amount_inr,
                "channel_type": "CNP_WEB",
                "is_fraud": 1,
                "scenario_tag": FraudScenario.IN_ADV_INTL_NON_3DS_BYPASS.value,
                "ip_distance_km": ip_dist,
                "preferred_mcc": 5732,
                "is_cross_border": True,
                "eci": "07",
                "asn_type": asn,
                "cvv_match_flag": 1,
            }

        # 11. India: Rent Portal Liquidation
        elif scenario_val == FraudScenario.IN_ADV_RENT_PORTAL_CASHOUT.value:
            remaining = max(2000.0, card.credit_limit - card.current_balance)
            if is_mimicry:
                probe_ratio = float(self.rng.choice([0.08, 0.15, 0.25]))
                amount = round(float(np.clip(probe_ratio * remaining, 1500.0, 18000.0)), 2)
                ip_dist = stealth_ip
            else:
                probe_ratio = float(self.rng.choice([0.15, 0.35, 0.65, 0.85]))
                amount = round(float(np.clip(probe_ratio * remaining, 2000.0, 75000.0)), 2)
                ip_dist = float(self.rng.uniform(25.0, 450.0))
            target.current_probe_amount = amount
            target.target_mcc = 6513
            return {
                "amount": amount,
                "channel_type": "CNP_WEB",
                "is_fraud": 1,
                "scenario_tag": FraudScenario.IN_ADV_RENT_PORTAL_CASHOUT.value,
                "ip_distance_km": ip_dist,
                "preferred_mcc": 6513,
                "is_cross_border": False,
                "otp_submitted": True,
                "asn_type": "residential",
                "cvv_match_flag": 1,
            }

        # Fallback
        target.current_probe_amount = 250.0
        return {
            "amount": 250.0,
            "channel_type": "CNP_WEB",
            "is_fraud": 1,
            "scenario_tag": scenario_val,
            "ip_distance_km": stealth_ip if is_mimicry else float(self.rng.uniform(15.0, 85.0)),
        }

    def receive_feedback(
        self,
        response_code: ISO8583Response,
        trans_status_3ds: Optional[str],
        sim_time_seconds: float,
        card_id: Optional[str] = None,
    ) -> None:
        """Adapts adversarial policy with per-target memory and stateful closed-loop updates."""
        target = self.target_states.get(card_id) if card_id else None

        # Slice 11: Information-Directed Belief State Update
        belief = self.belief_states.get(card_id) if card_id else None
        if belief:
            resp_str = response_code.value if hasattr(response_code, "value") else str(response_code)
            self.intent_optimizer.observe_outcome(
                belief=belief,
                amount=(target.current_probe_amount / target.usd_rate) if target else 100.0,
                response_code=resp_str,
            )

        if response_code in (ISO8583Response.APPROVED_00, ISO8583Response.PARTIAL_APPROVAL_10):
            self.consecutive_declines = 0
            if target:
                target.consecutive_declines = 0
                target.last_response_code = response_code
                if target.fsm_state == FraudsterState.MICRO_PROBING:
                    target.fsm_state = FraudsterState.ACTIVE_CASHOUT
                    target.current_probe_amount = float(self.rng.uniform(450.0, 1600.0)) * target.usd_rate
                else:
                    target.fsm_state = FraudsterState.SUCCESS_HARVEST
            if self.state == FraudsterState.MICRO_PROBING:
                self.state = FraudsterState.ACTIVE_CASHOUT
                self.current_amount = float(self.rng.uniform(450.0, 1600.0))
            else:
                self.state = FraudsterState.SUCCESS_HARVEST
                self.current_amount = max(50.0, self.current_amount * 0.85)

        elif response_code == ISO8583Response.INSUFFICIENT_FUNDS_51:
            if target:
                target.consecutive_declines += 1
                target.fsm_state = FraudsterState.AMOUNT_ADAPTATION
                target.last_response_code = response_code
                target.current_probe_amount = decay_after_insufficient_funds(target.current_probe_amount)
                target.cooldown_until_sec = sim_time_seconds + 30.0
            self.consecutive_declines += 1
            self.state = FraudsterState.AMOUNT_ADAPTATION
            self.current_amount = decay_after_insufficient_funds(self.current_amount)

        elif trans_status_3ds == "C":
            if target:
                target.fsm_state = FraudsterState.GATEWAY_HOP
                target.last_response_code = response_code
                target.target_mcc = 5815
                target.current_probe_amount = min(target.current_probe_amount, 28.0 * target.usd_rate)
            self.state = FraudsterState.GATEWAY_HOP
            self.active_merchant_tier = "TIER_C"

        elif response_code in (ISO8583Response.SUSPECTED_FRAUD_59, ISO8583Response.DO_NOT_HONOR_05):
            self.consecutive_declines += 1
            self.attack_interval_sec *= 2.5
            self.cooldown_until_sec = sim_time_seconds + self.attack_interval_sec
            self.state = FraudsterState.VELOCITY_BACKOFF
            if target:
                target.consecutive_declines += 1
                target.last_response_code = response_code
                if target.consecutive_declines >= 2:
                    target.is_burned = True
                    target.fsm_state = FraudsterState.CARD_PURGE
                else:
                    target.fsm_state = FraudsterState.VELOCITY_BACKOFF
                    target.cooldown_until_sec = sim_time_seconds + 300.0

        elif response_code in (ISO8583Response.INVALID_CARD_14, ISO8583Response.EXPIRED_CARD_54):
            self.state = FraudsterState.CARD_PURGE
            if target:
                target.is_burned = True
                target.fsm_state = FraudsterState.CARD_PURGE

    def select_intent_action(
        self,
        card: CardholderProfile,
        sim_time_seconds: float,
        world_center_lat: float,
        world_center_lon: float,
    ) -> Dict[str, Any]:
        """Emergent attack generation driven by first-principles intent and physical constraints (Slice 11)."""
        target = self.get_or_create_target(card)
        belief = self.belief_states[card.card_id]
        dossier = self.dossiers[card.card_id]
        current_hour = int((sim_time_seconds % 86400) // 3600)

        action = self.intent_optimizer.select_optimal_action(
            belief=belief,
            dossier=dossier,
            current_hour_local=current_hour,
            preferred_channel=self._attack_channel_preference(dossier.tier.value),
        )
        # spec/07 section 3: the optimiser's ladder is in USD; the authorisation carries the card's currency.
        is_probe = action.macro_option == MacroOptionType.OMEGA_PROBE
        amount_usd = float(action.amount)
        if amount_usd > 0.0:
            if is_probe:
                if self.rng.random() < self.adversary_mimicry:
                    low, high = self.telemetry.mimicry_probe_ticket_range_usd
                else:
                    low, high = self.telemetry.micro_probe_ticket_range_usd
                amount_usd = float(self.rng.uniform(low, high))
            elif self.rng.random() < self.adversary_mimicry:
                # Mimicking cashout camouflaged to cardholder's typical spend ticket
                typical_spend_usd = card.sample_spend_amount(self.rng) / target.usd_rate
                amount_usd = float(self.rng.uniform(1.5, 3.2)) * typical_spend_usd
            elif self.telemetry.ladder_jitter_fraction > 0.0:
                jitter = self.telemetry.ladder_jitter_fraction
                amount_usd *= float(self.rng.uniform(1.0 - jitter, 1.0 + jitter))
        amount = round(amount_usd * target.usd_rate, 2)
        target.current_probe_amount = amount
        target.target_mcc = action.mcc
        is_cross_border = bool(
            is_probe
            and card.region == "IN"
            and self.telemetry.route_indian_probes_cross_border
            and card.international_enabled
            and not dossier.has_live_otp
        )

        return {
            "amount": amount,
            "channel_type": ENGINE_CHANNEL_NAMES.get(action.channel, action.channel),
            "victim_device": self._uses_victim_device(tier=dossier.tier.value),
            "is_fraud": 1,
            "scenario_tag": f"INTENT_{action.macro_option.value}",
            "ip_distance_km": float(self.rng.uniform(15.0, 85.0)),
            "preferred_mcc": action.mcc,
            "macro_option": action.macro_option.value,
            "incubation_seconds": action.incubation_seconds,
            "is_cross_border": is_cross_border,
        }

    def inject_attack(
        self,
        scenario: Any,
        card: CardholderProfile,
        sim_time_seconds: float,
        world_center_lat: float,
        world_center_lon: float,
    ) -> Dict[str, Any]:
        """Backward-compatible inject_attack interface."""
        scen_str = scenario.value if hasattr(scenario, "value") else str(scenario)
        return self.select_attack_playbook(
            card=card,
            sim_time_seconds=sim_time_seconds,
            world_center_lat=world_center_lat,
            world_center_lon=world_center_lon,
            scenario_override=scen_str,
        )


FraudsterAgent = AdaptiveFraudsterAgent


class BankDecisionEngine:
    """Multi-tier issuer authorization switch implementing real-world ISO 8583 and regulatory rails."""

    def __init__(
        self,
        tau_decline: float = 0.88,
        tau_challenge: float = 0.45,
        max_speed_kmh: float = 900.0,
        max_hourly_velocity: int = 15,
        region: str = "US",
    ):
        self.tau_decline = tau_decline
        self.tau_challenge = tau_challenge
        self.max_speed_kmh = max_speed_kmh
        self.max_hourly_velocity = max_hourly_velocity
        self.region = region.upper()
        self._switches: Dict[str, Any] = {}

    def evaluate_authorization(
        self,
        card: CardholderProfile,
        amount: float,
        channel: str,
        sim_time: float,
        velocity_kph: float,
        count_1h: int,
        cvv_valid: bool = True,
        is_card_format_valid: bool = True,
        is_expired: bool = False,
        ml_risk_score: float = 0.05,
        mcc: int = 5411,
        is_cross_border: bool = False,
        otp_provided: bool = True,
        vaai_score: int = 20,
    ) -> Tuple[ISO8583Response, Optional[str], Optional[float]]:
        """Evaluates authorization request and returns (ISO_Field_39, 3DS_trans_status, approved_amount)."""
        # 1. Syntax & Card Lifecycle Invariants
        if not is_card_format_valid:
            return ISO8583Response.INVALID_CARD_14, None, 0.0
        if is_expired:
            return ISO8583Response.EXPIRED_CARD_54, None, 0.0

        # Delegate to unified institutional RailVerifierSwitch
        from .rails import CandidateTransactionIntent, RailVerifierSwitch
        reg = card.region or self.region
        if reg not in self._switches:
            self._switches[reg] = RailVerifierSwitch(
                region=reg,
                tau_decline=self.tau_decline,
                tau_challenge_high=self.tau_challenge,
            )
        switch = self._switches[reg]
        intent = CandidateTransactionIntent(
            tx_id=f"TX_LEGACY_{int(sim_time)}",
            card_id=card.card_id,
            sim_time_sec=sim_time,
            amount=amount,
            currency=card.currency,
            channel_type=channel,
            merchant_id=f"M_{mcc}_LEGACY",
            mcc=mcc,
            merchant_lat=card.home_lat,
            merchant_lon=card.home_lon,
            is_cross_border=is_cross_border,
            otp_submitted=otp_provided,
            cvv_provided=cvv_valid,
            pin_entered=(channel == "CP_POS_CHIP"),
            emv_chip_present=(channel == "CP_POS_CHIP"),
            emv_cryptogram_valid=True,
            three_ds_requested=channel.startswith("CNP"),
            risk_score=ml_risk_score,
            vaai_score=vaai_score,
            haversine_velocity_kph=velocity_kph,
            tx_count_1h=count_1h,
        )
        res = switch.verify_intent(intent, card)
        try:
            code = ISO8583Response(res.iso_response_code)
        except ValueError:
            code = ISO8583Response.DO_NOT_HONOR_05

        return code, res.trans_status_3ds, (res.approved_amount if res.approved else 0.0)
