"""Runtime Specification Loader and Schema Validator for Prism Generator.

Loads, validates, and caches canonical banking specifications from spec/:
- 01_financial_instruments.yaml
- 02_human_personas.yaml
- 03_payment_rail_gaps.yaml
- 04_adversarial_playbooks.yaml
- 05_india_payment_rails.yaml
- 07_export_leakage_gate.yaml
- 08_india_calibration_targets.yaml
"""

from __future__ import annotations

import functools
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml


def _find_spec_dir() -> Path:
    """Locates the canonical spec/ directory with fallback search."""
    env_dir = os.environ.get("PRISM_GENERATOR_SPEC_DIR")
    if env_dir and Path(env_dir).is_dir():
        return Path(env_dir)

    # Relative to this file: prism_generator/ -> root -> spec/
    candidate = Path(__file__).resolve().parent.parent / "spec"
    if candidate.is_dir():
        return candidate

    # Search current working directory ancestors
    curr = Path.cwd()
    for p in [curr, curr.parent, curr.parent.parent]:
        candidate = p / "spec"
        if candidate.is_dir() and (candidate / "01_financial_instruments.yaml").exists():
            return candidate

    raise FileNotFoundError("Canonical spec/ directory could not be located.")


# ----------------------------------------------------------------------
# 1. Financial Instruments Models (spec/01)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class CardProductSpec:
    id: str
    name: str
    category: str
    regulatory_framework: str
    network_product_codes: List[str]
    credit_limit_min_usd: float
    credit_limit_median_usd: float
    credit_limit_max_usd: float
    daily_spend_limit_usd: float
    daily_velocity_max_tx: int
    stip_floor_limit_usd: float
    stip_max_consecutive_approvals: int
    cash_advance_permitted: bool
    cross_border_permitted: bool
    zero_liability_eligible: bool
    primary_adversarial_vulnerabilities: List[str]


# ----------------------------------------------------------------------
# 2. Human Persona & Cohort Models (spec/02)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SpendDistributionSpec:
    model: str  # "LogNormal" or "Spliced_LogNormal_GPD"
    mu_log: float
    sigma_log: float
    mean_cents: float
    threshold_u_cents: Optional[float] = None
    gpd_xi: Optional[float] = None
    gpd_beta: Optional[float] = None
    tail_prob: Optional[float] = None


@dataclass(frozen=True)
class VigilanceTierWeights:
    tier_1_sms_push: float
    tier_2_app_checker: float
    tier_3_statement: float
    tier_4_passive: float

    def as_simplex_array(self) -> List[float]:
        return [self.tier_1_sms_push, self.tier_2_app_checker, self.tier_3_statement, self.tier_4_passive]


@dataclass(frozen=True)
class CohortPersonaSpec:
    id: str
    name: str
    population_weight: float
    annual_income_usd: Tuple[float, float]
    monthly_tx_volume_mean: float
    monthly_tx_volume_std: float
    spend_distribution: SpendDistributionSpec
    primary_channels: Dict[str, float]
    dominant_mccs: List[int]
    default_assigned_products: List[str]
    vigilance_weights: VigilanceTierWeights
    hawkes_dynamics: Optional[Dict[str, float]] = None
    annual_income_inr: Optional[Tuple[float, float]] = None


# ----------------------------------------------------------------------
# 3. Payment Rail Gaps & Plumbing Models (spec/03)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class StipConfigSpec:
    timeout_threshold_ms: int
    subprime_floor_usd: float
    rewards_floor_usd: float
    ultra_premium_floor_usd: float
    max_consecutive_offline_approvals: int
    fourth_consecutive_action: str


@dataclass(frozen=True)
class AvsCodeSpec:
    match: str
    street: Optional[bool]
    zip5: Optional[bool]
    risk: str


@dataclass(frozen=True)
class ScaExemptionsSpec:
    low_value_max_single_eur: float
    low_value_cumulative_eur: float
    low_value_consecutive_tx: int
    tra_tiers: List[Dict[str, float]]


# ----------------------------------------------------------------------
# 4. Adversarial Playbooks Models (spec/04)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class AdversarialPlaybookSpec:
    id: str
    name: str
    target_products: List[str]
    target_clusters: List[str]
    ticket_size_min_cents: float
    ticket_size_max_cents: float
    target_mccs: List[int]
    action_sequence: Dict[str, Any]
    response_handling: Dict[str, str]


# ----------------------------------------------------------------------
# 5. India Payment Rails Models (spec/05)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class IndiaRegulatoryRailsSpec:
    otp_validity_window_seconds: int
    contactless_pin_free_ceiling_inr: float
    contactless_max_consecutive_pinless: int
    contactless_cumulative_ceiling_inr: float
    e_mandate_standard_ceiling_inr: float
    e_mandate_high_value_ceiling_inr: float
    offline_micropayment_ceiling_inr: float
    upi_lite_per_tx_ceiling_inr: float
    default_enabled_controls: List[str]
    default_disabled_controls: List[str]
    rupay_zero_mdr_ceiling_inr: float
    liability_zero_working_days: int
    provisional_credit_mandate_days: int
    delayed_tat_penalty_per_day_inr: float


@dataclass(frozen=True)
class IndianCardProductSpec:
    id: str
    name: str
    category: str
    credit_limit_min_inr: float
    credit_limit_median_inr: float
    credit_limit_max_inr: float
    primary_adversarial_vulnerabilities: List[str]
    spend_mean_log: float = 7.60
    spend_sigma_log: float = 0.70


# ----------------------------------------------------------------------
# 5b. Operational Supervision Queue Models (spec/16)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class OperationalSupervisionSpec:
    default_k_daily: int = 50
    alert_threshold: float = 0.70
    default_priority_strategy: str = "RISK_SCORE"
    weibull_k: float = 1.35
    weibull_scale_hours: float = 18.0
    min_investigation_hours: float = 0.5
    max_investigation_hours: float = 72.0
    lognormal_mu_days: float = 3.40
    lognormal_sigma: float = 0.45
    min_chargeback_days: float = 3.0
    max_chargeback_days: float = 120.0
    v0_usd: float = 15.0
    v0_inr: float = 1250.0
    dark_smoothness: float = 0.40
    clean_maturity_days: float = 90.0


# ----------------------------------------------------------------------
# 6. Circadian Diurnal & Point-Process Models (spec/02)
# ----------------------------------------------------------------------
def bessel_i0(x: float, max_terms: int = 25) -> float:
    """Modified Bessel function of the first kind of order 0, I_0(x), via Taylor series.

    Converges to machine epsilon (< 1e-15) in <= 25 iterations for kappa <= 6.0
    with zero external C-dependencies (eliminates scipy requirement).
    """
    total = 1.0
    term = 1.0
    x_half_sq = (x * 0.5) ** 2
    for m in range(1, max_terms):
        term *= x_half_sq / (m * m)
        total += term
        if term < 1e-15:
            break
    return total


@dataclass(frozen=True)
class VonMisesComponent:
    name: str
    mu_hour: float
    kappa: float
    weight: float
    i0_kappa: float


@dataclass(frozen=True)
class DiurnalProfileSpec:
    nocturnal_floor_beta_0: float
    components: List[VonMisesComponent]
    phi_max: float

    def evaluate_density(self, hour: float) -> float:
        """Evaluates continuous diurnal relative arrival density phi(t) at given hour in [0, 24)."""
        t = hour % 24.0
        two_pi = 2.0 * math.pi
        comp_sum = 0.0
        for comp in self.components:
            cos_angle = math.cos(two_pi * (t - comp.mu_hour) / 24.0)
            comp_sum += comp.weight * math.exp(comp.kappa * cos_angle) / (24.0 * comp.i0_kappa)
        return self.nocturnal_floor_beta_0 + (24.0 - 24.0 * self.nocturnal_floor_beta_0) * comp_sum


@dataclass(frozen=True)
class CircadianDiurnalSpec:
    mathematical_form: str
    formula: str
    weekday_profile: DiurnalProfileSpec
    weekend_profile: DiurnalProfileSpec

    def evaluate_density(self, hour: float, is_weekend: bool) -> float:
        """Evaluates relative arrival intensity phi(t) given hour and weekend status."""
        profile = self.weekend_profile if is_weekend else self.weekday_profile
        return profile.evaluate_density(hour)

    def sample_next_arrival_nhpp(
        self,
        t_curr_sec: float,
        mean_inter_arrival_sec: float,
        rng: np.random.Generator,
        macro_rate_multiplier: float = 1.0,
        max_rejections: int = 200,
    ) -> float:
        """Samples next arrival timestamp using Lewis-Shedler thinning on the periodic von Mises NHPP."""
        effective_mean_sec = max(1.0, mean_inter_arrival_sec / max(0.01, macro_rate_multiplier))
        lambda_bar = 1.0 / effective_mean_sec
        phi_sup = max(self.weekday_profile.phi_max, self.weekend_profile.phi_max)
        lambda_sup = lambda_bar * phi_sup

        t = t_curr_sec
        for _ in range(max_rejections):
            dt = float(rng.exponential(scale=1.0 / lambda_sup))
            t += dt
            h = (t / 3600.0) % 24.0
            day_idx = int(t // 86400) % 7
            is_weekend = (day_idx in (5, 6))
            phi_val = self.evaluate_density(h, is_weekend)
            p_accept = min(1.0, phi_val / phi_sup)
            if rng.random() <= p_accept:
                return t
        return t


# ----------------------------------------------------------------------
# 7. Authentic Hard Negatives & Life-Stage Episodes (spec/02)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class AuthenticHardNegativeSpec:
    id: str
    name: str
    episodic_window_seconds: Optional[float] = None
    velocity_multiplier: Optional[float] = None
    spend_percentile_target: Optional[str] = None
    characteristic_mccs: List[int] = field(default_factory=list)
    ground_truth_discriminators: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Master Unified Specification Registry
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# 7. Point-in-Time Telemetry Calibration (spec/07)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class TelemetryCalibrationSpec:
    """Class-conditional telemetry parameters that keep the exported views free of label shortcuts."""
    network_risk_score_shapes: Dict[str, Tuple[float, float]]
    network_decline_threshold: int
    reference_rate_inr_per_usd: float
    micro_probe_ticket_range_usd: Tuple[float, float]
    mimicry_probe_ticket_range_usd: Tuple[float, float]
    ladder_jitter_fraction: float
    route_indian_probes_cross_border: bool
    legit_afa_failure_share: Dict[str, float]
    legit_challenge_share: Dict[str, float]
    legit_cross_border_cnp_share: Dict[str, float]
    avs_unavailable_probability_cross_border: float
    fraud_avs_weights: Dict[str, float]
    legit_avs_weights: Dict[str, float]
    validated_card_follow_up_probability: float
    full_record_dossier_tiers: Tuple[str, ...]
    full_record_playbooks: Tuple[str, ...]
    indian_intent_live_otp_probability: float
    legit_uncovered_mcc_share: Dict[int, Dict[str, float]]
    legit_micro_ticket_share: Dict[str, float]
    legit_micro_ticket_range: Dict[str, Tuple[float, float]]
    legit_ip_distance_lognormal: Dict[str, Tuple[float, float]]
    otp_theft_cashout_mccs: Dict[int, float]
    # spec/07 sections 16 to 19
    address_space: Dict[str, Any] = field(default_factory=dict)
    device_sharing: Dict[str, Any] = field(default_factory=dict)
    attack_channel_mix: Dict[str, Any] = field(default_factory=dict)
    prevalence_warm_up_rows: int = 25
    prevalence_window_hours: float = 24.0

    def usd_rate(self, currency: str) -> float:
        """Card-currency units per US dollar."""
        return self.reference_rate_inr_per_usd if currency == "INR" else 1.0


# ----------------------------------------------------------------------
# 8. Annual Seasonality & Macro Drift Models (spec/13)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class AnnualSeasonalitySpec:
    """Canonical 365-day annual seasonality, holiday calendars, and DoW dynamics."""
    day_of_week_multipliers: Dict[int, float]
    payday_cycles: Dict[str, Dict[str, Any]]
    mcc_monthly_scaling: Dict[str, Dict[str, Any]]
    calendar_macro_shocks: Dict[str, List[Dict[str, Any]]]
    wallet_hierarchy: Dict[str, Any]
    adversarial_drift_schedule: Dict[str, Any]
    topology_seasonality: Dict[str, Any] = field(default_factory=dict)
    cross_bank_smurfing: Dict[str, Any] = field(default_factory=dict)



@dataclass(frozen=True)
class InterbankSwitchSpec:
    """Multi-bank routing mesh, central switches, ISO 8583 MTI family, and multilateral net settlement."""
    commercial_banks: Dict[str, List[Dict[str, Any]]]
    central_switching_networks: Dict[str, Dict[str, Any]]
    routing_rules: Dict[str, Dict[str, Any]]
    iso8583_message_types: Dict[str, str]
    clearing_and_settlement: Dict[str, Any]


# ----------------------------------------------------------------------
# 9. Calibration Targets (spec/08)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class CalibrationTargetSpec:
    """One public figure the generated stream is compared with."""
    id: str
    statistic: str
    target: float
    source: str
    gate: bool = False
    tolerance_relative: Optional[float] = None
    tolerance_absolute: Optional[float] = None
    reason_not_gated: str = ""
    # Below this many fraud rows the batch cannot support a verdict on this target, because the
    # statistic's own sampling spread exceeds the tolerance band. None means no limit has been
    # measured, NOT that no limit exists -- see AGENTS.md 4, an unmeasured limit is a hypothesis.
    reliable_from_fraud_rows: Optional[int] = None
    reliability_basis: str = ""


@dataclass(frozen=True)
class CalibrationProfileSpec:
    """A dated set of calibration targets for one region."""
    id: str
    region: str
    as_of: str
    scope: str
    fraud_prevalence: float
    fraud_prevalence_range: Tuple[float, float]
    demo_rate_policy: str
    targets: Tuple[CalibrationTargetSpec, ...]
    out_of_scope: Tuple[Dict[str, str], ...]


class SpecRegistry:
    """Immutable, thread-safe memory container holding validated specifications."""

    def __init__(
        self,
        products: Dict[str, CardProductSpec],
        cohorts: Dict[str, CohortPersonaSpec],
        stip: StipConfigSpec,
        avs: Dict[str, AvsCodeSpec],
        sca: ScaExemptionsSpec,
        playbooks: Dict[str, AdversarialPlaybookSpec],
        india_rails: IndiaRegulatoryRailsSpec,
        indian_products: Dict[str, IndianCardProductSpec],
        circadian_raw: Dict[str, Any],
        raw_specs: Dict[str, Dict[str, Any]],
        circadian: Optional[CircadianDiurnalSpec] = None,
        hard_negatives: Optional[Dict[str, AuthenticHardNegativeSpec]] = None,
        telemetry: Optional[TelemetryCalibrationSpec] = None,
        calibration_profiles: Optional[Dict[str, CalibrationProfileSpec]] = None,
        supervision: Optional[OperationalSupervisionSpec] = None,
        credit_limit_steps: Optional[Dict[str, List[Tuple[Optional[float], float]]]] = None,
        seasonality: Optional[AnnualSeasonalitySpec] = None,
        interbank: Optional[InterbankSwitchSpec] = None,
    ):
        self.products = products
        self.cohorts = cohorts
        self.stip = stip
        self.avs = avs
        self.sca = sca
        self.playbooks = playbooks
        self.india_rails = india_rails
        self.indian_products = indian_products
        self.circadian_raw = circadian_raw
        self.raw_specs = raw_specs
        self.circadian = circadian
        self.hard_negatives = hard_negatives or {}
        self.telemetry = telemetry
        self.calibration_profiles = calibration_profiles or {}
        self.supervision = supervision or OperationalSupervisionSpec()
        # Per region, (upper bound or None, step) in major units: the grid credit lines are assigned on.
        self.credit_limit_steps: Dict[str, List[Tuple[Optional[float], float]]] = credit_limit_steps or {}
        self.seasonality = seasonality
        self.interbank = interbank

    def get_cohort(self, cohort_id: str) -> CohortPersonaSpec:
        """Looks up a cohort specification, handling legacy alias names."""
        if cohort_id == "C7_HNI_LUXURY_TRAVELER":
            cohort_id = "C7_LUXURY_AFFLUENT"
        if cohort_id not in self.cohorts:
            raise KeyError(f"Cohort '{cohort_id}' not found in canonical spec. Available: {list(self.cohorts.keys())}")
        return self.cohorts[cohort_id]

    def get_product(self, product_id: str) -> Union[CardProductSpec, IndianCardProductSpec]:
        """Looks up a card product specification across US and Indian catalogues."""
        if product_id in self.products:
            return self.products[product_id]
        if product_id in self.indian_products:
            return self.indian_products[product_id]
        raise KeyError(f"Product '{product_id}' not found in US or India specs.")

    def get_playbook(self, playbook_id: str) -> AdversarialPlaybookSpec:
        """Looks up an adversarial playbook specification."""
        if playbook_id not in self.playbooks:
            raise KeyError(f"Playbook '{playbook_id}' not found in canonical playbooks.")
        return self.playbooks[playbook_id]


@functools.lru_cache(maxsize=1)
def load_all_specs(spec_dir: Optional[str] = None) -> SpecRegistry:
    """Loads, parses, validates, and caches all specification YAML files."""
    base_dir = Path(spec_dir) if spec_dir else _find_spec_dir()

    def _read_yaml(fname: str) -> Dict[str, Any]:
        p = base_dir / fname
        if not p.exists():
            raise FileNotFoundError(f"Missing canonical spec file: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Spec file {fname} root must be a YAML mapping.")
        return data

    raw_01 = _read_yaml("01_financial_instruments.yaml")
    raw_02 = _read_yaml("02_human_personas.yaml")
    raw_03 = _read_yaml("03_payment_rail_gaps.yaml")
    raw_04 = _read_yaml("04_adversarial_playbooks.yaml")
    raw_05 = _read_yaml("05_india_payment_rails.yaml")
    raw_07 = _read_yaml("07_export_leakage_gate.yaml")
    raw_08 = _read_yaml("08_india_calibration_targets.yaml")
    raw_09 = _read_yaml("09_us_calibration_targets.yaml") if (base_dir / "09_us_calibration_targets.yaml").exists() else {}
    raw_13 = _read_yaml("13_annual_seasonality_and_drift.yaml") if (base_dir / "13_annual_seasonality_and_drift.yaml").exists() else {}
    raw_14 = _read_yaml("14_interbank_switch_and_clearing.yaml") if (base_dir / "14_interbank_switch_and_clearing.yaml").exists() else {}
    raw_16 = _read_yaml("16_operational_supervision.yaml")

    # 1. Parse 01_financial_instruments.yaml
    products: Dict[str, CardProductSpec] = {}
    for p_raw in raw_01.get("card_products", []):
        pid = p_raw["id"]
        cl = p_raw.get("credit_limit_cents", {})
        dd = p_raw.get("demand_deposit_balance_cents", {})
        lim_min = cl.get("min", dd.get("min", 0)) / 100.0
        lim_med = cl.get("median", dd.get("median", 0)) / 100.0
        lim_max = cl.get("max", dd.get("max", 0)) / 100.0

        daily_spend = p_raw.get("daily_spend_limit_cents", p_raw.get("daily_pos_spend_limit_cents", 0)) / 100.0
        stip_floor = p_raw.get("stip_floor_limit_cents", 0) / 100.0

        products[pid] = CardProductSpec(
            id=pid,
            name=p_raw["name"],
            category=p_raw["category"],
            regulatory_framework=p_raw["regulatory_framework"],
            network_product_codes=p_raw.get("network_product_codes", []),
            credit_limit_min_usd=lim_min,
            credit_limit_median_usd=lim_med,
            credit_limit_max_usd=lim_max,
            daily_spend_limit_usd=daily_spend,
            daily_velocity_max_tx=p_raw.get("daily_velocity_max_tx", 10),
            stip_floor_limit_usd=stip_floor,
            stip_max_consecutive_approvals=p_raw.get("stip_max_consecutive_approvals", 3),
            cash_advance_permitted=p_raw.get("cash_advance_permitted", False),
            cross_border_permitted=p_raw.get("cross_border_permitted", False),
            zero_liability_eligible=p_raw.get("zero_liability_eligible", True),
            primary_adversarial_vulnerabilities=p_raw.get("primary_adversarial_vulnerabilities", []),
        )

    # 2. Parse 02_human_personas.yaml
    cohorts: Dict[str, CohortPersonaSpec] = {}
    for c_raw in raw_02.get("consumer_clusters", []):
        cid = c_raw["id"]
        sp_raw = c_raw["spend_ticket_distribution"]
        params = sp_raw.get("params", {})
        spend_spec = SpendDistributionSpec(
            model=sp_raw.get("model", "LogNormal"),
            mu_log=float(params["mu_log"]),
            sigma_log=float(params["sigma_log"]),
            mean_cents=float(sp_raw.get("mean_cents", 0.0)),
            threshold_u_cents=float(params["threshold_u_cents"]) if "threshold_u_cents" in params else None,
            gpd_xi=float(params["gpd_xi"]) if "gpd_xi" in params else None,
            gpd_beta=float(params["gpd_beta"]) if "gpd_beta" in params else None,
            tail_prob=float(params["tail_prob"]) if "tail_prob" in params else None,
        )

        vw_raw = c_raw.get("vigilance_tier_weights", {})
        vigilance = VigilanceTierWeights(
            tier_1_sms_push=float(vw_raw.get("tier_1_sms_push", 0.35)),
            tier_2_app_checker=float(vw_raw.get("tier_2_app_checker", 0.45)),
            tier_3_statement=float(vw_raw.get("tier_3_statement", 0.12)),
            tier_4_passive=float(vw_raw.get("tier_4_passive", 0.08)),
        )

        channels = c_raw.get("primary_channels", {})
        cohorts[cid] = CohortPersonaSpec(
            id=cid,
            name=c_raw["name"],
            population_weight=float(c_raw["population_weight"]),
            annual_income_usd=tuple(c_raw.get("annual_income_usd", [0, 0])),
            annual_income_inr=(tuple(c_raw.get("annual_income_inr")) if c_raw.get("annual_income_inr") else None),
            monthly_tx_volume_mean=float(c_raw["monthly_card_tx_volume"]["mean"]),
            monthly_tx_volume_std=float(c_raw["monthly_card_tx_volume"]["std"]),
            spend_distribution=spend_spec,
            primary_channels=channels,
            dominant_mccs=list(c_raw.get("dominant_mccs", [])),
            default_assigned_products=list(c_raw.get("default_assigned_products", [])),
            vigilance_weights=vigilance,
            hawkes_dynamics=c_raw.get("hawkes_dynamics"),
        )

    # 3. Parse 03_payment_rail_gaps.yaml
    stip_raw = raw_03.get("stand_in_processing_stip", {})
    stip_floors = stip_raw.get("network_floor_limits_cents", {})
    stip_spec = StipConfigSpec(
        timeout_threshold_ms=int(stip_raw.get("trigger_conditions", {}).get("timeout_threshold_ms", 2000)),
        subprime_floor_usd=float(stip_floors.get("subprime_classic", 10000)) / 100.0,
        rewards_floor_usd=float(stip_floors.get("rewards_signature", 50000)) / 100.0,
        ultra_premium_floor_usd=float(stip_floors.get("ultra_premium", 250000)) / 100.0,
        max_consecutive_offline_approvals=int(stip_raw.get("controls", {}).get("max_consecutive_offline_approvals", 3)),
        fourth_consecutive_action=str(stip_raw.get("controls", {}).get("fourth_consecutive_action", "HARD_DECLINE_05")),
    )

    avs_matrix: Dict[str, AvsCodeSpec] = {}
    for code, meta in raw_03.get("address_verification_service_avs", {}).get("response_matrix", {}).items():
        avs_matrix[code] = AvsCodeSpec(
            match=meta["match"],
            street=meta.get("street"),
            zip5=meta.get("zip5"),
            risk=meta["risk"],
        )

    sca_raw = raw_03.get("emv_3ds_sca_exemptions", {})
    sca_spec = ScaExemptionsSpec(
        low_value_max_single_eur=float(sca_raw.get("low_value_article_16", {}).get("max_single_tx_eur", 30.0)),
        low_value_cumulative_eur=float(sca_raw.get("low_value_article_16", {}).get("cumulative_spend_circuit_breaker_eur", 100.0)),
        low_value_consecutive_tx=int(sca_raw.get("low_value_article_16", {}).get("consecutive_tx_circuit_breaker", 5)),
        tra_tiers=sca_raw.get("transaction_risk_analysis_article_18", {}).get("tiers", []),
    )

    # 4. Parse 04_adversarial_playbooks.yaml
    playbooks: Dict[str, AdversarialPlaybookSpec] = {}
    for p_raw in raw_04.get("adversarial_playbooks", []):
        pid = p_raw["id"]
        t_range = p_raw.get("ticket_size_range_cents", [0, 0])
        playbooks[pid] = AdversarialPlaybookSpec(
            id=pid,
            name=p_raw["name"],
            target_products=p_raw.get("target_products", []),
            target_clusters=p_raw.get("target_clusters", []),
            ticket_size_min_cents=float(t_range[0]),
            ticket_size_max_cents=float(t_range[1]),
            target_mccs=p_raw.get("target_mccs", []),
            action_sequence=p_raw.get("action_sequence", {}),
            response_handling=p_raw.get("response_handling", {}),
        )

    # 5. Parse 05_india_payment_rails.yaml
    in_rails = raw_05.get("india_regulatory_rails", {})
    afa = in_rails.get("additional_factor_of_authentication_afa", {})
    nfc = afa.get("exemptions", {}).get("contactless_nfc_tap", {})
    emandate = afa.get("exemptions", {}).get("e_mandate_recurring", {})
    offline = afa.get("exemptions", {}).get("offline_micropayments", {})
    controls = in_rails.get("card_controls_default_state", {})
    rupay = in_rails.get("rupay_credit_on_upi", {})
    cust_prot = in_rails.get("customer_protection_limiting_liability", {})
    tat = in_rails.get("harmonisation_tat_failed_transactions", {})

    india_rails_spec = IndiaRegulatoryRailsSpec(
        otp_validity_window_seconds=int(afa.get("otp_validity_window_seconds", 180)),
        contactless_pin_free_ceiling_inr=float(nfc.get("pin_free_ceiling_paisa", 500000)) / 100.0,
        contactless_max_consecutive_pinless=int(nfc.get("max_consecutive_pinless_tx", 5)),
        contactless_cumulative_ceiling_inr=float(nfc.get("cumulative_pinless_ceiling_paisa", 1500000)) / 100.0,
        e_mandate_standard_ceiling_inr=float(emandate.get("standard_ceiling_paisa", 1500000)) / 100.0,
        e_mandate_high_value_ceiling_inr=float(emandate.get("high_value_category_ceiling_paisa", 10000000)) / 100.0,
        offline_micropayment_ceiling_inr=float(offline.get("per_tx_ceiling_paisa", 50000)) / 100.0,
        upi_lite_per_tx_ceiling_inr=float(offline.get("upi_lite_per_tx_ceiling_paisa", 100000)) / 100.0,
        default_enabled_controls=controls.get("default_enabled", ["DOMESTIC_ATM", "DOMESTIC_POS_CONTACT"]),
        default_disabled_controls=controls.get("default_disabled", ["DOMESTIC_CNP", "INTERNATIONAL_POS", "CONTACTLESS_NFC"]),
        rupay_zero_mdr_ceiling_inr=float(rupay.get("small_merchant_zero_mdr_ceiling_paisa", 200000)) / 100.0,
        liability_zero_working_days=int(cust_prot.get("zero_liability_scenarios", {}).get("third_party_breach_reported_within_working_days", 3)),
        provisional_credit_mandate_days=int(cust_prot.get("provisional_shadow_credit_mandate_working_days", 10)),
        delayed_tat_penalty_per_day_inr=float(tat.get("delayed_reversal_statutory_penalty_per_day_paisa", 10000)) / 100.0,
    )

    indian_products: Dict[str, IndianCardProductSpec] = {}
    for ip_raw in raw_05.get("indian_card_products", []):
        ipid = ip_raw["id"]
        cl_paisa = ip_raw.get("credit_limit_paisa", ip_raw.get("account_balance_paisa", {}))
        sp_dist = ip_raw.get("spend_distribution", {})
        indian_products[ipid] = IndianCardProductSpec(
            id=ipid,
            name=ip_raw["name"],
            category=ip_raw["category"],
            credit_limit_min_inr=float(cl_paisa.get("min", 0)) / 100.0,
            credit_limit_median_inr=float(cl_paisa.get("median", 0)) / 100.0,
            credit_limit_max_inr=float(cl_paisa.get("max", 0)) / 100.0,
            primary_adversarial_vulnerabilities=ip_raw.get("primary_adversarial_vulnerabilities", []),
            spend_mean_log=float(sp_dist.get("mu_log", 7.60)),
            spend_sigma_log=float(sp_dist.get("sigma_log", 0.70)),
        )

    # 6. Parse circadian_diurnal_model (von Mises mixture on S^1)
    circ_raw = raw_02.get("circadian_diurnal_model", {})

    def _parse_profile(p_dict: Dict[str, Any]) -> DiurnalProfileSpec:
        beta_0 = float(p_dict.get("nocturnal_floor_beta_0", 0.025))
        comps: List[VonMisesComponent] = []
        for c in p_dict.get("components", []):
            k = float(c["kappa"])
            comps.append(
                VonMisesComponent(
                    name=str(c.get("name", "")),
                    mu_hour=float(c["mu_hour"]),
                    kappa=k,
                    weight=float(c["weight"]),
                    i0_kappa=bessel_i0(k),
                )
            )
        # Precompute supremum phi_max across 24 hours
        test_hours = [i * 0.05 for i in range(480)]
        phi_max = 1.0
        two_pi = 2.0 * math.pi
        for th in test_hours:
            val = beta_0 + (24.0 - 24.0 * beta_0) * sum(
                comp.weight * math.exp(comp.kappa * math.cos(two_pi * (th - comp.mu_hour) / 24.0)) / (24.0 * comp.i0_kappa)
                for comp in comps
            )
            if val > phi_max:
                phi_max = val
        return DiurnalProfileSpec(
            nocturnal_floor_beta_0=beta_0,
            components=comps,
            phi_max=phi_max,
        )

    weekday_spec = _parse_profile(circ_raw.get("weekday_profile", {}))
    weekend_spec = _parse_profile(circ_raw.get("weekend_profile", {}))
    circadian_spec = CircadianDiurnalSpec(
        mathematical_form=str(circ_raw.get("mathematical_form", "von_Mises_mixture_on_S1")),
        formula=str(circ_raw.get("formula", "")),
        weekday_profile=weekday_spec,
        weekend_profile=weekend_spec,
    )

    # 7. Parse authentic_hard_negatives
    hard_negatives: Dict[str, AuthenticHardNegativeSpec] = {}
    for hn_raw in raw_02.get("authentic_hard_negatives", []):
        hn_id = hn_raw["id"]
        hard_negatives[hn_id] = AuthenticHardNegativeSpec(
            id=hn_id,
            name=hn_raw.get("name", ""),
            episodic_window_seconds=float(hn_raw["episodic_window_seconds"]) if "episodic_window_seconds" in hn_raw else None,
            velocity_multiplier=float(hn_raw["velocity_multiplier"]) if "velocity_multiplier" in hn_raw else None,
            spend_percentile_target=hn_raw.get("spend_percentile_target"),
            characteristic_mccs=list(hn_raw.get("characteristic_mccs", hn_raw.get("transit_mccs", []))),
            ground_truth_discriminators=dict(hn_raw.get("ground_truth_discriminators", {})),
        )

    # 7. Parse 07_export_leakage_gate.yaml
    nrs = raw_07.get("network_risk_score", {})
    amounts = raw_07.get("attack_amounts", {})
    afa_fail = raw_07.get("legitimate_authentication_failure", {})
    xb_cnp = raw_07.get("legitimate_cross_border_cnp", {})
    avs_u = raw_07.get("avs_unavailable_international", {})
    in_dossier = raw_07.get("indian_intent_dossier", {})
    uncovered = raw_07.get("legitimate_uncovered_mccs", {})
    follow_up = raw_07.get("validated_card_follow_up", {})
    completeness = raw_07.get("credential_completeness", {})
    micro = raw_07.get("legitimate_micro_tickets", {})
    ip_dist = raw_07.get("legitimate_ip_distance_km", {})
    cashout = raw_07.get("otp_theft_cashout_mccs", {})
    address_space = raw_07.get("client_ip_address_space", {})
    device_sharing = raw_07.get("device_sharing", {})
    channel_mix = raw_07.get("attack_channel_mix", {})
    pacing = raw_07.get("prevalence_pacing", {})
    probe_range = amounts.get("micro_probe_ticket_range_usd", [0.50, 1.99])
    mimicry_range = amounts.get("mimicry_probe_ticket_range_usd", [3.50, 18.50])
    telemetry_spec = TelemetryCalibrationSpec(
        network_risk_score_shapes={
            str(k): (float(v["alpha"]), float(v["beta"])) for k, v in nrs.get("distributions", {}).items()
        },
        network_decline_threshold=int(nrs.get("decline_threshold", 75)),
        reference_rate_inr_per_usd=float(amounts.get("reference_rate_inr_per_usd", 83.5)),
        micro_probe_ticket_range_usd=(float(probe_range[0]), float(probe_range[1])),
        mimicry_probe_ticket_range_usd=(float(mimicry_range[0]), float(mimicry_range[1])),
        ladder_jitter_fraction=float(amounts.get("ladder_jitter_fraction", 0.0)),
        route_indian_probes_cross_border=bool(
            amounts.get("probe_for_indian_cards", {}).get("route_cross_border_when_international_enabled", True)
        ),
        legit_afa_failure_share={str(k): float(v) for k, v in afa_fail.get("failure_share_when_challenged", {}).items()},
        legit_challenge_share={str(k): float(v) for k, v in afa_fail.get("challenge_share_of_frictionless_eligible", {}).items()},
        legit_cross_border_cnp_share={
            str(k): float(v) for k, v in xb_cnp.get("share_of_cnp_on_international_enabled_cards", {}).items()
        },
        avs_unavailable_probability_cross_border=float(avs_u.get("probability_when_cross_border_cnp", 0.0)),
        fraud_avs_weights={str(k): float(v) for k, v in avs_u.get("fraud_avs_draw_without_u", {}).items()},
        legit_avs_weights={str(k): float(v) for k, v in avs_u.get("legitimate_avs_draw", {"Y": 0.92, "N": 0.05, "A": 0.02, "Z": 0.01}).items()},
        validated_card_follow_up_probability=float(follow_up.get("probability_next_attack_targets_validated_card", 0.0)),
        full_record_dossier_tiers=tuple(str(t) for t in completeness.get("full_record_dossier_tiers", [])),
        full_record_playbooks=tuple(str(p) for p in completeness.get("full_record_playbooks", [])),
        indian_intent_live_otp_probability=float(in_dossier.get("probability_live_otp", 0.0)),
        legit_uncovered_mcc_share={
            int(mcc): {str(r): float(v) for r, v in shares.items()}
            for mcc, shares in uncovered.items()
            if isinstance(shares, dict)
        },
        legit_micro_ticket_share={str(k): float(v) for k, v in micro.get("share_of_routine_transactions", {}).items()},
        legit_micro_ticket_range={
            str(c): (float(r[0]), float(r[1])) for c, r in micro.get("ticket_range_by_currency", {}).items()
        },
        legit_ip_distance_lognormal={
            str(ch): (float(p["median_km"]), float(p["sigma_log"])) for ch, p in ip_dist.get("by_channel", {}).items()
        },
        otp_theft_cashout_mccs={int(m): float(w) for m, w in cashout.items() if isinstance(w, (int, float)) and not isinstance(w, bool)},
        address_space={str(k): v for k, v in address_space.items() if k not in ("hypothesis", "note")},
        device_sharing={str(k): v for k, v in device_sharing.items() if k not in ("hypothesis", "note")},
        attack_channel_mix={
            str(table): {str(key): {str(ch): float(w) for ch, w in mix.items()} for key, mix in entries.items()}
            for table, entries in channel_mix.items()
            if isinstance(entries, dict)
        },
        prevalence_warm_up_rows=int(pacing.get("warm_up_rows", 25)),
        prevalence_window_hours=float(pacing.get("window_hours", 24.0)),
    )

    def _credit_limit_steps(raw: Dict[str, Any], key: str) -> List[Tuple[Optional[float], float]]:
        steps: List[Tuple[Optional[float], float]] = []
        for tier in raw.get("credit_limit_assignment", {}).get(key, []):
            up_to = tier.get("up_to")
            steps.append((None if up_to is None else float(up_to) / 100.0, float(tier["step"]) / 100.0))
        return steps

    credit_limit_steps = {
        "US": _credit_limit_steps(raw_01, "rounding_steps_cents"),
        "IN": _credit_limit_steps(raw_05, "rounding_steps_paisa"),
    }

    # 8. Parse 08_india_calibration_targets.yaml and 09_us_calibration_targets.yaml
    calibration_profiles: Dict[str, CalibrationProfileSpec] = {}
    for prof in raw_08.get("profiles", []) + raw_09.get("profiles", []):
        targets = tuple(
            CalibrationTargetSpec(
                id=str(t["id"]),
                statistic=str(t.get("statistic", "")),
                target=float(t["target"]),
                source=str(t.get("source", "")),
                gate=bool(t.get("gate", False)),
                tolerance_relative=float(t["tolerance_relative"]) if "tolerance_relative" in t else None,
                tolerance_absolute=float(t["tolerance_absolute"]) if "tolerance_absolute" in t else None,
                reason_not_gated=str(t.get("reason_not_gated", "")),
                reliable_from_fraud_rows=int(t["reliable_from_fraud_rows"]) if "reliable_from_fraud_rows" in t else None,
                reliability_basis=str(t.get("reliability_basis", "")),
            )
            for t in prof.get("targets", [])
        )
        for t in targets:
            if t.reliable_from_fraud_rows is not None and not t.reliability_basis.strip():
                raise ValueError(
                    f"Target {t.id} declares reliable_from_fraud_rows={t.reliable_from_fraud_rows} "
                    f"without a reliability_basis. A threshold nobody derived is an invented number "
                    f"(AGENTS.md 4): state the measurement it came from, or drop it."
                )
        rng_lo, rng_hi = prof.get("fraud_prevalence_range_2025_26", [0.0, 0.0])
        calibration_profiles[str(prof["id"])] = CalibrationProfileSpec(
            id=str(prof["id"]),
            region=str(prof.get("region", "IN")).upper(),
            as_of=str(prof.get("as_of", "")),
            scope=str(prof.get("scope", "")).strip(),
            fraud_prevalence=float(prof["fraud_prevalence_per_transaction"]),
            fraud_prevalence_range=(float(rng_lo), float(rng_hi)),
            demo_rate_policy=str(prof.get("demo_rate_policy", "")).strip(),
            targets=targets,
            out_of_scope=tuple({str(k): str(v) for k, v in item.items()} for item in prof.get("out_of_scope", [])),
        )

    # 9. Parse 16_operational_supervision.yaml
    sup_raw = raw_16.get("operational_supervision", {})
    queue_raw = sup_raw.get("investigation_queue", {})
    analyst_raw = sup_raw.get("analyst_review_latency", {})
    dispute_raw = sup_raw.get("customer_dispute_latency", {})
    dark_raw = sup_raw.get("dark_fraud_non_reporting", {})

    supervision_spec = OperationalSupervisionSpec(
        default_k_daily=int(queue_raw.get("default_k_daily", 50)),
        alert_threshold=float(queue_raw.get("alert_threshold", 0.70)),
        default_priority_strategy=str(queue_raw.get("default_priority_strategy", "RISK_SCORE")),
        weibull_k=float(analyst_raw.get("shape_k", 1.35)),
        weibull_scale_hours=float(analyst_raw.get("scale_hours", 18.0)),
        min_investigation_hours=float(analyst_raw.get("min_hours", 0.5)),
        max_investigation_hours=float(analyst_raw.get("max_hours", 72.0)),
        lognormal_mu_days=float(dispute_raw.get("mu_days", 3.40)),
        lognormal_sigma=float(dispute_raw.get("sigma_days", 0.45)),
        min_chargeback_days=float(dispute_raw.get("min_days", 3.0)),
        max_chargeback_days=float(dispute_raw.get("max_days", 120.0)),
        v0_usd=float(dark_raw.get("v0_usd", 15.0)),
        v0_inr=float(dark_raw.get("v0_inr", 1250.0)),
        dark_smoothness=float(dark_raw.get("smoothness_gamma", 0.40)),
        clean_maturity_days=float(dark_raw.get("clean_maturity_days", 90.0)),
    )

    # 10. Parse 13_annual_seasonality_and_drift.yaml
    seasonality_spec = None
    if raw_13:
        seasonality_spec = AnnualSeasonalitySpec(
            day_of_week_multipliers={int(k): float(v) for k, v in raw_13.get("day_of_week_multipliers", {}).items()},
            payday_cycles=dict(raw_13.get("payday_cycles", {})),
            mcc_monthly_scaling=dict(raw_13.get("mcc_monthly_scaling", {})),
            calendar_macro_shocks=dict(raw_13.get("calendar_macro_shocks", {})),
            wallet_hierarchy=dict(raw_13.get("wallet_hierarchy", {})),
            adversarial_drift_schedule=dict(raw_13.get("adversarial_drift_schedule", {})),
            topology_seasonality=dict(raw_13.get("topology_seasonality", {})),
            cross_bank_smurfing=dict(raw_13.get("cross_bank_smurfing", {})),
        )


    # 11. Parse 14_interbank_switch_and_clearing.yaml
    interbank_spec = None
    if raw_14:
        interbank_spec = InterbankSwitchSpec(
            commercial_banks=dict(raw_14.get("commercial_banks", {})),
            central_switching_networks=dict(raw_14.get("central_switching_networks", {})),
            routing_rules=dict(raw_14.get("routing_rules", {})),
            iso8583_message_types=dict(raw_14.get("iso8583_message_types", {})),
            clearing_and_settlement=dict(raw_14.get("clearing_and_settlement", {})),
        )

    return SpecRegistry(
        products=products,
        cohorts=cohorts,
        stip=stip_spec,
        avs=avs_matrix,
        sca=sca_spec,
        playbooks=playbooks,
        india_rails=india_rails_spec,
        indian_products=indian_products,
        circadian_raw=circ_raw,
        raw_specs={
            "01": raw_01,
            "02": raw_02,
            "03": raw_03,
            "04": raw_04,
            "05": raw_05,
            "07": raw_07,
            "08": raw_08,
            "13": raw_13,
            "14": raw_14,
            "16": raw_16,
        },
        circadian=circadian_spec,
        hard_negatives=hard_negatives,
        telemetry=telemetry_spec,
        calibration_profiles=calibration_profiles,
        supervision=supervision_spec,
        credit_limit_steps=credit_limit_steps,
        seasonality=seasonality_spec,
        interbank=interbank_spec,
    )
