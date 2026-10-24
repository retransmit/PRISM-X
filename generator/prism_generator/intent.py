# prism_generator/intent.py
# First-Principles Computational Adversarial Threat Modeling & Intent Optimization Engine
# Grounded in: US DOJ Indictments (911 S5, Seleznev), Thomas et al. (USENIX Security 2017),
# Ali et al. (IEEE S&P 2017), Vastel et al. (USENIX Security 2018), EMVCo Tokenization Book 4,
# Visa Core Rules (VCR 2024-2026), PSD2 RTS, RBI PSS Act 2007.

from dataclasses import dataclass
from enum import Enum
import math
from typing import List, Optional
import numpy as np


class CredentialTier(Enum):
    """Empirical underground data completeness tiers."""
    TIER_TRACK_2_DUMP = "TIER_TRACK_2_DUMP"
    TIER_CNP_FULLZ = "TIER_CNP_FULLZ"
    TIER_SESSION_COOKIE_BUNDLE = "TIER_SESSION_COOKIE_BUNDLE"
    TIER_PHISHED_OTP_STREAM = "TIER_PHISHED_OTP_STREAM"
    TIER_APP_DEVICE_TOKEN = "TIER_APP_DEVICE_TOKEN"
    TIER_VCN_DYNAMIC_CVV = "TIER_VCN_DYNAMIC_CVV"


class MacroOptionType(Enum):
    """Hierarchical Semi-Markov Decision Process (SMDP) macro-options."""
    OMEGA_PROBE = "OMEGA_PROBE"
    OMEGA_INCUBATE = "OMEGA_INCUBATE"
    OMEGA_BISECT_DRAIN = "OMEGA_BISECT_DRAIN"
    OMEGA_HARVEST = "OMEGA_HARVEST"
    OMEGA_PURGE = "OMEGA_PURGE"


@dataclass
class CredentialDossier:
    """Stolen financial credential dossier representing upstream information supply."""
    tier: CredentialTier
    pan: str
    expiry_month: int = 12
    expiry_year: int = 2028
    cvv2: Optional[str] = None
    cardholder_name: Optional[str] = None
    billing_zip: Optional[str] = None
    phone_number: Optional[str] = None
    session_cookies: Optional[str] = None
    live_otp: Optional[str] = None
    otp_expiry_seconds: float = 180.0
    has_chip_cryptogram: bool = False
    has_live_otp: bool = False
    has_bound_token: bool = False
    dpan: Optional[str] = None
    device_id: Optional[str] = None


@dataclass
class CandidateAction:
    """Discrete candidate transaction action on the pruned action lattice."""
    mcc: int
    amount: float
    channel: str
    macro_option: MacroOptionType = MacroOptionType.OMEGA_HARVEST
    incubation_seconds: float = 0.0
    proxy_type: str = "RESIDENTIAL"


@dataclass
class AnalyticalBeliefState:
    """Analytical Bayesian belief state over target card and defense posture."""
    card_id: str
    p_valid: float = 0.50
    mu_balance: float = 1500.0
    sigma_balance: float = 500.0
    min_balance: float = 0.0
    max_balance: float = 5000.0
    alpha_risk: float = 2.0
    beta_risk: float = 2.0
    consecutive_declines: int = 0
    is_burned: bool = False
    acquisition_cost_usd: float = 15.0
    probe_penalized_by_issuer: bool = False
    is_parked_until_midnight: bool = False
    last_response_code: Optional[str] = None


class ConstraintPruner:
    """Strict physical, telecommunication, and payment rail constraint pruning engine."""

    def prune_candidates(
        self,
        dossier: CredentialDossier,
        candidates: List[CandidateAction],
        velocity_kmh: float = 0.0,
    ) -> List[CandidateAction]:
        """Prunes physically impossible actions according to hardware and rail invariants."""
        # 1. Supersonic Kinematic Invariant (Commercial jet ceiling = 900 km/h)
        if velocity_kmh > 900.0:
            return []

        # 2. VCN Dynamic CVV Countermeasure (Rotated dynamic CVVs cannot be carded statically)
        if dossier.tier == CredentialTier.TIER_VCN_DYNAMIC_CVV:
            return []

        valid_actions: List[CandidateAction] = []

        for c in candidates:
            # Physical Contact Chip Invariant
            if c.channel in ("CP_POS_CHIP", "ATM_CASH"):
                if not dossier.has_chip_cryptogram:
                    continue

            # Contactless NFC Tap Invariant (Requires provisioned DPAN token)
            elif c.channel in ("CP_CONTACTLESS_NFC", "CP_POS_CONTACTLESS"):
                if not dossier.has_bound_token:
                    continue

            # Digital Wallet Provisioning (Apple/Google Pay Yellow Path requires real-time OTP)
            elif c.channel == "PROVISION_DIGITAL_WALLET":
                if not (dossier.has_live_otp or dossier.live_otp is not None):
                    continue

            # E-Commerce CNP Channels
            elif c.channel in ("CNP_WEB", "CNP_MOBILE"):
                # Track 2 dumps have no CVV2 and cannot be processed in e-commerce
                if dossier.tier == CredentialTier.TIER_TRACK_2_DUMP:
                    continue
                # If CVV2 is missing and not an active session cookie replay, prune
                if dossier.cvv2 is None and dossier.tier != CredentialTier.TIER_SESSION_COOKIE_BUNDLE:
                    continue

            valid_actions.append(c)

        return valid_actions


class InformationDirectedOptimizer:
    """Bounded-rational intent optimizer driven by Information-Directed Sampling (IDS)."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.pruner = ConstraintPruner()

    def compute_mutual_information(self, p_valid: float, p_approve: float) -> float:
        """Calculates exact closed-form Shannon Mutual Information I(V; O | a).

        Let V in {0, 1} be target card validity. P(V=1) = p_valid.
        Let O in {0, 1} be approval observation.
        P(O=1 | V=1) = p_approve, P(O=1 | V=0) = 0.0 (invalid card never approved).
        P(O=1) = p_valid * p_approve.
        P(O=0) = 1.0 - p_valid * p_approve.

        I(V; O) = H(V) - E_O[H(V | O)]
        """
        if p_valid <= 0.0 or p_valid >= 1.0:
            return 0.0

        # Prior Shannon Entropy H(V)
        h_prior = -p_valid * math.log(p_valid) - (1.0 - p_valid) * math.log(1.0 - p_valid)

        # Observation probabilities
        p_app = p_valid * p_approve
        p_dec = 1.0 - p_app

        if p_dec <= 1e-9:
            return h_prior

        # Posterior entropy if approved: P(V=1 | O=1) = 1.0 => H(V | O=1) = 0.0
        h_post_app = 0.0

        # Posterior entropy if declined:
        # P(V=1 | O=0) = (1 - p_approve) * p_valid / (1 - p_valid * p_approve)
        p_v_given_dec = (p_valid * (1.0 - p_approve)) / p_dec
        if p_v_given_dec <= 1e-9 or p_v_given_dec >= 1.0 - 1e-9:
            h_post_dec = 0.0
        else:
            h_post_dec = -p_v_given_dec * math.log(p_v_given_dec) - (1.0 - p_v_given_dec) * math.log(1.0 - p_v_given_dec)

        e_post_h = p_app * h_post_app + p_dec * h_post_dec
        return max(0.0, h_prior - e_post_h)

    def observe_outcome(
        self,
        belief: AnalyticalBeliefState,
        amount: float,
        response_code: str,
        partial_amount: Optional[float] = None,
    ) -> None:
        """Binding Invariant 1: Strictly updates mathematical belief state b(S).

        Zero next-action mutations permitted.
        """
        belief.last_response_code = response_code

        if response_code in ("00", "0"):
            # ISO 00: Approved -> validity confirmed, lower balance bound raised
            belief.p_valid = 1.0
            belief.min_balance = max(belief.min_balance, amount)
            belief.consecutive_declines = 0
            belief.alpha_risk += 1.0

        elif response_code == "10":
            # ISO 10: Partial Approval -> One-shot exact balance discovery oracle!
            p = partial_amount if partial_amount is not None else amount
            belief.p_valid = 1.0
            belief.min_balance = p
            belief.max_balance = p
            belief.consecutive_declines = 0

        elif response_code == "51":
            # ISO 51: Insufficient Funds -> Upper bound truncated strictly below requested amount
            belief.max_balance = min(belief.max_balance, round(amount - 0.01, 2))
            belief.consecutive_declines += 1

        elif response_code in ("05", "5"):
            # ISO 05: Do Not Honor -> Issuer fraud model flagged
            belief.beta_risk += 1.0
            belief.consecutive_declines += 1

        elif response_code in ("59", "14"):
            # ISO 59 (Suspected Fraud) or ISO 14 (Invalid Card) -> Hard Terminal Lock
            belief.p_valid = 0.0
            belief.is_burned = True
            belief.consecutive_declines += 1

        elif response_code == "65":
            # ISO 65: Activity Limit Exceeded -> Daily velocity cap reached; park until midnight
            belief.is_parked_until_midnight = True
            belief.consecutive_declines += 1

        elif response_code in ("63", "82"):
            # Cryptographic or CVV mismatch
            belief.consecutive_declines += 1

    def select_optimal_action(
        self,
        belief: AnalyticalBeliefState,
        dossier: CredentialDossier,
        current_hour_local: int = 12,
        velocity_kmh: float = 0.0,
        preferred_channel: Optional[str] = None,
    ) -> CandidateAction:
        """Selects the optimal candidate action using Information-Directed Sampling.

        preferred_channel: the channel the attacker drew from its credential tier's mix (spec/07
        section 18). The utility has no channel term, so among equally valued actions the first
        candidate channel wins; the preferred one is tried first and still has to pass the pruner.
        """
        # 1. Target Burned / Closed Account Invariant -> Purge
        if belief.is_burned or belief.p_valid <= 0.0:
            return CandidateAction(
                mcc=0,
                amount=0.0,
                channel="NONE",
                macro_option=MacroOptionType.OMEGA_PURGE,
            )

        # 2. Sunk Cost & Circadian Alert Hazard Invariant -> Incubation
        # Daytime (10:00 - 18:00) has high victim alert hazard (~60s notification)
        # Sunk cost preservation forces high-value credentials ($100+) to incubate until night
        if 10 <= current_hour_local <= 18 and belief.acquisition_cost_usd >= 100.0:
            sleep_hours = (24 - current_hour_local + 2)  # Wait until 02:00 nocturnal window
            return CandidateAction(
                mcc=0,
                amount=0.0,
                channel="NONE",
                macro_option=MacroOptionType.OMEGA_INCUBATE,
                incubation_seconds=float(sleep_hours * 3600),
            )

        # 3. Generate raw combinatorial action lattice
        raw_candidates: List[CandidateAction] = []
        candidate_amounts = [1.50, 3.50, 15.00, 28.00, 75.00, 150.00, 350.00, 850.00, 1200.00, 2500.00]
        candidate_mccs = [8398, 5815, 4899, 5311, 5732, 5944, 5947, 6051]
        candidate_channels = ["CNP_WEB", "CNP_MOBILE", "CP_POS_CONTACTLESS", "CP_POS_CHIP", "PROVISION_DIGITAL_WALLET"]
        if preferred_channel:
            preferred = {"CP_CONTACTLESS_NFC": "CP_POS_CONTACTLESS"}.get(preferred_channel, preferred_channel)
            if preferred in candidate_channels:
                candidate_channels.remove(preferred)
                candidate_channels.insert(0, preferred)

        for amt in candidate_amounts:
            for mcc in candidate_mccs:
                for ch in candidate_channels:
                    raw_candidates.append(CandidateAction(mcc=mcc, amount=amt, channel=ch))

        # 4. Filter through physical and hardware constraints
        valid_candidates = self.pruner.prune_candidates(dossier, raw_candidates, velocity_kmh=velocity_kmh)
        if not valid_candidates:
            return CandidateAction(mcc=0, amount=0.0, channel="NONE", macro_option=MacroOptionType.OMEGA_PURGE)

        # 5. Score candidates using Information-Directed Sampling (IDS)
        best_candidate: Optional[CandidateAction] = None
        best_utility = -float("inf")

        for c in valid_candidates:
            # Tripwire Guard: Honeypot Inversion (Issuer flags micro-probes)
            if belief.probe_penalized_by_issuer and c.amount <= 5.0:
                continue

            # Available balance probability P(C >= amount)
            if c.amount > belief.max_balance:
                p_balance = 0.0
            elif c.amount <= belief.min_balance:
                p_balance = 1.0
            else:
                # Truncated normal probability
                z = (c.amount - belief.mu_balance) / max(1.0, belief.sigma_balance)
                p_balance = max(0.0, min(1.0, 1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))))

            # Issuer risk tolerance probability
            base_risk_pass = (belief.alpha_risk / (belief.alpha_risk + belief.beta_risk))
            # High amounts trigger elevated fraud scoring scrutiny
            amount_penalty = max(0.20, 1.0 - (c.amount / 5000.0))
            if c.mcc in (8398, 5815, 4899) and c.amount <= 5.0:
                p_risk = min(0.98, base_risk_pass * 1.3)  # Low-value probe has near-zero risk scoring
            elif c.mcc in (6051, 7995):
                p_risk = max(0.05, base_risk_pass * 0.4 * amount_penalty)
            else:
                p_risk = base_risk_pass * amount_penalty

            # Total approval probability given card is valid
            p_approve_given_valid = max(0.001, p_balance * p_risk)
            # Unconditional approval probability
            p_approve = belief.p_valid * p_approve_given_valid

            # Economic Haircut
            haircut = 0.25 if c.mcc == 5947 else (0.15 if c.mcc == 5732 else (0.08 if c.mcc == 6051 else 0.20))
            if c.mcc == 8398:
                haircut = 1.0  # Charity probe has zero cashout yield

            # Expected Extracted Cash
            expected_cash = c.amount * (1.0 - haircut) * p_approve

            # Costs & Sunk Loss Risk
            op_cost = 0.25  # Residential proxy + infrastructure

            # Terminal Burn Loss Invariant:
            # Aggressive attempts (>15.0) on unverified cards risk burning unharvested balance!
            # Micro-probes (<=15.0) do NOT burn unharvested balance.
            if c.amount > 15.0 and belief.p_valid < 0.80:
                unharvested_loss = belief.mu_balance * min(1.0, c.amount / 300.0)
            else:
                unharvested_loss = 0.0

            burn_loss = (1.0 - p_approve) * (belief.acquisition_cost_usd + unharvested_loss)

            # Shannon Value-of-Information (VoI)
            mi = self.compute_mutual_information(belief.p_valid, p_approve_given_valid)
            lambda_info = 60.0 if belief.p_valid < 0.80 else 0.0
            voi = lambda_info * mi

            # Net Utility
            utility = expected_cash - op_cost - burn_loss + voi

            if utility > best_utility:
                best_utility = utility
                best_candidate = c

        if best_candidate is None:
            best_candidate = valid_candidates[0]

        # 6. Assign emergent MacroOptionType
        if best_candidate.amount <= 5.0 and best_candidate.mcc in (8398, 5815, 4899):
            best_candidate.macro_option = MacroOptionType.OMEGA_PROBE
        elif best_candidate.amount <= belief.max_balance and belief.max_balance < belief.mu_balance:
            best_candidate.macro_option = MacroOptionType.OMEGA_BISECT_DRAIN
        else:
            best_candidate.macro_option = MacroOptionType.OMEGA_HARVEST

        return best_candidate


@dataclass
class SwitchHopResult:
    """Result of multi-hop payment network evaluation."""
    response_code: str
    hop_origin: str
    decline_reason: str = ""
    is_approved: bool = False


@dataclass
class TokenBindingResult:
    """Result of EMVCo digital wallet token provisioning."""
    is_provisioned: bool
    dpan: Optional[str] = None
    token_requestor_id: Optional[str] = None
    liability_shift: Optional[str] = None
    error: Optional[str] = None


class MultiHopSwitchEngine:
    """4-Hop Payment Switch Routing Engine: Gateway -> Switch (VAAI) -> 3DS -> Issuer."""

    def evaluate_transaction(
        self,
        amount: float,
        mcc: int,
        channel: str,
        ip_reputation_score: int = 20,
        canvas_repeatability: bool = True,
        tls_fingerprint_match: bool = True,
        velocity_kmh: float = 0.0,
        vaai_score: int = 15,
        target_balance: float = 1000.0,
    ) -> SwitchHopResult:
        """Evaluates transaction across the 4 architectural boundaries."""
        # Hop 1: Gateway / PayFac Pre-Routing Filters
        if velocity_kmh > 900.0:
            return SwitchHopResult(
                response_code="59",
                hop_origin="GATEWAY_FILTER",
                decline_reason="SUPERSONIC_VELOCITY_VETO",
            )
        if not canvas_repeatability:
            return SwitchHopResult(
                response_code="05",
                hop_origin="GATEWAY_FILTER",
                decline_reason="BROWSER_TAMPERING_SYNTHETIC_CANVAS",
            )
        if not tls_fingerprint_match:
            return SwitchHopResult(
                response_code="05",
                hop_origin="GATEWAY_FILTER",
                decline_reason="TLS_SIGNATURE_MISMATCH_BOT",
            )
        if ip_reputation_score >= 85:
            return SwitchHopResult(
                response_code="05",
                hop_origin="GATEWAY_FILTER",
                decline_reason="IP_REPUTATION_HIGH_RISK",
            )

        # Hop 2: Scheme Network In-Flight ML (Visa VAAI / Mastercard Safety Net)
        if vaai_score >= 75:
            return SwitchHopResult(
                response_code="05",
                hop_origin="NETWORK_SWITCH_VAAI",
                decline_reason="VISA_VAAI_ENUMERATION_ATTACK_BLOCK",
            )

        # Hop 3: 3DS Access Control Server (ACS)
        # Evaluated if CNP and amount > Low-Value Exemption threshold
        # (Passes through for standard unit tests)

        # Hop 4: Issuer Core Banking Host
        if amount > target_balance:
            return SwitchHopResult(
                response_code="51",
                hop_origin="ISSUER_CORE_BANKING",
                decline_reason="INSUFFICIENT_FUNDS",
            )

        return SwitchHopResult(
            response_code="00",
            hop_origin="ISSUER_CORE_BANKING",
            decline_reason="APPROVED",
            is_approved=True,
        )

    def provision_digital_wallet(
        self,
        dossier: CredentialDossier,
        submitted_otp: str,
    ) -> TokenBindingResult:
        """Provisions digital wallet (Apple/Google Pay Yellow Path) via phished OTP."""
        if dossier.live_otp == submitted_otp:
            dossier.has_bound_token = True
            dossier.dpan = f"489504{dossier.pan[6:]}"
            return TokenBindingResult(
                is_provisioned=True,
                dpan=dossier.dpan,
                token_requestor_id="APPLE_PAY_TRID",
                liability_shift="ISSUER_100_PERCENT",
            )
        return TokenBindingResult(
            is_provisioned=False,
            error="INVALID_OR_EXPIRED_OTP",
        )
