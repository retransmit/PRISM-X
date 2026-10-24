"""Boundary Layer 1: Decoupled Payment Rail Verifier Switch for Prism Generator.

Implements:
1. CandidateTransactionIntent: Decoupled contract for proposed agent transaction intents.
2. RailVerificationResult: Complete banking switch authorization payload.
3. RailVerifierSwitch: Deterministic institutional boundary layer enforcing:
   - Multi-party solvency checks (credit limit, posted balance, overdraft limit).
   - Regulatory velocity & transaction ceilings (RBI ₹5k PIN-free contactless, 5-tx limit; US STIP floors).
   - 3DS 2.x protocol state machine & exemption evaluation (Low-Value, TRA, Whitelist, Challenge).
   - EMV 4.3 Bit 55 cryptogram & TVR validation.
   - AVS (Address Verification Service) and CVV2 validation.
   - Exact ISO 8583 response code generation (00, 05, 10, 14, 51, 54, 59, 63, 65, 82).
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import zlib

import numpy as np

from .agents import CardholderProfile, ISO8583Response
from .spec_loader import load_all_specs


@dataclass
class CandidateTransactionIntent:
    """Decoupled candidate transaction intent proposed by Generative Layer 2."""
    tx_id: str
    card_id: str
    sim_time_sec: float
    amount: float
    currency: str
    channel_type: str
    merchant_id: str
    mcc: int
    merchant_lat: float
    merchant_lon: float
    is_cross_border: bool = False
    is_fraud: int = 0
    scenario_tag: str = "ORGANIC_NORMAL"
    # Proposed authorization credentials
    otp_submitted: bool = True
    cvv_provided: bool = True
    avs_code: str = "Y"
    billing_shipping_match: int = 1
    pin_entered: bool = False
    emv_chip_present: bool = False
    emv_cryptogram_valid: bool = True
    three_ds_requested: bool = False
    risk_score: float = 0.02
    vaai_score: int = 25
    haversine_velocity_kph: float = 0.0
    tx_count_1h: int = 0
    tx_count_24h: int = 0
    tx_attempts_1h: int = 0
    distinct_mids_30m: int = 1
    subminute_attempts_60s: int = 0
    ip_distance_km: float = 5.0
    asn_type: str = "residential"
    # Set by the caller when `authenticate()` has already run for this intent (the engine
    # does: the causal ground truth consumes the outcome, so both subsystems must read
    # one authentication decision). A caller that leaves it None gets the outcome computed
    # inside `verify_intent`, which is what unit tests and direct callers have always done.
    authentication: AuthenticationOutcome | None = None


@dataclass(frozen=True)
class BankEntity:
    """Institutional commercial bank entity in the multi-bank network."""
    id: str
    name: str
    routing_number: str
    bin_prefixes: Tuple[str, ...]
    asset_tier: str
    central_bank_rtgs_id: str
    market_share_weight: float
    ifsc_prefix: str = ""


@dataclass
class RailVerificationResult:
    """Complete institutional banking switch evaluation result."""
    approved: bool
    iso_response_code: str  # ISO 8583 response code
    approved_amount: float
    trans_status_3ds: str   # 3DS 2.x status: "Y", "A", "C", "N", "R"
    eci: str                # Electronic Commerce Indicator ("05", "06", "07", "")
    auth_code: str
    decline_reason: str = ""
    hold_placed: bool = False
    hold_amount: float = 0.0
    regulatory_rule_triggered: str = ""
    interchange_fee: float = 0.0
    settlement_amount: float = 0.0
    pos_entry_mode: str = "051"
    pos_condition_code: str = "00"
    hop_origin: str = "ISSUER_HOST"
    # Multi-bank routing fields
    issuing_bank_id: str = ""
    acquiring_bank_id: str = ""
    is_on_us: bool = False
    central_switch_id: str = ""
    switch_network_fee: float = 0.0
    interbank_latency_ms: float = 0.0
    mti_request: str = "0100"
    mti_response: str = "0110"
    hop_sequence: List[str] = field(default_factory=list)
    is_timeout_reversal: bool = False


@dataclass(frozen=True)
class AuthenticationOutcome:
    """What the terminal/ACS established before the issuer's authorization decision.

    This is the authentication half of a payment authorization, split out of the
    authorization rules so that a single label-free decision feeds both the risk model
    (spec/21's explained columns) and the rail result. Every field is either a terminal
    capability (POS entry mode, EMV cryptogram outcome) or an ACS 3DS 2.x status from the
    exemption/challenge engine; none of them may read the transaction's fraud label
    (spec/07's no-synthetic-oracle-leak rule).

    `failed_reason` non-empty means authentication itself failed or was abandoned, which
    ISO 8583 answers with response code 63 before any authorization consideration.
    """

    pos_entry_mode: str
    pos_condition_code: str
    trans_status_3ds: str   # 3DS 2.x transStatus: "Y" authenticated, "C" challenge, "A" attempted, "N" not authenticated
    eci: str                # Electronic Commerce Indicator ("05" authenticated, "07" not)
    arqc_verified: bool     # EMV Bit 55 cryptogram validated by the issuer HSM
    pin_verified: bool      # offline/online PIN verified (intent.pin_entered)
    tvr_clean: bool         # Terminal Verification Results clean (no exception condition)
    three_ds_authenticated: bool  # transStatus in ("Y", "C")
    three_ds_attempted: bool      # transStatus in ("A", "C")
    failed_reason: str = ""
    regulatory_rule_triggered: str = ""


class BankRoutingMesh:
    """Manages multi-bank topology, BIN routing tables, On-Us vs Off-Us detection,
    interbank hop routing, and ISO 8583 message envelope creation.
    """

    def __init__(self, region: str = "US", seed: int = 42):
        self.region = region.upper()
        self.rng = np.random.default_rng(seed)
        self.specs = load_all_specs()
        self.banks: List[BankEntity] = self._load_banks()
        self._build_cum_weights()

    def _load_banks(self) -> List[BankEntity]:
        interbank = self.specs.interbank
        banks_raw = interbank.commercial_banks.get(self.region, []) if interbank else []
        if not banks_raw:
            if self.region == "IN":
                return [
                    BankEntity("BANK_IN_SBI", "State Bank of India", "SBIN0000001", ("400012",), "PUBLIC_SECTOR_ANCHOR", "RBI_RTGS_SBI_01", 0.26, "SBIN"),
                    BankEntity("BANK_IN_HDFC", "HDFC Bank Limited", "HDFC0000001", ("412801",), "PRIVATE_SECTOR_LEADER", "RBI_RTGS_HDFC_02", 0.24, "HDFC"),
                    BankEntity("BANK_IN_ICICI", "ICICI Bank Limited", "ICIC0000001", ("524181",), "PRIVATE_SECTOR_MAJOR", "RBI_RTGS_ICICI_03", 0.20, "ICIC"),
                    BankEntity("BANK_IN_AXIS", "Axis Bank Limited", "UTIB0000001", ("607152",), "PRIVATE_SECTOR_MAJOR", "RBI_RTGS_AXIS_04", 0.16, "UTIB"),
                    BankEntity("BANK_IN_PNB", "Punjab National Bank", "PUNB0000001", ("459111",), "PUBLIC_SECTOR_MAJOR", "RBI_RTGS_PNB_05", 0.14, "PUNB"),
                ]
            else:
                return [
                    BankEntity("BANK_US_CHASE", "JPMorgan Chase Bank, N.A.", "021000021", ("400000",), "TIER_1_MONEY_CENTER", "FEDWIRE_CHASE_01", 0.28, ""),
                    BankEntity("BANK_US_BOFA", "Bank of America, N.A.", "011000138", ("412800",), "TIER_1_MONEY_CENTER", "FEDWIRE_BOFA_02", 0.24, ""),
                    BankEntity("BANK_US_WELLS", "Wells Fargo Bank, N.A.", "121000247", ("541234",), "TIER_1_MONEY_CENTER", "FEDWIRE_WELLS_03", 0.20, ""),
                    BankEntity("BANK_US_CITI", "Citibank, N.A.", "021000089", ("524180",), "TIER_1_MONEY_CENTER", "FEDWIRE_CITI_04", 0.16, ""),
                    BankEntity("BANK_US_CAPITALONE", "Capital One, N.A.", "051405515", ("459110",), "TIER_2_SUPER_REGIONAL", "FEDWIRE_CAPONE_05", 0.12, ""),
                ]
        entities = []
        for b in banks_raw:
            entities.append(
                BankEntity(
                    id=b["id"],
                    name=b["name"],
                    routing_number=b.get("routing_number", b.get("ifsc_prefix", "ROUTING_DEFAULT")),
                    bin_prefixes=tuple(b.get("bin_prefixes", [])),
                    asset_tier=b.get("asset_tier", "COMMERCIAL_BANK"),
                    central_bank_rtgs_id=b.get("central_bank_rtgs_id", f"RTGS_{b['id']}"),
                    market_share_weight=float(b.get("market_share_weight", 0.20)),
                    ifsc_prefix=b.get("ifsc_prefix", ""),
                )
            )
        return entities

    def _build_cum_weights(self) -> None:
        total = sum(b.market_share_weight for b in self.banks)
        self.cum_weights = []
        acc = 0.0
        for b in self.banks:
            acc += b.market_share_weight / max(1e-6, total)
            self.cum_weights.append(acc)

    def resolve_issuing_bank(self, card_id: str) -> BankEntity:
        """Deterministically resolves cardholder to issuing bank based on card identifier."""
        u = (zlib.crc32(f"ISSUER_{card_id}".encode("utf-8")) % 1_000_000) / 1_000_000.0
        idx = bisect.bisect_right(self.cum_weights, u)
        return self.banks[min(idx, len(self.banks) - 1)]

    def resolve_acquiring_bank(self, merchant_id: str) -> BankEntity:
        """Deterministically resolves merchant/terminal to acquiring bank based on merchant identifier."""
        u = (zlib.crc32(f"ACQUIRER_{merchant_id}".encode("utf-8")) % 1_000_000) / 1_000_000.0
        idx = bisect.bisect_right(self.cum_weights, u)
        return self.banks[min(idx, len(self.banks) - 1)]

    def route_transaction(
        self,
        card_id: str,
        merchant_id: str,
        channel_type: str,
        amount: float,
        is_fraud: int = 0,
    ) -> Dict[str, Any]:
        issuing_bank = self.resolve_issuing_bank(card_id)
        if merchant_id.startswith("P2P_"):
            recip_card = merchant_id[4:]
            acquiring_bank = self.resolve_issuing_bank(recip_card)
        else:
            acquiring_bank = self.resolve_acquiring_bank(merchant_id)
        is_on_us = (issuing_bank.id == acquiring_bank.id)


        interbank = self.specs.interbank
        cfg_switch = interbank.central_switching_networks.get(self.region, {}) if interbank else {}
        cfg_rules = interbank.routing_rules if interbank else {}

        if is_on_us:
            central_switch_id = "ON_US_INTERNAL"
            switch_network_fee = 0.0
            interchange_fee = 0.0
            latency_ms = float(self.rng.uniform(8.0, 16.0))
            hops = ["MERCHANT", "ACQUIRER_CBS", "ISSUER_CORE"]
            is_timeout = False
            mti_req = "0100"
            mti_resp = "0110"
        else:
            if self.region == "IN":
                if "UPI" in channel_type or merchant_id.startswith("P2P_"):
                    central_switch_id = cfg_switch.get("instant_payments_switch", "NPCI_UPI_SWITCH")
                    switch_network_fee = 0.0
                    interchange_fee = 0.0
                else:
                    central_switch_id = cfg_switch.get("primary_card_switch", "NPCI_NFS_RUPAY")
                    switch_network_fee = float(cfg_switch.get("switch_network_fee_inr", 0.20))
                    if "DEBIT" in channel_type:
                        rate = float(cfg_switch.get("off_us_interchange_rate_debit", 0.0090))
                        cap = float(cfg_switch.get("off_us_interchange_cap_debit_inr", 150.0))
                        interchange_fee = round(min(amount * rate, cap), 2)
                    else:
                        rate = float(cfg_switch.get("off_us_interchange_rate_credit", 0.0180))
                        interchange_fee = round(amount * rate, 2)
            else:  # US
                if "P2P" in channel_type or merchant_id.startswith("P2P_"):
                    central_switch_id = cfg_switch.get("instant_payments_switch", "FEDNOW_SERVICE")
                    switch_network_fee = 0.0010
                    interchange_fee = 0.0
                else:
                    central_switch_id = cfg_switch.get("primary_card_switch", "VISANET_BASE_I_II")
                    switch_network_fee = float(cfg_switch.get("switch_network_fee_usd", 0.0025))
                    if "DEBIT" in channel_type:
                        rate = float(cfg_switch.get("off_us_interchange_rate_debit", 0.0080))
                        cap = float(cfg_switch.get("off_us_interchange_cap_debit_usd", 0.22))
                        interchange_fee = round(min(amount * rate, cap), 2)
                    else:
                        rate = float(cfg_switch.get("off_us_interchange_rate_credit", 0.0175))
                        interchange_fee = round(amount * rate, 2)

            hops = ["MERCHANT", "ACQUIRER_SWITCH", central_switch_id, "ISSUER_SWITCH"]
            latency_ms = float(self.rng.uniform(45.0, 115.0))
            timeout_prob = float(cfg_rules.get("off_us", {}).get("timeout_rate", 0.0005))
            is_timeout = bool(self.rng.random() < timeout_prob)
            if is_timeout:
                latency_ms = float(self.rng.uniform(2050.0, 3100.0))
                mti_req = "0400"
                mti_resp = "0410"
            else:
                mti_req = "0100"
                mti_resp = "0110"

        return {
            "issuing_bank": issuing_bank,
            "acquiring_bank": acquiring_bank,
            "is_on_us": is_on_us,
            "central_switch_id": central_switch_id,
            "switch_network_fee": switch_network_fee,
            "interchange_fee": interchange_fee,
            "interbank_latency_ms": latency_ms,
            "hop_sequence": hops,
            "is_timeout_reversal": is_timeout,
            "mti_request": mti_req,
            "mti_response": mti_resp,
        }


class RailVerifierSwitch:
    """Boundary Layer 1 Deterministic Rail Switch enforcing payment plumbing invariants."""

    def __init__(
        self,
        region: str = "US",
        tau_decline: float = 0.88,
        tau_challenge_low: float = 0.08,
        tau_challenge_high: float = 0.45,
        seed: int = 42,
    ):
        self.region = region.upper()
        self.tau_decline = tau_decline
        self.tau_challenge_low = tau_challenge_low
        self.tau_challenge_high = tau_challenge_high
        self.rng = np.random.default_rng(seed)
        # Authentication has its own stream: the ACS is a different party from the switch,
        # and drawing challenge randomness from `self.rng` would couple the authentication
        # decision to the routing/latency draw order (which the engine would then perturb
        # by authenticating before routing). 7919 is an arbitrary but fixed offset so the
        # stream is reproducible from the rail's seed like every other stream here.
        self.auth_rng = np.random.default_rng(seed + 7919)
        self.telemetry = load_all_specs().telemetry
        self.routing_mesh = BankRoutingMesh(region=self.region, seed=seed)

    @staticmethod
    def _pos_credentials(channel_type: str) -> Tuple[str, str]:
        """ISO 8583 field 22/25 POS entry-mode and condition codes for a channel.

        A terminal capability, not an outcome: it is a pure function of the channel.
        """
        if channel_type in ("CP_POS_CHIP", "ATM_CASH"):
            pos_entry = "051"
        elif channel_type in ("CP_POS_CONTACTLESS", "CP_CONTACTLESS_NFC"):
            pos_entry = "071"
        elif channel_type == "CP_POS_MAGSTRIPE":
            pos_entry = "901"
        elif channel_type == "CNP_WEB":
            pos_entry = "012"
        elif channel_type == "CNP_MOBILE":
            pos_entry = "102"
        elif channel_type == "UPI_QR_CREDIT":
            pos_entry = "031"
        else:
            pos_entry = "812"
        return pos_entry, ("00" if channel_type.startswith("CP") else "59")

    @staticmethod
    def _hardware_crypto_verified(intent: CandidateTransactionIntent, channel_type: str) -> bool:
        """EMV Bit 55 cryptogram validation (Tag 9F26 checked by the issuer HSM).

        Visa Core Rules / Mastercard Chapter 17: a non-replayable ARQC is only produced by
        chip-capable terminals, so this is a function of the terminal and channel security
        level, never of the transaction's legitimacy.
        """
        return bool(
            (intent.emv_chip_present or channel_type == "CP_POS_CHIP")
            and intent.emv_cryptogram_valid
            and (channel_type in ("CP_POS_CHIP", "CP_POS_CONTACTLESS"))
        )

    def authenticate(
        self,
        intent: CandidateTransactionIntent,
        card: CardholderProfile,
    ) -> AuthenticationOutcome:
        """Runs the pre-authorization authentication step for one intent.

        Reads only pre-authentication signals: the channel's cryptographic capability and
        `intent.risk_score`, which the engine sets to the risk assessment made *before* any
        authentication result exists. That ordering is what keeps the causal ground truth
        (which consumes these outcomes as mitigating evidence) from depending on its own
        output, and it removes the old circular fallback that read the fraud label.
        """
        ch = intent.channel_type
        pos_entry, pos_condition = self._pos_credentials(ch)
        arqc_verified = self._hardware_crypto_verified(intent, ch)

        trans_status_3ds = ""
        eci = ""
        failed_reason = ""
        regulatory_rule_triggered = ""
        if ch.startswith("CNP"):
            # Strict RBI Statutory Rule (India Domestic): 100% Mandatory AFA (OTP)
            if (self.region == "IN" or card.currency == "INR") and not intent.is_cross_border:
                if intent.otp_submitted:
                    trans_status_3ds = "Y"
                    eci = "05"
                else:
                    trans_status_3ds = "N"
                    eci = "07"
                    failed_reason = "RBI_MANDATORY_AFA_OTP_REQUIRED"
                    regulatory_rule_triggered = "RBI_MANDATORY_AFA_2FA"
            else:
                # Global / US PSD2 & EMVCo 3DS 2.x Exemption Engine
                challenge = False
                if intent.amount < 30.0:
                    trans_status_3ds = "Y"
                    eci = "05"  # Low-Value Exemption (LVE)
                elif intent.risk_score < self.tau_challenge_low and intent.amount < 100.0:
                    trans_status_3ds = "Y"
                    eci = "05"  # TRA Exemption
                elif self.auth_rng.random() < self.telemetry.legit_challenge_share.get(self.region, 0.0):
                    # spec/07 section 4: the ACS challenges a share of frictionless-eligible traffic
                    challenge = True
                elif intent.risk_score < self.tau_challenge_high:
                    trans_status_3ds = "Y"
                    eci = "05"  # Frictionless
                else:
                    # Step-Up Challenge (risk_score >= tau_challenge_high)
                    challenge = True
                if challenge:
                    if intent.otp_submitted:
                        trans_status_3ds = "C"
                        eci = "05"
                    else:
                        trans_status_3ds = "N"
                        eci = "07"
                        failed_reason = "3DS_CHALLENGE_FAILED_OR_BYPASSED"

        return AuthenticationOutcome(
            pos_entry_mode=pos_entry,
            pos_condition_code=pos_condition,
            trans_status_3ds=trans_status_3ds,
            eci=eci,
            arqc_verified=arqc_verified,
            pin_verified=bool(arqc_verified and intent.pin_entered),
            tvr_clean=bool(arqc_verified),
            three_ds_authenticated=trans_status_3ds in ("Y", "C"),
            three_ds_attempted=trans_status_3ds in ("A", "C"),
            failed_reason=failed_reason,
            regulatory_rule_triggered=regulatory_rule_triggered,
        )

    def _enrich_result(
        self,
        result: RailVerificationResult,
        routing: Dict[str, Any],
    ) -> RailVerificationResult:
        """Enriches verification result with multi-bank routing and network switch telemetry."""
        result.issuing_bank_id = routing["issuing_bank"].id
        result.acquiring_bank_id = routing["acquiring_bank"].id
        result.is_on_us = routing["is_on_us"]
        result.central_switch_id = routing["central_switch_id"]
        result.switch_network_fee = routing["switch_network_fee"]
        result.interbank_latency_ms = round(routing["interbank_latency_ms"], 2)
        result.mti_request = routing["mti_request"]
        result.mti_response = routing["mti_response"]
        result.hop_sequence = list(routing["hop_sequence"])
        result.is_timeout_reversal = routing["is_timeout_reversal"]

        if routing["is_timeout_reversal"]:
            result.approved = False
            result.iso_response_code = "68"
            result.decline_reason = "NETWORK_TIMEOUT_REVERSAL_MTI_0400"
            result.hold_placed = False
            result.hold_amount = 0.0
            result.mti_request = "0400"
            result.mti_response = "0410"

        if result.approved:
            if routing["is_on_us"]:
                result.interchange_fee = 0.0
                result.settlement_amount = result.approved_amount
            else:
                result.interchange_fee = routing["interchange_fee"]
                result.settlement_amount = round(result.approved_amount - result.interchange_fee, 2)
        return result

    def verify_intent(
        self,
        intent: CandidateTransactionIntent,
        card: CardholderProfile,
    ) -> RailVerificationResult:
        """Evaluates candidate intent against banking rails, solvency, and multi-bank routing."""
        routing = self.routing_mesh.route_transaction(
            card_id=card.card_id,
            merchant_id=intent.merchant_id,
            channel_type=intent.channel_type,
            amount=intent.amount,
            is_fraud=intent.is_fraud,
        )
        raw_result = self._evaluate_intent_rules(intent, card)
        return self._enrich_result(raw_result, routing)

    def _evaluate_intent_rules(
        self,
        intent: CandidateTransactionIntent,
        card: CardholderProfile,
    ) -> RailVerificationResult:
        """Evaluates candidate intent against banking rails, solvency, and regulatory limits."""
        # 1. Terminal POS Entry Mode & Condition Codes
        ch = intent.channel_type
        pos_entry, pos_condition = self._pos_credentials(ch)

        # 1b. Authentication outcome. The terminal/ACS authenticates before the issuer's
        # authorization decision. The engine passes the outcome in on the intent (it needs
        # it for the causal ground truth too); a direct caller without one gets it computed
        # here, from the same pre-authentication signals.
        auth = intent.authentication
        if auth is None:
            auth = self.authenticate(intent, card)

        # 2. Card Status / Freeze Checks (Cardholder app lock returns ISO 57)
        if card.is_frozen or card.check_freeze_status(intent.sim_time_sec):
            return RailVerificationResult(
                approved=False,
                iso_response_code=ISO8583Response.NOT_PERMITTED_57.value if hasattr(ISO8583Response, 'NOT_PERMITTED_57') else "57",
                approved_amount=0.0,
                trans_status_3ds=auth.trans_status_3ds,
                eci=auth.eci,
                auth_code="",
                decline_reason="CARD_FROZEN_OR_BLOCKED_BY_CARDHOLDER",
                pos_entry_mode=pos_entry,
                pos_condition_code=pos_condition,
                hop_origin="ISSUER_HOST",
            )

        # 3. Hardware Cryptographic Primacy (Visa Core Rules / Mastercard Chapter 17)
        # Authentic EMV Contact Chip + PIN or Biometric CDCVM Wallet:
        # ARQC cryptogram (Tag 9F26) is non-replayable and validated by issuer cryptographic security module (HSM).
        is_verified_hardware_crypto = auth.arqc_verified

        # 4. Channel Controls (E-Commerce / International enabled)
        if ch.startswith("CNP") and not card.domestic_cnp_enabled:
            return RailVerificationResult(
                approved=False,
                iso_response_code=ISO8583Response.NOT_PERMITTED_57.value if hasattr(ISO8583Response, 'NOT_PERMITTED_57') else "57",
                approved_amount=0.0,
                trans_status_3ds=auth.trans_status_3ds,
                eci=auth.eci,
                auth_code="",
                decline_reason="DOMESTIC_CNP_DISABLED_BY_CARDHOLDER",
                pos_entry_mode=pos_entry,
                pos_condition_code=pos_condition,
                hop_origin="ISSUER_HOST",
            )

        if intent.is_cross_border and not card.international_enabled:
            return RailVerificationResult(
                approved=False,
                iso_response_code=ISO8583Response.NOT_PERMITTED_57.value if hasattr(ISO8583Response, 'NOT_PERMITTED_57') else "57",
                approved_amount=0.0,
                trans_status_3ds=auth.trans_status_3ds,
                eci=auth.eci,
                auth_code="",
                decline_reason="INTERNATIONAL_TRANSACTIONS_DISABLED",
                pos_entry_mode=pos_entry,
                pos_condition_code=pos_condition,
                hop_origin="ISSUER_HOST",
            )

        # 5. Kinematic Space-Time Geofencing (Supersonic Velocity Veto)
        # Commercial aviation maximum speed ceiling (~900 km/h).
        if ch.startswith("CP") and intent.haversine_velocity_kph > 900.0:
            return RailVerificationResult(
                approved=False,
                iso_response_code=ISO8583Response.SUSPECTED_FRAUD_59.value,
                approved_amount=0.0,
                trans_status_3ds=auth.trans_status_3ds,
                eci=auth.eci,
                auth_code="",
                decline_reason="SUPERSONIC_PHYSICAL_VELOCITY",
                pos_entry_mode=pos_entry,
                pos_condition_code=pos_condition,
                hop_origin="GATEWAY_FILTER",
            )

        # 6. Channel-Differentiated Velocity & Botnet Enumeration Limits (ISO 65)
        # A. Core banking daily count limit (FIS / Fiserv / TSYS standard: 35 tx/24h)
        if intent.tx_count_24h >= 35:
            return RailVerificationResult(
                approved=False,
                iso_response_code=ISO8583Response.ACTIVITY_COUNT_EXCEEDED_65.value,
                approved_amount=0.0,
                trans_status_3ds=auth.trans_status_3ds,
                eci=auth.eci,
                auth_code="",
                decline_reason="DAILY_TRANSACTION_COUNT_LIMIT_EXCEEDED",
                pos_entry_mode=pos_entry,
                pos_condition_code=pos_condition,
                hop_origin="ISSUER_HOST",
            )

        # B. Card-Not-Present Card-Testing & Botnet Enumeration Checks
        if ch.startswith("CNP"):
            # Multi-merchant automated enumeration (e.g. automated checker bot testing across distinct MIDs).
            # distinct_mids_30m is a PRIOR-WINDOW count (excludes current MID per ledger.py point-in-time semantics).
            # Grounded in Visa VAAI / Mastercard SafetyNet spec/03 channel_differentiated_velocity_ceilings:
            #   card_not_present_max_distinct_mids: 3 (prior unique MIDs in 30m = spec threshold).
            # Compound condition: spec threshold of 3 prior MIDs combined with a fraud signal indicator.
            # Unconditional veto at 5 prior distinct MIDs: even without VAAI signal, 5+ unique CNP merchants
            # in 30 minutes is statistically outside normal human shopping behavior.
            is_enumeration_attack = (
                (intent.distinct_mids_30m >= 3 and (
                    intent.vaai_score >= 60 or
                    intent.subminute_attempts_60s >= 2 or
                    (intent.amount <= 15.0 and intent.tx_attempts_1h >= 3)
                ))
                or intent.distinct_mids_30m >= 5
            )
            if is_enumeration_attack:
                return RailVerificationResult(
                    approved=False,
                    iso_response_code=ISO8583Response.ACTIVITY_COUNT_EXCEEDED_65.value,
                    approved_amount=0.0,
                    trans_status_3ds=auth.trans_status_3ds,
                    eci=auth.eci,
                    auth_code="",
                    decline_reason="CNP_MERCHANT_ENUMERATION_DETECTED",
                    pos_entry_mode=pos_entry,
                    pos_condition_code=pos_condition,
                    hop_origin="NETWORK_SWITCH_VAAI",
                )
            # High-frequency sub-minute automated script burst (>= 3 attempts in 60s)
            if intent.subminute_attempts_60s >= 3:
                return RailVerificationResult(
                    approved=False,
                    iso_response_code=ISO8583Response.ACTIVITY_COUNT_EXCEEDED_65.value,
                    approved_amount=0.0,
                    trans_status_3ds=auth.trans_status_3ds,
                    eci=auth.eci,
                    auth_code="",
                    decline_reason="HIGH_FREQUENCY_AUTOMATED_PROBE",
                    pos_entry_mode=pos_entry,
                    pos_condition_code=pos_condition,
                    hop_origin="GATEWAY_FILTER",
                )
            # Normal CNP hourly throttle (calibrated to multi-stop shopping bursts)
            if intent.tx_count_1h >= 10:
                return RailVerificationResult(
                    approved=False,
                    iso_response_code=ISO8583Response.ACTIVITY_COUNT_EXCEEDED_65.value,
                    approved_amount=0.0,
                    trans_status_3ds=auth.trans_status_3ds,
                    eci=auth.eci,
                    auth_code="",
                    decline_reason="CNP_HOURLY_FREQUENCY_LIMIT_EXCEEDED",
                    pos_entry_mode=pos_entry,
                    pos_condition_code=pos_condition,
                    hop_origin="ISSUER_HOST",
                )
        else:
            # Card-Present: Verified Contact Chip allows up to 15 tx/h (shopping trips, food courts, parking, transit)
            max_cp_hourly = 15 if is_verified_hardware_crypto else 10
            if intent.tx_count_1h >= max_cp_hourly:
                return RailVerificationResult(
                    approved=False,
                    iso_response_code=ISO8583Response.ACTIVITY_COUNT_EXCEEDED_65.value,
                    approved_amount=0.0,
                    trans_status_3ds=auth.trans_status_3ds,
                    eci=auth.eci,
                    auth_code="",
                    decline_reason="CP_HOURLY_FREQUENCY_LIMIT_EXCEEDED",
                    pos_entry_mode=pos_entry,
                    pos_condition_code=pos_condition,
                    hop_origin="ISSUER_HOST",
                )

        # 7. Contactless Regulatory Rails (India RBI vs Global)
        if ch == "CP_POS_CONTACTLESS":
            if self.region == "IN" or card.currency == "INR":
                # RBI Master Direction: ₹5,000 PIN-free ceiling
                if intent.amount > 5000.0 and not intent.pin_entered:
                    return RailVerificationResult(
                        approved=False,
                        iso_response_code=ISO8583Response.ACTIVITY_LIMIT_EXCEEDED_65.value,
                        approved_amount=0.0,
                        trans_status_3ds=auth.trans_status_3ds,
                        eci=auth.eci,
                        auth_code="",
                        decline_reason="RBI_NFC_PIN_REQUIRED_OVER_5000_INR",
                        regulatory_rule_triggered="RBI_NFC_PIN_MANDATE",
                        pos_entry_mode=pos_entry,
                        pos_condition_code=pos_condition,
                        hop_origin="TERMINAL_SCA",
                    )
                # 5 consecutive PINless transaction ceiling
                if card.consecutive_pinless_contactless_count >= 5 and not intent.pin_entered:
                    return RailVerificationResult(
                        approved=False,
                        iso_response_code=ISO8583Response.ACTIVITY_LIMIT_EXCEEDED_65.value,
                        approved_amount=0.0,
                        trans_status_3ds=auth.trans_status_3ds,
                        eci=auth.eci,
                        auth_code="",
                        decline_reason="RBI_NFC_CONSECUTIVE_LIMIT_REACHED",
                        regulatory_rule_triggered="RBI_NFC_VELOCITY_CAP",
                        pos_entry_mode=pos_entry,
                        pos_condition_code=pos_condition,
                        hop_origin="ISSUER_HOST",
                    )
                # Cumulative ₹15,000 PINless transaction ceiling
                if (card.cumulative_pinless_contactless_amount + intent.amount > 15000.0) and not intent.pin_entered:
                    return RailVerificationResult(
                        approved=False,
                        iso_response_code=ISO8583Response.ACTIVITY_LIMIT_EXCEEDED_65.value,
                        approved_amount=0.0,
                        trans_status_3ds=auth.trans_status_3ds,
                        eci=auth.eci,
                        auth_code="",
                        decline_reason="RBI_NFC_CUMULATIVE_LIMIT_EXCEEDED_15000_INR",
                        regulatory_rule_triggered="RBI_NFC_CUMULATIVE_CAP",
                        pos_entry_mode=pos_entry,
                        pos_condition_code=pos_condition,
                        hop_origin="ISSUER_HOST",
                    )

        # 8. Monetary Bounds Validation
        max_cap = 20_000_000.0 if (self.region == "IN" or card.currency == "INR") else 250_000.0
        if intent.amount <= 0.0 or intent.amount > max_cap:
            return RailVerificationResult(
                approved=False,
                iso_response_code=ISO8583Response.DO_NOT_HONOR_05.value,
                approved_amount=0.0,
                trans_status_3ds=auth.trans_status_3ds,
                eci=auth.eci,
                auth_code="",
                decline_reason="INVALID_AMOUNT_BOUNDS",
                pos_entry_mode=pos_entry,
                pos_condition_code=pos_condition,
                hop_origin="ISSUER_HOST",
            )

        # 9. VAAI Network Intelligence Check. A verified EMV cryptogram keeps hardware primacy (step 3):
        # the network score does not override an ARQC the issuer's HSM has validated.
        if intent.vaai_score >= self.telemetry.network_decline_threshold and not is_verified_hardware_crypto:
            return RailVerificationResult(
                approved=False,
                iso_response_code=ISO8583Response.SUSPECTED_FRAUD_59.value,
                approved_amount=0.0,
                trans_status_3ds=auth.trans_status_3ds,
                eci=auth.eci,
                auth_code="",
                decline_reason="VAAI_NETWORK_FRAUD_SCORE_EXCEEDED",
                pos_entry_mode=pos_entry,
                pos_condition_code=pos_condition,
                hop_origin="NETWORK_SWITCH_VAAI",
            )

        # 10. 3DS 2.x Authentication Outcome (decided once, in `authenticate()`)
        # The ACS decision is already made: it must not be re-decided here from the
        # post-authentication risk score, or the ground truth that consumed it and the
        # authorization that re-derived it could disagree on the same transaction.
        trans_status_3ds = auth.trans_status_3ds
        eci = auth.eci
        if auth.failed_reason:
            return RailVerificationResult(
                approved=False,
                iso_response_code=ISO8583Response.SECURITY_VIOLATION_63.value,
                approved_amount=0.0,
                trans_status_3ds=trans_status_3ds,
                eci=eci,
                auth_code="",
                decline_reason=auth.failed_reason,
                regulatory_rule_triggered=auth.regulatory_rule_triggered,
                pos_entry_mode=pos_entry,
                pos_condition_code=pos_condition,
                hop_origin="ACS_3DS",
            )

        # 11. Cryptographic & Security Credentials Verification
        if ch.startswith("CNP") and not intent.cvv_provided:
            return RailVerificationResult(
                approved=False,
                iso_response_code=ISO8583Response.INVALID_CVV_82.value if hasattr(ISO8583Response, 'INVALID_CVV_82') else "82",
                approved_amount=0.0,
                trans_status_3ds=trans_status_3ds,
                eci=eci,
                auth_code="",
                decline_reason="CVV_MATCH_FAILED",
                pos_entry_mode=pos_entry,
                pos_condition_code=pos_condition,
                hop_origin="ISSUER_HOST",
            )

        # 12. Solvency & Balance Evaluation (Available Credit / Balance)
        avail = card.get_available_balance()
        if intent.amount > avail:
            min_afd_avail = 500.0 if (self.region == "IN" or card.currency == "INR") else 10.0
            # Partial approval strictly negotiated for Automated Fuel Dispensers (MCC 5542) or Transit (MCC 4111, 4784)
            if intent.mcc in (5542, 4111, 4784) and avail >= min_afd_avail:
                approved_amt = round(avail, 2)
                auth_code = f"A{self.rng.integers(10000, 99999)}"
                return RailVerificationResult(
                    approved=True,
                    iso_response_code=ISO8583Response.PARTIAL_APPROVAL_10.value,
                    approved_amount=approved_amt,
                    trans_status_3ds=trans_status_3ds or "Y",
                    eci=eci,
                    auth_code=auth_code,
                    hold_placed=True,
                    hold_amount=approved_amt,
                    pos_entry_mode=pos_entry,
                    pos_condition_code=pos_condition,
                    hop_origin="ISSUER_HOST",
                )
            else:
                return RailVerificationResult(
                    approved=False,
                    iso_response_code=ISO8583Response.INSUFFICIENT_FUNDS_51.value,
                    approved_amount=0.0,
                    trans_status_3ds=trans_status_3ds,
                    eci=eci,
                    auth_code="",
                    decline_reason="INSUFFICIENT_FUNDS_OR_CREDIT_LIMIT_EXCEEDED",
                    pos_entry_mode=pos_entry,
                    pos_condition_code=pos_condition,
                    hop_origin="ISSUER_HOST",
                )

        # 13. Real-Time Risk Score Thresholding (WITHOUT SYNTHETIC ORACLE LEAK)
        # Authentic EMV Contact Chip + valid cryptogram carries statutory counterfeit dispute protection
        if is_verified_hardware_crypto:
            pass  # Hardware cryptogram verified
        elif intent.risk_score >= self.tau_decline:
            return RailVerificationResult(
                approved=False,
                iso_response_code=ISO8583Response.SUSPECTED_FRAUD_59.value,
                approved_amount=0.0,
                trans_status_3ds=trans_status_3ds or "N",
                eci=eci,
                auth_code="",
                decline_reason="BANK_ML_DECISION_ENGINE_FRAUD_DECLINE",
                pos_entry_mode=pos_entry,
                pos_condition_code=pos_condition,
                hop_origin="ISSUER_HOST",
            )

        # 14. Transaction Approval
        auth_code = f"A{self.rng.integers(10000, 99999)}"
        interchange_rate = 0.0175
        interchange = round(intent.amount * interchange_rate, 2)
        settlement = round(intent.amount - interchange, 2)

        return RailVerificationResult(
            approved=True,
            iso_response_code=ISO8583Response.APPROVED_00.value,
            approved_amount=intent.amount,
            trans_status_3ds=trans_status_3ds or "Y",
            eci=eci,
            auth_code=auth_code,
            hold_placed=True,
            hold_amount=intent.amount,
            interchange_fee=interchange,
            settlement_amount=settlement,
            pos_entry_mode=pos_entry,
            pos_condition_code=pos_condition,
            hop_origin="ISSUER_HOST",
        )
