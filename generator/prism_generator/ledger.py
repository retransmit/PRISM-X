"""Real-time streaming ledger and rolling feature accumulator for Prism Generator.

Enforces:
1. Point-in-time causality: Features are computed strictly BEFORE mutating state (zero lookahead).
2. Strict temporal monotonicity: Operates synchronously with discrete-event priority queue.
3. Kinematic state protection: Foreign attacker coordinates do not poison legitimate card anchors.
4. Robust Welford accumulator with calibrated prior moments.
5. Multi-resolution financial units: Minor integer units (Cents/Paisa) and display amounts.
"""

from __future__ import annotations

import collections
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from typing import Any, Deque, Dict, List, Optional, Tuple
import zlib

import numpy as np

from .agents import CardholderProfile
from .invariants import haversine_distance_km
from .network import ClientAddressSpace
from .spec_loader import load_all_specs


class WelfordAccumulator:
    """Online running mean and sample variance computation via Welford's algorithm."""

    def __init__(self, count: int, mean: float, m2: float):
        self.count = max(1, count)
        self.mean = mean
        self.m2 = max(0.0, m2)

    def update(self, x: float) -> None:
        """Updates mean and M2 sum of squared differences with a new observation."""
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        """Sample variance s^2 = M2 / (count - 1)."""
        if self.count <= 1:
            return 0.0
        return max(0.0, self.m2 / (self.count - 1))

    @property
    def std(self) -> float:
        """Sample standard deviation s = sqrt(s^2)."""
        return math.sqrt(self.variance)

    def compute_z_score(self, x: float) -> float:
        """Standardized score z = (x - mean) / std. Regularized with Bayesian tail-shrinkage to avoid division by zero and heavy-tail blowups."""
        s = max(self.std, 0.15 * max(self.mean, 1.0), 0.08 * x, 3.0)
        return (x - self.mean) / s


@dataclass
class CardholderLedgerState:
    """Stateful tracking container for an individual cardholder."""
    card_id: str
    welford_30d: WelfordAccumulator
    tx_history_authorized_1h: Deque[Tuple[float, float]] = field(default_factory=collections.deque)        # (timestamp, amount) of approved transactions
    tx_history_attempts_1h: Deque[Tuple[float, float, str]] = field(default_factory=collections.deque)      # (timestamp, amount, merchant_id) of all attempts
    tx_history_24h: Deque[Tuple[float, float, str]] = field(default_factory=collections.deque)             # (timestamp, amount, merchant_id) of approved transactions
    last_tx_time: float = -1.0
    last_tx_lat: float = 0.0
    last_tx_lon: float = 0.0

    @property
    def tx_history_1h(self) -> Deque[Tuple[float, float]]:
        """Backwards-compatibility property returning authorized transactions."""
        return self.tx_history_authorized_1h

    def prune_expired(self, current_time: float) -> None:
        """Prunes historical entries older than rolling time horizons (strictly monotonic)."""
        cutoff_1h = current_time - 3600.0
        while self.tx_history_authorized_1h and self.tx_history_authorized_1h[0][0] <= cutoff_1h:
            self.tx_history_authorized_1h.popleft()

        while self.tx_history_attempts_1h and self.tx_history_attempts_1h[0][0] <= cutoff_1h:
            self.tx_history_attempts_1h.popleft()

        cutoff_24h = current_time - 86400.0
        while self.tx_history_24h and self.tx_history_24h[0][0] <= cutoff_24h:
            self.tx_history_24h.popleft()


class StreamingLedger:
    """Thread-safe streaming feature store maintaining point-in-time feature states."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.card_states: Dict[str, CardholderLedgerState] = {}
        self.telemetry = load_all_specs().telemetry
        # spec/07 sections 16 and 17: the prefixes this run's traffic comes from, and how devices are shared
        self.address_space = ClientAddressSpace(seed=seed)
        self.device_sharing = dict(self.telemetry.device_sharing)
        self.double_entry = DoubleEntryWorldLedger()

    def _cardholder_device_seed(self, card: CardholderProfile, channel_type: str, tx_time: float) -> str:
        """Which of the cardholder's devices this session is on (spec/07 section 17)."""
        spec = self.device_sharing
        draw = float(self.rng.random())
        household = float(spec.get("legitimate_household_device_share", 0.0))
        secondary = float(spec.get("legitimate_secondary_device_share", {}).get(channel_type, 0.0))
        one_off = float(spec.get("legitimate_one_off_device_share", 0.0))
        if draw < household:
            digits = re.sub(r"\D", "", card.card_id)
            index = int(digits) if digits else zlib.crc32(card.card_id.encode("utf-8"))
            return f"HOUSEHOLD_{card.region}_{index // max(1, int(spec.get('household_size', 3)))}"
        if draw < household + secondary:
            return f"{card.card_id}_{card.home_lat:.3f}_second"
        if draw < household + secondary + one_off:
            return f"{card.card_id}_{card.home_lat:.3f}_{tx_time}"
        return f"{card.card_id}_{card.home_lat:.3f}"

    def get_or_create_state(self, card: CardholderProfile) -> CardholderLedgerState:
        """Retrieves cardholder state or initializes with theoretical lognormal priors."""
        if card.card_id not in self.card_states:
            prior_mean = card.get_theoretical_mean_spend()
            sig2 = card.spend_sigma_log ** 2
            ln_var = (math.exp(sig2) - 1.0) * math.exp(2.0 * card.spend_mean_log + sig2)
            if card.is_spliced_gpd:
                u = card.gpd_threshold_u
                xi = max(0.05, min(0.45, card.gpd_xi))
                beta = card.gpd_beta
                p = card.gpd_tail_prob
                ln_mean = math.exp(card.spend_mean_log + 0.5 * sig2)
                e_gpd2 = (u ** 2) + 2.0 * u * beta / (1.0 - xi) + (2.0 * (beta ** 2)) / ((1.0 - xi) * (1.0 - 2.0 * xi))
                e_ln2 = ln_var + (ln_mean ** 2)
                total_e2 = (1.0 - p) * e_ln2 + p * e_gpd2
                prior_var = max(ln_var, total_e2 - (prior_mean ** 2))
            else:
                prior_var = ln_var
            prior_m2 = 11.0 * prior_var

            accumulator = WelfordAccumulator(count=12, mean=prior_mean, m2=prior_m2)
            self.card_states[card.card_id] = CardholderLedgerState(
                card_id=card.card_id,
                welford_30d=accumulator,
                last_tx_time=-1.0,
                last_tx_lat=card.home_lat,
                last_tx_lon=card.home_lon,
            )
        return self.card_states[card.card_id]

    def enrich_transaction(
        self,
        tx_id: str,
        card: CardholderProfile,
        merchant_id: str,
        merchant_name: str,
        mcc: int,
        merchant_category: str,
        merchant_lat: float,
        merchant_lon: float,
        amount: float,
        tx_time: float,
        channel_type: str,
        is_fraud: int,
        scenario_tag: str,
        mid: str = "",
        tid: str = "",
        acquirer_bin: str = "",
        gateway_provider: str = "",
        merchant_country: str = "US",
        postal_code: str = "10001",
        is_cross_border: bool = False,
        ip_distance_km: float = 0.0,
        asn_type: str = "residential",
        override_cvv_match: Optional[int] = None,
        override_avs_code: Optional[str] = None,
        override_billing_match: Optional[int] = None,
        override_client_ip: Optional[str] = None,
        syndicate_id: str = "",
        botnet_cluster_id: str = "",
        mule_ring_id: str = "",
        beneficiary_account_id: str = "",
        ip_subnet_prefix: str = "",
        device_fingerprint_id: str = "",
        credentials_complete: bool = False,
        victim_device: bool = False,
    ) -> Dict[str, Any]:
        """Enriches raw transaction parameters into full institutional telemetry record.

        credentials_complete: the attacker holds the full cardholder record (spec/07 section 11), so
        AVS, CVV and billing outcomes follow the legitimate distribution.
        victim_device: the attack runs on the cardholder's own device and connection (spec/07 section 17).
        """
        state = self.get_or_create_state(card)

        # 1. Prune expired entries from rolling windows prior to computing features
        state.prune_expired(tx_time)

        # 2. Point-in-time calculation: Read features BEFORE appending current transaction
        count_1h = len(state.tx_history_authorized_1h)
        attempts_1h = len(state.tx_history_attempts_1h)
        count_24h = len(state.tx_history_24h)
        sum_24h = sum(a for _, a, _ in state.tx_history_24h)
        distinct_merchants_24h = len(set(m for _, _, m in state.tx_history_24h))

        # Recent 30m distinct MIDs and 60s subminute attempts from attempt history.
        # distinct_mids_30m is a PRIOR-WINDOW measure: it counts unique MIDs seen before
        # this transaction attempt (point-in-time). The current merchant_id is intentionally
        # excluded; it has not yet been attempted. This matches the spec's enumeration
        # threshold semantics: a cardholder with 3 prior distinct MIDs in 30m = spec ceiling.
        cutoff_30m = tx_time - 1800.0
        prior_mids_30m = set(m for t, _, m in state.tx_history_attempts_1h if t > cutoff_30m)
        distinct_mids_30m = len(prior_mids_30m)
        cutoff_60s = tx_time - 60.0
        subminute_attempts_60s = len([t for t, _, _ in state.tx_history_attempts_1h if t > cutoff_60s])

        user_mean_30d = state.welford_30d.mean
        user_std_30d = state.welford_30d.std
        z_score_amount = state.welford_30d.compute_z_score(amount)

        # 3. Kinematic velocity computation
        velocity_kph = 0.0
        distance_from_last_tx_km = 0.0
        time_since_last_tx_sec = 0.0

        if channel_type.startswith("CP") or scenario_tag in ("IMPOSSIBLE_TRAVEL", "COUNTERFEIT_CLONE"):
            if card.last_physical_time >= 0.0:
                time_since_last_tx_sec = max(0.0, tx_time - card.last_physical_time)
                distance_from_last_tx_km = haversine_distance_km(
                    card.last_physical_lat,
                    card.last_physical_lon,
                    merchant_lat,
                    merchant_lon,
                )
                delta_hours = max(time_since_last_tx_sec / 3600.0, 1e-5)
                velocity_kph = distance_from_last_tx_km / delta_hours

        # 4. Digital Telemetry Sampling (AVS, CVV, Geolocation, Device Fingerprint)
        if channel_type.startswith("CP"):
            billing_shipping_match = 1
            avs_match_code = "Y"
            cvv_match_flag = 1
            asn_type = "residential"
            geo_risk_score = 0
            client_ip = "127.0.0.1"
        else:
            legit_like = (is_fraud == 0) or credentials_complete
            if override_cvv_match is not None:
                cvv_match_flag = override_cvv_match
            else:
                cvv_match_flag = 1 if (self.rng.random() < 0.98 if legit_like else self.rng.random() < 0.82) else 0

            if override_avs_code is not None:
                avs_match_code = override_avs_code
            elif is_cross_border and self.rng.random() < self.telemetry.avs_unavailable_probability_cross_border:
                # spec/03 U = UNAVAILABLE_INTERNATIONAL: a property of the merchant, not of the label (spec/07 section 6)
                avs_match_code = "U"
            elif legit_like:
                codes = list(self.telemetry.legit_avs_weights.keys())
                weights = np.array([self.telemetry.legit_avs_weights[c] for c in codes], dtype=float)
                avs_match_code = str(self.rng.choice(codes, p=weights / weights.sum()))
            else:
                codes = list(self.telemetry.fraud_avs_weights.keys())
                weights = np.array([self.telemetry.fraud_avs_weights[c] for c in codes], dtype=float)
                avs_match_code = str(self.rng.choice(codes, p=weights / weights.sum()))

            if override_billing_match is not None:
                billing_shipping_match = override_billing_match
            else:
                billing_shipping_match = 1 if (self.rng.random() < 0.93 if legit_like else self.rng.random() < 0.60) else 0

            # 5-way Geolocation mismatch score from spec/03 (independent of target label)
            geo_risk_score = 0
            if is_cross_border:
                geo_risk_score += 35
            if billing_shipping_match == 0:
                geo_risk_score += 25
            if avs_match_code == "N":
                geo_risk_score += 20
            if asn_type == "datacenter":
                geo_risk_score += int(self.rng.choice([25, 30, 35]))
            geo_risk_score = min(100, geo_risk_score)

            # The client address: the attacker's proxy when given, else the cardholder's own
            # connection on the prefixes this run's traffic shares (spec/07 section 16)
            if override_client_ip:
                client_ip = override_client_ip
            else:
                client_ip = self.address_space.legitimate_ip(card.card_id, card.region, channel_type, asn_type, self.rng)

        # Device canvas hash: the botnet's device, the attacker's own (seen once), or one of the
        # cardholder's devices, which a victim-device attack also uses (spec/07 section 17).
        # A card-present payment comes from a terminal, not a device (section 20).
        if channel_type.startswith("CP"):
            canvas_hash = ""
        elif device_fingerprint_id:
            canvas_hash = hashlib.md5(device_fingerprint_id.encode("utf-8")).hexdigest()[:16]
        elif is_fraud == 1 and not victim_device:
            canvas_hash = hashlib.md5(f"{card.card_id}_{card.home_lat:.3f}_{tx_time}".encode("utf-8")).hexdigest()[:16]
        else:
            canvas_hash = hashlib.md5(self._cardholder_device_seed(card, channel_type, tx_time).encode("utf-8")).hexdigest()[:16]

        # Construct ISO-8601 UTC timestamp
        dt_utc = datetime.fromtimestamp(tx_time, tz=timezone.utc)
        hour_of_day = dt_utc.hour
        day_of_week = dt_utc.weekday()

        # Minor currency unit conversion (cents for USD, paisa for INR)
        amount_minor = int(round(amount * 100))

        # Build complete transaction dictionary matching institutional production schemas
        record: Dict[str, Any] = {
            "transaction_id": tx_id,
            "card_id": card.card_id,
            "pan_masked": card.pan_masked,
            "product_id": card.product_id,
            "cohort_id": card.cohort_id,
            "merchant_id": merchant_id,
            "merchant_name": merchant_name,
            "mid": mid if mid else f"MID_{merchant_id}",
            "tid": tid if tid else f"TID_{merchant_id[-4:]}",
            "mcc": mcc,
            "merchant_category": merchant_category,
            "merchant_lat": merchant_lat,
            "merchant_lon": merchant_lon,
            "acquirer_bin": acquirer_bin if acquirer_bin else "400012",
            "gateway_provider": gateway_provider if gateway_provider else "STRIPE",
            "country_code": merchant_country,
            "postal_code": postal_code,
            "timestamp_utc": dt_utc.isoformat(),
            "tx_time_seconds": round(tx_time, 2),
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "amount": round(amount, 2),
            "amount_minor": amount_minor,
            "currency": card.currency,
            "channel_type": channel_type,
            "credit_limit": card.credit_limit,
            "current_balance": round(card.current_balance, 2),
            "available_balance": round(card.get_available_balance(), 2),
            "user_avg_tx_amount_30d": round(user_mean_30d, 2),
            "user_std_tx_amount_30d": round(user_std_30d, 2),
            "z_score_amount_30d": round(z_score_amount, 3),
            "tx_count_1h": count_1h,
            "tx_count_24h": count_24h,
            "tx_attempts_1h": attempts_1h,
            "distinct_mids_30m": distinct_mids_30m,
            "subminute_attempts_60s": subminute_attempts_60s,
            "tx_amount_sum_24h": round(sum_24h, 2),
            "distinct_merchants_24h": distinct_merchants_24h,
            "distance_from_last_tx_km": round(distance_from_last_tx_km, 3),
            "time_since_last_tx_seconds": round(time_since_last_tx_sec, 1),
            "haversine_velocity_kph": round(velocity_kph, 2),
            "ip_distance_from_home_km": round(ip_distance_km, 2),
            "client_ip": client_ip,
            "asn_type": asn_type,
            "geo_risk_score": geo_risk_score,
            "device_canvas_hash": canvas_hash,
            "is_cross_border": is_cross_border,
            "billing_shipping_match": billing_shipping_match,
            "avs_match_code": avs_match_code,
            "cvv_match_flag": cvv_match_flag,
            "is_fraud": is_fraud,
            "scenario_tag": scenario_tag,
            "syndicate_id": syndicate_id,
            "botnet_cluster_id": botnet_cluster_id,
            "mule_ring_id": mule_ring_id,
            "beneficiary_account_id": beneficiary_account_id,
            "ip_subnet_prefix": ip_subnet_prefix,
            "device_fingerprint_id": device_fingerprint_id,
        }

        # 5. Record attempt and spatial state (Point-in-time discipline)
        state.tx_history_attempts_1h.append((tx_time, amount, merchant_id))
        state.last_tx_time = tx_time
        state.last_tx_lat = merchant_lat
        state.last_tx_lon = merchant_lon

        # Physical location anchor protection for cardholder (legitimate CP transactions)
        if channel_type.startswith("CP") and is_fraud == 0:
            card.last_physical_lat = merchant_lat
            card.last_physical_lon = merchant_lon
            card.last_physical_time = tx_time
            card.last_merchant_id = merchant_id

        # Actively release any expired pre-auth holds
        expired_ids = card.prune_expired_holds(tx_time)
        for exp_tx_id in expired_ids:
            self.double_entry.release_hold(exp_tx_id, card.card_id, sim_time_sec=tx_time)

        return record

    def record_authorization_outcome(
        self,
        card: CardholderProfile,
        amount: float,
        merchant_id: str,
        tx_time: float,
        is_approved: bool,
        response_code: str = "00",
    ) -> None:
        """Records authorization outcome into the ledger's authorized transaction histories."""
        state = self.get_or_create_state(card)
        if not state.tx_history_attempts_1h or state.tx_history_attempts_1h[-1][0] != tx_time:
            state.tx_history_attempts_1h.append((tx_time, amount, merchant_id))
        if is_approved or response_code in ("00", "10"):
            state.tx_history_authorized_1h.append((tx_time, amount))
            state.tx_history_24h.append((tx_time, amount, merchant_id))
            state.welford_30d.update(amount)
        # Actively release any expired pre-auth holds
        expired_ids = card.prune_expired_holds(tx_time)
        for exp_tx_id in expired_ids:
            self.double_entry.release_hold(exp_tx_id, card.card_id, sim_time_sec=tx_time)


class DoubleEntryWorldLedger:
    """Multi-party double-entry ledger whose conservation invariant can actually fail.

    Sign convention: `accounts` holds signed balances in **minor integer units**
    (cents / paisa), positive meaning value held by that party. Value moving between
    parties leaves one account and enters another, so the accounts sum to exactly
    zero - integer arithmetic, not floating-point luck.

    Two account families:
    - *Conservation accounts* (`accounts`): escrow, card availability, merchant
      settlement, issuer interchange, network assessment. Only these are summed by
      `verify_global_balance_conservation`.
    - *Memo accounts* (`memo_accounts`): bookkeeping notes such as a cardholder's
      cumulative posted spend. A memo reports on transactions but holds no value of
      its own; recording it as a conservation account is what used to mint money at
      every settlement, because the cardholder's outflow was already booked when the
      hold was placed and the posted-balance line had no counterparty.

    Every mutation goes through `_post`, which refuses an unbalanced posting, and
    `verify_global_balance_conservation` re-derives the invariant from stored state
    (journal legs, account sum, journal replay) instead of trusting a running total
    that was incremented by the same code that performed the mutation.
    """

    def __init__(self, max_journal_entries: int = 2000):
        # Conservation accounts: signed balances in minor units (cents/paisa)
        self.accounts: Dict[str, int] = collections.defaultdict(int)
        # Memo accounts: reporting-only figures in minor units, excluded from conservation
        self.memo_accounts: Dict[str, int] = collections.defaultdict(int)
        # Pre-authorization hold state machine: tx_id -> (card_id, hold_amount, status)
        self.active_holds: Dict[str, Tuple[str, float, str]] = {}
        # Bounded double-entry journal ring buffer for auditing/tests
        self.journal: collections.deque[Dict[str, Any]] = collections.deque(maxlen=max_journal_entries)
        # Postings ever applied; equals len(journal) only while the ring is unevicted
        self._postings: int = 0

    @staticmethod
    def _to_minor(amount_major: float) -> int:
        """Converts a major-unit amount (USD/INR) to integer minor units (cents/paisa).

        Amounts are 2-decimal in this simulator, so this is an exact conversion; it
        raises on anything that cannot be represented in whole minor units rather
        than silently truncating value.
        """
        minor = int(round(float(amount_major) * 100.0))
        if not math.isclose(float(amount_major) * 100.0, minor, abs_tol=1e-6):
            raise ValueError(f"{amount_major!r} is not a whole number of minor units")
        return minor

    def _post(
        self,
        tx_id: str,
        action: str,
        sim_time_sec: float,
        debits: Dict[str, float],
        credits: Dict[str, float],
        memo: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Applies one balanced posting and appends it to the journal.

        Debits reduce an account, credits increase it. A posting whose legs do not
        sum to the same total would create or destroy value, so it is rejected here
        with an explicit error instead of being written.
        """
        debit_minor = {acct: self._to_minor(v) for acct, v in debits.items()}
        credit_minor = {acct: self._to_minor(v) for acct, v in credits.items()}

        total_d = sum(debit_minor.values())
        total_c = sum(credit_minor.values())
        if total_d != total_c:
            raise ValueError(
                f"Unbalanced double-entry posting {action} on {tx_id}: "
                f"debits {total_d} minor units != credits {total_c} minor units"
            )

        for acct, amount in debit_minor.items():
            self.accounts[acct] -= amount
        for acct, amount in credit_minor.items():
            self.accounts[acct] += amount
        for acct, amount in (memo or {}).items():
            self.memo_accounts[acct] += self._to_minor(amount)

        entry: Dict[str, Any] = {
            "tx_id": tx_id,
            "timestamp": sim_time_sec,
            "action": action,
            "debits": dict(debits),
            "credits": dict(credits),
        }
        if memo:
            entry["memo"] = dict(memo)
        self.journal.append(entry)
        self._postings += 1
        return entry

    def place_pre_auth_hold(
        self,
        tx_id: str,
        card_id: str,
        hold_amount: float,
        sim_time_sec: float,
    ) -> bool:
        """Places pre-authorization hold reserving available funds in escrow.
        
        Double-Entry Leg:
        - Debit: CARD_AVAILABLE:{card_id} (-hold_amount)
        - Credit: ESCROW_HOLD:{card_id} (+hold_amount)
        """
        hold_amount = round(hold_amount, 2)

        self._post(
            tx_id=tx_id,
            action="PRE_AUTH_HOLD",
            sim_time_sec=sim_time_sec,
            debits={f"card_available:{card_id}": hold_amount},
            credits={f"escrow_hold:{card_id}": hold_amount},
        )
        self.active_holds[tx_id] = (card_id, hold_amount, "HELD")
        return True

    def settle_hold(
        self,
        tx_id: str,
        card_id: str,
        merchant_id: str,
        settled_amount: float,
        interchange_rate: float = 0.0175,
        network_fee_rate: float = 0.0015,
        sim_time_sec: float = 0.0,
    ) -> Tuple[float, float, float]:
        """Settles clearing presentment with exact multi-party conservation.

        Double-Entry Legs (all value-bearing, all part of the conservation set):
        - Debit ESCROW_HOLD:{card_id} (hold_amount) - escrow gives up the reserved value
        - Credit CARD_AVAILABLE:{card_id} (hold - settled) - unused reservation returns
        - Credit MERCHANT_SETTLEMENT:{merchant_id} (settled - interchange - network)
        - Credit ISSUER_INTERCHANGE (interchange)
        - Credit NETWORK_ASSESSMENT (network)

        Conservation: hold == (hold - settled) + settled, and settled decomposes into
        merchant net + interchange + network by construction.

        The cardholder's cumulative posted spend is written to the *memo* ledger for
        reporting. It is not a conservation account: the cardholder's value already
        left `card_available` when the hold was placed, so booking `CARD_POSTED` as a
        value-holding debit as well would double-count the charge.
        """
        settled_amount = round(settled_amount, 2)
        interchange_fee = round(settled_amount * interchange_rate, 2)
        network_fee = round(settled_amount * network_fee_rate, 2)
        merchant_net = round(settled_amount - interchange_fee - network_fee, 2)

        # Retrieve hold
        hold_data = self.active_holds.pop(tx_id, (card_id, settled_amount, "HELD"))
        _, hold_amount, _ = hold_data

        escrow_acct = f"escrow_hold:{card_id}"
        card_avail_acct = f"card_available:{card_id}"
        card_posted_acct = f"card_posted:{card_id}"
        merchant_acct = f"merchant_settlement:{merchant_id}"

        unused_hold = round(hold_amount - settled_amount, 2)

        debits = {escrow_acct: hold_amount}
        credits: Dict[str, float] = {}
        if unused_hold != 0.0:
            credits[card_avail_acct] = unused_hold
        credits[merchant_acct] = merchant_net
        credits["issuer_interchange"] = interchange_fee
        credits["network_assessment"] = network_fee

        self._post(
            tx_id=tx_id,
            action="CLEARING_SETTLEMENT",
            sim_time_sec=sim_time_sec,
            debits=debits,
            credits=credits,
            memo={card_posted_acct: settled_amount},
        )
        return merchant_net, interchange_fee, network_fee

    def release_hold(self, tx_id: str, card_id: str, sim_time_sec: float = 0.0) -> float:
        """Releases hold upon decline, reversal, or expiry, returning reserved value."""
        if tx_id not in self.active_holds:
            return 0.0

        _, hold_amount, _ = self.active_holds.pop(tx_id)
        self._post(
            tx_id=tx_id,
            action="HOLD_RELEASE",
            sim_time_sec=sim_time_sec,
            debits={f"escrow_hold:{card_id}": hold_amount},
            credits={f"card_available:{card_id}": hold_amount},
        )
        return hold_amount

    def verify_global_balance_conservation(self, abs_tol: float = 1e-6) -> Tuple[bool, float]:
        """Re-derives conservation from stored state instead of trusting a counter.

        Three independent residuals are computed, and the largest is returned in
        major units (USD/INR):

        1. **Journal legs** - any retained entry whose debits and credits differ is a
           posting that minted value, even if it was written by code that believed
           it was balanced.
        2. **Account sum** - `sum(accounts)` must be exactly zero; a non-zero value
           means value entered or left the system outside a posting (the classic
           being a reporting figure booked as a conserved balance).
        3. **Journal replay** - while the journal ring buffer has dropped no entries,
           replaying every entry must reproduce `accounts` exactly, which fails if an
           account was mutated by anything other than `_post`.

        Returns `(is_conserved, discrepancy)`; `is_conserved` is true only when
        `discrepancy < abs_tol`.
        """
        worst = 0.0

        def minor(value: Any) -> Optional[int]:
            """Minor units, or None if the stored amount is not a whole minor unit.

            A value that cannot be converted is itself a defect in stored state, so
            it is reported as a failure below rather than raising out of an invariant
            checker (a checker that throws cannot say whether the invariant holds).
            """
            try:
                return self._to_minor(value)
            except (TypeError, ValueError):
                return None

        # 1. Per-entry leg balance
        for entry in self.journal:
            leg_totals = []
            for leg in ("debits", "credits"):
                values = [minor(v) for v in entry.get(leg, {}).values()]
                if any(v is None for v in values):
                    return False, float("inf")
                leg_totals.append(sum(v for v in values if v is not None))
            worst = max(worst, abs(leg_totals[0] - leg_totals[1]) / 100.0)

        # 2. Conservation accounts must sum to zero (minor units are exact)
        worst = max(worst, abs(sum(self.accounts.values())) / 100.0)

        # 3. Replay the journal against live accounts, but only while it is complete
        if self._postings == len(self.journal):
            replayed: Dict[str, int] = collections.defaultdict(int)
            for entry in self.journal:
                for leg, sign in (("debits", -1), ("credits", 1)):
                    for acct, amount in entry.get(leg, {}).items():
                        value = minor(amount)
                        if value is None:
                            return False, float("inf")
                        replayed[acct] += sign * value  # type: ignore[operator]
            for acct in set(replayed) | set(self.accounts):
                worst = max(worst, abs(replayed.get(acct, 0) - self.accounts.get(acct, 0)) / 100.0)

        return worst < abs_tol, worst


@dataclass
class NetSettlementReport:
    """Multilateral Net Settlement Position (MNSP) for a commercial bank in a clearing window."""
    bank_id: str
    cycle_id: str
    region: str
    gross_receivable: float
    gross_payable: float
    net_position: float  # gross_receivable - gross_payable (positive = creditor, negative = debtor)
    settlement_account_id: str
    interchange_net: float
    switch_fee_payable: float
    transaction_count: int


class InterbankClearingLedger:
    """Manages bilateral clearing accumulation and executes Multilateral Net Settlement (MNSP).
    Enforces the zero-sum conservation invariant: sum(net_positions) == 0.00 across all clearing windows.
    """

    def __init__(self, region: str = "US"):
        self.region = region.upper()
        self.specs = load_all_specs()
        # cycle_id -> (acquirer_bank, issuer_bank) -> {amount: float, interchange: float, count: int}
        self.cycle_flows: Dict[str, Dict[Tuple[str, str], Dict[str, float]]] = collections.defaultdict(
            lambda: collections.defaultdict(lambda: {"amount": 0.0, "interchange": 0.0, "count": 0.0})
        )
        self.settlement_history: List[Dict[str, Any]] = []

    def record_clearing_presentment(
        self,
        cycle_id: str,
        acquiring_bank_id: str,
        issuing_bank_id: str,
        settled_amount: float,
        interchange_fee: float,
        is_on_us: bool = False,
    ) -> None:
        """Records a clearing presentment (MTI 0200) into the active clearing cycle."""
        if is_on_us or acquiring_bank_id == issuing_bank_id:
            return

        flow = self.cycle_flows[cycle_id][(acquiring_bank_id, issuing_bank_id)]
        flow["amount"] = round(flow["amount"] + settled_amount, 2)
        flow["interchange"] = round(flow["interchange"] + interchange_fee, 2)
        flow["count"] += 1

    def execute_multilateral_netting(
        self,
        cycle_id: str,
        banks: Optional[List[Dict[str, Any]]] = None,
        sim_time_sec: float = 0.0,
        abs_tol: float = 1e-4,
    ) -> List[NetSettlementReport]:
        """Calculates Multilateral Net Settlement Positions (MNSP) for all participating banks.
        Asserts zero-sum conservation invariant: sum(net_positions) == 0.00.
        """
        if banks is None:
            interbank = self.specs.interbank
            banks = interbank.commercial_banks.get(self.region, []) if interbank else []
            if not banks:
                banks = [
                    {"id": "BANK_US_CHASE", "central_bank_rtgs_id": "FEDWIRE_CHASE_01"},
                    {"id": "BANK_US_BOFA", "central_bank_rtgs_id": "FEDWIRE_BOFA_02"},
                    {"id": "BANK_US_WELLS", "central_bank_rtgs_id": "FEDWIRE_WELLS_03"},
                    {"id": "BANK_US_CITI", "central_bank_rtgs_id": "FEDWIRE_CITI_04"},
                ]

        flows = self.cycle_flows.get(cycle_id, {})
        reports: List[NetSettlementReport] = []
        net_sum = 0.0

        for b in banks:
            bid = b["id"]
            receivable = 0.0
            payable = 0.0
            interchange_net = 0.0
            tx_count = 0

            for (acq, iss), data in flows.items():
                amt = data["amount"]
                ich = data["interchange"]
                net_transfer = amt - ich
                cnt = int(data["count"])

                if acq == bid:
                    receivable += net_transfer
                    interchange_net -= ich
                    tx_count += cnt
                if iss == bid:
                    payable += net_transfer
                    interchange_net += ich
                    tx_count += cnt

            receivable = round(receivable, 2)
            payable = round(payable, 2)
            net_pos = round(receivable - payable, 2)
            interchange_net = round(interchange_net, 2)
            net_sum += net_pos

            rtgs_id = b.get("central_bank_rtgs_id", f"RTGS_{bid}")
            reports.append(
                NetSettlementReport(
                    bank_id=bid,
                    cycle_id=cycle_id,
                    region=self.region,
                    gross_receivable=receivable,
                    gross_payable=payable,
                    net_position=net_pos,
                    settlement_account_id=rtgs_id,
                    interchange_net=interchange_net,
                    switch_fee_payable=0.0,
                    transaction_count=tx_count,
                )
            )

        net_sum = round(net_sum, 2)
        if abs(net_sum) > abs_tol:
            raise ValueError(
                f"Multilateral Net Settlement zero-sum invariant violated in cycle {cycle_id}: "
                f"net sum = {net_sum} != 0.00"
            )

        self.settlement_history.append({
            "cycle_id": cycle_id,
            "sim_time_sec": sim_time_sec,
            "zero_sum_conserved": True,
            "residual": net_sum,
            "reports": reports,
        })
        return reports


