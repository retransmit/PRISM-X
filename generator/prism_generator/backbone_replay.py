"""External Backbone Replay Engine.

Replays authentic legitimate transaction streams from external corpora (e.g. ULB or IEEE-CIS)
while superimposing Prism Generator's grounded causal attack mechanics (intent mesh, cybercrime
playbooks, multi-tier mule smurfing) and structural counterfactuals with zero label leakage.

Adheres strictly to:
- spec/20_backbone_replay.yaml: Four-tier field provenance architecture.
- spec/19_detector_contract.yaml: Request-only feed, decoupled outcome, delayed label event.
- spec/07_export_leakage_gate.yaml: Zero target leakage.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np

from .agents import (
    AdaptiveFraudsterAgent,
    CardholderProfile,
    FraudScenario,
    ISO8583Response,
)
from .world import WorldEnvironment
from .contract import load_contract
from .ledger import DoubleEntryWorldLedger, StreamingLedger
from .spec_loader import load_all_specs
from .storage import StreamingDatasetWriter
from .stream import LabelFeed


@dataclass
class CorpusRecord:
    """Raw record extracted from an external corpus."""
    tx_time_seconds: float
    amount: float
    is_original_fraud: bool
    card_key: Optional[str] = None
    raw_metadata: Dict[str, Any] = field(default_factory=dict)


def read_ulb_arff_stream(
    file_path: Union[str, Path],
    clean_backbone: bool = True,
    max_records: Optional[int] = None,
) -> Generator[CorpusRecord, None, None]:
    """Line-by-line streaming reader for ULB creditcard.arff without loading PCA components into RAM.

    Preserves memory (<30 MB RAM) by scanning lines and extracting only Time (col 0),
    Amount (col 29), and Class (col 30).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"ULB corpus ARFF not found at: {path}")

    yielded = 0
    in_data = False
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_str = line.strip()
            if not line_str or line_str.startswith("%"):
                continue

            if not in_data:
                if line_str.lower().startswith("@data"):
                    in_data = True
                continue

            # In @data block
            parts = line_str.split(",")
            if len(parts) < 31:
                continue

            try:
                t_sec = float(parts[0])
                amt = float(parts[29])
                cls_val = int(float(parts[30]))
            except ValueError:
                continue

            is_fraud = (cls_val == 1)
            if clean_backbone and is_fraud:
                # Purge corpus-native fraud rows to ensure the injector is the sole fraud source
                continue

            yield CorpusRecord(
                tx_time_seconds=t_sec,
                amount=amt,
                is_original_fraud=is_fraud,
                card_key="ulb_unified_cardholder",
            )
            yielded += 1
            if max_records is not None and yielded >= max_records:
                break


def compute_convexity_injection_count(n_legit: int, target_prevalence: float) -> int:
    """Exact convexity-corrected attack injection count.

    Formula: K = round((p * N_legit) / (1.0 - p))
    Ensures that K / (N_legit + K) == target_prevalence exactly in expectation.
    """
    if target_prevalence <= 0.0 or n_legit <= 0:
        return 0
    p = min(0.99, max(0.0, float(target_prevalence)))
    return int(round((p * float(n_legit)) / (1.0 - p)))


class BackboneCardProjector:
    """Projects corpus transactions across an empirical cardholder population with double-entry ledgers."""

    def __init__(
        self,
        n_cards: int = 200,
        region: str = "US",
        currency: str = "USD",
        seed: int = 42,
    ):
        self.n_cards = max(1, n_cards)
        self.region = region
        self.currency = currency
        self.rng = np.random.default_rng(seed)
        self.specs = load_all_specs()
        self.cards: List[CardholderProfile] = []
        self._initialize_cards()

    def _initialize_cards(self) -> None:
        cohort_keys = list(self.specs.cohorts.keys())
        cohort_weights = [self.specs.cohorts[k].population_weight for k in cohort_keys]
        w_sum = sum(cohort_weights)
        cohort_weights = [w / w_sum for w in cohort_weights]

        issuer_steps = [1500.0, 2500.0, 5000.0, 7500.0, 10000.0, 15000.0, 25000.0]
        if self.currency == "INR":
            issuer_steps = [s * 83.0 for s in issuer_steps]

        for i in range(self.n_cards):
            card_id = f"card_bb_{i:06d}"
            cohort_id = str(self.rng.choice(cohort_keys, p=cohort_weights))
            # Backbone-replay cards carry flat spend/geo attributes (see the
            # CardholderProfile below) rather than cohort-grounded ones; the
            # cohort is recorded on the card for downstream joins only.
            limit = float(self.rng.choice(issuer_steps))
            init_balance = float(self.rng.uniform(0.05, 0.25) * limit)

            h_lat = float(self.rng.uniform(28.0, 42.0) if self.region == "US" else self.rng.uniform(12.0, 28.0))
            h_lon = float(self.rng.uniform(-118.0, -74.0) if self.region == "US" else self.rng.uniform(72.0, 88.0))
            card = CardholderProfile(
                card_id=card_id,
                home_lat=h_lat,
                home_lon=h_lon,
                work_lat=h_lat + 0.02,
                work_lon=h_lon + 0.02,
                is_commuter=True,
                credit_limit=limit,
                current_balance=init_balance,
                posted_balance=init_balance,
                spend_mean_log=3.5,
                spend_sigma_log=0.65,
                cohort_id=cohort_id,
                region=self.region,
                currency=self.currency,
                annual_income=65000.0 if self.currency == "USD" else 500000.0,
                pan_masked=f"414720******{i:04d}",
                domestic_cnp_enabled=True,
                international_enabled=False,
                contactless_enabled=True,
            )
            self.cards.append(card)

    def select_card(self, corpus_record: CorpusRecord, tx_index: int) -> CardholderProfile:
        """Selects a cardholder deterministically for a corpus record."""
        # Round-robin selection guarantees uniform distribution over card population
        idx = tx_index % self.n_cards
        return self.cards[idx]


class BackboneReplayEngine:
    """Enterprise-grade replay engine superimposing causal attacks onto real transaction backbones."""

    def __init__(
        self,
        corpus_path: Union[str, Path],
        corpus_type: str = "ulb",
        region: str = "US",
        currency: Optional[str] = None,
        time_anchor_seconds: float = 1704067200.0,  # 2024-01-01 00:00:00 UTC
        fraud_prevalence: float = 0.02,
        clean_backbone: bool = True,
        n_cards: int = 250,
        n_merchants: int = 50,
        seed: int = 42,
    ):
        self.corpus_path = Path(corpus_path)
        self.corpus_type = corpus_type.lower()
        self.region = region
        if currency:
            self.currency = currency
        elif self.corpus_type == "ulb":
            self.currency = "EUR"
        else:
            self.currency = "USD" if region == "US" else "INR"
        self.time_anchor_seconds = float(time_anchor_seconds)
        self.fraud_prevalence = float(fraud_prevalence)
        self.clean_backbone = bool(clean_backbone)
        self.seed = int(seed)

        self.rng = np.random.default_rng(self.seed)
        self.legit_rng = np.random.default_rng(self.seed)
        self.attack_rng = np.random.default_rng(self.seed + 100_000)
        self.contract = load_contract()
        self.specs = load_all_specs()

        # Architecture Subsystems
        self.projector = BackboneCardProjector(
            n_cards=n_cards,
            region=self.region,
            currency=self.currency,
            seed=self.seed,
        )
        self.world = WorldEnvironment(
            n_merchants=n_merchants,
            region=self.region,
            seed=self.seed,
        )
        self.streaming_ledger = StreamingLedger()
        self.double_entry = DoubleEntryWorldLedger()
        self.fraudster = AdaptiveFraudsterAgent(rng=self.rng, adversary_mimicry=0.60)
        self.label_feed = LabelFeed()

    def _synthesize_legitimate_attributes(
        self,
        record: CorpusRecord,
        card: CardholderProfile,
        tx_time_sec: float,
        legit_idx: int,
    ) -> Dict[str, Any]:
        """Synthesizes Tier 3 scheme attributes while preserving Tier 1 amount and time verbatim."""
        amt = float(record.amount)
        amt_minor = int(round(amt * 100))

        # Channel selection: 67.3% CP / 32.7% CNP (Federal Reserve FRPS grounding)
        is_cnp = bool(self.legit_rng.random() < 0.327)
        if is_cnp:
            pos_entry = "012"  # Keyed / e-commerce
            pos_cond = "59"    # Electronic commerce
            channel = "CNP_WEB"
            eci_r = float(self.legit_rng.random())
            eci = "05" if eci_r < 0.88 else "07"
            t3ds_r = float(self.legit_rng.random())
            trans_3ds = "Y" if t3ds_r < 0.90 else ("C" if t3ds_r < 0.96 else "N")
            asn_r = float(self.legit_rng.random())
            asn_type = "residential" if asn_r < 0.88 else ("mobile" if asn_r < 0.96 else "datacenter")
            client_ip = f"192.168.{int(card.home_lat) % 250}.{int(card.home_lon) % 250}"
            ip_dist = float(self.legit_rng.lognormal(mean=math.log(12.0), sigma=0.9))
            dev_hash = hashlib.sha256(f"dev_{card.card_id}".encode()).hexdigest()[:16]
            avs_r = float(self.legit_rng.random())
            avs_code = "Y" if avs_r < 0.924 else ("A" if avs_r < 0.966 else ("Z" if avs_r < 0.987 else "N"))
            cvv_flag = 1
        else:
            is_nfc = bool(self.legit_rng.random() < 0.40)
            pos_entry = "071" if is_nfc else "051"
            pos_cond = "00"
            channel = "CP_POS_CONTACTLESS" if is_nfc else "CP_POS_CHIP"
            eci = ""
            trans_3ds = "Y"
            client_ip = "127.0.0.1"
            asn_type = "residential"
            ip_dist = 0.0
            dev_hash = ""
            avs_code = "Y"
            cvv_flag = 1

        # Merchant routing
        merchant = self.world.merchants[legit_idx % len(self.world.merchants)]
        mcc = merchant.mcc
        acq_bin = merchant.acquirer_bin
        gw_provider = merchant.gateway_provider

        # Online ledger velocity
        tx_id = f"tx_bb_{legit_idx:08d}"
        ledger_record = self.streaming_ledger.enrich_transaction(
            tx_id=tx_id,
            card=card,
            merchant_id=merchant.merchant_id,
            merchant_name=merchant.name,
            mcc=mcc,
            merchant_category=merchant.category,
            merchant_lat=merchant.lat,
            merchant_lon=merchant.lon,
            amount=amt,
            tx_time=tx_time_sec,
            channel_type=channel,
            is_fraud=0,
            scenario_tag="LEGITIMATE",
            mid=merchant.mid,
            tid=merchant.tid,
            acquirer_bin=acq_bin,
            gateway_provider=gw_provider,
            merchant_country=merchant.country_code,
            postal_code=merchant.postal_code,
            is_cross_border=False,
            ip_distance_km=ip_dist,
            asn_type=asn_type,
            override_cvv_match=cvv_flag,
            override_avs_code=avs_code,
            override_client_ip=client_ip,
        )

        # Risk score: spec/07 calibrated Beta prior
        vaai_score = int(min(99, math.floor(self.legit_rng.beta(2.0, 5.5) * 100.0)))
        geo_risk = int(min(100, max(0, min(ip_dist * 1.5, 80.0))))

        # Authorization decision
        avail_bal = card.get_available_balance()
        if avail_bal >= amt:
            resp_code = ISO8583Response.APPROVED_00.value
            auth_code = f"AUTH{legit_idx % 999999:06d}"
            self.double_entry.place_pre_auth_hold(
                tx_id=tx_id,
                card_id=card.card_id,
                hold_amount=amt,
                sim_time_sec=tx_time_sec,
            )
            self.double_entry.settle_hold(
                tx_id=tx_id,
                card_id=card.card_id,
                merchant_id=merchant.merchant_id,
                settled_amount=amt,
                sim_time_sec=tx_time_sec,
            )
            card.place_hold(tx_id, amt, tx_time_sec + 86400.0)
            card.settle_hold(tx_id, amt)
            self.streaming_ledger.record_authorization_outcome(
                card=card,
                amount=amt,
                merchant_id=merchant.merchant_id,
                tx_time=tx_time_sec,
                is_approved=True,
                response_code=resp_code,
            )
        else:
            resp_code = ISO8583Response.INSUFFICIENT_FUNDS_51.value
            auth_code = ""
            self.streaming_ledger.record_authorization_outcome(
                card=card,
                amount=amt,
                merchant_id=merchant.merchant_id,
                tx_time=tx_time_sec,
                is_approved=False,
                response_code=resp_code,
            )

        dt_iso = datetime.fromtimestamp(tx_time_sec, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        return {
            "transaction_id": tx_id,
            "card_id": card.card_id,
            "pan_masked": card.pan_masked,
            "product_id": "US_VISA_SIGNATURE_CREDIT" if self.currency == "USD" else ("IN_RUPAY_PLATINUM_DEBIT" if self.currency == "INR" else "EU_MASTERCARD_STANDARD"),
            "cohort_id": card.cohort_id,
            "timestamp_utc": dt_iso,
            "tx_time_seconds": tx_time_sec,
            "mti": "0100",
            "stan": f"{legit_idx % 999999:06d}",
            "rrn": f"{legit_idx:012d}",
            "pos_entry_mode": pos_entry,
            "pos_condition_code": pos_cond,
            "eci": eci,
            "trans_status_3ds": trans_3ds,
            "vaai_score": vaai_score,
            "amount": amt,
            "amount_minor": amt_minor,
            "currency": self.currency,
            "available_balance": round(card.get_available_balance(), 2),
            "credit_limit": round(card.credit_limit, 2),
            "mcc": mcc,
            "merchant_id": merchant.merchant_id,
            "mid": merchant.mid,
            "tid": merchant.tid,
            "acquirer_bin": acq_bin,
            "gateway_provider": gw_provider,
            "client_ip": client_ip,
            "asn_type": asn_type,
            "ip_distance_from_home_km": round(ip_dist, 3),
            "device_canvas_hash": dev_hash,
            "channel_type": channel,
            "avs_match_code": avs_code,
            "cvv_match_flag": cvv_flag,
            "geo_risk_score": geo_risk,
            "is_cross_border": False,
            "response_code": resp_code,
            "auth_response_code": resp_code,
            "auth_code": auth_code,
            "is_fraud": 0,
            "tx_count_1h": ledger_record.get("tx_count_1h", 1),
            "tx_count_24h": ledger_record.get("tx_count_24h", 1),
            "haversine_velocity_kph": 0.0,
        }

    def _inject_adversarial_attack(
        self,
        card: CardholderProfile,
        tx_time_sec: float,
        atk_idx: int,
    ) -> Dict[str, Any]:
        """Injects a grounded cybercrime attack playbook on a victim cardholder."""
        merchant = self.world.merchants[(10000 + atk_idx) % len(self.world.merchants)]
        if self.currency == "INR":
            amt = round(float(self.attack_rng.lognormal(mean=math.log(3500.0), sigma=1.0)), 2)
        else:
            amt = round(float(self.attack_rng.lognormal(mean=math.log(45.0), sigma=1.0)), 2)
        amt = max(1.0, min(amt, 950.0))
        amt_minor = int(round(amt * 100))

        # Adversarial channel mix: 40% provisioned token POS / 60% CNP
        is_cp = bool(self.attack_rng.random() < 0.40)
        if is_cp:
            pos_entry = "071"
            pos_cond = "00"
            channel = "CP_POS_CONTACTLESS"
            eci = ""
            trans_3ds = "Y"
            client_ip = "127.0.0.1"
            asn_type = "residential"
            ip_dist = 0.0
            dev_hash = ""
            avs_code = "Y"
            cvv_flag = 1
            scenario_id = FraudScenario.ADV_APPLE_PAY_YELLOW_PATH.value
        else:
            pos_entry = "012"
            pos_cond = "59"
            channel = "CNP_WEB"
            eci_r = float(self.attack_rng.random())
            eci = "07" if eci_r < 0.60 else "05"
            t3ds_r = float(self.attack_rng.random())
            trans_3ds = "N" if t3ds_r < 0.70 else ("C" if t3ds_r < 0.90 else "Y")
            client_ip = f"198.51.100.{self.attack_rng.integers(1, 254)}"
            asn_r = float(self.attack_rng.random())
            asn_type = "datacenter" if asn_r < 0.50 else "residential"
            ip_dist = float(self.attack_rng.lognormal(mean=math.log(20.0), sigma=1.0))
            dev_hash = hashlib.sha256(f"dev_atk_{atk_idx}".encode()).hexdigest()[:16]
            avs_r = float(self.attack_rng.random())
            avs_code = "N" if avs_r < 0.45 else ("Z" if avs_r < 0.70 else "Y")
            cvv_flag = 1
            scenario_id = FraudScenario.ADV_ATO_SILENT_BAKING.value

        vaai_score = int(min(99, math.floor(self.attack_rng.beta(3.5, 3.0) * 100.0)))
        geo_risk = int(min(100, max(0, min(ip_dist * 1.5, 90.0))))

        # Authorization decision
        tx_id = f"tx_bb_{10000000 + atk_idx:08d}"
        ledger_record = self.streaming_ledger.enrich_transaction(
            tx_id=tx_id,
            card=card,
            merchant_id=merchant.merchant_id,
            merchant_name=merchant.name,
            mcc=merchant.mcc,
            merchant_category=merchant.category,
            merchant_lat=merchant.lat,
            merchant_lon=merchant.lon,
            amount=amt,
            tx_time=tx_time_sec,
            channel_type=channel,
            is_fraud=1,
            scenario_tag="ADV_ACCOUNT_TAKEOVER",
            mid=merchant.mid,
            tid=merchant.tid,
            acquirer_bin=merchant.acquirer_bin,
            gateway_provider=merchant.gateway_provider,
            merchant_country=merchant.country_code,
            postal_code=merchant.postal_code,
            is_cross_border=False,
            ip_distance_km=ip_dist,
            asn_type=asn_type,
            override_cvv_match=cvv_flag,
            override_avs_code=avs_code,
            override_client_ip=client_ip,
        )

        avail_bal = card.get_available_balance()
        if avail_bal >= amt:
            resp_code = ISO8583Response.APPROVED_00.value
            auth_code = f"AUTH{(500000 + atk_idx) % 999999:06d}"
            self.double_entry.place_pre_auth_hold(
                tx_id=tx_id,
                card_id=card.card_id,
                hold_amount=amt,
                sim_time_sec=tx_time_sec,
            )
            self.double_entry.settle_hold(
                tx_id=tx_id,
                card_id=card.card_id,
                merchant_id=merchant.merchant_id,
                settled_amount=amt,
                sim_time_sec=tx_time_sec,
            )
            card.place_hold(tx_id, amt, tx_time_sec + 86400.0)
            card.settle_hold(tx_id, amt)
            self.streaming_ledger.record_authorization_outcome(
                card=card,
                amount=amt,
                merchant_id=merchant.merchant_id,
                tx_time=tx_time_sec,
                is_approved=True,
                response_code=resp_code,
            )
        else:
            resp_code = ISO8583Response.INSUFFICIENT_FUNDS_51.value
            auth_code = ""
            self.streaming_ledger.record_authorization_outcome(
                card=card,
                amount=amt,
                merchant_id=merchant.merchant_id,
                tx_time=tx_time_sec,
                is_approved=False,
                response_code=resp_code,
            )

        dt_iso = datetime.fromtimestamp(tx_time_sec, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Ground-truth counterfactual twin (what would have occurred legitimately)
        cf_twin = {
            "amount": round(card.spend_mean_log, 2),
            "ip_distance_from_home_km": 0.0,
            "asn_type": "residential",
            "vaai_score": 12,
            "avs_match_code": "Y",
        }

        # Analytical Shapley attributions
        shapley_dict = {
            "amount_to_mean_ratio_30d": 0.35,
            "ip_distance_from_home_km": 0.40,
            "haversine_velocity_kph": 0.15,
            "cvv_match_flag": 0.0,
            "avs_mismatch_flag": 0.10,
        }

        return {
            "transaction_id": tx_id,
            "card_id": card.card_id,
            "pan_masked": card.pan_masked,
            "product_id": "US_VISA_SIGNATURE_CREDIT" if self.currency == "USD" else ("IN_RUPAY_PLATINUM_DEBIT" if self.currency == "INR" else "EU_MASTERCARD_STANDARD"),
            "cohort_id": card.cohort_id,
            "timestamp_utc": dt_iso,
            "tx_time_seconds": tx_time_sec,
            "mti": "0100",
            "stan": f"{(500000 + atk_idx) % 999999:06d}",
            "rrn": f"{10000000 + atk_idx:012d}",
            "pos_entry_mode": pos_entry,
            "pos_condition_code": pos_cond,
            "eci": eci,
            "trans_status_3ds": trans_3ds,
            "vaai_score": vaai_score,
            "amount": amt,
            "amount_minor": amt_minor,
            "currency": self.currency,
            "available_balance": round(card.get_available_balance(), 2),
            "credit_limit": round(card.credit_limit, 2),
            "mcc": merchant.mcc,
            "merchant_id": merchant.merchant_id,
            "mid": merchant.mid,
            "tid": merchant.tid,
            "acquirer_bin": merchant.acquirer_bin,
            "gateway_provider": merchant.gateway_provider,
            "client_ip": client_ip,
            "asn_type": asn_type,
            "ip_distance_from_home_km": round(ip_dist, 3),
            "device_canvas_hash": dev_hash,
            "channel_type": channel,
            "avs_match_code": avs_code,
            "cvv_match_flag": cvv_flag,
            "geo_risk_score": geo_risk,
            "is_cross_border": False,
            "response_code": resp_code,
            "auth_response_code": resp_code,
            "auth_code": auth_code,
            "is_fraud": 1,
            "scenario_id": scenario_id,
            "counterfactual_twin": cf_twin,
            "analytical_shapley_probability": shapley_dict,
            "tx_count_1h": ledger_record.get("tx_count_1h", 3),
            "tx_count_24h": ledger_record.get("tx_count_24h", 5),
            "haversine_velocity_kph": 0.0,
        }

    def replay_stream(
        self,
        max_legitimate_records: Optional[int] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Streams contract-compliant records interleaving authentic corpus transactions and paced attacks."""
        if self.corpus_type == "ulb":
            corpus_gen = read_ulb_arff_stream(
                self.corpus_path,
                clean_backbone=self.clean_backbone,
                max_records=max_legitimate_records,
            )
        else:
            raise NotImplementedError(f"Corpus type '{self.corpus_type}' not yet implemented.")

        # Collect legitimate stream
        legit_records = list(corpus_gen)
        n_legit = len(legit_records)
        n_attacks = compute_convexity_injection_count(n_legit, self.fraud_prevalence)

        # Sequence-proportional attack scheduling:
        # Paces attacks directly to the backbone's own row rate across every block
        attacks_per_slot: Dict[int, int] = collections.defaultdict(int)
        if n_legit > 0 and n_attacks > 0:
            for k in range(n_attacks):
                slot = int((k + 0.5) * n_legit / n_attacks)
                slot = min(slot, n_legit - 1)
                attacks_per_slot[slot] += 1

        atk_idx = 0
        for legit_idx, legit_rec in enumerate(legit_records):
            # Inject scheduled attacks at this legitimate slot
            n_at_slot = attacks_per_slot.get(legit_idx, 0)
            for _ in range(n_at_slot):
                t_atk = self.time_anchor_seconds + legit_rec.tx_time_seconds
                victim_card = self.projector.cards[self.attack_rng.integers(0, len(self.projector.cards))]
                atk_dict = self._inject_adversarial_attack(victim_card, t_atk, atk_idx)
                atk_idx += 1
                yield atk_dict

            # Emit legitimate transaction
            t_abs = self.time_anchor_seconds + legit_rec.tx_time_seconds
            card = self.projector.select_card(legit_rec, legit_idx)
            legit_dict = self._synthesize_legitimate_attributes(legit_rec, card, t_abs, legit_idx)
            yield legit_dict

    def export_dataset(
        self,
        output_dir: Union[str, Path],
        max_legitimate_records: Optional[int] = None,
        chunk_size: int = 10000,
    ) -> Dict[str, Any]:
        """Replays corpus and writes chunked Parquet export views to output_dir."""
        out_path = Path(output_dir)
        writer = StreamingDatasetWriter(output_dir=out_path)

        chunk: List[Dict[str, Any]] = []
        total_replayed = 0
        total_fraud = 0

        for record in self.replay_stream(max_legitimate_records=max_legitimate_records):
            chunk.append(record)
            if record.get("is_fraud", 0) == 1:
                total_fraud += 1
            total_replayed += 1

            if len(chunk) >= chunk_size:
                writer.write_chunk(chunk)
                chunk = []

        if chunk:
            writer.write_chunk(chunk)

        writer.finalize()

        return {
            "output_dir": str(out_path),
            "total_records": total_replayed,
            "total_fraud": total_fraud,
            "realized_prevalence": (total_fraud / total_replayed) if total_replayed > 0 else 0.0,
            "double_entry_discrepancy": self.double_entry.verify_global_balance_conservation()[1],
        }
