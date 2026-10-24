"""Adversarial Syndicate Graph Topologies & Persistent Infrastructure Entities.

Models real-world cybercrime underground infrastructure:
- Multi-threat syndicate ecology: IABs, Card Checkers, ATO crews, Synthetic Bust-outs, Reship/Gift drain, and FinCEN mule pipelines.
- Multi-ASN residential, mobile, and datacenter proxy pools with authentic ISP allocations (Comcast, AT&T, Charter, Jio, Airtel).
- FoxIO JA4 TLS signatures and passive p0f TCP stack telemetry for cross-layer anomaly modeling.
- 3-tier FinCEN/FATF money mule layering DAGs: Tier 1 Smurfing -> Tier 2 Aggregator LLCs -> Tier 3 Crypto Off-Ramp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .network import ClientAddressSpace, prefix_of


class MuleTier(str, Enum):
    TIER_1_SMURF = "TIER_1_SMURF"
    TIER_2_AGGREGATOR = "TIER_2_AGGREGATOR"
    TIER_3_OFFRAMP = "TIER_3_OFFRAMP"


@dataclass
class BotnetCluster:
    """Represents a distributed botnet or residential proxy pool executing coordinated attacks."""
    cluster_id: str
    name: str
    subnet_prefix: str
    asn: str
    isp: str
    proxy_type: str  # "RESIDENTIAL_STICKY", "MOBILE_4G_5G", "DATACENTER_ROTATING"
    ja4_signature: str
    tcp_os_profile: str  # "Windows_11", "Linux_Container", "macOS_Darwin", "Android_Linux"
    device_fingerprints: List[str]
    ip_pool: List[str]

    def sample_ip_and_device(self, rng: np.random.Generator) -> Tuple[str, str, Dict[str, Any]]:
        """Draws an IP, device fingerprint, and network telemetry from the cluster pool."""
        ip = str(rng.choice(self.ip_pool))
        device = str(rng.choice(self.device_fingerprints))
        telemetry = {
            "asn": self.asn,
            "isp": self.isp,
            "proxy_type": self.proxy_type,
            "ja4_signature": self.ja4_signature,
            "tcp_os_profile": self.tcp_os_profile,
        }
        return ip, device, telemetry


@dataclass
class SmurfTransferLeg:
    """Represents one discrete structured transfer leg in a cross-bank smurfing ring."""
    victim_account_id: str
    victim_bank_id: str
    mule_account_id: str
    mule_bank_id: str
    amount: float
    currency: str
    dispatch_time_sec: float
    liquidation_time_sec: float
    offramp_channel: str


class CrossBankSmurfingSynthesizer:
    """Synthesizes multi-stage cross-bank smurfing operations across commercial banks.

    Grounded in FinCEN Advisory FIN-2020-A003, BSA 31 CFR 1010.311, and I4C CFCFRMS 1930.
    Enforces strict institutional diversity: Bank(M_k) != Bank(M_j) and Bank(M_k) != Bank(V_0).
    """

    def __init__(self, region: str = "US", seed: int = 42):
        self.region = region.upper()
        self.rng = np.random.default_rng(seed)
        # Only an ImportError is tolerated here. A malformed or partial spec previously
        # raised here, was swallowed by a bare `except Exception: pass`, and the engine
        # silently continued on a hardcoded bank list with invented market-share weights.
        # A spec regression is exactly the thing that must not fail quietly, so it now
        # propagates. ImportError is the one case where an unavailable module is a
        # legitimate reason to fall back rather than a defect.
        try:
            from .spec_loader import load_all_specs
        except ImportError:  # pragma: no cover - only hit if the package is broken
            import warnings
            warnings.warn(
                "spec_loader could not be imported; cross-bank smurfing falls back to a "
                "built-in bank list with illustrative market shares",
                RuntimeWarning,
                stacklevel=2,
            )
            specs = None
        else:
            specs = load_all_specs()

        interbank = specs.interbank if specs else None
        banks_raw = interbank.commercial_banks.get(self.region, []) if interbank else []
        if banks_raw:
            self.bank_ids = [str(b["id"]) for b in banks_raw]
            weights = np.array([float(b.get("market_share_weight", 0.20)) for b in banks_raw])
            self.bank_weights = weights / weights.sum()
        else:
            if self.region == "IN":
                self.bank_ids = ["BANK_IN_SBI", "BANK_IN_HDFC", "BANK_IN_ICICI", "BANK_IN_AXIS", "BANK_IN_PNB"]
                self.bank_weights = np.array([0.26, 0.24, 0.20, 0.16, 0.14])
            else:
                self.bank_ids = ["BANK_US_CHASE", "BANK_US_BOFA", "BANK_US_WELLS", "BANK_US_CITI", "BANK_US_CAPITALONE"]
                self.bank_weights = np.array([0.28, 0.24, 0.20, 0.16, 0.12])

        # spec/13 cross_bank_smurfing.offramp_probabilities. This mix lived only in
        # generate_smurfing_ring_operation before 2026-10-13, so the spec declared the
        # set of channels and the engine chose their weights invisibly.
        if specs is not None:
            smurf_spec = getattr(specs, "cross_bank_smurfing", None) or {}
        else:
            smurf_spec = {}
        offramp_probs = dict(smurf_spec.get("offramp_probabilities", {}))
        if offramp_probs:
            self.offramp_channels = list(offramp_probs)
            p = np.array([float(offramp_probs[c]) for c in self.offramp_channels])
            self.offramp_weights = p / p.sum()
        else:
            self.offramp_channels = ["P2P_CRYPTO_USDT_ESCROW", "ATM_CLONED_CARD_CASHOUT"]
            self.offramp_weights = np.array([0.65, 0.35])

    def sample_fan_out_degree(self) -> int:
        """Samples number of branches K in {2, 3, 4, 5}. Expected value = 3.2."""
        degrees = [2, 3, 4, 5]
        probs = [0.25, 0.40, 0.25, 0.10]
        return int(self.rng.choice(degrees, p=probs))

    def sample_destination_banks(self, source_bank: str, k: int) -> List[str]:
        """Selects K distinct destination banks strictly excluding source bank."""
        available = [b for b in self.bank_ids if b != source_bank]
        if not available:
            return [self.bank_ids[0]]
        k_clamped = min(k, len(available))
        indices = [self.bank_ids.index(b) for b in available]
        weights = self.bank_weights[indices]
        norm_weights = weights / weights.sum()
        chosen = self.rng.choice(available, size=k_clamped, replace=False, p=norm_weights)
        return list(chosen)

    def sample_structuring_ticket(self) -> float:
        """Draws structuring ticket size from Truncated Log-Normal distribution.

        Strictly bounded below BSA ($10,000 USD) and PMLA (Rs 200,000 INR) reporting thresholds.
        """
        if self.region == "IN":
            mu, sigma, low, high = 11.150, 0.450, 25000.0, 180000.0
        else:
            mu, sigma, low, high = 8.150, 0.420, 1500.0, 8500.0

        for _ in range(50):
            val = float(self.rng.lognormal(mean=mu, sigma=sigma))
            if low <= val <= high:
                return round(val, 2)
        return round(float(np.clip(val, low, high)), 2)

    def sample_hop_latency_seconds(self) -> float:
        """Samples off-ramp liquidation hop latency in seconds. Median = 22.5 minutes (1,350s)."""
        mu = 7.2078  # ln(1350)
        sigma = 0.450
        for _ in range(50):
            val = float(self.rng.lognormal(mean=mu, sigma=sigma))
            if 600.0 <= val <= 3600.0:  # Truncated between 10m and 60m
                return round(val, 1)
        return round(float(np.clip(val, 600.0, 3600.0)), 1)

    def generate_smurfing_ring_operation(
        self,
        victim_id: str,
        victim_bank: str,
        t_ingress_sec: float,
    ) -> List[SmurfTransferLeg]:
        """Generates a complete multi-stage cross-bank smurfing operation."""
        k = self.sample_fan_out_degree()
        dest_banks = self.sample_destination_banks(victim_bank, k)
        currency = "INR" if self.region == "IN" else "USD"
        legs = []

        for idx, dest_bank in enumerate(dest_banks):
            ticket_amt = self.sample_structuring_ticket()
            # Stagger interbank transfers by 10 to 45 seconds to mimic bot/manual sequencing
            stagger_sec = float(self.rng.uniform(10.0, 45.0)) * idx
            dispatch_time = t_ingress_sec + stagger_sec
            hop_latency = self.sample_hop_latency_seconds()
            liquidation_time = dispatch_time + hop_latency
            mule_id = f"MULE_{dest_bank}_{self.rng.integers(1000, 9999)}"
            # Weights come from spec/13 cross_bank_smurfing.offramp_probabilities.
            # Written as a Bernoulli on the first channel's weight rather than an
            # rng.choice over the array, so the draw consumes one uniform exactly as
            # before and the generated stream is unchanged.
            offramp = str(
                self.offramp_channels[0]
                if self.rng.random() < self.offramp_weights[0]
                else self.offramp_channels[1]
            )

            legs.append(
                SmurfTransferLeg(
                    victim_account_id=victim_id,
                    victim_bank_id=victim_bank,
                    mule_account_id=mule_id,
                    mule_bank_id=dest_bank,
                    amount=ticket_amt,
                    currency=currency,
                    dispatch_time_sec=dispatch_time,
                    liquidation_time_sec=liquidation_time,
                    offramp_channel=offramp,
                )
            )
        return legs


@dataclass
class MuleRing:
    """Represents a coordinated money-mule laundering network receiving fraudulent proceeds."""
    ring_id: str
    tier: MuleTier
    account_type: str  # "PERSONAL_CHECKING", "COMMERCIAL_SHELL_LLC", "P2P_CRYPTO_ESCROW"
    liquidation_channel: str
    beneficiary_accounts: List[str]
    bank_routing: str
    fee_cut_ratio: float = 0.10
    layering_hop_latency_seconds: float = 300.0
    is_frozen: bool = False
    bank_id: str = ""

    def sample_beneficiary(self, rng: np.random.Generator) -> str:
        """Draws a target mule beneficiary account."""
        return str(rng.choice(self.beneficiary_accounts))


@dataclass
class SyndicateEntity:
    """A persistent cybercrime syndicate operating specialized campaigns across accounts."""
    syndicate_id: str
    name: str
    archetype: str
    primary_playbooks: List[str]
    target_mccs: List[int] = field(default_factory=list)
    credential_tiers: List[str] = field(default_factory=list)
    botnets: List[BotnetCluster] = field(default_factory=list)
    mule_rings: List[MuleRing] = field(default_factory=list)

    def sample_telemetry(self, rng: np.random.Generator, victim_bank_id: str = "") -> Dict[str, Any]:
        """Samples persistent technical markers and mule destination for an attack event."""
        res: Dict[str, Any] = {
            "syndicate_id": self.syndicate_id,
            "syndicate_name": self.name,
            "syndicate_archetype": self.archetype,
            "botnet_cluster_id": "",
            "ip_subnet_prefix": "",
            "client_ip": "",
            "device_fingerprint_id": "",
            "asn": "",
            "isp": "",
            "ja4_signature": "",
            "mule_ring_id": "",
            "mule_tier": "",
            "beneficiary_account_id": "",
            "liquidation_channel": "",
            "mule_bank_id": "",
            "layering_hop_latency_seconds": 0.0,
        }

        if self.botnets:
            botnet = rng.choice(self.botnets)
            ip, dev, net_tel = botnet.sample_ip_and_device(rng)
            res["botnet_cluster_id"] = botnet.cluster_id
            res["ip_subnet_prefix"] = prefix_of(ip)
            res["client_ip"] = ip
            res["device_fingerprint_id"] = dev
            res["asn"] = net_tel["asn"]
            res["isp"] = net_tel["isp"]
            res["proxy_type"] = net_tel["proxy_type"]
            res["ja4_signature"] = net_tel["ja4_signature"]

        if self.mule_rings:
            # Enforce institutional diversity: mule bank != victim bank if multiple rings available
            eligible = [r for r in self.mule_rings if not victim_bank_id or r.bank_id != victim_bank_id]
            ring = rng.choice(eligible) if eligible else rng.choice(self.mule_rings)
            res["mule_ring_id"] = ring.ring_id
            res["mule_tier"] = ring.tier.value
            res["beneficiary_account_id"] = ring.sample_beneficiary(rng)
            res["liquidation_channel"] = ring.liquidation_channel
            res["mule_bank_id"] = ring.bank_id
            res["layering_hop_latency_seconds"] = ring.layering_hop_latency_seconds

        return res



class SyndicateRegistry:
    """Initializes and manages persistent cybercrime syndicates for the simulation environment."""

    def __init__(self, region: str = "US", seed: int = 42):
        self.region = region.upper()
        self.rng = np.random.default_rng(seed)
        self.syndicates: List[SyndicateEntity] = []
        self._syndicate_by_id: Dict[str, SyndicateEntity] = {}
        self._playbook_to_syndicates: Dict[str, List[SyndicateEntity]] = {}
        # spec/07 section 16: the proxy pools ride on the prefixes the cardholders use
        self.address_space = ClientAddressSpace(seed=seed)
        self.smurfing_synthesizer = CrossBankSmurfingSynthesizer(region=self.region, seed=seed)
        self._initialize_syndicates()

    def _initialize_syndicates(self) -> None:
        """Instantiates grounded cybercrime organizations with realistic technical assets."""
        if self.region == "IN":
            self._init_india_syndicates()
        else:
            self._init_us_syndicates()

        for syn in self.syndicates:
            for botnet in syn.botnets:
                kind = "datacenter" if botnet.proxy_type == "DATACENTER_ROTATING" else "residential"
                botnet.subnet_prefix, botnet.ip_pool = self.address_space.proxy_pool(self.region, kind, self.rng)

        for syn in self.syndicates:
            self._syndicate_by_id[syn.syndicate_id] = syn
            for pb in syn.primary_playbooks:
                if pb not in self._playbook_to_syndicates:
                    self._playbook_to_syndicates[pb] = []
                self._playbook_to_syndicates[pb].append(syn)

    def _init_us_syndicates(self) -> None:
        """Configures 6 empirical US cybercrime threat actors grounded in DOJ/FinCEN reports."""
        # Common Mule Infrastructure: Tier 1 Smurf & Tier 2 Aggregator
        ato_mule_t1 = MuleRing(
            ring_id="MULE_US_SMURF_CHASE_01",
            tier=MuleTier.TIER_1_SMURF,
            account_type="PERSONAL_CHECKING",
            liquidation_channel="INSTANT_P2P_SMURF",
            beneficiary_accounts=[f"MULE_CHASE_{i:05d}" for i in range(100, 115)],
            bank_routing="021000021",
            fee_cut_ratio=0.10,
            layering_hop_latency_seconds=300.0,
            bank_id="BANK_US_CHASE",
        )
        ato_mule_t2 = MuleRing(
            ring_id="MULE_US_AGGREGATOR_LLC_02",
            tier=MuleTier.TIER_2_AGGREGATOR,
            account_type="COMMERCIAL_SHELL_LLC",
            liquidation_channel="SAME_DAY_ACH_CONSOLIDATION",
            beneficiary_accounts=[f"MULE_BOA_CORP_{i:05d}" for i in range(200, 208)],
            bank_routing="011000138",
            fee_cut_ratio=0.15,
            layering_hop_latency_seconds=1200.0,
            bank_id="BANK_US_BOFA",
        )


        # 1. SYN_US_CARDING_BOT: Automated micro-auth card testing botnet
        checker_botnet = BotnetCluster(
            cluster_id="BOTNET_US_CHECKER_01",
            name="OpenBullet High-Throughput Checker Pool",
            subnet_prefix="198.54.130.0/24",
            asn="AS16509",
            isp="Amazon AWS",
            proxy_type="DATACENTER_ROTATING",
            ja4_signature="t13d1516h2_8daaf6152771_e562703ab853",
            tcp_os_profile="Linux_Container",
            device_fingerprints=[f"DEV_CHECKER_BOT_{i:03d}" for i in range(25)],
            ip_pool=[f"198.54.130.{i}" for i in range(10, 100)],
        )
        syn_carding = SyndicateEntity(
            syndicate_id="SYN_US_CARDING_BOT",
            name="Automated Micro-Auth Probing Syndicate",
            archetype="CARD_CHECKER_BOTNET",
            primary_playbooks=[
                "ADV_MICRO_AUTH_PROBE",
                "ADV_DISTRIBUTED_BIN_ENUMERATION",
                "ADV_NOCTURNAL_BURST",
                "INTENT_OMEGA_PROBE",
            ],
            target_mccs=[8398, 4899, 5815],
            credential_tiers=["TIER_TRACK_2_DUMP", "TIER_CNP_FULLZ"],
            botnets=[checker_botnet],
            mule_rings=[ato_mule_t1, ato_mule_t2],  # Connects card checking hits to smurf ingress across Chase & BofA
        )

        # 2. SYN_US_ATO_SYNDICATE: Account Takeover & Session Hijacking Syndicate
        ato_botnet_comcast = BotnetCluster(
            cluster_id="BOTNET_US_RES_COMCAST_02",
            name="Comcast Xfinity Residential SOCKS5 Pool",
            subnet_prefix="73.162.10.0/24",
            asn="AS7922",
            isp="Comcast Cable Communications",
            proxy_type="RESIDENTIAL_STICKY",
            ja4_signature="t13d1516h2_dfa4a7752771_c212703ab899",
            tcp_os_profile="Windows_11",
            device_fingerprints=[f"DEV_RES_CHROME_WIN_{i:03d}" for i in range(20)],
            ip_pool=[f"73.162.10.{i}" for i in range(10, 60)],
        )
        ato_botnet_att = BotnetCluster(
            cluster_id="BOTNET_US_RES_ATT_03",
            name="AT&T Fiber Residential SOCKS5 Pool",
            subnet_prefix="108.204.45.0/24",
            asn="AS7018",
            isp="AT&T Services",
            proxy_type="RESIDENTIAL_STICKY",
            ja4_signature="t13d1516h2_dfa4a7752771_c212703ab899",
            tcp_os_profile="Windows_11",
            device_fingerprints=[f"DEV_RES_EDGE_WIN_{i:03d}" for i in range(20)],
            ip_pool=[f"108.204.45.{i}" for i in range(10, 60)],
        )
        syn_ato = SyndicateEntity(
            syndicate_id="SYN_US_ATO_SYNDICATE",
            name="Account Takeover & High-Ticket Retail Syndicate",
            archetype="ATO_RETAIL_CREW",
            primary_playbooks=[
                "ADV_ATO_SILENT_BAKING",
                "ADV_ACCOUNT_TAKEOVER",
                "ADV_APPLE_PAY_YELLOW_PATH",
                "ADV_TOKEN_PROVISIONING_FRAUD",
                "INTENT_OMEGA_HARVEST",
            ],
            target_mccs=[5732, 5311, 5944],
            credential_tiers=["TIER_SESSION_COOKIE", "TIER_PHISHED_OTP", "TIER_DEVICE_TOKEN"],
            botnets=[ato_botnet_comcast, ato_botnet_att],
            mule_rings=[ato_mule_t1, ato_mule_t2],
        )

        # 3. SYN_US_SYNTHETIC_BUSTOUT: Synthetic Identity Sleeper Bust-Out & ACH Float Syndicate
        bustout_mule_t2 = MuleRing(
            ring_id="MULE_US_BUSTOUT_WF_03",
            tier=MuleTier.TIER_2_AGGREGATOR,
            account_type="COMMERCIAL_SHELL_LLC",
            liquidation_channel="ACH_KITING_WHOLESALE",
            beneficiary_accounts=[f"MULE_WELLS_CORP_{i:05d}" for i in range(300, 310)],
            bank_routing="121000247",
            fee_cut_ratio=0.18,
            layering_hop_latency_seconds=1800.0,
            bank_id="BANK_US_WELLS",
        )
        syn_bustout = SyndicateEntity(
            syndicate_id="SYN_US_SYNTHETIC_BUSTOUT",
            name="Synthetic Identity Sleeper Bust-Out Ring",
            archetype="SYNTHETIC_BUSTOUT_RING",
            primary_playbooks=[
                "ADV_SLEEPER_BUST_OUT",
                "ADV_COLLUSIVE_BUST_OUT_MID",
                "INTENT_OMEGA_INCUBATE",
            ],
            target_mccs=[6051, 5944, 5732],
            credential_tiers=["TIER_CNP_FULLZ"],
            botnets=[ato_botnet_comcast],
            mule_rings=[bustout_mule_t2, ato_mule_t2],
        )

        # 4. SYN_US_RESHIP_GIFT: CNP Reship & Gift Card Drain Ring
        reship_botnet = BotnetCluster(
            cluster_id="BOTNET_US_RES_CHARTER_04",
            name="Charter Spectrum Residential SOCKS5 Pool",
            subnet_prefix="68.114.88.0/24",
            asn="AS20115",
            isp="Charter Communications",
            proxy_type="RESIDENTIAL_STICKY",
            ja4_signature="t13d1415h2_08a3f8541249_4387201bc910",
            tcp_os_profile="macOS_Darwin",
            device_fingerprints=[f"DEV_RES_MAC_SAFARI_{i:03d}" for i in range(15)],
            ip_pool=[f"68.114.88.{i}" for i in range(10, 50)],
        )
        reship_mule = MuleRing(
            ring_id="MULE_US_GIFT_LIQUIDATION_04",
            tier=MuleTier.TIER_1_SMURF,
            account_type="PERSONAL_CHECKING",
            liquidation_channel="SECONDARY_GIFT_CARD_P2P",
            beneficiary_accounts=[f"MULE_CAPONE_{i:05d}" for i in range(400, 408)],
            bank_routing="051405515",
            fee_cut_ratio=0.25,
            layering_hop_latency_seconds=600.0,
            bank_id="BANK_US_CAPITALONE",
        )
        syn_reship = SyndicateEntity(
            syndicate_id="SYN_US_RESHIP_GIFT",
            name="CNP Reship Logistics & Gift Card Drain Ring",
            archetype="RESHIP_GIFT_DRAIN",
            primary_playbooks=[
                "ADV_TRIANGULATION_FRAUD",
                "INTENT_OMEGA_BISECT_DRAIN",
            ],
            target_mccs=[5947, 5310, 5732],
            credential_tiers=["TIER_CNP_FULLZ", "TIER_TRACK_2_DUMP"],
            botnets=[reship_botnet],
            mule_rings=[reship_mule, ato_mule_t1],
        )

        # 5. SYN_US_IAB_STEALER: Initial Access Broker & Stealer Log Distributor
        stealer_botnet = BotnetCluster(
            cluster_id="BOTNET_US_MALWARE_C2_05",
            name="Infostealer C2 Exfiltration Gateway",
            subnet_prefix="185.220.101.0/24",
            asn="AS9009",
            isp="Offshore Bulletproof Hosting",
            proxy_type="DATACENTER_ROTATING",
            ja4_signature="t12i1212h1_7caa04519921_b124802fc712",
            tcp_os_profile="Linux_Container",
            device_fingerprints=[f"DEV_MALWARE_DROPPER_{i:03d}" for i in range(10)],
            ip_pool=[f"185.220.101.{i}" for i in range(5, 45)],
        )
        syn_iab = SyndicateEntity(
            syndicate_id="SYN_US_IAB_STEALER",
            name="Initial Access Broker & Stealer Log Distributor",
            archetype="IAB_STEALER_BROKER",
            primary_playbooks=["ADV_STEALER_LOG_TRIAGE", "ADV_INITIAL_ACCESS_DROP"],
            target_mccs=[5815, 8398],
            credential_tiers=["TIER_SESSION_COOKIE", "TIER_CNP_FULLZ"],
            botnets=[stealer_botnet],
            mule_rings=[ato_mule_t1, ato_mule_t2],
        )

        # 6. SYN_US_FINCEN_MULE_NET: Multi-Tier Layering & Crypto Off-Ramp Network
        fincen_crypto_offramp = MuleRing(
            ring_id="MULE_US_CRYPTO_OFFRAMP_05",
            tier=MuleTier.TIER_3_OFFRAMP,
            account_type="P2P_CRYPTO_ESCROW",
            liquidation_channel="BINANCE_BYBIT_P2P_USDT_TRC20",
            beneficiary_accounts=[f"MULE_CRYPTO_USDT_{i:05d}" for i in range(500, 515)],
            bank_routing="021000089",
            fee_cut_ratio=0.05,
            layering_hop_latency_seconds=900.0,
            bank_id="BANK_US_CITI",
        )
        syn_fincen_mule = SyndicateEntity(
            syndicate_id="SYN_US_FINCEN_MULE_NET",
            name="FinCEN 3-Tier Layering & Crypto Off-Ramp Network",
            archetype="FINCEN_MULE_NETWORK",
            primary_playbooks=["ADV_MULTI_HOP_LAYERING", "ADV_CRYPTO_SEVERANCE"],
            target_mccs=[6051, 6012],
            credential_tiers=["TIER_SESSION_COOKIE", "TIER_DEVICE_TOKEN"],
            botnets=[ato_botnet_att],
            mule_rings=[ato_mule_t1, ato_mule_t2, bustout_mule_t2, fincen_crypto_offramp, reship_mule],
        )

        self.syndicates = [
            syn_carding,
            syn_ato,
            syn_bustout,
            syn_reship,
            syn_iab,
            syn_fincen_mule,
        ]


    def _init_india_syndicates(self) -> None:
        """Configures 5 empirical India cybercrime threat actors grounded in I4C/CFCFRMS reports."""
        # 1. SYN_IN_MEWAT_VISH: Mewat Remote Vishing & APK Trojan Ring
        apk_botnet = BotnetCluster(
            cluster_id="BOTNET_IN_JIO_APK_01",
            name="Reliance Jio 4G Mobile Botnet Pool",
            subnet_prefix="103.251.167.0/24",
            asn="AS55836",
            isp="Reliance Jio Infocomm",
            proxy_type="MOBILE_4G_5G",
            ja4_signature="t13d1516h2_apk_android_dalvik",
            tcp_os_profile="Android_Linux",
            device_fingerprints=[f"DEV_APK_SPOOF_JIO_{i:03d}" for i in range(20)],
            ip_pool=[f"103.251.167.{i}" for i in range(10, 80)],
        )
        mewat_mule_t1 = MuleRing(
            ring_id="MULE_IN_SMURF_SBI_01",
            tier=MuleTier.TIER_1_SMURF,
            account_type="PERSONAL_CHECKING",
            liquidation_channel="INSTANT_UPI_P2P",
            beneficiary_accounts=[f"MULE_SBI_{i:05d}" for i in range(100, 115)],
            bank_routing="SBIN0001234",
            fee_cut_ratio=0.10,
            layering_hop_latency_seconds=120.0,
            bank_id="BANK_IN_SBI",
        )
        axis_mule_t1 = MuleRing(
            ring_id="MULE_IN_SMURF_AXIS_04",
            tier=MuleTier.TIER_1_SMURF,
            account_type="PERSONAL_CHECKING",
            liquidation_channel="INSTANT_UPI_P2P",
            beneficiary_accounts=[f"MULE_AXIS_{i:05d}" for i in range(400, 415)],
            bank_routing="UTIB0000123",
            fee_cut_ratio=0.10,
            layering_hop_latency_seconds=150.0,
            bank_id="BANK_IN_AXIS",
        )
        pnb_mule_t2 = MuleRing(
            ring_id="MULE_IN_AGGREGATOR_PNB_05",
            tier=MuleTier.TIER_2_AGGREGATOR,
            account_type="COMMERCIAL_SHELL_LLC",
            liquidation_channel="IMPS_COMMERCIAL_SETTLEMENT",
            beneficiary_accounts=[f"MULE_PNB_{i:05d}" for i in range(500, 510)],
            bank_routing="PUNB0000456",
            fee_cut_ratio=0.12,
            layering_hop_latency_seconds=240.0,
            bank_id="BANK_IN_PNB",
        )
        syn_mewat = SyndicateEntity(
            syndicate_id="SYN_IN_MEWAT_VISH",
            name="Mewat Remote Vishing & APK Trojan Syndicate",
            archetype="REMOTE_VISHING_APK",
            primary_playbooks=[
                "IN_ADV_REVERSE_PROXY_VISHING",
                "IN_ADV_APK_SMS_STEALER",
                "IN_ADV_SIM_SWAP_ESIM_HIJACK",
                "ADV_INDIAN_VISHING_OTP",
                "ADV_INDIAN_APK_FORWARDER",
                "INTENT_OMEGA_PROBE",
            ],
            target_mccs=[4899, 8398, 5815],
            credential_tiers=["TIER_PHISHED_OTP", "TIER_TRACK_2_DUMP"],
            botnets=[apk_botnet],
            mule_rings=[mewat_mule_t1, axis_mule_t1],
        )

        # 2. SYN_IN_JAMTARA_VISH: Jamtara Telephony Social Engineering Crew
        airtel_botnet = BotnetCluster(
            cluster_id="BOTNET_IN_AIRTEL_02",
            name="Bharti Airtel Mobile SOCKS5 Pool",
            subnet_prefix="49.32.12.0/24",
            asn="AS45609",
            isp="Bharti Airtel Limited",
            proxy_type="MOBILE_4G_5G",
            ja4_signature="t13d1516h2_dfa4a7752771_c212703ab899",
            tcp_os_profile="Android_Linux",
            device_fingerprints=[f"DEV_AIRTEL_MOB_{i:03d}" for i in range(15)],
            ip_pool=[f"49.32.12.{i}" for i in range(10, 50)],
        )
        syn_jamtara = SyndicateEntity(
            syndicate_id="SYN_IN_JAMTARA_VISH",
            name="Jamtara Social Engineering & KYC Phishing Crew",
            archetype="TELEPHONY_SOCIAL_ENGINEERING",
            primary_playbooks=["IN_ADV_KYC_EXPIRY_PHISHING", "IN_ADV_LOTTERY_PRIZE_SCAM"],
            target_mccs=[5311, 5732],
            credential_tiers=["TIER_PHISHED_OTP", "TIER_CNP_FULLZ"],
            botnets=[airtel_botnet],
            mule_rings=[mewat_mule_t1, pnb_mule_t2],
        )

        # 3. SYN_IN_RENT_DRAIN: Credit-to-Bank Rent Portal Arbitrage Ring
        rent_mule_t2 = MuleRing(
            ring_id="MULE_IN_RENT_HDFC_02",
            tier=MuleTier.TIER_2_AGGREGATOR,
            account_type="COMMERCIAL_SHELL_LLC",
            liquidation_channel="CRED_HOUSING_RENT_PORTAL",
            beneficiary_accounts=[f"MULE_HDFC_CORP_{i:05d}" for i in range(200, 210)],
            bank_routing="HDFC0000456",
            fee_cut_ratio=0.12,
            layering_hop_latency_seconds=180.0,
            bank_id="BANK_IN_HDFC",
        )
        syn_rent = SyndicateEntity(
            syndicate_id="SYN_IN_RENT_DRAIN",
            name="Credit-to-Bank Rent Portal Arbitrage Ring",
            archetype="RENT_PORTAL_DRAIN",
            primary_playbooks=[
                "IN_ADV_RENT_PORTAL_CASHOUT",
                "ADV_CREDIT_LINE_CASH_OUT",
                "INTENT_OMEGA_BISECT_DRAIN",
                "INTENT_OMEGA_HARVEST",
            ],
            target_mccs=[6513, 5311, 5732],
            credential_tiers=["TIER_CNP_FULLZ", "TIER_DEVICE_TOKEN"],
            botnets=[apk_botnet],
            mule_rings=[rent_mule_t2, axis_mule_t1],
        )

        # 4. SYN_IN_P2P_MULE_HERDER: P2P Crypto Laundering Ring
        crypto_mule_t3 = MuleRing(
            ring_id="MULE_IN_P2P_CRYPTO_03",
            tier=MuleTier.TIER_3_OFFRAMP,
            account_type="P2P_CRYPTO_ESCROW",
            liquidation_channel="BINANCE_P2P_USDT_ESCROW",
            beneficiary_accounts=[f"MULE_USDT_TRC_{i:05d}" for i in range(300, 312)],
            bank_routing="ICIC0000789",
            fee_cut_ratio=0.06,
            layering_hop_latency_seconds=600.0,
            bank_id="BANK_IN_ICICI",
        )
        syn_p2p_crypto = SyndicateEntity(
            syndicate_id="SYN_IN_P2P_MULE_HERDER",
            name="P2P USDT Crypto Off-Ramp Herder Network",
            archetype="P2P_CRYPTO_OFFRAMP",
            primary_playbooks=["IN_ADV_P2P_CRYPTO_LAUNDERING", "ADV_CRYPTO_SEVERANCE"],
            target_mccs=[6051, 6012],
            credential_tiers=["TIER_CNP_FULLZ"],
            botnets=[airtel_botnet],
            mule_rings=[mewat_mule_t1, rent_mule_t2, crypto_mule_t3, axis_mule_t1, pnb_mule_t2],
        )

        # 5. SYN_IN_INTL_BYPASS: Foreign Non-3DS Gateway Bypass Syndicate
        intl_botnet = BotnetCluster(
            cluster_id="BOTNET_IN_INTL_PROXY_03",
            name="Offshore Foreign Gateway Proxy Pool",
            subnet_prefix="185.220.101.0/24",
            asn="AS9009",
            isp="Offshore Bulletproof Hosting",
            proxy_type="DATACENTER_ROTATING",
            ja4_signature="t13d1516h2_dfa4a7752771_c212703ab899",
            tcp_os_profile="Linux_Container",
            device_fingerprints=[f"DEV_INTL_CHROME_{i:03d}" for i in range(12)],
            ip_pool=[f"185.220.101.{i}" for i in range(1, 35)],
        )
        syn_intl = SyndicateEntity(
            syndicate_id="SYN_IN_INTL_BYPASS",
            name="Foreign Gateway Non-3DS Bypass Syndicate",
            archetype="INTL_NON_3DS_BYPASS",
            primary_playbooks=[
                "IN_ADV_INTL_NON_3DS_BYPASS",
                "ADV_INDIAN_INTL_BYPASS",
                "INTENT_OMEGA_INCUBATE",
            ],
            target_mccs=[4722, 5999, 7011],
            credential_tiers=["TIER_CNP_FULLZ"],
            botnets=[intl_botnet],
            mule_rings=[crypto_mule_t3, pnb_mule_t2],
        )

        self.syndicates = [
            syn_mewat,
            syn_jamtara,
            syn_rent,
            syn_p2p_crypto,
            syn_intl,
        ]


    def get_syndicate_by_id(self, syndicate_id: str) -> Optional[SyndicateEntity]:
        """Returns the syndicate entity with the given ID, if known."""
        return self._syndicate_by_id.get(syndicate_id)

    def get_syndicate_for_intent(
        self,
        macro_option: str,
        mcc: int,
        amount: float,
        credential_tier: str = "",
        playbook_name: str = "",
    ) -> SyndicateEntity:
        """Dynamically attributes a candidate fraud transaction to the most specialized syndicate.
        
        Eliminates the monopolistic 2-syndicate baseline by evaluating the target MCC,
        ticket size, credential tier, and attack macro-option.
        """
        # If explicit legacy playbook matches exactly one syndicate (and is not an INTENT scenario tag), prioritize it
        if playbook_name and not playbook_name.startswith("INTENT_OMEGA_") and playbook_name in self._playbook_to_syndicates:
            candidates = self._playbook_to_syndicates[playbook_name]
            if len(candidates) == 1:
                return candidates[0]

        if self.region == "IN":
            if macro_option == "OMEGA_PROBE" or mcc in (8398, 4899, 5815):
                # Probing is shared between Mewat remote testing and Jamtara vishing testers
                return self._syndicate_by_id["SYN_IN_JAMTARA_VISH"] if self.rng.random() < 0.35 else self._syndicate_by_id["SYN_IN_MEWAT_VISH"]
            elif mcc in (6513, 5311) or amount > 25000.0:
                return self._syndicate_by_id["SYN_IN_RENT_DRAIN"]
            elif mcc in (6051, 6012) or amount > 50000.0:
                return self._syndicate_by_id["SYN_IN_P2P_MULE_HERDER"]
            elif mcc in (4722, 7011):
                return self._syndicate_by_id["SYN_IN_INTL_BYPASS"]
            else:
                # Weighted distribution across active India syndicates
                weights = [0.35, 0.20, 0.25, 0.15, 0.05]
                return self.rng.choice(self.syndicates, p=weights)

        # US Region Logic
        if macro_option == "OMEGA_PROBE" or (mcc in (8398, 4899, 5815) and amount <= 10.0):
            # Probing is shared between card-checking botnets (75%) and Initial Access Brokers (25%)
            if self.rng.random() < 0.25:
                return self._syndicate_by_id["SYN_US_IAB_STEALER"]
            return self._syndicate_by_id["SYN_US_CARDING_BOT"]

        if mcc == 5947 or mcc == 5310:
            # Gift card liquidation and department store drains
            return self._syndicate_by_id["SYN_US_RESHIP_GIFT"]

        if mcc in (6051, 6012):
            # Cryptocurrency purchase and financial off-ramps
            return self._syndicate_by_id["SYN_US_FINCEN_MULE_NET"]

        if amount >= 1500.0 and mcc in (5732, 5944):
            # High-ticket electronics and luxury goods bust-outs or ATO
            p_bustout = 0.45 if credential_tier == "TIER_CNP_FULLZ" else 0.20
            if self.rng.random() < p_bustout:
                return self._syndicate_by_id["SYN_US_SYNTHETIC_BUSTOUT"]
            return self._syndicate_by_id["SYN_US_ATO_SYNDICATE"]

        if macro_option == "OMEGA_BISECT_DRAIN":
            # Bisect drains span between ATO retail crews and Reship/Gift rings
            return self.rng.choice([
                self._syndicate_by_id["SYN_US_ATO_SYNDICATE"],
                self._syndicate_by_id["SYN_US_RESHIP_GIFT"],
            ])

        # General harvest: distributed across the 4 exploitation syndicates
        candidate_pool = [
            self._syndicate_by_id["SYN_US_ATO_SYNDICATE"],
            self._syndicate_by_id["SYN_US_RESHIP_GIFT"],
            self._syndicate_by_id["SYN_US_SYNTHETIC_BUSTOUT"],
            self._syndicate_by_id["SYN_US_FINCEN_MULE_NET"],
        ]
        weights = [0.40, 0.30, 0.15, 0.15]
        return self.rng.choice(candidate_pool, p=weights)

    def get_syndicate_for_playbook(self, playbook: str) -> Optional[SyndicateEntity]:
        """Returns a syndicate responsible for the specified playbook (backwards compatibility)."""
        if playbook in self._playbook_to_syndicates:
            matches = self._playbook_to_syndicates[playbook]
            return matches[0] if len(matches) == 1 else self.rng.choice(matches)
        if self.syndicates:
            return self.syndicates[0]
        return None
