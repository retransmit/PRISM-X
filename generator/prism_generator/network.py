"""Client address space shared by cardholders and attackers (spec/07 section 16).

A metro's consumer traffic arrives from a bounded set of /24 prefixes; a home broadband
connection keeps its prefix for weeks while a phone roams the carrier's pool. Residential
and mobile proxy pools are compromised consumer devices on those same prefixes, so the
botnets draw their pools from the table the cardholders use. Datacenter pools (cloud, Tor,
bulletproof hosting) come from a second table that legitimate VPN users also draw from.

The table is a deterministic function of the run's seed, so the ledger and the syndicate
registry, built apart, land on the same prefixes.
"""

from __future__ import annotations

import zlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .spec_loader import load_all_specs

_FALLBACK_FIRST_OCTETS = {
    ("US", "residential"): [72],
    ("IN", "residential"): [103],
    ("US", "datacenter"): [185],
    ("IN", "datacenter"): [185],
}


def prefix_of(ip: str) -> str:
    """The /24 an address belongs to, written the way the threat-intelligence view writes it."""
    return ".".join(ip.split(".")[:3]) + ".0/24"


class ClientAddressSpace:
    """The /24 prefixes one run's traffic comes from, per region and per kind of network."""

    def __init__(self, seed: int = 42, spec: Optional[Dict[str, Any]] = None):
        self.seed = int(seed)
        self.spec: Dict[str, Any] = spec if spec is not None else dict(load_all_specs().telemetry.address_space)
        self._tables: Dict[Tuple[str, str], List[str]] = {}
        self._weights: Dict[Tuple[str, str], np.ndarray] = {}

    def prefixes(self, region: str, kind: str = "residential") -> List[str]:
        """The prefix table of a region and kind, built once per run from the seed."""
        key = (region.upper(), kind)
        if key not in self._tables:
            self._tables[key] = self._build(*key)
        return self._tables[key]

    def weights(self, region: str, kind: str = "residential") -> np.ndarray:
        """How much of the traffic each prefix carries: a Zipf law over the table, so the large
        carriers' prefixes carry most of it and a proxy pool lands where the cardholders are."""
        key = (region.upper(), kind)
        if key not in self._weights:
            exponent = float(self.spec.get("prefix_popularity_exponent", 1.0))
            ranks = np.arange(1, len(self.prefixes(region, kind)) + 1, dtype=float)
            raw = ranks ** (-exponent)
            self._weights[key] = raw / raw.sum()
        return self._weights[key]

    def _build(self, region: str, kind: str) -> List[str]:
        default_count = 400 if kind == "residential" else 40
        count = int(self.spec.get(f"{kind}_prefix_count", {}).get(region, default_count))
        octets = [int(o) for o in self.spec.get(f"{kind}_first_octets", {}).get(region, [])]
        if not octets:
            octets = _FALLBACK_FIRST_OCTETS.get((region, kind), [72])
        rng = np.random.default_rng([self.seed, zlib.crc32(f"{region}:{kind}".encode("utf-8"))])
        table: List[str] = []
        seen = set()
        while len(table) < count:
            prefix = f"{int(rng.choice(octets))}.{int(rng.integers(0, 256))}.{int(rng.integers(0, 256))}"
            if prefix not in seen:
                seen.add(prefix)
                table.append(prefix)
        return table

    def home_prefix(self, card_id: str, region: str) -> str:
        """The /24 a cardholder's home connection sits on, fixed for the run and drawn by the
        prefixes' share of the traffic."""
        table = self.prefixes(region, "residential")
        cumulative = np.cumsum(self.weights(region, "residential"))
        u = zlib.crc32(card_id.encode("utf-8")) / 2 ** 32
        return table[min(len(table) - 1, int(np.searchsorted(cumulative, u, side="right")))]

    def legitimate_ip(
        self,
        card_id: str,
        region: str,
        channel_type: str,
        asn_type: str,
        rng: np.random.Generator,
    ) -> str:
        """A cardholder's own address: the home prefix most of the time on broadband, a roaming
        prefix on mobile, a datacenter prefix when the cardholder is behind a VPN."""
        if asn_type == "datacenter":
            prefix = str(rng.choice(self.prefixes(region, "datacenter"), p=self.weights(region, "datacenter")))
        else:
            share = float(self.spec.get("home_prefix_share", {}).get(channel_type, 0.5))
            if rng.random() < share:
                prefix = self.home_prefix(card_id, region)
            else:
                prefix = str(rng.choice(self.prefixes(region, "residential"), p=self.weights(region, "residential")))
        return f"{prefix}.{int(rng.integers(1, 255))}"

    def proxy_pool(self, region: str, kind: str, rng: np.random.Generator) -> Tuple[str, List[str]]:
        """A botnet's primary subnet and its address pool, drawn from the shared table of its kind."""
        kind = "datacenter" if kind == "datacenter" else "residential"
        default_prefixes = 3 if kind == "datacenter" else 8
        n_prefixes = max(1, int(self.spec.get(f"{kind}_prefixes_per_botnet", default_prefixes)))
        hosts = max(1, min(254, int(self.spec.get("hosts_per_prefix", 60))))
        table = self.prefixes(region, kind)
        picks = rng.choice(len(table), size=min(n_prefixes, len(table)), replace=False, p=self.weights(region, kind))
        chosen = [table[int(i)] for i in picks]
        pool = [
            f"{prefix}.{int(h)}"
            for prefix in chosen
            for h in rng.choice(np.arange(1, 255), size=hosts, replace=False)
        ]
        return f"{chosen[0]}.0/24", pool
