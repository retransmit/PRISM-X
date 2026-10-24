"""Spatial, economic, and merchant topology environment for Prism Generator.

Simulates:
1. Dual-region geography: US Metro (New York USD) and India Metro (Mumbai/Bengaluru INR).
2. Institutional merchant banking plumbing: MID (DE 42), TID (DE 41), Acquirer BIN (DE 32),
   Payment Gateway/Aggregator provisioning (Stripe, Razorpay, CyberSource, Adyen).
3. ISO 18245 Merchant Category Codes (MCC) and Zipf-distributed retail popularity.
4. Multimodal transaction channels (Card-Present EMV, Contactless NFC, E-Commerce, UPI-Credit).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class MerchantProfile:
    merchant_id: str
    name: str
    mcc: int
    category: str
    lat: float
    lon: float
    popularity_weight: float
    supported_channels: List[str]
    open_hour: int
    close_hour: int
    avg_ticket_size: float
    # Institutional Banking Attributes
    mid: str
    tid: str
    acquirer_bin: str
    gateway_provider: str
    country_code: str
    postal_code: str
    supports_3ds: bool = True
    supports_partial_auth: bool = True


# ISO 18245 Merchant Category Definitions with Institutional Risk Metrics
MCC_TAXONOMY: Dict[int, Dict[str, object]] = {
    5411: {"category": "Grocery / Supermarket", "avg_ticket_usd": 48.50, "avg_ticket_inr": 1850.00, "risk_tier": "LOW", "is_24h": False},
    5812: {"category": "Restaurants / Dining", "avg_ticket_usd": 34.00, "avg_ticket_inr": 1200.00, "risk_tier": "LOW", "is_24h": False},
    5814: {"category": "Fast Food", "avg_ticket_usd": 12.50, "avg_ticket_inr": 350.00, "risk_tier": "LOW", "is_24h": False},
    5541: {"category": "Fuel / Service Stations", "avg_ticket_usd": 45.00, "avg_ticket_inr": 2200.00, "risk_tier": "MEDIUM", "is_24h": True},
    5542: {"category": "Automated Fuel Dispenser (AFD)", "avg_ticket_usd": 45.00, "avg_ticket_inr": 2200.00, "risk_tier": "MEDIUM", "is_24h": True},
    5912: {"category": "Pharmacies", "avg_ticket_usd": 28.00, "avg_ticket_inr": 650.00, "risk_tier": "LOW", "is_24h": False},
    5311: {"category": "Department Stores", "avg_ticket_usd": 85.00, "avg_ticket_inr": 3500.00, "risk_tier": "MEDIUM", "is_24h": False},
    5732: {"category": "Consumer Electronics", "avg_ticket_usd": 420.00, "avg_ticket_inr": 28000.00, "risk_tier": "HIGH", "is_24h": False},
    5094: {"category": "Precious Stones / Wholesale Jewelry", "avg_ticket_usd": 1250.00, "avg_ticket_inr": 185000.00, "risk_tier": "CRITICAL", "is_24h": False},
    5944: {"category": "Jewelry Stores Retail (Gold)", "avg_ticket_usd": 850.00, "avg_ticket_inr": 125000.00, "risk_tier": "CRITICAL", "is_24h": False},
    7011: {"category": "Hotels / Lodging", "avg_ticket_usd": 240.00, "avg_ticket_inr": 8500.00, "risk_tier": "MEDIUM", "is_24h": True},
    7995: {"category": "Gambling / Casinos", "avg_ticket_usd": 200.00, "avg_ticket_inr": 5000.00, "risk_tier": "CRITICAL", "is_24h": True},
    6051: {"category": "Quasi-Cash / Crypto On-Ramp", "avg_ticket_usd": 500.00, "avg_ticket_inr": 25000.00, "risk_tier": "CRITICAL", "is_24h": True},
    4816: {"category": "Computer Network / SaaS", "avg_ticket_usd": 25.00, "avg_ticket_inr": 999.00, "risk_tier": "LOW", "is_24h": True},
    5999: {"category": "Miscellaneous Retail", "avg_ticket_usd": 55.00, "avg_ticket_inr": 1500.00, "risk_tier": "LOW", "is_24h": False},
    6513: {"category": "Real Estate / Rent Portals", "avg_ticket_usd": 1800.00, "avg_ticket_inr": 35000.00, "risk_tier": "HIGH", "is_24h": True},
    8398: {"category": "Charitable Organizations (Card Testing)", "avg_ticket_usd": 5.00, "avg_ticket_inr": 100.00, "risk_tier": "HIGH", "is_24h": True},
}


def compute_reachable_radius_km(delta_t_sec: float, radius_metro_km: float = 35.0) -> float:
    """Computes maximum travel distance R_max(delta_t) based on elapsed time.
    
    Regimes:
    - 0 to 120s: Intra-facility / checkout dwell (50m bounds)
    - 120s to 900s: Pedestrian walking (v_eff = 5.0 km/h)
    - 900s to 3600s: Urban surface driving / transit (v_eff = 35.0 km/h)
    - > 3600s: Regional highway travel (v_eff = 80.0 km/h) capped by metro diameter
    """
    if delta_t_sec <= 0.0:
        return 0.05

    if delta_t_sec <= 120.0:
        return 0.05

    if delta_t_sec <= 900.0:
        dt_hours = (delta_t_sec - 120.0) / 3600.0
        return 0.05 + 5.0 * dt_hours

    r_ped_max = 0.05 + 5.0 * (780.0 / 3600.0)  # ~1.133 km
    if delta_t_sec <= 3600.0:
        dt_hours = (delta_t_sec - 900.0) / 3600.0
        return r_ped_max + 35.0 * dt_hours

    r_urban_max = r_ped_max + 35.0 * (2700.0 / 3600.0)  # ~27.383 km
    dt_hours = (delta_t_sec - 3600.0) / 3600.0
    r_hwy = r_urban_max + 80.0 * dt_hours
    return min(2.0 * radius_metro_km, r_hwy)


class WorldEnvironment:
    """Simulates spatial distribution of merchants and distance-weighted merchant selection."""

    def __init__(
        self,
        n_merchants: int = 150,
        region: str = "US",
        center_lat: Optional[float] = None,
        center_lon: Optional[float] = None,
        radius_km: float = 35.0,
        seed: int = 42,
    ):
        self.region = region.upper()
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Grounded Regional Geographic Anchors
        if self.region == "IN":
            self.center_lat = center_lat if center_lat is not None else 19.0760  # Mumbai
            self.center_lon = center_lon if center_lon is not None else 72.8777
            self.currency = "INR"
            self.country_code = "IN"
            self.gateways = ["RAZORPAY", "CASHFREE", "PAYU", "BILLDESK"]
            self.acquirer_bins = ["459110", "524180", "607152", "400814"]  # HDFC, ICICI, SBI, Axis
        else:
            self.center_lat = center_lat if center_lat is not None else 40.7580  # NYC Midtown
            self.center_lon = center_lon if center_lon is not None else -73.9855
            self.currency = "USD"
            self.country_code = "US"
            self.gateways = ["STRIPE", "CYBERSOURCE", "ADYEN", "BRAINTREE"]
            self.acquirer_bins = ["400012", "412800", "541234", "370001"]  # Chase, Wells Fargo, BofA, Amex

        self.radius_km = radius_km
        self.merchants: List[MerchantProfile] = []
        self._initialize_merchant_ecosystem(n_merchants)

    def _initialize_merchant_ecosystem(self, n_merchants: int) -> None:
        """Generates realistic merchant pool with Zipf-distributed popularity and spatial dispersion."""
        mcc_keys = list(MCC_TAXONOMY.keys())

        # Base MCC frequencies aligned with economic surveys
        base_mcc_weights = np.array([
            0.24,  # 5411 Grocery
            0.18,  # 5812 Restaurant
            0.14,  # 5814 Fast Food
            0.08,  # 5541 Fuel
            0.04,  # 5542 AFD Fuel Pump
            0.07,  # 5912 Pharmacy
            0.05,  # 5311 Dept Store
            0.03,  # 5732 Electronics
            0.01,  # 5094 Jewelry Wholesale
            0.02,  # 5944 Jewelry Retail (Gold)
            0.02,  # 7011 Hotel Lodging
            0.01,  # 7995 Gambling
            0.01,  # 6051 Crypto / Quasi-Cash
            0.03,  # 4816 SaaS / Cloud
            0.04,  # 5999 Misc Retail
            0.02,  # 6513 Rent Portals
            0.01,  # 8398 Charities (Testing)
        ], dtype=np.float64)
        base_mcc_weights /= base_mcc_weights.sum()

        chosen_mccs = self.rng.choice(mcc_keys, size=n_merchants, p=base_mcc_weights)

        # Zipf popularity distribution (s = 1.15)
        ranks = np.arange(1, n_merchants + 1, dtype=np.float64)
        zipf_weights = 1.0 / (ranks ** 1.15)
        zipf_weights /= zipf_weights.sum()
        self.rng.shuffle(zipf_weights)

        # Spatial distribution: log-polar distribution centered at metropolitan core
        angles = self.rng.uniform(0.0, 2.0 * math.pi, size=n_merchants)
        radii_km = self.radius_km * (self.rng.power(0.65, size=n_merchants))

        # Geodesic lat/lon projection
        d_lat_deg = (radii_km * np.sin(angles)) / 111.139
        mean_cos = math.cos(math.radians(self.center_lat))
        d_lon_deg = (radii_km * np.cos(angles)) / (111.139 * mean_cos)

        lats = self.center_lat + d_lat_deg
        lons = self.center_lon + d_lon_deg

        for i in range(n_merchants):
            mcc = int(chosen_mccs[i])
            meta = MCC_TAXONOMY[mcc]
            is_24h = bool(meta["is_24h"])

            # Channel assignment based on merchant category
            if mcc in (4816, 6051, 6513, 8398):
                channels = ["CNP_WEB", "CNP_API", "CNP_MOBILE"]
                open_h, close_h = 0, 24
            elif mcc == 5542:  # AFD automated pump
                channels = ["CP_POS_CHIP", "CP_POS_CONTACTLESS", "CP_POS_MAGSTRIPE"]
                open_h, close_h = 0, 24
            elif is_24h:
                channels = ["CP_POS_CHIP", "CP_POS_CONTACTLESS", "CP_POS_MAGSTRIPE", "CNP_MOBILE"]
                open_h, close_h = 0, 24
            else:
                channels = ["CP_POS_CHIP", "CP_POS_CONTACTLESS", "CNP_WEB", "CNP_MOBILE"]
                open_h = int(self.rng.integers(6, 9))
                close_h = int(self.rng.integers(21, 24))

            # Postal code simulation
            if self.region == "IN":
                postal_code = f"{400001 + (i % 99):06d}"
                avg_ticket = float(meta["avg_ticket_inr"])
            else:
                postal_code = f"{10001 + (i % 99):05d}"
                avg_ticket = float(meta["avg_ticket_usd"])

            gateway = str(self.rng.choice(self.gateways))
            acq_bin = str(self.rng.choice(self.acquirer_bins))

            profile = MerchantProfile(
                merchant_id=f"M_{mcc}_{i:05d}",
                name=f"{meta['category']} #{i+1:03d}",
                mcc=mcc,
                category=str(meta["category"]),
                lat=float(lats[i]),
                lon=float(lons[i]),
                popularity_weight=float(zipf_weights[i]),
                supported_channels=channels,
                open_hour=open_h,
                close_hour=close_h,
                avg_ticket_size=avg_ticket,
                mid=f"MID_{acq_bin}_{i:06d}",
                tid=f"TID_{i:04d}",
                acquirer_bin=acq_bin,
                gateway_provider=gateway,
                country_code=self.country_code,
                postal_code=postal_code,
                supports_3ds=True,
                supports_partial_auth=(mcc == 5542),
            )
            self.merchants.append(profile)

        # Precompute vectorized coordinates and weights
        self.merchant_lats = np.array([m.lat for m in self.merchants], dtype=np.float64)
        self.merchant_lons = np.array([m.lon for m in self.merchants], dtype=np.float64)
        self.merchant_weights = np.array([m.popularity_weight for m in self.merchants], dtype=np.float64)

    def route_merchant_by_gravity(
        self,
        agent_lat: float,
        agent_lon: float,
        hour_of_day: int,
        channel_type: str = "CP_POS_CHIP",
        preferred_mcc: Optional[int] = None,
        delta_t_sec: Optional[float] = None,
        last_merchant_id: Optional[str] = None,
        last_lat: Optional[float] = None,
        last_lon: Optional[float] = None,
    ) -> MerchantProfile:
        """Selects merchant using distance-weighted probabilities filtered by kinematic travel constraints."""
        n = len(self.merchants)
        eligible_mask = np.ones(n, dtype=bool)

        for i, m in enumerate(self.merchants):
            if channel_type not in m.supported_channels:
                eligible_mask[i] = False
                continue

            if channel_type.startswith("CP"):
                if m.open_hour < m.close_hour:
                    if not (m.open_hour <= hour_of_day < m.close_hour):
                        eligible_mask[i] = False
                else:
                    if not (hour_of_day >= m.open_hour or hour_of_day < m.close_hour):
                        eligible_mask[i] = False

            if preferred_mcc is not None and m.mcc != preferred_mcc:
                eligible_mask[i] = False

        if not np.any(eligible_mask):
            for i, m in enumerate(self.merchants):
                if channel_type in m.supported_channels:
                    eligible_mask[i] = True
            if not np.any(eligible_mask):
                eligible_mask[:] = True

        eligible_indices = np.where(eligible_mask)[0]

        sub_lats = self.merchant_lats[eligible_indices]
        sub_lons = self.merchant_lons[eligible_indices]
        sub_weights = self.merchant_weights[eligible_indices]

        d_lat = (sub_lats - agent_lat) * 111.139
        mean_lat_rad = np.radians(0.5 * (sub_lats + agent_lat))
        d_lon = (sub_lons - agent_lon) * (111.139 * np.cos(mean_lat_rad))
        dist_km = np.sqrt(d_lat * d_lat + d_lon * d_lon)

        # Kinematic Reachable Radius Filtering for Card-Present Transactions
        if channel_type.startswith("CP") and delta_t_sec is not None:
            # Physical origin of the cardholder: where they physically transacted most recently
            orig_lat = last_lat if last_lat is not None else agent_lat
            orig_lon = last_lon if last_lon is not None else agent_lon

            d_lat_orig = (sub_lats - orig_lat) * 111.139
            mean_lat_rad_orig = np.radians(0.5 * (sub_lats + orig_lat))
            d_lon_orig = (sub_lons - orig_lon) * (111.139 * np.cos(mean_lat_rad_orig))
            dist_from_orig_km = np.sqrt(d_lat_orig * d_lat_orig + d_lon_orig * d_lon_orig)

            # Cap velocity at 720 km/h (ensuring safety margin below 900 km/h commercial flight ceiling)
            v_cap_kph = 720.0
            r_max = max(0.05, min(compute_reachable_radius_km(delta_t_sec, radius_metro_km=self.radius_km), delta_t_sec * (v_cap_kph / 3600.0)))
            kinematic_mask = (dist_from_orig_km <= r_max)

            if np.any(kinematic_mask):
                eligible_indices = eligible_indices[kinematic_mask]
                sub_weights = sub_weights[kinematic_mask]
                dist_km = dist_km[kinematic_mask]
                dist_from_orig_km = dist_from_orig_km[kinematic_mask]
            else:
                # Dwell fallback: cardholder remains at the same merchant facility
                if last_merchant_id is not None:
                    for m in self.merchants:
                        if m.merchant_id == last_merchant_id:
                            return m
                closest_sub_idx = np.argmin(dist_from_orig_km)
                return self.merchants[eligible_indices[closest_sub_idx]]

        if not channel_type.startswith("CP"):
            gravity_scores = sub_weights
        else:
            dist_clamped = np.maximum(dist_km, 0.05)
            gravity_scores = sub_weights / (dist_clamped ** 1.35)

        total_score = np.sum(gravity_scores)
        if total_score <= 0.0 or math.isnan(total_score):
            probs = np.ones(len(eligible_indices), dtype=np.float64) / len(eligible_indices)
        else:
            probs = gravity_scores / total_score

        probs = probs / np.sum(probs)
        chosen_idx = self.rng.choice(eligible_indices, p=probs)
        return self.merchants[chosen_idx]
