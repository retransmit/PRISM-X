"""Physical, mathematical, financial, and temporal invariant verification suite for Prism Generator.

Certifies that generated synthetic transaction records strictly conform to:
1. Geodesic laws and antipodal-safe Haversine metrics.
2. Space-time kinematic velocity limits (< 900 km/h for card-present commercial travel).
3. Temporal monotonicity per cardholder account (no retrograde time travel).
4. Monetary conservation and positive currency bounds.
5. Analytical Shapley efficiency axioms.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

EARTH_RADIUS_KM: float = 6371.0088


def haversine_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Computes great-circle distance between two geographic coordinates on WGS-84 sphere.

    Mathematically robust against floating-point roundoff errors (antipodal points
    where a > 1.0, identical points where d = 0, and polar singularities).
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    sin_half_dphi = math.sin(delta_phi * 0.5)
    sin_half_dlambda = math.sin(delta_lambda * 0.5)

    a = sin_half_dphi * sin_half_dphi + math.cos(phi1) * math.cos(phi2) * sin_half_dlambda * sin_half_dlambda

    # Strict clamping to handle numerical precision drift on antipodal coordinates
    a_clamped = max(0.0, min(1.0, a))
    one_minus_a = max(0.0, 1.0 - a_clamped)

    c = 2.0 * math.atan2(math.sqrt(a_clamped), math.sqrt(one_minus_a))
    return EARTH_RADIUS_KM * c


def verify_transaction_invariants(
    tx: Dict[str, Any],
    prev_tx: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Verifies physical, financial, and kinematic invariants on a synthesized transaction.

    Args:
        tx: Current transaction dictionary.
        prev_tx: Previous transaction dictionary for the same cardholder (if available).

    Returns:
        List of violation strings (empty list indicates transaction passed all invariants).
    """
    violations: List[str] = []

    # 1. Schema & Non-Nullability Invariants
    mandatory_fields = [
        "transaction_id",
        "card_id",
        "merchant_id",
        "timestamp_utc",
        "tx_time_seconds",
        "amount",
        "currency",
        "mcc",
        "channel_type",
        "is_fraud",
        "scenario_tag",
    ]
    for field in mandatory_fields:
        if field not in tx or tx[field] is None:
            violations.append(f"MISSING_FIELD: Mandatory attribute '{field}' is absent or null.")

    # 2. Monetary Bounds Invariants
    amount = tx.get("amount", 0.0)
    currency = tx.get("currency", "USD")
    # Sanity ceilings: USD 250,000 aligns with PROD_ULTRA_PREMIUM_INFINITE line (spec/01) and large-ticket T&E holds.
    # INR 20,000,000 (₹2 Crore) provides exact USD-INR parity (~$240k USD) for private wealth cards.
    max_cap = 20000000.0 if currency == "INR" else 250000.0
    if not isinstance(amount, (int, float)) or math.isnan(amount) or math.isinf(amount):
        violations.append(f"INVALID_AMOUNT_TYPE: Amount {amount} is not a valid finite float.")
    elif amount <= 0.0:
        violations.append(f"NON_POSITIVE_AMOUNT: Amount {amount:.2f} {currency} must be strictly positive.")
    elif amount > max_cap:
        violations.append(f"UNREALISTIC_EXCESSIVE_AMOUNT: Amount {amount:.2f} {currency} exceeds physical scheme ceiling ({max_cap:.2f} {currency}).")

    # 3. Coordinate Bounds Invariants
    lat = tx.get("merchant_lat", 0.0)
    lon = tx.get("merchant_lon", 0.0)
    if not (-90.0 <= lat <= 90.0):
        violations.append(f"COORDINATE_LAT_OUT_OF_BOUNDS: Latitude {lat} outside [-90, +90].")
    if not (-180.0 <= lon <= 180.0):
        violations.append(f"COORDINATE_LON_OUT_OF_BOUNDS: Longitude {lon} outside [-180, +180].")

    # 4. Binary & Categorical Field Invariants
    is_fraud = tx.get("is_fraud", 0)
    if is_fraud not in (0, 1):
        violations.append(f"INVALID_LABEL: is_fraud must be binary integer 0 or 1, got {is_fraud}.")

    channel = str(tx.get("channel_type", ""))
    valid_channels = {"CP_POS_CHIP", "CP_POS_CONTACTLESS", "CP_POS_MAGSTRIPE", "CNP_WEB", "CNP_MOBILE", "CNP_API"}
    if channel not in valid_channels:
        violations.append(f"INVALID_CHANNEL: channel_type '{channel}' not in recognized enum.")

    # 5. Temporal & Kinematic Sequence Invariants (Across consecutive transactions)
    if prev_tx is not None and prev_tx.get("card_id") == tx.get("card_id"):
        t_prev = float(prev_tx.get("tx_time_seconds", 0.0))
        t_curr = float(tx.get("tx_time_seconds", 0.0))
        delta_t = t_curr - t_prev

        if delta_t < 0.0:
            violations.append(
                f"RETROGRADE_TIME_TRAVEL: Card {tx.get('card_id')} moved backwards in time: "
                f"prev_t={t_prev:.1f}s, curr_t={t_curr:.1f}s (delta={delta_t:.1f}s)."
            )

        # Kinematic velocity check: ONLY strictly enforced between consecutive legitimate Card-Present (CP) in-person swipes
        # (If previous transaction was fraudulent, legitimate cardholder was not physically at that location)
        prev_channel = str(prev_tx.get("channel_type", ""))
        prev_is_fraud = int(prev_tx.get("is_fraud", 0))

        if channel.startswith("CP") and prev_channel.startswith("CP") and is_fraud == 0 and prev_is_fraud == 0:
            prev_lat = float(prev_tx.get("merchant_lat", 0.0))
            prev_lon = float(prev_tx.get("merchant_lon", 0.0))
            dist_km = haversine_distance_km(prev_lat, prev_lon, lat, lon)
            delta_hours = delta_t / 3600.0

            if delta_hours > 0.0:
                velocity_kph = dist_km / delta_hours
                if velocity_kph > 900.0:
                    violations.append(
                        f"IMPOSSIBLE_TRAVEL_VELOCITY: Card-Present transaction for card {tx.get('card_id')} "
                        f"traveled {dist_km:.2f} km in {delta_t:.1f}s ({velocity_kph:.1f} km/h > 900 km/h) "
                        f"between two legitimate in-person transactions."
                    )

    # 6. Shapley Efficiency Axiom Check (if causal metadata is attached)
    if "analytical_shapley_probability" in tx and "risk_score" in tx and "base_risk" in tx:
        shapley_dict = tx["analytical_shapley_probability"]
        risk_score = float(tx["risk_score"])
        base_risk = float(tx["base_risk"])
        sum_shapley = sum(float(v) for v in shapley_dict.values())
        diff = abs(sum_shapley - (risk_score - base_risk))
        if diff > 1e-4:
            violations.append(
                f"SHAPLEY_EFFICIENCY_VIOLATION: Probability Shapley sum ({sum_shapley:.6f}) "
                f"does not equal risk_score - base_risk ({risk_score - base_risk:.6f}), gap={diff:.6e}."
            )

    return violations
