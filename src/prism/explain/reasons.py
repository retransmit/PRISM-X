"""Reason codes: short, stable codes per driver family, whichever model produced the
score, for a cardholder's message and the audit log (design section 8)."""

from __future__ import annotations

from collections.abc import Iterable

# code -> (label, the track features that belong to it). Column lists are code, not
# configuration: a typo here fails a test, not a customer. A feature no family names is
# an anonymised component.
FAMILIES: dict[str, tuple[str, tuple[str, ...]]] = {
    "VEL": ("card velocity", ("card_tx_1h", "card_tx_24h", "card_amount_24h", "seconds_since_last", "card_distinct_mcc_30m")),
    "AMT": ("amount against the card's limit and history", ("amount", "amount_to_limit", "utilization", "Amount")),
    "NEW": ("first use of a merchant category or channel", ("first_mcc_30d", "first_cnp_30d")),
    "NET": ("network risk score and verification results", ("vaai_score", "geo_risk_score", "cvv_match_flag", "avs_match_code", "eci")),
    "LOC": ("location", ("ip_distance_from_home_km", "is_cross_border")),
    "CHN": ("channel and entry mode", ("channel_type", "pos_entry_mode", "pos_condition_code", "asn_type")),
    "MER": ("merchant category, acquirer and gateway", ("mcc", "acquirer_bin", "gateway_provider")),
    "PRD": ("card product", ("product_id",)),
    "TIM": ("time of day", ("hour_of_day", "Time")),
    "LAT": ("anonymised components", ()),
}
_BY_FEATURE = {feature: code for code, (_, features) in FAMILIES.items() for feature in features}


def family_of(feature: str) -> tuple[str, str]:
    """The code and label of the family a feature belongs to."""
    code = _BY_FEATURE.get(feature, "LAT")
    return code, FAMILIES[code][0]


# A family whose share of the push would show as zero percent gets no code.
MIN_SHARE = 0.01


def reason_codes(contributions: Iterable, limit: int = 4) -> list[dict]:
    """The families whose net push is towards fraud, strongest first, each with its share
    of that push and the members that pushed that way (never one that argued against).
    ``contributions`` carry ``feature`` and ``shap_value``; nothing else about them is read."""
    net: dict[str, float] = {}
    members: dict[str, list] = {}
    for c in contributions:
        code, _ = family_of(c.feature)
        net[code] = net.get(code, 0.0) + float(c.shap_value)
        members.setdefault(code, []).append(c)
    positive = {code: value for code, value in net.items() if value > 0.0}
    total = sum(positive.values())
    out = []
    for code, value in sorted(positive.items(), key=lambda item: item[1], reverse=True):
        share = float(value / total)
        if share < MIN_SHARE:
            continue
        towards = sorted((c for c in members[code] if c.shap_value > 0.0), key=lambda c: c.shap_value, reverse=True)
        out.append({
            "code": code,
            "label": FAMILIES[code][0],
            "share": share,
            "features": [c.feature for c in towards[:3]],
        })
        if len(out) == limit:
            break
    return out
