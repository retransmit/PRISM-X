"""Reason codes: short stable codes per driver family, whichever model produced the score."""

from __future__ import annotations

import pytest

from prism.explain import reasons
from prism.explain.shap_engine import Explanation, FeatureContribution


def _c(feature, value):
    return FeatureContribution(feature=feature, value=1.0, shap_value=value, interpretable=True)


def test_families_cover_every_indian_feature_and_the_anonymised_components():
    from prism import tracks

    for feature in tracks.INDIA.feature_columns:
        code, label = reasons.family_of(feature)
        assert code in reasons.FAMILIES and code != "LAT" and label, feature
    assert reasons.family_of("V14")[0] == "LAT"
    assert reasons.family_of("Amount")[0] == "AMT"
    assert reasons.family_of("Time")[0] == "TIM"


def test_codes_aggregate_by_family_and_order_by_the_push_towards_fraud():
    contributions = [
        _c("card_tx_1h", 0.5), _c("card_tx_24h", 0.3), _c("seconds_since_last", -0.1), _c("amount", 0.6),
        _c("vaai_score", -0.4), _c("ip_distance_from_home_km", 0.1), _c("mcc", 0.05), _c("product_id", 0.001),
    ]
    codes = reasons.reason_codes(contributions, limit=3)
    assert [c["code"] for c in codes] == ["VEL", "AMT", "LOC"]
    # a code names the members that pushed towards fraud, never one that argued against
    assert codes[0]["features"] == ["card_tx_1h", "card_tx_24h"]
    total = 0.7 + 0.6 + 0.1 + 0.05 + 0.001
    assert codes[0]["share"] == pytest.approx(0.7 / total)
    assert sum(c["share"] for c in codes) == pytest.approx((0.7 + 0.6 + 0.1) / total)
    assert all(c["label"] for c in codes)
    # a family whose share would show as zero percent gets no code
    assert "PRD" not in [c["code"] for c in reasons.reason_codes(contributions, limit=8)]
    assert "MER" in [c["code"] for c in reasons.reason_codes(contributions, limit=8)]


def test_no_push_towards_fraud_means_no_codes():
    assert reasons.reason_codes([_c("amount", -0.2), _c("mcc", 0.0)]) == []


def test_the_explanation_payload_carries_the_codes():
    explanation = Explanation(model="xgboost", base_value=0.0, prediction=0.9, contributions=[_c("amount", 0.9)])
    payload = explanation.as_dict()
    assert payload["reason_codes"][0]["code"] == "AMT"
    assert payload["reason_codes"][0]["share"] == 1.0
