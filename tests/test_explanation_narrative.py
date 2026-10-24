"""Tests for analyst-readable explanation text (Option A, PCA limitation)."""

from __future__ import annotations

import pytest

from prism.explain.shap_engine import Explanation, FeatureContribution


def make_explanation(contributions: list[FeatureContribution]) -> Explanation:
    return Explanation(
        model="xgboost",
        base_value=-4.0,
        prediction=1.5,
        contributions=contributions,
    )


@pytest.fixture
def amount_driven() -> Explanation:
    return make_explanation(
        [
            FeatureContribution("Amount", 4200.0, 3.0, interpretable=True),
            FeatureContribution("V14", -6.1, 1.5, interpretable=False),
            FeatureContribution("V4", 2.2, 0.5, interpretable=False),
            FeatureContribution("Time", 3600.0, 0.05, interpretable=True),
        ]
    )


@pytest.fixture
def latent_driven() -> Explanation:
    return make_explanation(
        [
            FeatureContribution("V14", -6.1, 3.0, interpretable=False),
            FeatureContribution("V12", -4.0, 2.0, interpretable=False),
            FeatureContribution("V10", -3.2, 1.0, interpretable=False),
            FeatureContribution("Amount", 12.0, 0.1, interpretable=True),
        ]
    )


def test_concentration_is_a_fraction_of_total_attribution(amount_driven):
    # top 3 of |3.0| + |1.5| + |0.5| + |0.05| = 5.05 -> 5.0 / 5.05
    assert amount_driven.concentration(3) == pytest.approx(5.0 / 5.05)


def test_concentration_of_everything_is_one(amount_driven):
    assert amount_driven.concentration(99) == pytest.approx(1.0)


def test_named_features_are_described_in_business_terms(amount_driven):
    text = amount_driven.narrative(k=3)
    assert "Amount" in text
    assert "4,200" in text or "4200" in text


def test_fallback_narrative_counts_components_and_never_names_them(latent_driven):
    """The attribution-only fallback (served when the reference artifacts are
    absent) must follow the same policy as the measured-facts narrative: a PCA
    component is counted as anonymised, never named."""
    import re

    text = latent_driven.narrative(k=3)
    assert re.search("V[0-9]+", text) is None, text
    assert "3 anonymised factors" in text


def test_narrative_quotes_the_concentration(latent_driven):
    text = latent_driven.narrative(k=3)
    assert "3" in text
    assert "%" in text


def test_narrative_direction_matches_the_sign_of_attribution():
    pushed_down = make_explanation(
        [
            FeatureContribution("Amount", 2.0, -3.0, interpretable=True),
            FeatureContribution("V14", 0.1, -1.0, interpretable=False),
        ]
    )
    assert "lower" in pushed_down.narrative(k=2).lower()


def test_as_dict_carries_the_narrative_to_the_dashboard(amount_driven):
    payload = amount_driven.as_dict(k=3)
    assert isinstance(payload["narrative"], str)
    assert payload["narrative"]


def test_concentration_handles_an_all_zero_attribution():
    flat = make_explanation(
        [FeatureContribution("Amount", 1.0, 0.0, interpretable=True)]
    )
    assert flat.concentration(1) == 0.0
    assert isinstance(flat.narrative(k=1), str)


def test_driver_sentence_names_a_numeric_driver_against_its_band():
    import numpy as np

    from prism import tracks
    from prism.explain.narrative import driver_sentence, plain_english

    class Reference:
        amount_column = "amount"
        amounts = np.array([100.0])
        characteristics = ["vaai_score", "card_tx_1h"]
        categorical_share = {"channel_type": {"CNP_WEB": 0.21, "CP_POS_CHIP": 0.79}}
        band = 0.995
        driver_band = 0.95

        def numeric_band(self, feature):
            return (2.0, 70.0)

        def numeric_driver_band(self, feature):
            return (5.0, 52.0)

    explanation = Explanation(
        model="xgboost", base_value=-4.0, prediction=2.0,
        contributions=[
            FeatureContribution("vaai_score", 71.0, 3.1, True),
            FeatureContribution("channel_type", 2.0, 0.4, True, display="CNP_WEB"),
        ],
    )
    sentence = driver_sentence(explanation, Reference(), {"vaai_score": 71.0, "channel_type": "CNP_WEB"}, tracks.INDIA)
    assert sentence == "The strongest push towards fraud came from the network risk score: 71, above the 52 that covers 95% of transactions."

    facts = {
        "amount": 4200.0, "amount_pct_below": 0.974, "cycle_share": 0.042, "typical_cycle_share": 0.042,
        "unusual_count": 1, "n_characteristics": 2, "band": 0.995, "similar_fraud": 41, "similar_total": 50,
        "base_rate": 0.005, "clearing_amount": None,
    }
    assert plain_english(explanation, facts, driver=sentence).endswith(sentence)
    assert plain_english(explanation, facts) == plain_english(explanation, facts, driver=None)


def test_driver_sentence_names_a_categorical_driver_by_its_share():
    import numpy as np

    from prism import tracks
    from prism.explain.narrative import driver_sentence

    class Reference:
        amount_column = "amount"
        amounts = np.array([100.0])
        characteristics = ["vaai_score"]
        categorical_share = {"channel_type": {"CNP_WEB": 0.21, "MOTO": 0.0046, "KIOSK": 0.0004}}
        band = 0.995
        driver_band = 0.95

        def numeric_band(self, feature):
            return (0.0, 1.0)

        def numeric_driver_band(self, feature):
            return (0.0, 1.0)

    explanation = Explanation(
        model="xgboost", base_value=-4.0, prediction=-1.0,
        contributions=[FeatureContribution("channel_type", 2.0, -1.5, True, display="CNP_WEB")],
    )
    sentence = driver_sentence(explanation, Reference(), {"channel_type": "CNP_WEB"}, tracks.INDIA)
    assert sentence == "The strongest push away from fraud came from the channel: CNP_WEB, seen in 21% of the transactions the system learned from."

    rare = Explanation(model="xgboost", base_value=-4.0, prediction=-1.0,
                       contributions=[FeatureContribution("channel_type", 3.0, 1.0, True, display="MOTO")])
    assert driver_sentence(rare, Reference(), {"channel_type": "MOTO"}, tracks.INDIA).endswith(
        "MOTO, seen in 0.5% of the transactions the system learned from."
    )
    rarer = Explanation(model="xgboost", base_value=-4.0, prediction=-1.0,
                        contributions=[FeatureContribution("channel_type", 4.0, 1.0, True, display="KIOSK")])
    assert driver_sentence(rarer, Reference(), {"channel_type": "KIOSK"}, tracks.INDIA).endswith(
        "KIOSK, seen in fewer than 0.1% of the transactions the system learned from."
    )


def test_driver_sentence_for_the_amount_does_not_repeat_the_amount_fact():
    """The narrative's first clause already places the amount against the
    population; an amount-led alert read the same comparison twice."""
    import numpy as np

    from prism import tracks
    from prism.explain.narrative import driver_sentence

    class Reference:
        amount_column = "amount"
        amounts = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        characteristics = []
        categorical_share = {}
        band = 0.995

    explanation = Explanation(model="xgboost", base_value=0.0, prediction=2.0,
                              contributions=[FeatureContribution("amount", 450.0, 2.0, True)])
    sentence = driver_sentence(explanation, Reference(), {"amount": 450.0}, tracks.INDIA)
    assert sentence == "The strongest push towards fraud came from the amount."

    small = Explanation(model="xgboost", base_value=0.0, prediction=-2.0,
                        contributions=[FeatureContribution("amount", 150.0, -2.0, True)])
    assert driver_sentence(small, Reference(), {"amount": 150.0}, tracks.INDIA) == (
        "The strongest push away from fraud came from the amount."
    )


def test_no_driver_sentence_on_the_anonymised_track():
    from prism import tracks
    from prism.explain.narrative import driver_sentence

    explanation = Explanation(model="xgboost", base_value=0.0, prediction=1.0,
                              contributions=[FeatureContribution("V14", 3.0, 2.0, False)])
    assert driver_sentence(explanation, None, {"V14": 3.0}, tracks.ULB) is None
