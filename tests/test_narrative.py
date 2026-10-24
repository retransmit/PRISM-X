"""The sentence an analyst reads.

It must never name a PCA component, must change when the facts change, and
must say only things the data can support: no cardholder history, no clock
time, no invented cause.
"""

from __future__ import annotations

import re

import pytest

from prism.explain.narrative import plain_english
from prism.explain.shap_engine import Explanation, FeatureContribution


def explanation() -> Explanation:
    contributions = [FeatureContribution("Amount", 2.28, 0.4, True)]
    contributions += [
        FeatureContribution(f"V{i}", 0.1 * i, 3.0 if i in (14, 4, 10) else 0.05, False)
        for i in range(1, 29)
    ]
    return Explanation(model="xgboost", base_value=-4.0, prediction=2.0, contributions=contributions)


def facts(**over) -> dict:
    base = {
        "amount": 2.28,
        "amount_pct_below": 0.17,
        "cycle_share": 0.012,
        "typical_cycle_share": 1 / 24,
        "unusual_count": 19,
        "n_characteristics": 28,
        "band": 0.995,
        "similar_fraud": 45,
        "similar_total": 50,
        "base_rate": 0.0017,
        "clearing_amount": None,
    }
    base.update(over)
    return base


def test_never_names_a_pca_component():
    text = plain_english(explanation(), facts())
    assert not re.search(r"\bV\d{1,2}\b", text), text
    assert "latent" not in text.lower()


def test_a_small_amount_is_described_as_small():
    text = plain_english(explanation(), facts(amount=2.28, amount_pct_below=0.17))
    assert "2.28" in text
    assert "small" in text
    assert "83% of transactions are larger" in text


def test_a_large_amount_is_ranked_against_the_population():
    text = plain_english(explanation(), facts(amount=4_200.0, amount_pct_below=0.997))
    assert "4,200.00" in text
    assert "larger than 99.7% of all transactions" in text


def test_a_quiet_part_of_the_daily_cycle_is_said_plainly():
    text = plain_english(explanation(), facts(cycle_share=0.012))
    assert "quiet part of the daily cycle" in text
    assert "1.2% of transactions" in text
    assert "4.2%" in text  # what a typical hour sees


def test_an_ordinary_time_is_not_dressed_up():
    text = plain_english(explanation(), facts(cycle_share=0.05))
    assert "ordinary point in the daily cycle" in text
    assert "quiet" not in text


def test_counts_unusual_characteristics_without_naming_them():
    text = plain_english(explanation(), facts(unusual_count=19))
    assert "unusual on 19 of its 28 transaction characteristics" in text
    assert "99.5% of transactions" in text


def test_says_when_nothing_is_out_of_range():
    text = plain_english(explanation(), facts(unusual_count=0))
    assert "None of its 28 transaction characteristics is outside the normal range" in text


def test_quotes_the_similar_cases_against_the_base_rate():
    text = plain_english(explanation(), facts(similar_fraud=45, similar_total=50, base_rate=0.0017))
    assert "Of the 50 most similar transactions the system learned from, 45 were confirmed fraud" in text
    assert "overall rate is 0.17%" in text


def test_states_how_concentrated_the_decision_is():
    text = plain_english(explanation(), facts())
    share = explanation().concentration(3)
    assert f"Three factors account for {share:.0%} of the decision" in text


def test_amount_counterfactual_appears_only_when_it_fires():
    without = plain_english(explanation(), facts(clearing_amount=None))
    assert "would have been cleared" not in without
    with_cf = plain_english(explanation(), facts(amount=4_200.0, amount_pct_below=0.99, clearing_amount=118.5, clearing_is_monotone=True))
    assert "Had the amount been below 118.50, with everything else unchanged, it would have been cleared" in with_cf


def test_different_facts_produce_different_sentences():
    a = plain_english(explanation(), facts())
    b = plain_english(explanation(), facts(amount=252.92, amount_pct_below=0.921, cycle_share=0.05,
                                            unusual_count=10, similar_fraud=36))
    assert a != b


@pytest.mark.parametrize("missing", ["amount_pct_below", "similar_fraud", "unusual_count"])
def test_refuses_incomplete_facts_rather_than_guessing(missing):
    incomplete = facts()
    del incomplete[missing]
    with pytest.raises(KeyError):
        plain_english(explanation(), incomplete)


def test_a_non_monotone_counterfactual_is_a_point_claim():
    text = plain_english(
        explanation(), facts(amount=4_200.0, amount_pct_below=0.99, clearing_amount=118.5, clearing_is_monotone=False)
    )
    assert "At 118.50, with everything else unchanged, it would have been cleared; some lower amounts would still be flagged." in text
    assert "Had the amount been below" not in text
