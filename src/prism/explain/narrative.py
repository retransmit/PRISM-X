"""The sentence an analyst reads for a flagged transaction.

Every clause is measured against the training population, none names a PCA
component, and nothing claims a cardholder history or a clock time the dataset
does not carry. The point is that the sentence changes with the case: the
previous narrative named the same three components for every alert, which
told the reader nothing.
"""

from __future__ import annotations

from typing import Mapping

from prism.explain.shap_engine import Explanation


def _percent(share: float) -> str:
    """A share as a percentage that never rounds a rare value to nothing."""
    if share < 0.001:
        return "fewer than 0.1%"
    return f"{share * 100:.1f}".rstrip("0").rstrip(".") + "%"


def driver_sentence(explanation: Explanation, reference, raw: Mapping[str, object], schema) -> str | None:
    """One sentence naming the strongest driver, for tracks whose features have names.

    Numerics are placed against the training band the reference statistics
    hold, categoricals against the share of training rows that carried the
    same value. The amount is only named: the narrative's first clause has
    already placed it against the population, and an amount-led alert read
    the same comparison twice. The anonymised track returns None: "V14 drove
    this alert" is true and useless, and the sentence is not written.
    """
    if schema.interpretable is not None or not explanation.contributions:
        return None
    lead = explanation.top(1)[0]
    direction = "towards" if lead.shap_value >= 0 else "away from"
    label = schema.label(lead.feature)
    if lead.feature == reference.amount_column:
        return f"The strongest push {direction} fraud came from the {label}."
    if lead.feature in reference.characteristics:
        low, high = reference.numeric_driver_band(lead.feature)
        coverage = getattr(reference, "driver_band", None) or reference.band
        value = float(raw[lead.feature])
        if value > high:
            comparison = f"above the {high:,.0f} that covers {coverage:.0%} of transactions"
        elif value < low:
            comparison = f"below the {low:,.0f} that covers {coverage:.0%} of transactions"
        else:
            comparison = "within its usual range"
        return f"The strongest push {direction} fraud came from the {label}: {value:,.0f}, {comparison}."
    if lead.feature in reference.categorical_share:
        value = str(raw[lead.feature])
        share = float(reference.categorical_share[lead.feature].get(value, 0.0))
        return (
            f"The strongest push {direction} fraud came from the {label}: {value}, "
            f"seen in {_percent(share)} of the transactions the system learned from."
        )
    return None


def plain_english(explanation: Explanation, facts: Mapping[str, object], driver: str | None = None) -> str:
    """Compose the explanation from measured facts. Missing facts raise rather
    than being papered over: a sentence built on a guess is worse than none."""
    amount = float(facts["amount"])
    pct_below = float(facts["amount_pct_below"])
    if pct_below >= 0.5:
        amount_clause = f"The amount, {amount:,.2f}, is larger than {pct_below:.1%} of all transactions."
    else:
        amount_clause = f"The amount, {amount:,.2f}, is small: {1 - pct_below:.0%} of transactions are larger."

    share = float(facts["cycle_share"])
    typical = float(facts["typical_cycle_share"])
    if share < 0.5 * typical:
        cycle_clause = (
            f"It came during a quiet part of the daily cycle, when only {share:.1%} of "
            f"transactions occur (a typical hour sees {typical:.1%})."
        )
    else:
        cycle_clause = "It came at an ordinary point in the daily cycle."

    unusual = int(facts["unusual_count"])
    total = int(facts["n_characteristics"])
    if unusual:
        unusual_clause = (
            f"It is unusual on {unusual} of its {total} transaction characteristics, each "
            f"outside the range that covers {float(facts['band']):.1%} of transactions."
        )
    else:
        unusual_clause = f"None of its {total} transaction characteristics is outside the normal range."

    similar_clause = (
        f"Of the {int(facts['similar_total'])} most similar transactions the system learned from, "
        f"{int(facts['similar_fraud'])} were confirmed fraud "
        f"(the overall rate is {float(facts['base_rate']):.2%})."
    )
    concentration_clause = f"Three factors account for {explanation.concentration(3):.0%} of the decision."

    clauses = [amount_clause, cycle_clause, unusual_clause, similar_clause, concentration_clause]

    clearing = facts.get("clearing_amount")
    if clearing is not None:
        # Unknown counts as not shown: the range is claimed only when the sweep said so.
        if facts.get("clearing_is_monotone") is True:
            clauses.append(
                f"Had the amount been below {float(clearing):,.2f}, with everything else unchanged, "
                f"it would have been cleared."
            )
        else:
            # The sweep found lower amounts the model still flags: claim the
            # amount that was tested, not a range it never was.
            clauses.append(
                f"At {float(clearing):,.2f}, with everything else unchanged, it would have been cleared; "
                f"some lower amounts would still be flagged."
            )
    if driver:
        clauses.append(driver)
    return " ".join(clauses)
