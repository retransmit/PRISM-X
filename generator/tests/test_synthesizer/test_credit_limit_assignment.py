"""spec/01 and spec/05 credit_limit_assignment: issuers assign lines in steps.

A continuous triangular draw gave every card its own limit, so credit_limit was a card
identifier in the export (1,000 distinct values on 1,000 cards).
"""

from __future__ import annotations

from collections import Counter

import pytest

from prism_generator import SimulationEngine
from prism_generator.spec_loader import load_all_specs


def _step_for(limit: float, steps) -> float:
    for up_to, step in steps:
        if up_to is None or limit <= up_to + 1e-6:
            return step
    return steps[-1][1]


@pytest.mark.parametrize("region", ["IN", "US"])
def test_credit_limits_fall_on_issuer_steps(region):
    steps = load_all_specs().credit_limit_steps[region]
    assert steps and steps[-1][0] is None, "the last tier must be open-ended"
    engine = SimulationEngine(n_cards=1000, n_merchants=50, region=region, seed=42)
    limits = [card.credit_limit for card in engine.cards]
    support = Counter(limits)
    # a limit is not a card identifier: few cards carry a value no other card carries
    assert len(support) <= 0.15 * len(limits), f"{len(support)} distinct limits on {len(limits)} cards"
    singletons = sum(1 for limit in limits if support[limit] == 1)
    assert singletons <= 0.05 * len(limits), f"{singletons} of {len(limits)} cards carry a limit of their own"
    for limit in limits:
        step = _step_for(limit, steps)
        assert abs(limit / step - round(limit / step)) < 1e-6, (limit, step)
        assert limit > 0.0


@pytest.mark.parametrize("region", ["IN", "US"])
def test_rounded_limits_stay_inside_the_product_range(region):
    specs = load_all_specs()
    engine = SimulationEngine(n_cards=600, n_merchants=50, region=region, seed=3)
    for card in engine.cards:
        product = specs.get_product(card.product_id)
        if region == "IN":
            low, high = product.credit_limit_min_inr, product.credit_limit_max_inr
        else:
            low, high = product.credit_limit_min_usd, product.credit_limit_max_usd
        if high <= 0.0:
            continue
        step = _step_for(card.credit_limit, specs.credit_limit_steps[region])
        assert low - step <= card.credit_limit <= high + step, (card.product_id, card.credit_limit, low, high)
