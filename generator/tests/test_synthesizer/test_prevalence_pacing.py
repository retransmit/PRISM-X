"""spec/07 section 19: the competing fraud stream follows the realised legitimate rate.

generate_batch paced the fraud stream to the planned span, but the legitimate stream runs
ahead of the plan (self-excitation and shopping trips), so a 1,000-row batch asked for
6 percent fraud delivered 3.3 percent and stopped on day 17 of 30.
"""

from __future__ import annotations

import pytest

from prism_generator import SimulationEngine


@pytest.mark.parametrize("n, cards, prevalence, seed", [
    (1000, 200, 0.06, 123),
    (1500, 150, 0.03, 7),
    (2500, 400, 0.10, 42),
])
def test_small_batches_deliver_the_requested_prevalence(n, cards, prevalence, seed):
    engine = SimulationEngine(n_cards=cards, n_merchants=60, seed=seed)
    records = engine.generate_batch(n_transactions=n, fraud_prevalence=prevalence, time_span_days=30)
    share = sum(int(r["is_fraud"]) for r in records) / len(records)
    assert abs(share - prevalence) / prevalence <= 0.25, f"{share:.4f} against {prevalence}"


def test_the_fraud_stream_stays_a_competing_stream():
    """Pacing by the realised rate must not turn attacks into a function of the victim's own
    activity: an attack lands on any active card, not only on cards that just paid."""
    engine = SimulationEngine(n_cards=200, n_merchants=60, seed=9)
    records = engine.generate_batch(n_transactions=2000, fraud_prevalence=0.08, time_span_days=30)
    previous = None
    same_card_as_previous = 0
    fraud = 0
    for r in records:
        if r["is_fraud"] == 1:
            fraud += 1
            if previous is not None and previous["card_id"] == r["card_id"] and previous["is_fraud"] == 0:
                same_card_as_previous += 1
        previous = r
    assert fraud >= 100
    assert same_card_as_previous / fraud <= 0.10
