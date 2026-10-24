"""Tests verifying numerical stability and lognormal priors in Welford accumulator."""

from prism_generator import SimulationEngine


def test_welford_cold_start_prior_variance():
    """Welford accumulator with theoretical lognormal prior must have non-zero variance."""
    engine = SimulationEngine(n_cards=10, n_merchants=20, seed=42)
    card = engine.cards[0]
    state = engine.ledger.get_or_create_state(card)

    # Initial variance must be strictly positive and reflective of theoretical distribution
    assert state.welford_30d.variance > 0.0
    assert state.welford_30d.std > 0.0

    # A standard purchase near the mean must have a bounded z-score in [-2.5, +2.5]
    typical_amount = card.get_theoretical_mean_spend()
    z = state.welford_30d.compute_z_score(typical_amount)
    assert abs(z) < 2.5, f"Initial z-score on typical purchase was excessively high: {z}"


def test_welford_first_ten_transactions_bounded():
    """First 10 transactions of all cardholders must not generate astronomical z-scores (e.g. +250)."""
    engine = SimulationEngine(n_cards=50, n_merchants=30, seed=42)
    records = engine.generate_batch(n_transactions=500, fraud_prevalence=0.0)

    for r in records:
        z = abs(float(r["z_score_amount_30d"]))
        assert z < 10.0, f"Normal transaction generated unrealistic z-score: {z} for amount ${r['amount']}"
