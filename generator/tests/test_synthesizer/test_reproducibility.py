"""Tests verifying deterministic bit-for-bit reproducibility of simulation runs."""

from prism_generator import SimulationEngine


def test_batch_reproducibility_identical_seeds():
    """Two simulation engines instantiated with identical seeds must yield identical records."""
    engine1 = SimulationEngine(n_cards=100, n_merchants=50, seed=42)
    batch1 = engine1.generate_batch(n_transactions=500, fraud_prevalence=0.04)

    engine2 = SimulationEngine(n_cards=100, n_merchants=50, seed=42)
    batch2 = engine2.generate_batch(n_transactions=500, fraud_prevalence=0.04)

    assert len(batch1) == len(batch2)

    for i in range(len(batch1)):
        r1, r2 = batch1[i], batch2[i]
        assert r1["transaction_id"] == r2["transaction_id"]
        assert r1["card_id"] == r2["card_id"]
        assert r1["merchant_id"] == r2["merchant_id"]
        assert r1["amount"] == r2["amount"]
        assert r1["tx_time_seconds"] == r2["tx_time_seconds"]
        assert r1["is_fraud"] == r2["is_fraud"]
        assert r1["scenario_tag"] == r2["scenario_tag"]
        assert r1["risk_score"] == r2["risk_score"]
        assert r1["dominant_causal_driver"] == r2["dominant_causal_driver"]
        assert r1["analytical_shapley_probability"] == r2["analytical_shapley_probability"]


def test_streaming_reproducibility_with_deterministic_clock():
    """Streaming with deterministic start time and identical seeds must yield identical streams."""
    engine1 = SimulationEngine(n_cards=50, n_merchants=25, seed=999)
    stream1 = list(engine1.stream_continuous(duration_seconds=10.0, target_tps=5.0, start_time_seconds=1000.0))

    engine2 = SimulationEngine(n_cards=50, n_merchants=25, seed=999)
    stream2 = list(engine2.stream_continuous(duration_seconds=10.0, target_tps=5.0, start_time_seconds=1000.0))

    assert len(stream1) == len(stream2)
    for i in range(len(stream1)):
        assert stream1[i]["transaction_id"] == stream2[i]["transaction_id"]
        assert stream1[i]["amount"] == stream2[i]["amount"]
        assert stream1[i]["risk_score"] == stream2[i]["risk_score"]


def test_streaming_real_time_pacing():
    """Streaming with real_time=True paces emissions according to inter-arrival delta."""
    import itertools
    import time

    engine = SimulationEngine(n_cards=20, n_merchants=10, seed=42)
    t0 = time.perf_counter()
    # Pull 3 transactions with target_tps=50.0 (inter-arrival ~0.02s)
    stream = list(
        itertools.islice(
            engine.stream_continuous(
                duration_seconds=1.0,
                target_tps=50.0,
                start_time_seconds=0.0,
                real_time=True,
            ),
            3,
        )
    )
    elapsed = time.perf_counter() - t0
    assert len(stream) == 3
    # Two inter-arrival intervals of 1.0 / 50.0 = 0.02s each -> elapsed >= 0.03s
    assert elapsed >= 0.03
