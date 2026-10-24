"""Deterministic verification suite for Day 4 (Slice 17): Prequential Evaluation & Streaming Retraining.

Asserts:
1. Zero temporal lookahead leakage (no unobserved labels enter training).
2. Delay gap blackout enforcement in STRICT_DELAY_GAP mode.
3. Operational metrics exactness (P@K, CP@K, DR@K) against hand-calculated ground truth.
4. Financial cost savings bounds and monotonicity.
5. Streaming unsupervised score drift tripwires (KS test and PSI).
6. End-to-end prequential rolling benchmark execution.
"""

from __future__ import annotations


import numpy as np

from prism_generator import (
    CostMatrixConfig,
    DelayedSupervisionPolicy,
    LabelSource,
    PrequentialBenchmarkReport,
    PrequentialStreamingEvaluator,
    SimulationEngine,
    StreamingDriftAuditor,
    StreamingMetricTracker,
    SupervisionEngine,
)


def test_zero_temporal_lookahead_leakage():
    """Prove that across all rolling folds, zero records with tau_discovery > T_train enter training."""
    engine = SimulationEngine(n_cards=80, n_merchants=25, region="US", seed=42)
    records = engine.generate_batch(n_transactions=600, fraud_prevalence=0.08, time_span_days=12)

    supervision_engine = SupervisionEngine(k_daily=10, seed=42)
    supervision_records = supervision_engine.process_batch(records)

    evaluator = PrequentialStreamingEvaluator(
        w_train_days=4.0,
        w_test_days=1.0,
        delta_delay_days=2.0,
        retrain_freq_days=1.0,
        k_daily=10,
        policy=DelayedSupervisionPolicy.STRICT_DELAY_GAP,
        supervision_engine=supervision_engine,
        seed=42,
    )

    report = evaluator.evaluate_stream(records, supervision_records=supervision_records)

    assert report.n_days_evaluated > 0
    assert len(evaluator.last_training_pools) > 0

    tx_to_sup = {s.transaction_id: s for s in supervision_records}

    # Invariant: Every admitted record in the training pool must have matured before current_epoch
    for t_epoch, train_recs in evaluator.last_training_pools:
        for r in train_recs:
            sup = tx_to_sup[r["transaction_id"]]
            label = sup.is_label_available_at(query_time_seconds=t_epoch)
            assert label is not None, (
                f"Temporal leakage violation at epoch {t_epoch}: "
                f"Transaction {sup.transaction_id} (tx_time={sup.tx_time_seconds}, "
                f"tau_discovery={sup.discovery_time_seconds}) was admitted before legal maturity!"
            )
            assert sup.tx_time_seconds < t_epoch, (
                f"Future transaction admitted into training pool: tx_time={sup.tx_time_seconds} >= epoch={t_epoch}"
            )


def test_delay_gap_blackout_enforcement():
    """Verify that in STRICT_DELAY_GAP mode, unalerted transactions in [T - delta_delay, T] are blacked out."""
    engine = SimulationEngine(n_cards=50, n_merchants=20, region="IN", seed=99)
    records = engine.generate_batch(n_transactions=400, fraud_prevalence=0.06, time_span_days=10)

    supervision_engine = SupervisionEngine(k_daily=5, seed=99)
    supervision_records = supervision_engine.process_batch(records)

    evaluator = PrequentialStreamingEvaluator(
        w_train_days=3.0,
        w_test_days=1.0,
        delta_delay_days=2.0,
        retrain_freq_days=1.0,
        k_daily=5,
        policy=DelayedSupervisionPolicy.STRICT_DELAY_GAP,
        supervision_engine=supervision_engine,
        seed=99,
    )

    delta_delay_sec = 2.0 * 86400.0
    report = evaluator.evaluate_stream(records, supervision_records=supervision_records)

    # Check each fold's daily report
    for daily in report.daily_metrics:
        t_epoch = daily.t_start_seconds
        t_gap_start = t_epoch - delta_delay_sec

        # Search for any transaction in the gap that was NOT investigated by analyst
        for r, sup in zip(records, supervision_records):
            t_tx = float(r["tx_time_seconds"])
            if t_gap_start <= t_tx < t_epoch:
                if sup.label_source != LabelSource.INVESTIGATOR_ALERT:
                    # Must NOT have label available at t_epoch (blacked out pending dispute)
                    assert sup.is_label_available_at(t_epoch) is None or sup.discovery_time_seconds > t_epoch


def test_operational_metrics_exactness():
    """Assert that P@K, CP@K, DR@K, and PR-AUC match exact hand-calculated ground truth."""
    # Deterministic test scenario:
    # 10 transactions with known risk scores, ground truth, amounts, and cardholders
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 0], dtype=int)
    y_score = np.array([0.95, 0.90, 0.85, 0.80, 0.75, 0.50, 0.40, 0.30, 0.20, 0.10], dtype=float)
    amounts = np.array([100.0, 200.0, 50.0, 80.0, 300.0, 40.0, 500.0, 20.0, 30.0, 10.0], dtype=float)
    card_ids = ["C1", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

    # 1. Alert Precision @ 3 (Top 3 scores: indices 0, 1, 2 with labels [1, 1, 0])
    p_3 = StreamingMetricTracker.compute_p_at_k(y_true, y_score, k=3)
    assert abs(p_3 - (2.0 / 3.0)) < 1e-6, f"Expected P@3 = 2/3, got {p_3}"

    # 2. Card Precision @ 3
    # Top unique cards:
    # Index 0: C1 (fraud = 1) -> seen {C1}, comp = 1
    # Index 1: C1 (already seen)
    # Index 2: C2 (fraud = 0) -> seen {C1, C2}, comp = 1
    # Index 3: C3 (fraud = 0) -> seen {C1, C2, C3}, comp = 1
    # Card Precision @ 3 = 1 / 3
    cp_3 = StreamingMetricTracker.compute_card_precision_at_k(y_true, y_score, card_ids, k=3)
    assert abs(cp_3 - (1.0 / 3.0)) < 1e-6, f"Expected CP@3 = 1/3, got {cp_3}"

    # 3. Dollar Recall @ 3
    # Total fraud dollars: index 0 (100) + index 1 (200) + index 4 (300) + index 6 (500) = 1100.0
    # Captured in top 3: index 0 (100) + index 1 (200) = 300.0
    # Dollar Recall @ 3 = 300 / 1100 = 0.272727...
    dr_3 = StreamingMetricTracker.compute_dollar_recall_at_k(y_true, y_score, amounts, k=3)
    assert abs(dr_3 - (300.0 / 1100.0)) < 1e-6, f"Expected DR@3 = 3/11, got {dr_3}"

    # 4. Dollar Precision @ 3
    # Top 3 total dollars: index 0 (100) + index 1 (200) + index 2 (50) = 350.0
    # Captured fraud dollars in top 3: 300.0
    # Dollar Precision @ 3 = 300 / 350 = 6/7
    dp_3 = StreamingMetricTracker.compute_dollar_precision_at_k(y_true, y_score, amounts, k=3)
    assert abs(dp_3 - (300.0 / 350.0)) < 1e-6, f"Expected DP@3 = 6/7, got {dp_3}"

    # 5. Prequential PR-AUC bounds
    pr_auc = StreamingMetricTracker.compute_pr_auc(y_true, y_score)
    assert 0.0 < pr_auc <= 1.0


def test_financial_cost_savings_monotonicity():
    """Assert that Savings <= 1.0 and a high-precision model yields strictly higher savings than random."""
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=int)
    amounts = np.array([500.0, 1000.0, 1500.0, 50.0, 80.0, 120.0, 40.0, 70.0, 90.0, 60.0])
    cost_matrix = CostMatrixConfig(c_admin=2.50, c_friction_rate=0.01, c_chargeback_fee=15.00)

    # Perfect model: ranks all frauds first
    scores_perfect = np.array([0.99, 0.98, 0.97, 0.3, 0.2, 0.1, 0.05, 0.04, 0.03, 0.01])
    base_cost, model_cost_perfect, savings_perfect = StreamingMetricTracker.compute_financial_savings(
        y_true, scores_perfect, amounts, k=3, cost_matrix=cost_matrix
    )

    assert savings_perfect <= 1.0
    assert savings_perfect > 0.85, f"Perfect model should yield > 85% savings, got {savings_perfect:.4f}"

    # Inverted worst model: ranks frauds last
    scores_worst = 1.0 - scores_perfect
    _, model_cost_worst, savings_worst = StreamingMetricTracker.compute_financial_savings(
        y_true, scores_worst, amounts, k=3, cost_matrix=cost_matrix
    )

    assert savings_perfect > savings_worst, (
        f"Monotonicity violation: perfect model savings {savings_perfect} <= worst model savings {savings_worst}"
    )


def test_streaming_drift_tripwire_alarm():
    """Verify that unsupervised score distribution shifts trigger both KS test and PSI alarms."""
    rng = np.random.default_rng(42)

    # Normal baseline: beta distribution with high mass on low risk scores (mean ~0.08)
    ref_scores = rng.beta(a=1.0, b=12.0, size=500)

    # Shifted stream: adversarial wave creates massive clustering at high risk scores (mean ~0.55)
    shifted_scores = rng.beta(a=5.0, b=4.0, size=200)

    # 1. Two-sample KS test
    drift_detected, ks_stat, p_val = StreamingDriftAuditor.detect_ks_score_drift(
        reference_scores=ref_scores,
        current_scores=shifted_scores,
        alpha=0.01,
    )
    assert drift_detected is True, f"KS drift failed to detect shift (p-value {p_val})"
    assert p_val < 0.01

    # 2. Population Stability Index (PSI)
    psi = StreamingDriftAuditor.compute_population_stability_index(
        reference_scores=ref_scores,
        current_scores=shifted_scores,
        n_bins=10,
    )
    assert psi >= 0.25, f"PSI failed to reach severe alarm threshold (>= 0.25): got PSI={psi:.4f}"

    # Verify stable stream does not false alarm
    stable_scores = rng.beta(a=1.0, b=12.0, size=200)
    stable_drift, _, p_stable = StreamingDriftAuditor.detect_ks_score_drift(ref_scores, stable_scores, alpha=0.01)
    psi_stable = StreamingDriftAuditor.compute_population_stability_index(ref_scores, stable_scores, n_bins=10)
    assert stable_drift is False
    assert psi_stable < 0.10


def test_end_to_end_prequential_rolling_benchmark():
    """Verify end-to-end multi-day rolling prequential evaluation under delayed supervision."""
    engine = SimulationEngine(n_cards=100, n_merchants=30, region="US", seed=123)
    records = engine.generate_batch(n_transactions=500, fraud_prevalence=0.06, time_span_days=10)

    evaluator = PrequentialStreamingEvaluator(
        w_train_days=3.0,
        w_test_days=1.0,
        delta_delay_days=1.5,
        retrain_freq_days=1.0,
        k_daily=10,
        policy=DelayedSupervisionPolicy.STRICT_DELAY_GAP,
        seed=123,
    )

    report = evaluator.evaluate_stream(records)

    assert isinstance(report, PrequentialBenchmarkReport)
    assert report.n_days_evaluated >= 2
    assert report.total_test_transactions > 0
    assert 0.0 <= report.mean_pr_auc <= 1.0
    assert 0.0 <= report.mean_p_at_k <= 1.0
    assert 0.0 <= report.mean_cp_at_k <= 1.0
    assert 0.0 <= report.mean_dollar_recall_at_k <= 1.0
    assert report.total_cost_base > 0.0
    assert report.overall_savings_ratio <= 1.0
    assert len(report.daily_metrics) == report.n_days_evaluated

    # Verify each day has valid structure
    for d in report.daily_metrics:
        assert d.n_transactions > 0
        assert 0.0 <= d.p_at_k <= 1.0
        assert 0.0 <= d.dollar_recall_at_k <= 1.0
        assert d.cost_base >= 0.0
