"""Multi-Seed and Multi-Timeline Leakage Verification Test Suite.

Verifies that the Prism Generator synthesis engine produces zero label, shortcut, or
temporal lookahead leakage across diverse random seeds and calendar horizons.

Grounded in spec/07_export_leakage_gate.yaml:
1. Exported views strictly exclude target labels and causal ground truth.
2. No observable *input* column acts as a deterministic shortcut (single-column ROC-AUC <= 0.95).
   The issuer's own response codes (response_code, auth_response_code) are the decision system's
   output, not an input a detector can consume; they stay in the export and in the learner's
   feature matrix, where the learner ceiling covers them. See spec/07's decision_columns_* key.
3. No high-support categorical attribute is fraud-only (support >= 20).
4. Strictly causal point-in-time features with zero lookahead into future events.
5. Model PR-AUC bounded between 0.15 (learnability floor) and 0.97 (anti-shortcut ceiling).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from prism_generator import SimulationEngine
from tests.test_synthesizer.test_export_leakage_gate import (
    audited_single_column_scores,
    build_matrix,
    export_and_join,
    fraud_only_values,
    single_column_scores,
)

BANNED_EXPORT_COLUMNS = (
    "is_fraud",
    "scenario_tag",
    "syndicate_id",
    "risk_score",
    "base_risk",
    "dominant_causal_driver",
    "analytical_shapley_probability",
    "analytical_shapley_log_odds",
    "counterfactual_input_deltas",
    "counterfactual_mode",
    "counterfactual_twin",
    "normative_baseline",
    "explanation_narrative",
)


@pytest.mark.parametrize("seed", [42, 101, 777, 2024, 9999])
@pytest.mark.parametrize("region", ["US", "IN"])
def test_leakage_across_seeds_fixed_timeline(seed: int, region: str) -> None:
    """Verifies that varied random seeds do not introduce label shortcuts or single-column leakage."""
    engine = SimulationEngine(
        n_cards=200,
        n_merchants=60,
        region=region,
        adversary_mode="intent",
        seed=seed,
    )
    records = engine.generate_batch(
        n_transactions=1500,
        time_span_days=14,
        fraud_prevalence=0.04,
    )
    assert len(records) == 1500

    label_of = {r["transaction_id"]: int(r["is_fraud"]) for r in records}
    with tempfile.TemporaryDirectory() as tmp_dir:
        rows = export_and_join(records, Path(tmp_dir))
        assert len(rows) == len(records)

        # 1. Gate Check: Banned ground-truth columns must never appear in exported views
        for banned in BANNED_EXPORT_COLUMNS:
            assert banned not in rows[0], f"Seed {seed} ({region}): {banned} leaked into exported view"

        labels = np.array([label_of[r["transaction_id"]] for r in rows], dtype=int)
        assert labels.sum() >= 15, f"Seed {seed} ({region}): insufficient fraud rows to evaluate"

        # 2. Gate Check: No categorical value with support >= 20 is fraud-only
        leaks = fraud_only_values(rows, labels)
        filtered_leaks = [lk for lk in leaks if lk[2] >= 20]
        assert not filtered_leaks, f"Seed {seed} ({region}): deterministic fraud-only categories: {filtered_leaks}"

        # 3. Gate Check: Single-column ROC-AUC ceiling <= 0.95
        X, is_categorical, names, y, cut = build_matrix(rows, labels)
        assert y[cut:].sum() >= 5, f"Seed {seed} ({region}): test holdout lacks sufficient fraud rows"

        ranked = single_column_scores(X, is_categorical, names, y, cut)
        top_auc, top_col = audited_single_column_scores(ranked)[0]
        assert top_auc <= 0.95, (
            f"Seed {seed} ({region}): single input column {top_col} leaked ground truth with "
            f"ROC-AUC {top_auc:.4f} > 0.95"
        )

        # 4. Gate Check: Supervised model PR-AUC ceiling <= 0.97 and floor >= 0.15
        clf = HistGradientBoostingClassifier(max_iter=50, random_state=42)
        clf.fit(X[:cut], y[:cut])
        probs = clf.predict_proba(X[cut:])[:, 1]
        pr_auc = float(average_precision_score(y[cut:], probs))

        assert pr_auc <= 0.97, (
            f"Seed {seed} ({region}): model PR-AUC {pr_auc:.4f} exceeds 0.97 anti-shortcut ceiling"
        )
        assert pr_auc >= 0.15, (
            f"Seed {seed} ({region}): model PR-AUC {pr_auc:.4f} is below 0.15 learnability floor"
        )


@pytest.mark.parametrize("time_span_days", [3, 7, 21, 45])
@pytest.mark.parametrize("adversary_mode", ["intent", "playbook"])
def test_leakage_across_timelines_fixed_seed(time_span_days: int, adversary_mode: str) -> None:
    """Verifies that short, medium, and multi-cycle monthly timelines preserve zero-leakage invariants."""
    engine = SimulationEngine(
        n_cards=250,
        n_merchants=70,
        region="IN",
        adversary_mode=adversary_mode,
        seed=101,
    )
    records = engine.generate_batch(
        n_transactions=1800,
        time_span_days=time_span_days,
        fraud_prevalence=0.045,
    )
    assert len(records) == 1800

    # Verify temporal monotonicity across the timeline
    times = [float(r["tx_time_seconds"]) for r in records]
    assert all(times[i] <= times[i + 1] for i in range(len(times) - 1)), (
        f"Timeline {time_span_days}d ({adversary_mode}): temporal monotonicity violated"
    )

    label_of = {r["transaction_id"]: int(r["is_fraud"]) for r in records}
    with tempfile.TemporaryDirectory() as tmp_dir:
        rows = export_and_join(records, Path(tmp_dir))
        labels = np.array([label_of[r["transaction_id"]] for r in rows], dtype=int)

        X, is_categorical, names, y, cut = build_matrix(rows, labels)
        ranked = single_column_scores(X, is_categorical, names, y, cut)
        top_auc, top_col = audited_single_column_scores(ranked)[0]

        assert top_auc <= 0.95, (
            f"Timeline {time_span_days}d ({adversary_mode}): input column {top_col} ROC-AUC "
            f"{top_auc:.4f} exceeds 0.95"
        )


@pytest.mark.slow
@pytest.mark.parametrize("days", [7, 14])
@pytest.mark.parametrize("seed", [42, 303, 808])
def test_leakage_in_days_simulation_mode_across_seeds(days: int, seed: int) -> None:
    """Verifies that unconstrained Hawkes calendar horizon mode does not introduce arrival rate leakage."""
    cards_count = max(100, min(500, days * 35))
    merchants_count = max(40, cards_count // 4)
    engine = SimulationEngine(
        n_cards=cards_count,
        n_merchants=merchants_count,
        region="US",
        adversary_mode="intent",
        seed=seed,
    )
    records = engine.generate_batch(
        simulation_mode="days",
        time_span_days=days,
        fraud_prevalence=0.04,
        pace_to_sample_budget=False,
    )
    assert len(records) >= 500, f"Days mode {days}d (seed {seed}): expected >= 500 emergent transactions"

    # Verify calendar horizon span
    times = [float(r["tx_time_seconds"]) for r in records]
    horizon_days = (max(times) - min(times)) / 86400.0
    assert abs(horizon_days - days) <= 1.0, f"Simulated calendar horizon {horizon_days:.2f}d deviates from {days}d"

    label_of = {r["transaction_id"]: int(r["is_fraud"]) for r in records}
    with tempfile.TemporaryDirectory() as tmp_dir:
        rows = export_and_join(records, Path(tmp_dir))
        labels = np.array([label_of[r["transaction_id"]] for r in rows], dtype=int)

        # No deterministic single-column separator
        X, is_categorical, names, y, cut = build_matrix(rows, labels)
        ranked = single_column_scores(X, is_categorical, names, y, cut)
        top_auc, top_col = audited_single_column_scores(ranked)[0]
        assert top_auc <= 0.95, (
            f"Days mode {days}d (seed {seed}): input column {top_col} ROC-AUC {top_auc:.4f} exceeds 0.95"
        )


def test_no_lookahead_temporal_leakage_across_seeds_and_timelines() -> None:
    """Verifies that point-in-time velocity and counter features never access future transactions."""
    for test_seed in [42, 999]:
        engine = SimulationEngine(n_cards=150, n_merchants=40, region="IN", seed=test_seed)
        records = engine.generate_batch(n_transactions=1000, time_span_days=14, fraud_prevalence=0.05)

        card_auth_history: Dict[str, List[float]] = {}
        card_attempt_history: Dict[str, List[float]] = {}
        last_global_time = -1.0

        for r in records:
            cid = r["card_id"]
            tx_t = float(r["tx_time_seconds"])
            assert tx_t >= last_global_time, f"Seed {test_seed}: global time reversal detected {tx_t} < {last_global_time}"
            last_global_time = tx_t

            count_1h = int(r.get("tx_count_1h", 0))
            attempts_1h = int(r.get("tx_attempts_1h", 0))

            prior_auths_1h = sum(1 for t in card_auth_history.get(cid, []) if tx_t - 3600.0 < t < tx_t)
            prior_attempts_1h = sum(1 for t in card_attempt_history.get(cid, []) if tx_t - 3600.0 < t < tx_t)

            assert count_1h == prior_auths_1h, (
                f"Seed {test_seed}: tx_count_1h discrepancy on card {cid}: "
                f"feature={count_1h}, actual strictly prior authorized={prior_auths_1h}"
            )
            assert attempts_1h == prior_attempts_1h, (
                f"Seed {test_seed}: tx_attempts_1h discrepancy on card {cid}: "
                f"feature={attempts_1h}, actual strictly prior attempts={prior_attempts_1h}"
            )

            # Update histories post-authorization
            card_attempt_history.setdefault(cid, []).append(tx_t)
            if r.get("response_code") in ("00", "10"):
                card_auth_history.setdefault(cid, []).append(tx_t)
