"""Empirical XAI and Tripartite Industrial Benchmark Suite for Transaction Fraud Detection.

Evaluates synthetic payment streams across three industrial quality dimensions plus causal XAI:
1. Dimension 1: Statistical Fidelity (Wasserstein-1, Jensen-Shannon divergence, Spearman Frobenius norm)
2. Dimension 2: Machine Learning Utility (TSTR: Train on Synthetic, Test on Real / Reference)
3. Dimension 3: Adversarial Privacy & Non-Memorization (DCR 5th percentile, NNDR, MIA resistance)
4. Dimension 4: Causal XAI Concordance (Quantus & OpenXAI protocols: Kendall tau, Spearman rho, Directional Cosine, AUDC)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.stats

from .causal_contract import (
    feature_names,
    feature_vector,
    ground_truth_attributions,
    intervention_support,
)
from .engine import DiscreteEventEngine
from .evaluation import GroundTruthXAIEvaluator


# The eleven explained columns and the per-playbook intervention supports live in
# spec/21 and are read through prism_generator/causal_contract.py; benchmark.py and
# benchmark_reporter.py used to declare them independently and had drifted apart.
#
# The contract is read lazily (never at import time), so importing this module does not
# require spec/ to be locatable; `causal_contract.feature_names()` returns the columns.


# =====================================================================
# DATACLASS SUMMARIES FOR ALL THREE INDUSTRIAL DIMENSIONS
# =====================================================================

@dataclass
class StatisticalFidelitySummary:
    """Summary metrics for Dimension 1: Statistical Fidelity."""
    wasserstein_amount_log: float
    wasserstein_arrival_log: float
    js_divergence_mcc: float
    js_divergence_channel: float
    js_divergence_response_code: float
    spearman_frobenius_error: float
    pearson_frobenius_error: float
    amount_passed: bool
    arrival_passed: bool
    mcc_passed: bool
    channel_passed: bool
    correlation_passed: bool
    fidelity_passed: bool


@dataclass
class MLUtilitySummary:
    """Summary metrics for Dimension 2: Machine Learning Utility (TSTR Protocol)."""
    tstr_pr_auc: float
    trtr_pr_auc: float
    relative_pr_auc_retention: float
    tstr_roc_auc: float
    trtr_roc_auc: float
    relative_roc_auc_retention: float
    tstr_f1_score: float
    trtr_f1_score: float
    trts_pr_auc: float
    utility_passed: bool
    precision_at_k: float = 0.0
    cost_savings: float = 0.0
    temporal_split_fraction: float = 0.60


@dataclass
class AdversarialPrivacySummary:
    """Summary metrics for Dimension 3: Adversarial Privacy & Non-Memorization."""
    dcr_5th_percentile: float
    dcr_reference_5th_percentile: float
    dcr_reference_ratio: float
    dcr_median: float
    dcr_min: float
    nndr_mean: float
    nndr_median: float
    mia_attack_roc_auc: float
    dcr_passed: bool
    nndr_passed: bool
    mia_passed: bool
    privacy_passed: bool


@dataclass
class ModelBenchmarkSummary:
    """Aggregated benchmark metrics for a single model and explainer combination."""
    model_name: str
    explainer_name: str
    n_evaluated_samples: int
    mean_kendall_tau: float
    mean_spearman_rho: float
    mean_cosine_similarity: float
    mean_precision_at_3: float
    mean_relative_attribution_error: float
    auc_roc: float
    pr_auc: float
    mean_intervention_precision_at_3: float = 0.0
    mean_intervention_recall_at_3: float = 0.0
    interventional_kendall_tau: float = 0.0
    scorer_kendall_tau: float = 0.0
    anti_leak_tripwire_passed: bool = True
    mean_normalized_l1_distance: float = 0.0
    mean_normalized_l2_distance: float = 0.0
    metrics_by_k: Dict[str, float] = field(default_factory=dict)


@dataclass
class TripartiteBenchmarkSummary:
    """Consolidated industrial benchmark certification summary."""
    fidelity: StatisticalFidelitySummary
    utility: MLUtilitySummary
    privacy: AdversarialPrivacySummary
    xai_summary: Optional[ModelBenchmarkSummary]
    n_synthetic_samples: int
    n_reference_samples: int
    region: str
    all_passed: bool


# =====================================================================
# DIMENSION 1: STATISTICAL FIDELITY EVALUATOR
# =====================================================================

class StatisticalFidelityEvaluator:
    """Evaluates multi-dimensional empirical geometry fidelity between synthetic and reference streams."""

    @staticmethod
    def compute_wasserstein_1d(u: np.ndarray, v: np.ndarray, log_scale: bool = True) -> float:
        """Computes Earth Mover's Distance (Wasserstein-1) with optional log10 scaling for heavy tails."""
        if len(u) == 0 or len(v) == 0:
            return 0.0
        if log_scale:
            u_eval = np.log10(np.maximum(0.0, u) + 1.0)
            v_eval = np.log10(np.maximum(0.0, v) + 1.0)
        else:
            u_eval = u
            v_eval = v
        return float(scipy.stats.wasserstein_distance(u_eval, v_eval))

    @staticmethod
    def compute_jensen_shannon_discrete(
        u_cat: Sequence[Any],
        v_cat: Sequence[Any],
        alpha: float = 1e-4,
    ) -> float:
        """Computes symmetric, Laplace-smoothed Jensen-Shannon divergence over discrete categories."""
        all_keys = sorted(list(set(u_cat) | set(v_cat)))
        if not all_keys:
            return 0.0

        u_counts = {k: 0 for k in all_keys}
        for k in u_cat:
            u_counts[k] += 1
        v_counts = {k: 0 for k in all_keys}
        for k in v_cat:
            v_counts[k] += 1

        K = len(all_keys)
        u_total = len(u_cat) + alpha * K
        v_total = len(v_cat) + alpha * K

        p = np.array([(u_counts[k] + alpha) / u_total for k in all_keys], dtype=np.float64)
        q = np.array([(v_counts[k] + alpha) / v_total for k in all_keys], dtype=np.float64)

        m = 0.5 * (p + q)
        kl_p = np.sum(p * np.log(p / m))
        kl_q = np.sum(q * np.log(q / m))
        jsd = 0.5 * (kl_p + kl_q)
        return float(max(0.0, jsd))

    @staticmethod
    def compute_correlation_frobenius_error(
        X_syn: np.ndarray,
        X_ref: np.ndarray,
        method: str = "spearman",
    ) -> float:
        """Computes relative Frobenius norm error between feature correlation matrices."""
        if len(X_syn) < 2 or len(X_ref) < 2 or X_syn.shape[1] < 2:
            return 0.0

        with np.errstate(divide="ignore", invalid="ignore"):
            if method == "spearman":
                r_syn = np.apply_along_axis(scipy.stats.rankdata, 0, X_syn)
                r_ref = np.apply_along_axis(scipy.stats.rankdata, 0, X_ref)
                C_syn = np.corrcoef(r_syn, rowvar=False)
                C_ref = np.corrcoef(r_ref, rowvar=False)
            else:
                C_syn = np.corrcoef(X_syn, rowvar=False)
                C_ref = np.corrcoef(X_ref, rowvar=False)

        C_syn = np.nan_to_num(C_syn, nan=0.0)
        C_ref = np.nan_to_num(C_ref, nan=0.0)

        ref_norm = float(np.linalg.norm(C_ref, "fro"))
        diff_norm = float(np.linalg.norm(C_syn - C_ref, "fro"))
        return float(diff_norm / max(1e-6, ref_norm))

    def evaluate(
        self,
        syn_records: List[Dict[str, Any]],
        ref_records: List[Dict[str, Any]],
        X_syn: np.ndarray,
        X_ref: np.ndarray,
    ) -> StatisticalFidelitySummary:
        """Executes full Dimension 1 statistical fidelity audit."""
        # 1. Amounts (log-scaled)
        amounts_syn = np.array([float(r.get("amount", 0.0)) for r in syn_records], dtype=np.float64)
        amounts_ref = np.array([float(r.get("amount", 0.0)) for r in ref_records], dtype=np.float64)
        w1_amt = self.compute_wasserstein_1d(amounts_syn, amounts_ref, log_scale=True)

        # 2. Inter-arrival times (log-scaled)
        times_syn = sorted([float(r.get("tx_time_seconds", 0.0)) for r in syn_records])
        times_ref = sorted([float(r.get("tx_time_seconds", 0.0)) for r in ref_records])
        arr_syn = np.diff(times_syn) if len(times_syn) > 1 else np.array([1.0])
        arr_ref = np.diff(times_ref) if len(times_ref) > 1 else np.array([1.0])
        w1_arr = self.compute_wasserstein_1d(arr_syn, arr_ref, log_scale=True)

        # 3. Categorical distributions
        mcc_syn = [str(r.get("mcc", "0000")) for r in syn_records]
        mcc_ref = [str(r.get("mcc", "0000")) for r in ref_records]
        js_mcc = self.compute_jensen_shannon_discrete(mcc_syn, mcc_ref)

        chan_syn = [str(r.get("channel_type", "CP_POS_EMV_CHIP")) for r in syn_records]
        chan_ref = [str(r.get("channel_type", "CP_POS_EMV_CHIP")) for r in ref_records]
        js_chan = self.compute_jensen_shannon_discrete(chan_syn, chan_ref)

        resp_syn = [str(r.get("response_code", "00")) for r in syn_records]
        resp_ref = [str(r.get("response_code", "00")) for r in ref_records]
        js_resp = self.compute_jensen_shannon_discrete(resp_syn, resp_ref)

        # 4. Correlation matrix Frobenius errors
        frob_spearman = self.compute_correlation_frobenius_error(X_syn, X_ref, method="spearman")
        frob_pearson = self.compute_correlation_frobenius_error(X_syn, X_ref, method="pearson")

        # Grounded Threshold Acceptance Gates
        amt_pass = bool(w1_amt <= 0.050)
        arr_pass = bool(w1_arr <= 0.060)
        mcc_pass = bool(js_mcc <= 0.070)
        chan_pass = bool(js_chan <= 0.035)
        corr_pass = bool(frob_spearman <= 0.160)
        fidelity_pass = bool(amt_pass and arr_pass and mcc_pass and chan_pass and corr_pass)

        return StatisticalFidelitySummary(
            wasserstein_amount_log=w1_amt,
            wasserstein_arrival_log=w1_arr,
            js_divergence_mcc=js_mcc,
            js_divergence_channel=js_chan,
            js_divergence_response_code=js_resp,
            spearman_frobenius_error=frob_spearman,
            pearson_frobenius_error=frob_pearson,
            amount_passed=amt_pass,
            arrival_passed=arr_pass,
            mcc_passed=mcc_pass,
            channel_passed=chan_pass,
            correlation_passed=corr_pass,
            fidelity_passed=fidelity_pass,
        )


# =====================================================================
# DIMENSION 2: MACHINE LEARNING UTILITY (TSTR PROTOCOL)
# =====================================================================

class MLUtilityEvaluator:
    """Evaluates ML downstream utility via Train on Synthetic, Test on Real (TSTR) protocol."""

    @staticmethod
    def evaluate_tstr(
        X_syn: np.ndarray,
        y_syn: np.ndarray,
        X_ref_train: np.ndarray,
        y_ref_train: np.ndarray,
        X_ref_test: np.ndarray,
        y_ref_test: np.ndarray,
        random_state: int = 42,
    ) -> MLUtilitySummary:
        """Trains LightGBM on synthetic data and benchmarks against reference-trained model on held-out test data."""
        try:
            import lightgbm as lgb
            from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
        except ImportError as e:
            raise ImportError("ML utility evaluation requires lightgbm and scikit-learn.") from e

        # Ensure class diversity
        if len(np.unique(y_syn)) < 2 or len(np.unique(y_ref_train)) < 2 or len(np.unique(y_ref_test)) < 2:
            return MLUtilitySummary(
                tstr_pr_auc=0.0,
                trtr_pr_auc=0.0,
                relative_pr_auc_retention=0.0,
                tstr_roc_auc=0.5,
                trtr_roc_auc=0.5,
                relative_roc_auc_retention=0.0,
                tstr_f1_score=0.0,
                trtr_f1_score=0.0,
                trts_pr_auc=0.0,
                utility_passed=False,
            )

        # 1. Train on Synthetic (TSTR)
        clf_syn = lgb.LGBMClassifier(
            n_estimators=60,
            max_depth=4,
            random_state=random_state,
            verbose=-1,
        )
        clf_syn.fit(X_syn, y_syn)
        p_syn_on_ref = clf_syn.predict_proba(X_ref_test)[:, 1]

        # 2. Train on Reference Train (TRTR)
        clf_ref = lgb.LGBMClassifier(
            n_estimators=60,
            max_depth=4,
            random_state=random_state,
            verbose=-1,
        )
        clf_ref.fit(X_ref_train, y_ref_train)
        p_ref_on_ref = clf_ref.predict_proba(X_ref_test)[:, 1]

        # 3. Train on Reference, Test on Synthetic (TRTS)
        p_ref_on_syn = clf_ref.predict_proba(X_syn)[:, 1]

        tstr_pr = float(average_precision_score(y_ref_test, p_syn_on_ref))
        trtr_pr = float(average_precision_score(y_ref_test, p_ref_on_ref))
        rel_pr = float(tstr_pr / max(1e-6, trtr_pr))

        tstr_roc = float(roc_auc_score(y_ref_test, p_syn_on_ref))
        trtr_roc = float(roc_auc_score(y_ref_test, p_ref_on_ref))
        rel_roc = float(tstr_roc / max(1e-6, trtr_roc))

        trts_pr = float(average_precision_score(y_syn, p_ref_on_syn))

        # Best F1-scores
        thresholds = np.linspace(0.1, 0.9, 17)
        tstr_f1 = max(float(f1_score(y_ref_test, (p_syn_on_ref >= t).astype(int), zero_division=0)) for t in thresholds)
        trtr_f1 = max(float(f1_score(y_ref_test, (p_ref_on_ref >= t).astype(int), zero_division=0)) for t in thresholds)

        # Gate: Relative PR-AUC retention >= 85% and non-trivial absolute PR-AUC >= 0.60
        utility_pass = bool(rel_pr >= 0.85 and tstr_pr >= 0.60)

        return MLUtilitySummary(
            tstr_pr_auc=tstr_pr,
            trtr_pr_auc=trtr_pr,
            relative_pr_auc_retention=rel_pr,
            tstr_roc_auc=tstr_roc,
            trtr_roc_auc=trtr_roc,
            relative_roc_auc_retention=rel_roc,
            tstr_f1_score=tstr_f1,
            trtr_f1_score=trtr_f1,
            trts_pr_auc=trts_pr,
            utility_passed=utility_pass,
        )


class TSTRHarness:
    """TSTR (Train on Synthetic, Test on Real) & Temporal Self-TSTR Protocol Harness.

    Provides standardized evaluation of synthetic data downstream utility under
    time-ordered sequential splits with delayed supervision considerations.
    """

    @staticmethod
    def evaluate_tstr(
        X_syn: np.ndarray,
        y_syn: np.ndarray,
        X_ref_train: np.ndarray,
        y_ref_train: np.ndarray,
        X_ref_test: np.ndarray,
        y_ref_test: np.ndarray,
        random_state: int = 42,
    ) -> MLUtilitySummary:
        return MLUtilityEvaluator.evaluate_tstr(
            X_syn=X_syn,
            y_syn=y_syn,
            X_ref_train=X_ref_train,
            y_ref_train=y_ref_train,
            X_ref_test=X_ref_test,
            y_ref_test=y_ref_test,
            random_state=random_state,
        )

    @staticmethod
    def evaluate_temporal_self_tstr(
        records: List[Dict[str, Any]],
        train_fraction: float = 0.60,
        k_daily: int = 50,
        random_state: int = 42,
    ) -> MLUtilitySummary:
        """Executes the 60/40 time-ordered self-TSTR protocol on a generated transaction stream."""
        if not records:
            raise ValueError("records list cannot be empty")

        # 1. Sort strictly by timestamp to ensure zero lookahead
        order = np.argsort([float(r.get("tx_time_seconds", 0.0)) for r in records], kind="stable")
        sorted_records = [records[i] for i in order]

        n = len(sorted_records)
        cut = int(n * train_fraction)
        train_records = sorted_records[:cut]
        test_records = sorted_records[cut:]

        def _extract(recs: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
            X, y = [], []
            for r in recs:
                ch = str(r.get("channel_type", "CNP_WEB"))
                asn = str(r.get("asn_type", "residential"))
                row = [
                    float(r.get("amount", 0.0)),
                    float(r.get("tx_count_1h", 0)),
                    float(r.get("tx_count_24h", 0)),
                    float(r.get("haversine_velocity_kph", 0.0)),
                    float(r.get("ip_distance_from_home_km", 0.0)),
                    float(r.get("vaai_score", 50)),
                    1.0 if ch.startswith("CP") else 0.0,
                    1.0 if "MOBILE" in ch else 0.0,
                    1.0 if asn == "residential" else 0.0,
                    1.0 if r.get("is_cross_border") else 0.0,
                    float(r.get("cvv_match_flag", 1)),
                ]
                X.append(row)
                y.append(int(r.get("is_fraud", 0)))
            return np.array(X, dtype=np.float64), np.array(y, dtype=np.int32)

        X_train, y_train = _extract(train_records)
        X_test, y_test = _extract(test_records)

        # Downstream learner (LightGBM if available, else HistGradientBoostingClassifier)
        try:
            import lightgbm as lgb
            from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
            clf = lgb.LGBMClassifier(n_estimators=50, max_depth=4, random_state=random_state, verbose=-1)
        except (ImportError, AttributeError):
            from sklearn.ensemble import HistGradientBoostingClassifier
            from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
            clf = HistGradientBoostingClassifier(max_iter=50, random_state=random_state)

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            return MLUtilitySummary(
                tstr_pr_auc=0.0,
                trtr_pr_auc=0.0,
                relative_pr_auc_retention=0.0,
                tstr_roc_auc=0.5,
                trtr_roc_auc=0.5,
                relative_roc_auc_retention=0.0,
                tstr_f1_score=0.0,
                trtr_f1_score=0.0,
                trts_pr_auc=0.0,
                utility_passed=False,
                precision_at_k=0.0,
                cost_savings=0.0,
                temporal_split_fraction=train_fraction,
            )

        clf.fit(X_train, y_train)
        p_test = clf.predict_proba(X_test)[:, 1]

        pr_auc = float(average_precision_score(y_test, p_test))
        roc_auc = float(roc_auc_score(y_test, p_test))

        # Precision at top-K daily alert capacity
        k = min(k_daily, len(y_test))
        top_k_indices = np.argsort(p_test)[::-1][:k]
        prec_at_k = float(np.mean(y_test[top_k_indices])) if k > 0 else 0.0

        # Cost utility calculation under standard bank cost matrix ($2.50 admin, 1% customer friction, loss = amount)
        test_amounts = np.array([float(r.get("amount", 0.0)) for r in test_records], dtype=float)
        baseline_cost = float(np.sum(test_amounts[y_test == 1]))
        pred_fraud = (p_test >= 0.50).astype(int)
        fp = (pred_fraud == 1) & (y_test == 0)
        fn = (pred_fraud == 0) & (y_test == 1)
        model_cost = float(np.sum(pred_fraud) * 2.50 + np.sum(0.01 * test_amounts[fp]) + np.sum(test_amounts[fn]))
        net_savings = max(0.0, baseline_cost - model_cost)

        thresholds = np.linspace(0.1, 0.9, 17)
        f1 = max(float(f1_score(y_test, (p_test >= t).astype(int), zero_division=0)) for t in thresholds)

        return MLUtilitySummary(
            tstr_pr_auc=pr_auc,
            trtr_pr_auc=pr_auc,
            relative_pr_auc_retention=1.0,
            tstr_roc_auc=roc_auc,
            trtr_roc_auc=roc_auc,
            relative_roc_auc_retention=1.0,
            tstr_f1_score=f1,
            trtr_f1_score=f1,
            trts_pr_auc=pr_auc,
            utility_passed=bool(pr_auc >= 0.20 and roc_auc >= 0.65),
            precision_at_k=prec_at_k,
            cost_savings=net_savings,
            temporal_split_fraction=train_fraction,
        )


# =====================================================================
# DIMENSION 3: ADVERSARIAL PRIVACY & NON-MEMORIZATION EVALUATOR
# =====================================================================

class AdversarialPrivacyEvaluator:
    """Evaluates privacy guarantees: Distance to Closest Record (DCR), NNDR, and Membership Inference Attack (MIA)."""

    @staticmethod
    def compute_privacy_metrics(
        X_syn: np.ndarray,
        X_ref_train: np.ndarray,
        X_ref_test: np.ndarray,
        random_state: int = 42,
    ) -> AdversarialPrivacySummary:
        """Calculates DCR, NNDR, and executes Membership Inference Attack classifier."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.neighbors import NearestNeighbors

        # 1. Feature normalization to [0, 1] using reference training range
        col_min = np.min(X_ref_train, axis=0)
        col_max = np.max(X_ref_train, axis=0)
        col_range = col_max - col_min
        col_range[col_range == 0] = 1.0

        X_ref_norm = np.clip((X_ref_train - col_min) / col_range, 0.0, 1.0)
        X_syn_norm = np.clip((X_syn - col_min) / col_range, -0.2, 1.2)
        X_test_norm = np.clip((X_ref_test - col_min) / col_range, -0.2, 1.2)

        # 2. Nearest Neighbors on normalized reference set
        nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
        nn.fit(X_ref_norm)
        distances, _ = nn.kneighbors(X_syn_norm)
        d1 = distances[:, 0]
        d2 = distances[:, 1]
        nndr = d1 / np.maximum(1e-6, d2)

        dcr_5th = float(np.percentile(d1, 5))
        dcr_median = float(np.median(d1))
        dcr_min = float(np.min(d1))
        nndr_mean = float(np.mean(nndr))
        nndr_median = float(np.median(nndr))

        # Reference: how far the reference records sit from *each other*, measured with
        # the same neighbours and the same normalisation. Memorisation is the synthetic
        # distance relative to that reference instead of an absolute floor: a floor in
        # min-max normalised space is unit-dependent and passes any generator that
        # reproduces common records, which is exactly what a non-memorising gate must
        # fail. Same construction as spec/18 pillar 2 (dcr_reference_ratio_min).
        ref_nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
        ref_nn.fit(X_ref_norm)
        ref_dist, _ = ref_nn.kneighbors(X_ref_norm)
        real_dcr_5th = float(np.percentile(ref_dist[:, 1], 5))
        dcr_ratio = float(dcr_5th / max(real_dcr_5th, 1e-9))

        # 3. Membership Inference Attack (MIA)
        n_eval = min(len(X_ref_norm), len(X_test_norm), 400)
        members = X_ref_norm[:n_eval]
        non_members = X_test_norm[:n_eval]

        nn_syn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn_syn.fit(X_syn_norm)
        dist_members, _ = nn_syn.kneighbors(members)
        dist_non_members, _ = nn_syn.kneighbors(non_members)

        X_attack = np.vstack([dist_members, dist_non_members])
        y_attack = np.concatenate([np.ones(n_eval, dtype=int), np.zeros(n_eval, dtype=int)])

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        mia_probs = np.zeros(len(y_attack))
        for train_idx, val_idx in skf.split(X_attack, y_attack):
            clf = LogisticRegression(random_state=random_state)
            clf.fit(-X_attack[train_idx], y_attack[train_idx])
            mia_probs[val_idx] = clf.predict_proba(-X_attack[val_idx])[:, 1]

        mia_roc = float(roc_auc_score(y_attack, mia_probs)) if len(np.unique(y_attack)) > 1 else 0.50

        # Grounded Acceptance Gates
        dcr_pass = bool(dcr_ratio >= 0.500)  # dimensionless DCR ratio to the real-to-real reference
        nndr_pass = bool(0.50 <= nndr_mean <= 0.98)
        mia_pass = bool(mia_roc <= 0.58)
        privacy_pass = bool(dcr_pass and nndr_pass and mia_pass)

        return AdversarialPrivacySummary(
            dcr_5th_percentile=dcr_5th,
            dcr_reference_5th_percentile=real_dcr_5th,
            dcr_reference_ratio=dcr_ratio,
            dcr_median=dcr_median,
            dcr_min=dcr_min,
            nndr_mean=nndr_mean,
            nndr_median=nndr_median,
            mia_attack_roc_auc=mia_roc,
            dcr_passed=dcr_pass,
            nndr_passed=nndr_pass,
            mia_passed=mia_pass,
            privacy_passed=privacy_pass,
        )


# =====================================================================
# DIMENSION 4: CLASSIC XAI BENCHMARK HARNESS (PRESERVED)
# =====================================================================

class XAIBenchmarkHarness:
    """Runs standardized XAI benchmark evaluation comparing post-hoc explainers against causal ground truth."""

    def __init__(
        self,
        n_transactions: int = 3000,
        fraud_prevalence: float = 0.06,
        region: str = "US",
        adversary_mimicry: float = 0.55,
        seed: int = 42,
    ):
        self.n_transactions = n_transactions
        self.fraud_prevalence = fraud_prevalence
        self.region = region
        self.adversary_mimicry = adversary_mimicry
        self.seed = seed
        self.evaluator = GroundTruthXAIEvaluator()

    def generate_and_prepare_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Synthesizes transactions and extracts feature matrix, labels, and ground-truth Shapley attributions."""
        engine = DiscreteEventEngine(
            n_cards=max(100, int(self.n_transactions / 10)),
            n_merchants=max(30, int(self.n_transactions / 40)),
            region=self.region,
            adversary_mimicry=self.adversary_mimicry,
            seed=self.seed,
        )
        records = engine.generate_batch(
            n_transactions=self.n_transactions,
            fraud_prevalence=self.fraud_prevalence,
        )

        X: List[List[float]] = []
        y: List[int] = []
        GT: List[List[float]] = []

        for r in records:
            X.append(feature_vector(r))
            y.append(int(r["is_fraud"]))
            GT.append(ground_truth_attributions(r, scale="auto"))

        return np.array(X, dtype=np.float64), np.array(y, dtype=np.int32), np.array(GT, dtype=np.float64), records

    def _build_interventional_ground_truth(self, record: Dict[str, Any]) -> np.ndarray:
        """Builds normalized L1 interventional ground truth vector for a transaction.

        Maps the record's scenario_tag to the modelled physical levers spec/21 declares
        for that playbook. A playbook spec/21 does not declare contributes nothing.
        Returns a vector of length len(feature_names()) where sum(v) == 1.0 (or all 0 if
        legitimate or undeclared).
        """
        columns = feature_names()
        vec = np.zeros(len(columns), dtype=np.float64)
        if int(float(record.get("is_fraud", 0))) == 0:
            return vec

        target_set = intervention_support(str(record.get("scenario_tag", "")))
        if not target_set:
            return vec

        for j, feat_name in enumerate(columns):
            if feat_name in target_set:
                vec[j] = 1.0

        s = np.sum(vec)
        if s > 0:
            vec /= s
        return vec

    def calculate_intervention_support_metrics(
        self,
        shap_values: np.ndarray,
        test_records: List[Dict[str, Any]],
        fraud_indices: List[int],
        k: int = 3,
    ) -> Tuple[float, float]:
        """Evaluates whether explainer attributed risk to the causal features actively intervened by the attack script."""
        precisions: List[float] = []
        recalls: List[float] = []

        for idx in fraud_indices:
            rec = test_records[idx]
            target_set = intervention_support(str(rec.get("scenario_tag", "")))
            if not target_set:
                # spec/21 declares no intervention for this playbook, so the record is
                # not scored. tests/test_causal_contract.py pins which playbooks those
                # are, so a new playbook cannot enter the generator unscored and silent.
                continue

            phi_hat = shap_values[idx]
            ranked_indices = np.argsort(-np.abs(phi_hat))
            top_k_feats = {feature_names()[j] for j in ranked_indices[:k]}
            overlap = len(top_k_feats & target_set)
            k_eff = min(k, len(target_set))
            precisions.append(overlap / float(k_eff) if k_eff > 0 else 0.0)
            recalls.append(overlap / float(len(target_set)))

        mean_p = float(np.mean(precisions)) if precisions else 0.0
        mean_r = float(np.mean(recalls)) if recalls else 0.0
        return mean_p, mean_r

    @staticmethod
    def _positive_class_shap(shap_values: Any) -> np.ndarray:
        """Attributions for the fraud class whatever SHAP's return shape is.

        Older SHAP returns a list [negative, positive] for binary tree ensembles; SHAP 0.45 and later
        returns one array of shape (n, d, 2) for RandomForest, which used to leave a (d, 2) matrix per
        instance and a 2d-versus-d dimension mismatch against the causal ground truth.
        """
        if isinstance(shap_values, list):
            return np.asarray(shap_values[1])
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            return arr[:, :, -1]
        return arr

    def run_benchmark(
        self,
        model_type: str = "lightgbm",
        train_ratio: float = 0.70,
    ) -> ModelBenchmarkSummary:
        """Trains model and computes evaluation metrics between TreeSHAP and causal ground truth."""
        try:
            import lightgbm as lgb
            import shap
            from sklearn.metrics import average_precision_score, roc_auc_score
        except ImportError as e:
            raise ImportError(
                "Benchmarking requires shap, lightgbm, and scikit-learn. "
                "Install them via: pip install 'prism-generator[benchmark]'"
            ) from e

        X, y, GT, records = self.generate_and_prepare_dataset()

        n_train = int(len(X) * train_ratio)
        X_train, y_train = X[:n_train], y[:n_train]
        X_test, y_test = X[n_train:], y[n_train:]
        GT_test = GT[n_train:]
        test_records = records[n_train:]

        # Train model
        if model_type == "lightgbm":
            model = lgb.LGBMClassifier(
                n_estimators=60,
                max_depth=4,
                random_state=self.seed,
                verbose=-1,
            )
            model.fit(X_train, y_train)
            test_probs = model.predict_proba(X_test)[:, 1]
            explainer = shap.TreeExplainer(model)
            shap_values = self._positive_class_shap(explainer.shap_values(X_test))
            explainer_name = "TreeSHAP (Interventional)"
        else:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=self.seed)
            model.fit(X_train, y_train)
            test_probs = model.predict_proba(X_test)[:, 1]
            explainer = shap.TreeExplainer(model)
            shap_values = self._positive_class_shap(explainer.shap_values(X_test))
            explainer_name = "TreeSHAP (RandomForest)"

        auc_roc = float(roc_auc_score(y_test, test_probs)) if len(np.unique(y_test)) > 1 else 0.5
        pr_auc = float(average_precision_score(y_test, test_probs)) if len(np.unique(y_test)) > 1 else 0.0

        fraud_test_indices = [i for i, label in enumerate(y_test) if label == 1]
        if not fraud_test_indices:
            fraud_test_indices = list(range(len(y_test)))

        # Battery & CPU preservation: cap evaluated fraud samples to at most 50
        fraud_test_indices = fraud_test_indices[:50]

        taus: List[float] = []
        rhos: List[float] = []
        cosines: List[float] = []
        raes: List[float] = []
        l1_dists: List[float] = []
        l2_dists: List[float] = []
        precisions_3: List[float] = []
        interv_taus: List[float] = []

        for idx in fraud_test_indices:
            phi_hat = shap_values[idx]
            phi_star = GT_test[idx]
            phi_interv = self._build_interventional_ground_truth(test_records[idx])

            res = self.evaluator.evaluate_instance(
                phi_hat=phi_hat,
                phi_star=phi_star,
                k_values=(2, 3, 4),
            )
            taus.append(res.kendall_tau)
            rhos.append(res.spearman_rho)
            cosines.append(res.cosine_similarity)
            raes.append(res.relative_attribution_error)
            l1_dists.append(res.normalized_l1_distance)
            l2_dists.append(res.normalized_l2_distance)
            precisions_3.append(res.precision_at_k.get(3, 0.0))

            # Interventional Kendall tau: rank correlation between |phi_hat| and phi_interv
            if np.sum(phi_interv) > 0 and np.std(phi_hat) > 1e-9:
                kt, _ = scipy.stats.kendalltau(np.abs(phi_hat), phi_interv)
                interv_taus.append(float(kt) if not np.isnan(kt) else 0.0)
            else:
                interv_taus.append(0.0)

        interv_p3, interv_r3 = self.calculate_intervention_support_metrics(
            shap_values=shap_values,
            test_records=test_records,
            fraud_indices=fraud_test_indices,
            k=3,
        )

        # Scorer Kendall Tau: rank correlation between model risk score and SCM ground-truth risk
        if len(fraud_test_indices) > 1:
            gt_scores = [float(np.sum(GT_test[i])) for i in fraud_test_indices]
            model_scores = [float(test_probs[i]) for i in fraud_test_indices]
            if np.std(gt_scores) > 1e-9 and np.std(model_scores) > 1e-9:
                s_tau, _ = scipy.stats.kendalltau(model_scores, gt_scores)
                scorer_kendall_tau = float(s_tau) if not np.isnan(s_tau) else 0.0
            else:
                scorer_kendall_tau = 0.0
        else:
            scorer_kendall_tau = 0.0

        interv_kt = float(np.mean(interv_taus)) if interv_taus else 0.0

        pr_auc_passed = (pr_auc <= 0.985) or (len(fraud_test_indices) < 5)

        if len(fraud_test_indices) > 0:
            abs_shap = np.abs(shap_values[fraud_test_indices])
            sum_abs = np.sum(abs_shap, axis=1, keepdims=True)
            sum_abs[sum_abs == 0] = 1.0
            feat_shares = np.mean(abs_shap / sum_abs, axis=0)
            max_share = float(np.max(feat_shares))
            single_feat_passed = (max_share <= 0.70)
        else:
            single_feat_passed = True

        tripwire_passed = bool(pr_auc_passed and single_feat_passed)

        return ModelBenchmarkSummary(
            model_name=model_type,
            explainer_name=explainer_name,
            n_evaluated_samples=len(fraud_test_indices),
            mean_kendall_tau=float(np.mean(taus)),
            mean_spearman_rho=float(np.mean(rhos)),
            mean_cosine_similarity=float(np.mean(cosines)),
            mean_precision_at_3=float(np.mean(precisions_3)),
            mean_relative_attribution_error=float(np.mean(raes)),
            auc_roc=auc_roc,
            pr_auc=pr_auc,
            mean_intervention_precision_at_3=interv_p3,
            mean_intervention_recall_at_3=interv_r3,
            interventional_kendall_tau=interv_kt,
            scorer_kendall_tau=scorer_kendall_tau,
            anti_leak_tripwire_passed=tripwire_passed,
            mean_normalized_l1_distance=float(np.mean(l1_dists)) if l1_dists else 0.0,
            mean_normalized_l2_distance=float(np.mean(l2_dists)) if l2_dists else 0.0,
            metrics_by_k={
                "Precision@3": float(np.mean(precisions_3)),
                "Intervention_Precision@3": interv_p3,
                "Intervention_Recall@3": interv_r3,
                "Interventional_Kendall_Tau": interv_kt,
                "Scorer_Kendall_Tau": scorer_kendall_tau,
            },
        )


# =====================================================================
# TRIPARTITE INDUSTRIAL BENCHMARK HARNESS
# =====================================================================

class TripartiteBenchmarkHarness:
    """Master benchmark suite evaluating Statistical Fidelity, ML Utility (TSTR), and Adversarial Privacy."""

    def __init__(
        self,
        n_transactions: int = 3000,
        fraud_prevalence: float = 0.05,
        region: str = "US",
        adversary_mimicry: float = 0.55,
        seed: int = 42,
    ):
        self.n_transactions = n_transactions
        self.fraud_prevalence = fraud_prevalence
        self.region = region
        self.adversary_mimicry = adversary_mimicry
        self.seed = seed

        self.fidelity_evaluator = StatisticalFidelityEvaluator()
        self.utility_evaluator = MLUtilityEvaluator()
        self.privacy_evaluator = AdversarialPrivacyEvaluator()
        self.xai_harness = XAIBenchmarkHarness(
            n_transactions=n_transactions,
            fraud_prevalence=fraud_prevalence,
            region=region,
            adversary_mimicry=adversary_mimicry,
            seed=seed,
        )

    def extract_features(self, records: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts the eleven modelled columns and the target labels (spec/21 contract)."""
        X: List[List[float]] = []
        y: List[int] = []
        for r in records:
            X.append(feature_vector(r))
            y.append(int(r.get("is_fraud", 0)))
        return np.array(X, dtype=np.float64), np.array(y, dtype=np.int32)

    def run_tripartite_benchmark(
        self,
        custom_syn_records: Optional[List[Dict[str, Any]]] = None,
        custom_ref_records: Optional[List[Dict[str, Any]]] = None,
        include_xai: bool = True,
    ) -> TripartiteBenchmarkSummary:
        """Executes complete Tripartite Benchmark evaluation across all 3 dimensions + XAI concordance."""
        # 1. Generate or load synthetic and reference streams
        if custom_syn_records is not None and custom_ref_records is not None:
            syn_records = custom_syn_records
            ref_records = custom_ref_records
        else:
            # Generate bona fide synthetic stream (seed = self.seed)
            engine_syn = DiscreteEventEngine(
                n_cards=max(80, int(self.n_transactions / 15)),
                n_merchants=max(25, int(self.n_transactions / 50)),
                region=self.region,
                adversary_mimicry=self.adversary_mimicry,
                seed=self.seed,
            )
            syn_records = engine_syn.generate_batch(
                n_transactions=self.n_transactions,
                fraud_prevalence=self.fraud_prevalence,
            )

            # Generate independent reference stream (seed = self.seed + 10000)
            engine_ref = DiscreteEventEngine(
                n_cards=max(80, int(self.n_transactions / 15)),
                n_merchants=max(25, int(self.n_transactions / 50)),
                region=self.region,
                adversary_mimicry=self.adversary_mimicry,
                seed=self.seed + 10000,
            )
            ref_records = engine_ref.generate_batch(
                n_transactions=self.n_transactions,
                fraud_prevalence=self.fraud_prevalence,
            )

        X_syn, y_syn = self.extract_features(syn_records)
        X_ref, y_ref = self.extract_features(ref_records)

        # Split reference into Train (50%) and Test (50%) for TSTR
        n_ref_train = int(len(X_ref) * 0.50)
        X_ref_train, y_ref_train = X_ref[:n_ref_train], y_ref[:n_ref_train]
        X_ref_test, y_ref_test = X_ref[n_ref_train:], y_ref[n_ref_train:]

        # 2. Dimension 1: Statistical Fidelity
        fidelity_summary = self.fidelity_evaluator.evaluate(
            syn_records=syn_records,
            ref_records=ref_records,
            X_syn=X_syn,
            X_ref=X_ref,
        )

        # 3. Dimension 2: ML Utility (TSTR Protocol)
        utility_summary = self.utility_evaluator.evaluate_tstr(
            X_syn=X_syn,
            y_syn=y_syn,
            X_ref_train=X_ref_train,
            y_ref_train=y_ref_train,
            X_ref_test=X_ref_test,
            y_ref_test=y_ref_test,
            random_state=self.seed,
        )

        # 4. Dimension 3: Adversarial Privacy & Non-Memorization
        privacy_summary = self.privacy_evaluator.compute_privacy_metrics(
            X_syn=X_syn,
            X_ref_train=X_ref_train,
            X_ref_test=X_ref_test,
            random_state=self.seed,
        )

        # 5. Dimension 4: Ground-Truth Causal XAI Conformance
        xai_summary: Optional[ModelBenchmarkSummary] = None
        if include_xai:
            xai_summary = self.xai_harness.run_benchmark(model_type="lightgbm")

        all_passed = bool(
            fidelity_summary.fidelity_passed
            and utility_summary.utility_passed
            and privacy_summary.privacy_passed
            and (xai_summary is None or xai_summary.anti_leak_tripwire_passed)
        )

        return TripartiteBenchmarkSummary(
            fidelity=fidelity_summary,
            utility=utility_summary,
            privacy=privacy_summary,
            xai_summary=xai_summary,
            n_synthetic_samples=len(syn_records),
            n_reference_samples=len(ref_records),
            region=self.region,
            all_passed=all_passed,
        )

    def evaluate_temporal_self_tstr(
        self,
        records: Optional[List[Dict[str, Any]]] = None,
        train_fraction: float = 0.60,
        k_daily: int = 50,
        random_state: int = 42,
    ) -> MLUtilitySummary:
        """Executes the 60/40 time-ordered self-TSTR protocol on a generated or provided transaction stream."""
        if records is None:
            engine = DiscreteEventEngine(
                n_cards=max(80, int(self.n_transactions / 15)),
                n_merchants=max(25, int(self.n_transactions / 50)),
                region=self.region,
                adversary_mimicry=self.adversary_mimicry,
                seed=self.seed,
            )
            records = engine.generate_batch(
                n_transactions=self.n_transactions,
                fraud_prevalence=self.fraud_prevalence,
            )
        return TSTRHarness.evaluate_temporal_self_tstr(
            records=records,
            train_fraction=train_fraction,
            k_daily=k_daily,
            random_state=random_state,
        )


def generate_tripartite_markdown_report(summary: TripartiteBenchmarkSummary) -> str:
    """Renders a publication-grade GitHub Flavored Markdown report of the Tripartite Benchmark."""
    verdict_badge = "**PASS (CERTIFIED)**" if summary.all_passed else "**FAIL (TRIPWIRE TRIGGERED)**"

    lines = [
        "# Prism Generator Tripartite Industrial Benchmark Certification Report",
        "",
        f"**Audit Status:** {verdict_badge}  ",
        f"**Ecosystem Region:** `{summary.region}` | **Synthetic Samples:** `{summary.n_synthetic_samples}` | **Reference Samples:** `{summary.n_reference_samples}`",
        "",
        "---",
        "",
        "## Dimension 1: Statistical Fidelity Suite",
        "",
        "| Metric | Empirical Value | Acceptance Bound | Real-World Grounding / Rationale | Status |",
        "| :--- | :--- | :--- | :--- | :--- |",
        f"| **Wasserstein-1 (Log Amount)** | `{summary.fidelity.wasserstein_amount_log:.4f}` | <= 0.050 | Log10 scale accounts for 5 orders of magnitude spend tail | {'PASS' if summary.fidelity.amount_passed else 'FAIL'} |",
        f"| **Wasserstein-1 (Log Arrival Delta t)** | `{summary.fidelity.wasserstein_arrival_log:.4f}` | <= 0.060 | Captures Hawkes spree burstiness and circadian diurnal rhythms | {'PASS' if summary.fidelity.arrival_passed else 'FAIL'} |",
        f"| **Jensen-Shannon Divergence (MCC)** | `{summary.fidelity.js_divergence_mcc:.4f}` | <= 0.070 | Laplace-smoothed divergence over 50+ merchant categories | {'PASS' if summary.fidelity.mcc_passed else 'FAIL'} |",
        f"| **Jensen-Shannon Divergence (Channel)** | `{summary.fidelity.js_divergence_channel:.4f}` | <= 0.035 | Distributional fidelity across CP chip, contactless, e-comm | {'PASS' if summary.fidelity.channel_passed else 'FAIL'} |",
        f"| **Jensen-Shannon Divergence (ISO Code)** | `{summary.fidelity.js_divergence_response_code:.4f}` | <= 0.035 | Verifies realistic ISO 8583 response bitfield frequencies | {'PASS' if summary.fidelity.js_divergence_response_code <= 0.035 else 'FAIL'} |",
        f"| **Spearman Frobenius Error** | `{summary.fidelity.spearman_frobenius_error:.4f}` | <= 0.160 | Non-linear rank correlation matrix fidelity | {'PASS' if summary.fidelity.correlation_passed else 'FAIL'} |",
        "",
        "---",
        "",
        "## Dimension 2: Machine Learning Utility (TSTR Protocol)",
        "",
        "| Metric | Empirical Value | Benchmark Target | Description | Status |",
        "| :--- | :--- | :--- | :--- | :--- |",
        f"| **TSTR PR-AUC (Synthetic -> Ref)** | `{summary.utility.tstr_pr_auc:.4f}` | >= 0.600 | LightGBM trained on synthetic, tested on held-out reference | {'PASS' if summary.utility.tstr_pr_auc >= 0.600 else 'FAIL'} |",
        f"| **TRTR PR-AUC (Ref -> Ref)** | `{summary.utility.trtr_pr_auc:.4f}` | Baseline | Upper bound trained and tested on reference split | Baseline |",
        f"| **Relative PR-AUC Retention (Delta PR)** | `{summary.utility.relative_pr_auc_retention:.4f}` | >= 0.850 | Ratio PR-AUC_TSTR / PR-AUC_TRTR (State-of-the-art >= 85%) | {'PASS' if summary.utility.relative_pr_auc_retention >= 0.850 else 'FAIL'} |",
        f"| **TSTR ROC-AUC** | `{summary.utility.tstr_roc_auc:.4f}` | >= 0.700 | Area under ROC curve on reference test stream | {'PASS' if summary.utility.tstr_roc_auc >= 0.700 else 'FAIL'} |",
        f"| **TSTR Optimal F1-Score** | `{summary.utility.tstr_f1_score:.4f}` | >= 0.550 | F1 classification performance at optimal probability threshold | {'PASS' if summary.utility.tstr_f1_score >= 0.550 else 'FAIL'} |",
        f"| **TRTS PR-AUC (Ref -> Synthetic)** | `{summary.utility.trts_pr_auc:.4f}` | Informational | Bidirectional symmetry check | Informational |",
    ]

    if summary.utility.precision_at_k > 0:
        lines.append(f"| **Alert Precision@K** | `{summary.utility.precision_at_k:.4f}` | Capacity Bound | Precision among top daily investigation alerts | Informational |")
    if summary.utility.cost_savings > 0:
        lines.append(f"| **Net Operational Savings** | `${summary.utility.cost_savings:,.2f}` | Utility Target | Fraud loss prevented minus investigation cost | Informational |")

    lines.extend([
        "",
        "---",
        "",
        "## Dimension 3: Adversarial Privacy & Non-Memorization",
        "",
        "| Metric | Empirical Value | Safety Bound | Privacy Protection Mechanism | Status |",
        "| :--- | :--- | :--- | :--- | :--- |",
        f"| **5th Percentile DCR (DCR_0.05)** | `{summary.privacy.dcr_5th_percentile:.4f}` | Informational | Synthetic-to-nearest-reference distance, normalised feature space | Informational |",
        f"| **DCR ratio to real reference** | `{summary.privacy.dcr_reference_ratio:.4f}` (reference `{summary.privacy.dcr_reference_5th_percentile:.4f}`) | >= 0.500 | Synthetic records must sit at least half as far from the reference as reference records sit from each other; a verbatim copier scores 0 | {'PASS' if summary.privacy.dcr_passed else 'FAIL'} |",
        f"| **Median DCR** | `{summary.privacy.dcr_median:.4f}` | Informational | Median distance to nearest reference transaction | Informational |",
        f"| **Mean NNDR (d1 / d2)** | `{summary.privacy.nndr_mean:.4f}` | [0.50, 0.98] | Nearest Neighbor Distance Ratio: proves diffuse distribution | {'PASS' if summary.privacy.nndr_passed else 'FAIL'} |",
        f"| **MIA Attack ROC-AUC** | `{summary.privacy.mia_attack_roc_auc:.4f}` | <= 0.580 | Membership Inference Attack: ~0.50 proves zero leakage | {'PASS' if summary.privacy.mia_passed else 'FAIL'} |",
        "",
    ])

    if summary.xai_summary is not None:
        xs = summary.xai_summary
        lines.extend([
            "---",
            "",
            "## Dimension 4: Ground-Truth Causal XAI Conformance",
            "",
            "| Metric | Empirical Value | Target Protocol | Description |",
            "| :--- | :--- | :--- | :--- |",
            f"| **Ranking Concordance (Kendall tau)** | `{xs.mean_kendall_tau:.4f}` | Quantus / OpenXAI | Rank correlation with causal Shapley ground truth |",
            f"| **Rank Correlation (Spearman rho)** | `{xs.mean_spearman_rho:.4f}` | Quantus / OpenXAI | Monotonic feature importance concordance |",
            f"| **Directional Cosine Similarity** | `{xs.mean_cosine_similarity:.4f}` | >= 0.70 | Alignment between TreeSHAP vector and latent Aumann-Shapley |",
            f"| **Intervention Precision@3** | `{xs.mean_intervention_precision_at_3:.4f}` | Causal Graph | Explainer recovery of actively intervened attack levers |",
            f"| **Interventional Kendall tau** | `{xs.interventional_kendall_tau:.4f}` | Attack Vector | Concordance with attack script intervention vector |",
            f"| **Scorer Kendall tau** | `{xs.scorer_kendall_tau:.4f}` | Risk Ranking | Concordance between predicted score and true SCM risk |",
            f"| **Anti-Leak Tripwire Passed** | `{xs.anti_leak_tripwire_passed}` | Anti-Leak | Single feature dominance <= 70%, PR-AUC <= 0.985 |",
            "",
        ])

    lines.extend([
        "---",
        "",
        f"### Final Certification Decision: {verdict_badge}",
    ])

    return "\n".join(lines)


# =====================================================================
# CLI ENTRYPOINT
# =====================================================================

def main() -> None:
    """CLI entrypoint for running the empirical XAI and Tripartite benchmarks."""
    parser = argparse.ArgumentParser(prog="prism-benchmark", description="Tripartite Industrial Benchmark Harness")
    parser.add_argument("-n", "--samples", type=int, default=3000, help="Number of synthetic transactions to generate")
    parser.add_argument("--model", type=str, choices=["lightgbm", "rf"], default="lightgbm", help="ML model architecture")
    parser.add_argument("--region", type=str, choices=["US", "IN"], default="US", help="Banking ecosystem region")
    parser.add_argument("--fraud-rate", type=float, default=0.05, help="Fraud prevalence ratio")
    parser.add_argument("--mimicry", type=float, default=0.55, help="Adversary stealth mimicry ratio")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    parser.add_argument("--tripartite", action="store_true", help="Run full Tripartite Industrial Benchmark Suite")
    parser.add_argument("--output-report", type=str, default=None, help="Path to write Markdown certification report")
    parser.add_argument("--json", action="store_true", help="Output benchmark metrics in JSON format")

    args = parser.parse_args()

    if args.tripartite:
        print(f"Executing Tripartite Industrial Benchmark on {args.samples} transactions ({args.region})...", file=sys.stderr)
        harness = TripartiteBenchmarkHarness(
            n_transactions=args.samples,
            fraud_prevalence=args.fraud_rate,
            region=args.region,
            adversary_mimicry=args.mimicry,
            seed=args.seed,
        )
        summary = harness.run_tripartite_benchmark()

        if args.json:
            print(json.dumps(asdict(summary), indent=2))
        else:
            report = generate_tripartite_markdown_report(summary)
            print("\n" + report + "\n")

        if args.output_report:
            report_content = generate_tripartite_markdown_report(summary)
            Path(args.output_report).write_text(report_content, encoding="utf-8")
            print(f"Report saved to {args.output_report}", file=sys.stderr)

    else:
        harness = XAIBenchmarkHarness(
            n_transactions=args.samples,
            fraud_prevalence=args.fraud_rate,
            region=args.region,
            adversary_mimicry=args.mimicry,
            seed=args.seed,
        )

        print(f"Executing XAI Benchmark on {args.samples} transactions ({args.model})...", file=sys.stderr)
        summary = harness.run_benchmark(model_type=args.model)

        if args.json:
            print(json.dumps(asdict(summary), indent=2))
        else:
            print("\n" + "=" * 65)
            print("  PRISM GENERATOR EMPIRICAL XAI BENCHMARK RESULTS")
            print("=" * 65)
            print(f"  Model Architecture:           {summary.model_name.upper()}")
            print(f"  Explainer Method:             {summary.explainer_name}")
            print(f"  Evaluated Fraud Samples:      {summary.n_evaluated_samples}")
            print(f"  Classifier ROC-AUC:           {summary.auc_roc:.4f}")
            print(f"  Classifier PR-AUC:            {summary.pr_auc:.4f}")
            print("-" * 65)
            print(f"  Ranking Concordance (Kendall Tau):      {summary.mean_kendall_tau:.4f}")
            print(f"  Rank Correlation (Spearman Rho):        {summary.mean_spearman_rho:.4f}")
            print(f"  Directional Cosine Similarity:          {summary.mean_cosine_similarity:.4f}")
            print(f"  Top-3 Support Recovery (Precision@3):   {summary.mean_precision_at_3:.4f}")
            print(f"  Intervention Support (Precision@3):     {summary.mean_intervention_precision_at_3:.4f}")
            print(f"  Intervention Support (Recall@3):        {summary.mean_intervention_recall_at_3:.4f}")
            print(f"  Anti-Leak Tripwire Passed:              {summary.anti_leak_tripwire_passed}")
            print(f"  Relative Attribution Error (RAE):       {summary.mean_relative_attribution_error:.4f}")
            print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
