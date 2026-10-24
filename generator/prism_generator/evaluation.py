"""Evaluation module for Prism Generator.

Contains:
1. XAI & Causal Attribution Evaluation: GroundTruthXAIEvaluator, XAIBenchmarkResult, ranking correlations.
2. Prequential Time-Ordered Evaluation & Streaming Retraining:
   - DelayedSupervisionPolicy, CostMatrixConfig, DailyStreamingMetrics, PrequentialBenchmarkReport.
   - StreamingMetricTracker (PR-AUC, P@K, CP@K, DR@K, Cost Savings).
   - StreamingDriftAuditor (Two-sample KS score drift, Population Stability Index).
   - PrequentialStreamingEvaluator (Rolling predict-then-train loop under delayed supervision).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from .stream import LabelSource, SupervisionEngine, SupervisionRecord


# ==============================================================================
# SECTION 1: XAI & Causal Attribution Evaluation
# ==============================================================================

@dataclass
class XAIBenchmarkResult:
    """Standardized benchmark metrics for an XAI feature attribution evaluation."""
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    f1_at_k: Dict[int, float]
    kendall_tau: float
    spearman_rho: float
    cosine_similarity: float
    normalized_l1_distance: float
    normalized_l2_distance: float
    relative_attribution_error: float
    causal_faithfulness: Optional[float] = None
    deletion_auc: Optional[float] = None


def compute_spearman_rho_pure_numpy(a: np.ndarray, b: np.ndarray) -> float:
    """Computes Spearman rank correlation with fractional midrank tie handling."""
    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return 0.0
    res = stats.spearmanr(a, b)
    corr = float(res.correlation) if hasattr(res, "correlation") else float(res[0])
    return 0.0 if np.isnan(corr) else float(corr)


def compute_kendall_tau_pure_numpy(a: np.ndarray, b: np.ndarray) -> float:
    """Pure NumPy implementation of Kendall's tau-b."""
    n = len(a)
    if n < 2:
        return 0.0
    concordant = 0
    discordant = 0
    ties_a = 0
    ties_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            if da == 0 and db == 0:
                ties_a += 1
                ties_b += 1
            elif da == 0:
                ties_a += 1
            elif db == 0:
                ties_b += 1
            elif (da > 0 and db > 0) or (da < 0 and db < 0):
                concordant += 1
            else:
                discordant += 1

    total_pairs = n * (n - 1) / 2.0
    denom = math.sqrt((total_pairs - ties_a) * (total_pairs - ties_b))
    if denom <= 1e-12:
        return 0.0
    return float((concordant - discordant) / denom)


class GroundTruthXAIEvaluator:
    """Evaluates estimated attribution vectors against structural causal ground truth."""

    def __init__(self, eps: float = 1e-9, causal_threshold: float = 1e-4):
        self.eps = eps
        self.causal_threshold = causal_threshold

    def evaluate_instance(
        self,
        phi_hat: np.ndarray,
        phi_star: np.ndarray,
        k_values: Sequence[int] = (2, 3, 4),
        model_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        x_obs: Optional[np.ndarray] = None,
        x_cf: Optional[np.ndarray] = None,
    ) -> XAIBenchmarkResult:
        """Computes comprehensive evaluation metrics for a single explanation."""
        phi_h = np.asarray(phi_hat, dtype=np.float64).flatten()
        phi_s = np.asarray(phi_star, dtype=np.float64).flatten()
        d = len(phi_h)
        assert len(phi_s) == d, f"Dimension mismatch: phi_hat ({d}) != phi_star ({len(phi_s)})"

        # 1. Support Identification
        active_causal_indices = set(np.where(np.abs(phi_s) > self.causal_threshold)[0])
        ranked_hat_indices = np.argsort(-np.abs(phi_h))

        prec_at_k: Dict[int, float] = {}
        rec_at_k: Dict[int, float] = {}
        f1_at_k: Dict[int, float] = {}

        for k in k_values:
            k_clamped = min(k, d)
            top_k_hat = set(ranked_hat_indices[:k_clamped])
            true_positives = len(top_k_hat & active_causal_indices)

            p_k = true_positives / float(k_clamped) if k_clamped > 0 else 0.0
            prec_at_k[k] = round(p_k, 5)

            n_active = len(active_causal_indices)
            if n_active > 0:
                r_k = true_positives / float(n_active)
            else:
                r_k = 1.0 if len(top_k_hat) == 0 else 0.0
            rec_at_k[k] = round(r_k, 5)

            if (p_k + r_k) > 0.0:
                f1 = 2.0 * (p_k * r_k) / (p_k + r_k)
            else:
                f1 = 0.0
            f1_at_k[k] = round(f1, 5)

        # 2. Ranking Correlations
        abs_hat = np.abs(phi_h)
        abs_star = np.abs(phi_s)
        tau_b = compute_kendall_tau_pure_numpy(abs_hat, abs_star)
        rho = compute_spearman_rho_pure_numpy(abs_hat, abs_star)

        # 3. Continuous Attribution Error Metrics
        norm_hat_2 = float(np.linalg.norm(phi_h))
        norm_star_2 = float(np.linalg.norm(phi_s))

        # Directional Cosine Similarity
        if norm_hat_2 > self.eps and norm_star_2 > self.eps:
            cos_sim = float(np.dot(phi_h, phi_s) / (norm_hat_2 * norm_star_2))
        else:
            cos_sim = 1.0 if (norm_hat_2 <= self.eps and norm_star_2 <= self.eps) else 0.0

        # L1 Normalization (Total Variation Distance)
        l1_hat = float(np.sum(np.abs(phi_h)))
        l1_star = float(np.sum(np.abs(phi_s)))
        p_hat = phi_h / (l1_hat + self.eps)
        p_star = phi_s / (l1_star + self.eps)
        norm_l1_dist = float(0.5 * np.sum(np.abs(p_hat - p_star)))

        # L2 Normalization
        u_hat = phi_h / (norm_hat_2 + self.eps)
        u_star = phi_s / (norm_star_2 + self.eps)
        norm_l2_dist = float(np.linalg.norm(u_hat - u_star))

        # Relative Attribution Error (RAE)
        rae = float(np.linalg.norm(phi_h - phi_s) / (norm_star_2 + self.eps))

        # 4. Optional Faithfulness & Deletion Metrics
        causal_faithfulness: Optional[float] = None
        deletion_auc: Optional[float] = None

        if model_fn is not None and x_obs is not None and x_cf is not None:
            causal_faithfulness = self._compute_causal_faithfulness(model_fn, x_obs, x_cf, phi_h)
            deletion_auc = self._compute_deletion_auc(model_fn, x_obs, x_cf, ranked_hat_indices)

        return XAIBenchmarkResult(
            precision_at_k=prec_at_k,
            recall_at_k=rec_at_k,
            f1_at_k=f1_at_k,
            kendall_tau=round(tau_b, 5),
            spearman_rho=round(rho, 5),
            cosine_similarity=round(cos_sim, 5),
            normalized_l1_distance=round(norm_l1_dist, 5),
            normalized_l2_distance=round(norm_l2_dist, 5),
            relative_attribution_error=round(rae, 5),
            causal_faithfulness=round(causal_faithfulness, 5) if causal_faithfulness is not None else None,
            deletion_auc=round(deletion_auc, 5) if deletion_auc is not None else None,
        )

    def _compute_causal_faithfulness(
        self,
        model_fn: Callable[[np.ndarray], np.ndarray],
        x_obs: np.ndarray,
        x_cf: np.ndarray,
        phi_hat: np.ndarray,
    ) -> float:
        """Measures Pearson correlation between attribution and surgical score drop upon counterfactual restitution."""
        d = len(x_obs)
        delta_f = np.zeros(d, dtype=np.float64)
        base_pred = float(model_fn(x_obs.reshape(1, -1))[0])

        for i in range(d):
            if abs(x_obs[i] - x_cf[i]) < self.eps:
                delta_f[i] = 0.0
                continue
            x_pert = x_obs.copy()
            x_pert[i] = x_cf[i]
            pert_pred = float(model_fn(x_pert.reshape(1, -1))[0])
            delta_f[i] = base_pred - pert_pred

        if np.all(delta_f == 0.0) or np.all(phi_hat == 0.0):
            return 1.0 if np.all(delta_f == phi_hat) else 0.0

        mean_hat = np.mean(phi_hat)
        mean_delta = np.mean(delta_f)
        diff_hat = phi_hat - mean_hat
        diff_delta = delta_f - mean_delta
        denom = np.sqrt(np.sum(diff_hat ** 2) * np.sum(diff_delta ** 2))
        if denom <= 1e-12:
            return 0.0
        return float(np.sum(diff_hat * diff_delta) / denom)

    def _compute_deletion_auc(
        self,
        model_fn: Callable[[np.ndarray], np.ndarray],
        x_obs: np.ndarray,
        x_cf: np.ndarray,
        ranked_indices: np.ndarray,
    ) -> float:
        """Area under the prediction decay curve under counterfactual deletion."""
        d = len(x_obs)
        scores: List[float] = []
        x_curr = x_obs.copy()

        for step in range(d):
            idx = ranked_indices[step]
            x_curr[idx] = x_cf[idx]
            pred = float(model_fn(x_curr.reshape(1, -1))[0])
            scores.append(pred)

        return float(np.mean(scores))


# ==============================================================================
# SECTION 2: Prequential Time-Ordered Evaluation & Streaming Retraining
# ==============================================================================

class DelayedSupervisionPolicy(str, Enum):
    """Policy for handling pending transactions within the feedback delay gap [T - delta_delay, T]."""
    STRICT_DELAY_GAP = "STRICT_DELAY_GAP"
    ASSUME_NEGATIVE = "ASSUME_NEGATIVE"
    BIFURCATED_ENSEMBLE = "BIFURCATED_ENSEMBLE"


@dataclass(frozen=True)
class CostMatrixConfig:
    """Financial parameters for cost-sensitive loss evaluation in fraud detection.

    Values in nominal currency units (USD or INR).
    """
    c_admin: float = 2.50             # Cost of analyst verification call / SMS alert
    c_friction_rate: float = 0.01     # Customer friction cost (1% of transaction amount for false declines)
    c_chargeback_fee: float = 15.00   # Card network chargeback processing penalty fee

    @classmethod
    def for_region(cls, region: str) -> "CostMatrixConfig":
        if region.upper() == "IN":
            return cls(c_admin=200.0, c_friction_rate=0.01, c_chargeback_fee=1200.0)
        return cls(c_admin=2.50, c_friction_rate=0.01, c_chargeback_fee=15.00)


@dataclass
class DailyStreamingMetrics:
    """Metrics evaluated on a single forward test window [T, T + W_test)."""
    day_index: int
    t_start_seconds: float
    t_end_seconds: float
    n_transactions: int
    n_fraud_actual: int
    pr_auc: float
    average_precision: float
    p_at_k: float                    # Alert Precision @ K
    cp_at_k: float                   # Card Precision @ K
    dollar_recall_at_k: float        # Dollar Recall @ K (DR@K)
    dollar_precision_at_k: float     # Dollar Precision @ K (DP@K)
    cost_base: float                 # Cost without model (do nothing: all approved silently)
    cost_model: float                # Cost with model alert queue
    savings_ratio: float             # (cost_base - cost_model) / cost_base
    n_train_admissible: int
    ks_drift_p_value: Optional[float] = None
    psi_score: Optional[float] = None
    drift_detected: bool = False


@dataclass
class PrequentialBenchmarkReport:
    """Aggregated report across the entire prequential time-ordered evaluation trajectory."""
    n_days_evaluated: int
    total_test_transactions: int
    total_fraud_actual: int
    mean_pr_auc: float
    mean_average_precision: float
    mean_p_at_k: float
    mean_cp_at_k: float
    mean_dollar_recall_at_k: float
    mean_dollar_precision_at_k: float
    total_cost_base: float
    total_cost_model: float
    overall_savings_ratio: float
    daily_metrics: List[DailyStreamingMetrics]
    drift_alert_days: List[int] = field(default_factory=list)
    triage_k_summary: Dict[int, Dict[str, float]] = field(default_factory=dict)


class StreamingMetricTracker:
    """Vectorized calculation of operational, ranking, and financial fraud metrics."""

    @staticmethod
    def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Computes Precision-Recall AUC (Average Precision) via Riemann sum over sorted rank."""
        if len(y_true) == 0 or np.sum(y_true) == 0:
            return 0.0
        if np.sum(y_true) == len(y_true):
            return 1.0

        order = np.argsort(-y_score, kind="stable")
        y_sorted = y_true[order]

        n_pos = np.sum(y_sorted)
        true_pos = np.cumsum(y_sorted)
        total_pos = np.arange(1, len(y_sorted) + 1)

        precision = true_pos / total_pos
        recall = true_pos / n_pos

        # Compute area under PR curve (Average Precision)
        recall_diffs = np.diff(recall, prepend=0.0)
        ap = float(np.sum(precision * recall_diffs))
        return float(np.clip(ap, 0.0, 1.0))

    @staticmethod
    def compute_p_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
        """Computes Alert Precision @ K: fraction of top-K scored transactions that are true fraud."""
        n = len(y_true)
        if n == 0 or k <= 0:
            return 0.0
        k_eval = min(k, n)
        order = np.argsort(-y_score, kind="stable")
        top_k_labels = y_true[order[:k_eval]]
        return float(np.sum(top_k_labels) / float(k))

    @staticmethod
    def compute_card_precision_at_k(
        y_true: np.ndarray,
        y_score: np.ndarray,
        card_ids: List[str],
        k: int,
    ) -> float:
        """Computes Card Precision @ K (Le Borgne et al. 2021).

        Extracts the first K unique cardholders from the risk-ranked stream and evaluates
        whether each cardholder suffered at least one confirmed fraud event on that day.
        """
        if len(y_true) == 0 or k <= 0 or len(card_ids) != len(y_true):
            return 0.0

        order = np.argsort(-y_score, kind="stable")
        seen_cards: set = set()
        compromised_cards: int = 0

        # Group actual fraud presence per card for ground truth
        card_has_fraud: Dict[str, bool] = {}
        for cid, y in zip(card_ids, y_true):
            if int(y) == 1:
                card_has_fraud[cid] = True

        for idx in order:
            cid = card_ids[idx]
            if cid not in seen_cards:
                seen_cards.add(cid)
                if card_has_fraud.get(cid, False):
                    compromised_cards += 1
                if len(seen_cards) >= k:
                    break

        return float(compromised_cards / float(k))

    @staticmethod
    def compute_dollar_recall_at_k(
        y_true: np.ndarray,
        y_score: np.ndarray,
        amounts: np.ndarray,
        k: int,
    ) -> float:
        """Computes Dollar Recall @ K: fraud value captured in top-K divided by total fraud value."""
        total_fraud_dollars = float(np.sum(amounts[y_true == 1]))
        if total_fraud_dollars <= 0.0 or k <= 0 or len(y_true) == 0:
            return 0.0

        k_eval = min(k, len(y_true))
        order = np.argsort(-y_score, kind="stable")
        top_k_idx = order[:k_eval]
        captured_fraud_dollars = float(np.sum(amounts[top_k_idx] * y_true[top_k_idx]))
        return float(np.clip(captured_fraud_dollars / total_fraud_dollars, 0.0, 1.0))

    @staticmethod
    def compute_dollar_precision_at_k(
        y_true: np.ndarray,
        y_score: np.ndarray,
        amounts: np.ndarray,
        k: int,
    ) -> float:
        """Computes Dollar Precision @ K: fraud value captured divided by total dollar value in top-K."""
        if len(y_true) == 0 or k <= 0:
            return 0.0
        k_eval = min(k, len(y_true))
        order = np.argsort(-y_score, kind="stable")
        top_k_idx = order[:k_eval]
        top_k_total_dollars = float(np.sum(amounts[top_k_idx]))
        if top_k_total_dollars <= 0.0:
            return 0.0
        captured_fraud_dollars = float(np.sum(amounts[top_k_idx] * y_true[top_k_idx]))
        return float(np.clip(captured_fraud_dollars / top_k_total_dollars, 0.0, 1.0))

    @staticmethod
    def compute_financial_savings(
        y_true: np.ndarray,
        y_score: np.ndarray,
        amounts: np.ndarray,
        k: int,
        cost_matrix: CostMatrixConfig,
    ) -> Tuple[float, float, float]:
        """Calculates example-dependent operational cost savings (Bahnsen et al. 2016).

        Returns:
            Tuple of (cost_base, cost_model, savings_ratio).
        """
        n = len(y_true)
        if n == 0:
            return 0.0, 0.0, 0.0

        cost_base = float(np.sum((amounts + cost_matrix.c_chargeback_fee) * (y_true == 1)))
        if cost_base <= 0.0:
            return 0.0, 0.0, 0.0

        k_eval = min(k, n)
        order = np.argsort(-y_score, kind="stable")
        top_k_set = set(order[:k_eval])

        cost_model = 0.0
        for i in range(n):
            amt = float(amounts[i])
            is_fraud = int(y_true[i])

            if i in top_k_set:
                if is_fraud == 1:
                    cost_model += cost_matrix.c_admin
                else:
                    cost_model += cost_matrix.c_admin + (amt * cost_matrix.c_friction_rate)
            else:
                if is_fraud == 1:
                    cost_model += amt + cost_matrix.c_chargeback_fee
                else:
                    pass

        savings = (cost_base - cost_model) / cost_base
        return cost_base, cost_model, float(savings)


class StreamingDriftAuditor:
    """Unsupervised distribution shift auditing for streaming transaction scores."""

    @staticmethod
    def detect_ks_score_drift(
        reference_scores: np.ndarray,
        current_scores: np.ndarray,
        alpha: float = 0.01,
    ) -> Tuple[bool, float, float]:
        """Performs two-sample Kolmogorov-Smirnov test on risk score distributions.

        Returns:
            Tuple of (drift_detected, ks_stat, p_value).
        """
        if len(reference_scores) < 10 or len(current_scores) < 10:
            return False, 0.0, 1.0

        res = stats.ks_2samp(reference_scores, current_scores)
        stat = float(res.statistic)
        p_val = float(res.pvalue)
        return (p_val < alpha), stat, p_val

    @staticmethod
    def compute_population_stability_index(
        reference_scores: np.ndarray,
        current_scores: np.ndarray,
        n_bins: int = 10,
        epsilon: float = 1e-4,
    ) -> float:
        """Computes Population Stability Index (PSI) between reference and current inference scores."""
        if len(reference_scores) < 20 or len(current_scores) < 20:
            return 0.0

        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        bins = np.quantile(reference_scores, quantiles)
        bins[0] = -1e-6
        bins[-1] = 1.0 + 1e-6
        for j in range(1, len(bins)):
            if bins[j] <= bins[j - 1]:
                bins[j] = bins[j - 1] + 1e-5

        ref_counts, _ = np.histogram(reference_scores, bins=bins)
        curr_counts, _ = np.histogram(current_scores, bins=bins)

        ref_dist = ref_counts / float(len(reference_scores))
        curr_dist = curr_counts / float(len(current_scores))

        psi = np.sum((curr_dist - ref_dist) * np.log((curr_dist + epsilon) / (ref_dist + epsilon)))
        return float(max(0.0, psi))


class DefaultStreamingModel:
    """Fast, deterministic gradient boosted tree or logistic baseline for streaming evaluation."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DefaultStreamingModel":
        if len(np.unique(y)) < 2:
            self.model = "CONSTANT"
            self.constant_val = float(np.mean(y))
            return self

        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
            clf = HistGradientBoostingClassifier(
                max_iter=100,
                learning_rate=0.1,
                min_samples_leaf=10,
                random_state=self.random_state,
            )
            clf.fit(X, y)
            self.model = clf
        except Exception:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=200, random_state=self.random_state)
            clf.fit(X, y)
            self.model = clf
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model == "CONSTANT":
            p = np.full(len(X), self.constant_val)
            return np.column_stack([1.0 - p, p])
        if self.model is None:
            p = np.full(len(X), 0.01)
            return np.column_stack([1.0 - p, p])
        return self.model.predict_proba(X)


class PrequentialStreamingEvaluator:
    """Predict-then-train rolling evaluation engine under delayed supervision."""

    def __init__(
        self,
        w_train_days: float = 30.0,
        w_test_days: float = 1.0,
        delta_delay_days: float = 14.0,
        retrain_freq_days: float = 1.0,
        k_daily: int = 50,
        policy: DelayedSupervisionPolicy = DelayedSupervisionPolicy.STRICT_DELAY_GAP,
        cost_matrix: Optional[CostMatrixConfig] = None,
        feature_columns: Optional[List[str]] = None,
        supervision_engine: Optional[SupervisionEngine] = None,
        seed: int = 42,
    ):
        self.w_train_days = float(w_train_days)
        self.w_test_days = float(w_test_days)
        self.delta_delay_days = float(delta_delay_days)
        self.retrain_freq_days = float(retrain_freq_days)
        self.k_daily = int(k_daily)
        self.policy = policy
        self.cost_matrix = cost_matrix or CostMatrixConfig()
        self.supervision_engine = supervision_engine or SupervisionEngine(k_daily=k_daily, seed=seed)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.feature_columns = feature_columns or [
            "amount",
            "haversine_velocity_kph",
            "ip_distance_from_home_km",
            "tx_count_1h",
            "tx_count_24h",
            "tx_amount_sum_24h",
            "user_avg_tx_amount_30d",
            "cvv_match_flag",
            "billing_shipping_match",
        ]
        self.last_training_pools: List[Tuple[float, List[Dict[str, Any]]]] = []

    def _extract_feature_matrix(self, records: List[Dict[str, Any]]) -> np.ndarray:
        """Extracts numerical features with deterministic nan-handling and canonical alias resolution."""
        n = len(records)
        d = len(self.feature_columns)
        X = np.zeros((n, d), dtype=np.float64)

        aliases = {
            "user_tx_count_1h": "tx_count_1h",
            "user_tx_count_24h": "tx_count_24h",
            "user_tx_amount_sum_24h": "tx_amount_sum_24h",
            "tx_count_1h": "user_tx_count_1h",
            "tx_count_24h": "user_tx_count_24h",
            "tx_amount_sum_24h": "user_tx_amount_sum_24h",
        }

        for i, r in enumerate(records):
            for j, col in enumerate(self.feature_columns):
                val = r.get(col)
                if val is None and col in aliases:
                    val = r.get(aliases[col])
                if val is None or val == "":
                    X[i, j] = 0.0
                else:
                    try:
                        X[i, j] = float(val)
                    except (ValueError, TypeError):
                        X[i, j] = 0.0
        return X

    def evaluate_stream(
        self,
        records: List[Dict[str, Any]],
        model_factory: Optional[Callable[[], Any]] = None,
        supervision_records: Optional[List[SupervisionRecord]] = None,
    ) -> PrequentialBenchmarkReport:
        """Executes full rolling prequential evaluation over the transaction stream."""
        if not records:
            raise ValueError("Cannot evaluate an empty transaction record list")

        # Step 1: Ensure supervision timestamps are available
        if supervision_records is None:
            supervision_records = self.supervision_engine.process_batch(records)

        if len(supervision_records) != len(records):
            raise ValueError("Supervision records count does not match transaction records count")

        # Step 2: Sort records and supervision chronologically by tx_time_seconds
        order = np.argsort([float(r["tx_time_seconds"]) for r in records], kind="stable")
        sorted_records = [records[i] for i in order]
        sorted_supervision = [supervision_records[i] for i in order]

        t_min = float(sorted_records[0]["tx_time_seconds"])
        t_max = float(sorted_records[-1]["tx_time_seconds"])

        day_sec = 86400.0
        w_train_sec = self.w_train_days * day_sec
        delta_delay_sec = self.delta_delay_days * day_sec
        w_test_sec = self.w_test_days * day_sec
        retrain_step_sec = self.retrain_freq_days * day_sec

        # Initial training warmup requires at least w_train_sec + delta_delay_sec
        current_epoch = t_min + w_train_sec + delta_delay_sec
        if current_epoch + 2.0 * w_test_sec > t_max:
            # Scale down windows dynamically for short demonstration or unit test streams
            span_days = (t_max - t_min) / day_sec
            w_train_sec = max(0.5 * day_sec, span_days * 0.35 * day_sec)
            delta_delay_sec = max(0.25 * day_sec, span_days * 0.15 * day_sec)
            w_test_sec = min(w_test_sec, max(0.5 * day_sec, span_days * 0.20 * day_sec))
            retrain_step_sec = min(retrain_step_sec, w_test_sec)
            current_epoch = t_min + w_train_sec + delta_delay_sec

        if model_factory is None:
            model_factory = lambda: DefaultStreamingModel(random_state=self.seed)

        active_model = None
        reference_scores = np.array([])
        daily_metrics_list: List[DailyStreamingMetrics] = []
        drift_alert_days: List[int] = []
        all_days_triage: List[Dict[int, Dict[str, float]]] = []
        day_counter = 0

        # Step 3: Rolling prequential loop
        self.last_training_pools = []
        while current_epoch + w_test_sec <= t_max:
            t_test_start = current_epoch
            t_test_end = current_epoch + w_test_sec

            # Phase A: Form Admissible Training Set at Point-in-Time current_epoch
            train_records = []
            train_labels = []

            for r, sup in zip(sorted_records, sorted_supervision):
                t_tx = float(r["tx_time_seconds"])
                if not (current_epoch - (w_train_sec + delta_delay_sec) <= t_tx < current_epoch):
                    continue

                known_label = sup.is_label_available_at(query_time_seconds=current_epoch)

                if self.policy == DelayedSupervisionPolicy.STRICT_DELAY_GAP:
                    if t_tx >= current_epoch - delta_delay_sec:
                        if sup.label_source == LabelSource.INVESTIGATOR_ALERT and known_label is not None:
                            train_records.append(r)
                            train_labels.append(known_label)
                    else:
                        if known_label is not None:
                            train_records.append(r)
                            train_labels.append(known_label)
                elif self.policy == DelayedSupervisionPolicy.ASSUME_NEGATIVE:
                    if known_label is not None:
                        train_records.append(r)
                        train_labels.append(known_label)
                    elif t_tx >= current_epoch - delta_delay_sec:
                        train_records.append(r)
                        train_labels.append(0)

            self.last_training_pools.append((current_epoch, list(train_records)))

            # Fit new model on point-in-time training pool if enough samples
            if len(train_records) >= 10:
                X_train = self._extract_feature_matrix(train_records)
                y_train = np.array(train_labels, dtype=int)
                active_model = model_factory()
                active_model.fit(X_train, y_train)

                if len(train_records) >= 20:
                    train_probs = active_model.predict_proba(X_train)[:, 1]
                    reference_scores = train_probs

            # Phase B: Forward Evaluation on Unseen Future Test Window
            test_indices = [
                i for i, r in enumerate(sorted_records)
                if t_test_start <= float(r["tx_time_seconds"]) < t_test_end
            ]

            if test_indices:
                test_recs = [sorted_records[i] for i in test_indices]
                X_test = self._extract_feature_matrix(test_recs)
                y_true = np.array([int(r.get("is_fraud", 0)) for r in test_recs], dtype=int)
                amounts = np.array([float(r.get("amount", 0.0)) for r in test_recs], dtype=float)
                card_ids = [str(r.get("card_id", "")) for r in test_recs]

                if active_model is not None:
                    test_scores = active_model.predict_proba(X_test)[:, 1]
                else:
                    test_scores = np.full(len(test_recs), 0.01)

                pr_auc = StreamingMetricTracker.compute_pr_auc(y_true, test_scores)
                p_at_k = StreamingMetricTracker.compute_p_at_k(y_true, test_scores, self.k_daily)
                cp_at_k = StreamingMetricTracker.compute_card_precision_at_k(y_true, test_scores, card_ids, self.k_daily)
                dr_at_k = StreamingMetricTracker.compute_dollar_recall_at_k(y_true, test_scores, amounts, self.k_daily)
                dp_at_k = StreamingMetricTracker.compute_dollar_precision_at_k(y_true, test_scores, amounts, self.k_daily)
                cost_base, cost_model, savings = StreamingMetricTracker.compute_financial_savings(
                    y_true=y_true,
                    y_score=test_scores,
                    amounts=amounts,
                    k=self.k_daily,
                    cost_matrix=self.cost_matrix,
                )

                drift_detected = False
                ks_p_val = None
                psi_val = None
                if len(reference_scores) >= 20 and len(test_scores) >= 10:
                    drift_detected, _, ks_p_val = StreamingDriftAuditor.detect_ks_score_drift(
                        reference_scores=reference_scores,
                        current_scores=test_scores,
                        alpha=0.01,
                    )
                    psi_val = StreamingDriftAuditor.compute_population_stability_index(
                        reference_scores=reference_scores,
                        current_scores=test_scores,
                    )
                    if psi_val >= 0.25:
                        drift_detected = True

                if drift_detected:
                    drift_alert_days.append(day_counter)

                daily_m = DailyStreamingMetrics(
                    day_index=day_counter,
                    t_start_seconds=t_test_start,
                    t_end_seconds=t_test_end,
                    n_transactions=len(test_recs),
                    n_fraud_actual=int(np.sum(y_true)),
                    pr_auc=pr_auc,
                    average_precision=pr_auc,
                    p_at_k=p_at_k,
                    cp_at_k=cp_at_k,
                    dollar_recall_at_k=dr_at_k,
                    dollar_precision_at_k=dp_at_k,
                    cost_base=cost_base,
                    cost_model=cost_model,
                    savings_ratio=savings,
                    n_train_admissible=len(train_records),
                    ks_drift_p_value=ks_p_val,
                    psi_score=psi_val,
                    drift_detected=drift_detected,
                )
                daily_metrics_list.append(daily_m)

                # Multi-k triage curves for operational trade-off evaluation
                daily_triage: Dict[int, Dict[str, float]] = {}
                for k_val in (10, 25, 50, 100, 200):
                    p_k = StreamingMetricTracker.compute_p_at_k(y_true, test_scores, k_val)
                    cp_k = StreamingMetricTracker.compute_card_precision_at_k(y_true, test_scores, card_ids, k_val)
                    dr_k = StreamingMetricTracker.compute_dollar_recall_at_k(y_true, test_scores, amounts, k_val)
                    c_base_k, c_model_k, s_ratio_k = StreamingMetricTracker.compute_financial_savings(
                        y_true=y_true,
                        y_score=test_scores,
                        amounts=amounts,
                        k=k_val,
                        cost_matrix=self.cost_matrix,
                    )
                    daily_triage[k_val] = {
                        "p_at_k": p_k,
                        "cp_at_k": cp_k,
                        "dollar_recall_at_k": dr_k,
                        "savings_ratio": s_ratio_k,
                        "net_savings_nominal": max(0.0, c_base_k - c_model_k),
                    }
                all_days_triage.append(daily_triage)

            day_counter += 1
            current_epoch += retrain_step_sec

        if not daily_metrics_list:
            raise RuntimeError("Prequential evaluation produced 0 test windows. Check time span and window parameters.")

        n_days = len(daily_metrics_list)
        total_test_tx = sum(m.n_transactions for m in daily_metrics_list)
        total_fraud = sum(m.n_fraud_actual for m in daily_metrics_list)
        mean_pr_auc = float(np.mean([m.pr_auc for m in daily_metrics_list]))
        mean_ap = float(np.mean([m.average_precision for m in daily_metrics_list]))
        mean_p_at_k = float(np.mean([m.p_at_k for m in daily_metrics_list]))
        mean_cp_at_k = float(np.mean([m.cp_at_k for m in daily_metrics_list]))
        mean_dr_at_k = float(np.mean([m.dollar_recall_at_k for m in daily_metrics_list]))
        mean_dp_at_k = float(np.mean([m.dollar_precision_at_k for m in daily_metrics_list]))
        total_cost_base = float(sum(m.cost_base for m in daily_metrics_list))
        total_cost_model = float(sum(m.cost_model for m in daily_metrics_list))
        overall_savings = (total_cost_base - total_cost_model) / max(total_cost_base, 1e-9)

        triage_k_summary: Dict[int, Dict[str, float]] = {}
        if all_days_triage:
            for k_val in (10, 25, 50, 100, 200):
                triage_k_summary[k_val] = {
                    "p_at_k": float(np.mean([d[k_val]["p_at_k"] for d in all_days_triage])),
                    "cp_at_k": float(np.mean([d[k_val]["cp_at_k"] for d in all_days_triage])),
                    "dollar_recall_at_k": float(np.mean([d[k_val]["dollar_recall_at_k"] for d in all_days_triage])),
                    "savings_ratio": float(np.mean([d[k_val]["savings_ratio"] for d in all_days_triage])),
                    "net_savings_nominal": float(np.mean([d[k_val]["net_savings_nominal"] for d in all_days_triage])),
                }

        return PrequentialBenchmarkReport(
            n_days_evaluated=n_days,
            total_test_transactions=total_test_tx,
            total_fraud_actual=total_fraud,
            mean_pr_auc=mean_pr_auc,
            mean_average_precision=mean_ap,
            mean_p_at_k=mean_p_at_k,
            mean_cp_at_k=mean_cp_at_k,
            mean_dollar_recall_at_k=mean_dr_at_k,
            mean_dollar_precision_at_k=mean_dp_at_k,
            total_cost_base=total_cost_base,
            total_cost_model=total_cost_model,
            overall_savings_ratio=float(overall_savings),
            daily_metrics=daily_metrics_list,
            drift_alert_days=drift_alert_days,
            triage_k_summary=triage_k_summary,
        )
