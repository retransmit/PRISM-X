"""Module 3: metrics chosen for a 0.17%-positive problem.

Accuracy is deliberately not reported anywhere in this project. On this dataset
a model that predicts "not fraud" unconditionally scores 99.83% accuracy while
catching zero fraud (§3 of the proposal). ROC-AUC is recorded for completeness
but is not the optimisation target: with 284,315 negatives, a large absolute
number of false positives barely moves the false-positive rate, so ROC-AUC
looks flattering even when precision is unusable.

PR-AUC (average precision) is the primary metric because its baseline is the
positive class prevalence, 0.0017 here, so there is nowhere for a trivial
model to hide.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class EvalResult:
    model: str
    threshold: float
    pr_auc: float          # primary
    roc_auc: float         # reported, not optimised
    precision: float
    recall: float
    f1: float
    true_negatives: int
    false_positives: int
    false_negatives: int   # missed fraud, the expensive error
    true_positives: int

    @property
    def alerts_per_true_fraud(self) -> float:
        """How many alerts an analyst reviews per real fraud caught.

        The number that decides whether a fraud team would actually deploy this.
        1 / precision.
        """
        return (
            (self.true_positives + self.false_positives) / self.true_positives
            if self.true_positives
            else float("inf")
        )

    def as_dict(self) -> dict:
        """JSON-safe: a model that caught nothing has no ratio, and json.dumps
        would otherwise write Infinity, which no browser accepts."""
        data = asdict(self)
        data["alerts_per_true_fraud"] = self.alerts_per_true_fraud if self.true_positives else None
        return data

    def __str__(self) -> str:
        return (
            f"{self.model:<18} thr={self.threshold:.4f}  "
            f"PR-AUC={self.pr_auc:.4f}  P={self.precision:.4f}  "
            f"R={self.recall:.4f}  F1={self.f1:.4f}  "
            f"missed={self.false_negatives}"
        )


def evaluate(
    y_true: np.ndarray, scores: np.ndarray, threshold: float, model: str = "model"
) -> EvalResult:
    """Score a model's continuous risk output at one decision threshold."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return EvalResult(
        model=model,
        threshold=float(threshold),
        pr_auc=float(average_precision_score(y_true, scores)),
        roc_auc=float(roc_auc_score(y_true, scores)),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
    )


def adjusted_precision(tpr: float, fpr: float, prevalence: float) -> float:
    """Precision at a prevalence other than the window's own.

    A test window boosted to 0.5 percent fraud says nothing about how many
    alerts would be real at an Indian issuer. Its true-positive and
    false-positive rates do not depend on prevalence; precision does, and
    Bayes gives it: TPR * r / (TPR * r + FPR * (1 - r)).
    """
    hits = tpr * prevalence
    noise = fpr * (1.0 - prevalence)
    return float(hits / (hits + noise)) if (hits + noise) > 0 else 0.0


def adjusted_curve(
    y_true: np.ndarray, scores: np.ndarray, prevalence: float
) -> dict[str, list[float]]:
    """The precision-recall curve the report leads with for the Indian track."""
    fpr, tpr, thresholds = roc_curve(np.asarray(y_true).astype(int), np.asarray(scores, dtype=float))
    finite = np.isfinite(thresholds)
    return {
        "recall": [float(v) for v in tpr[finite]],
        "precision": [adjusted_precision(float(t), float(f), prevalence) for t, f in zip(tpr[finite], fpr[finite])],
        "thresholds": [float(v) for v in thresholds[finite]],
    }


def fpr_per_10k(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    """Legitimate transactions flagged, per 10,000 legitimate transactions."""
    y_true = np.asarray(y_true).astype(int)
    flagged = np.asarray(scores, dtype=float) >= threshold
    negatives = int((y_true == 0).sum())
    return float(10_000.0 * int((flagged & (y_true == 0)).sum()) / negatives) if negatives else 0.0


def value_caught(
    y_true: np.ndarray, scores: np.ndarray, threshold: float, amounts: np.ndarray, floor: float
) -> dict[str, float]:
    """Fraud rupees in flagged rows, overall and above the discussion paper's floor."""
    y_true = np.asarray(y_true).astype(int)
    amounts = np.asarray(amounts, dtype=float)
    flagged = np.asarray(scores, dtype=float) >= threshold
    fraud = y_true == 1
    total = float(amounts[fraud].sum())
    caught = float(amounts[fraud & flagged].sum())
    above = fraud & (amounts > floor)
    total_above = float(amounts[above].sum())
    caught_above = float(amounts[above & flagged].sum())
    return {
        "floor": float(floor),
        "fraud_value": total,
        "fraud_value_caught": caught,
        "share_of_fraud_value_caught": caught / total if total else 0.0,
        "fraud_value_above_floor": total_above,
        "share_above_floor_caught": caught_above / total_above if total_above else 0.0,
    }


def track_extras(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    amounts: np.ndarray,
    prevalence: float,
    floor: float,
    prevalence_range: tuple[float, float] | None = None,
) -> dict[str, float | None]:
    """The Indian track's extra lines (design D6): plain floats for JSON, and
    None for a ratio that does not exist because nothing was caught. With a
    ``prevalence_range`` the adjusted precision is also given at its two ends."""
    y_true = np.asarray(y_true).astype(int)
    flagged = np.asarray(scores, dtype=float) >= threshold
    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)
    tpr = float(int((flagged & (y_true == 1)).sum()) / positives) if positives else 0.0
    fpr = float(int((flagged & (y_true == 0)).sum()) / negatives) if negatives else 0.0
    precision = adjusted_precision(tpr, fpr, prevalence)
    out: dict[str, float | None] = {
        "prevalence_target": float(prevalence),
        "recall_at_threshold": tpr,
        "fpr_per_10k": float(10_000.0 * fpr),
        "adjusted_precision_at_prevalence": precision,
        # None, not inf: the files are served as JSON and inf is not JSON.
        "adjusted_alerts_per_true_fraud": float(1.0 / precision) if precision > 0 else None,
        **value_caught(y_true, scores, threshold, amounts, floor),
    }
    if prevalence_range is not None:
        low, high = sorted(float(p) for p in prevalence_range)
        out["prevalence_range"] = [low, high]
        out["adjusted_precision_low"] = adjusted_precision(tpr, fpr, low)
        out["adjusted_precision_high"] = adjusted_precision(tpr, fpr, high)
    return out


def paired_bootstrap(
    y_true: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray, *, resamples: int = 2000, seed: int = 0
) -> dict[str, float]:
    """A 95 percent interval of PR-AUC(a) minus PR-AUC(b) over stratified
    resamples of the same test rows, and the share of resamples in which b
    scores at least a. Two arms scored on one split differ by sampling alone
    unless the interval excludes zero."""
    from sklearn.metrics import average_precision_score

    y_true = np.asarray(y_true).astype(int)
    scores_a, scores_b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    rng = np.random.default_rng(seed)
    positives, negatives = np.flatnonzero(y_true == 1), np.flatnonzero(y_true == 0)
    differences = np.empty(resamples)
    for i in range(resamples):
        idx = np.concatenate([rng.choice(positives, len(positives)), rng.choice(negatives, len(negatives))])
        differences[i] = average_precision_score(y_true[idx], scores_a[idx]) - average_precision_score(
            y_true[idx], scores_b[idx]
        )
    return {
        "low": float(np.percentile(differences, 2.5)),
        "high": float(np.percentile(differences, 97.5)),
        "share_b_at_least_a": float(np.mean(differences <= 0.0)),
    }
