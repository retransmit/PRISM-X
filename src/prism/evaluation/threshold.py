"""Module 3: decision-threshold tuning (§5.3, cost-sensitive learning).

The default 0.5 cutoff is arbitrary for an imbalanced problem: it is the right
answer only when classes are balanced and errors cost the same. Here a missed
fraud costs the chargeback; a false positive costs an analyst's minute. The
threshold is where that trade-off is actually expressed, so it is tuned rather
than assumed.

Tuning happens on the VALIDATION split. Tuning on test and then reporting test
metrics is the same class of error as fitting the scaler before splitting:
the number stops meaning what it claims to mean.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from prism import config, tracks


def tune_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    strategy: str | None = None,
    target_recall: float | None = None,
    fpr_budget_per_10k: float | None = None,
) -> float:
    """Choose a decision threshold from validation scores.

    strategy="f1"         -> the threshold maximising F1
    strategy="recall_at"  -> the highest-precision threshold that still reaches
                             ``target_recall`` (use when the business requires a
                             minimum catch rate and precision is negotiable)
    strategy="fpr_budget" -> the most sensitive threshold whose false-positive
                             rate stays within ``fpr_budget_per_10k`` alerts per
                             10,000 legitimate transactions (design D5). F1
                             weighs a missed fraud and a false alert equally and
                             moves with the prevalence; at one fraud in 80,000
                             its optimum sits wherever the validation window
                             happens to show no false alert, which a window of a
                             few thousand legitimate rows cannot resolve. A
                             budget is how an issuer actually sets the cutoff.
    """
    strategy = strategy or tracks.active().setting(
        "threshold_strategy", config.get("evaluation.threshold_strategy", "f1")
    )
    target_recall = (
        target_recall
        if target_recall is not None
        else config.get("evaluation.target_recall", 0.85)
    )
    fpr_budget_per_10k = (
        fpr_budget_per_10k
        if fpr_budget_per_10k is not None
        else tracks.active().setting("fpr_budget_per_10k", config.get("evaluation.fpr_budget_per_10k", 5))
    )

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    if y_true.sum() == 0:
        raise ValueError("Cannot tune a threshold on a split containing no fraud.")

    if strategy == "fpr_budget":
        fpr, tpr, cutoffs = roc_curve(y_true, scores)
        eligible = np.flatnonzero(fpr <= float(fpr_budget_per_10k) / 10_000.0)
        # roc_curve's first point (fpr 0, tpr 0, threshold inf) is always
        # eligible, so there is always a choice; argmax takes the first of the
        # tied maxima, which is the lowest false-positive rate among them.
        best = eligible[int(np.argmax(tpr[eligible]))]
        threshold = float(cutoffs[best])
        # An infinite cutoff means nothing met the budget: flag nothing. Every
        # risk_score is strictly below 1.0, so 1.0 is that cutoff.
        return min(threshold, 1.0)

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision/recall have one more element than thresholds; drop the trailing
    # point, which corresponds to recall=0 and has no threshold attached.
    precision, recall = precision[:-1], recall[:-1]

    if len(thresholds) == 0:
        return 0.5

    if strategy == "f1":
        denominator = precision + recall
        f1 = np.divide(
            2 * precision * recall,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )
        return float(thresholds[int(np.argmax(f1))])

    if strategy == "recall_at":
        eligible = np.flatnonzero(recall >= target_recall)
        if eligible.size == 0:
            # Target unreachable at any cutoff: fall back to maximum recall and
            # let the caller see it in the reported metrics rather than crash.
            return float(thresholds[int(np.argmax(recall))])
        # Among thresholds meeting the recall floor, take the most precise.
        best = eligible[int(np.argmax(precision[eligible]))]
        return float(thresholds[best])

    raise ValueError(f"Unknown threshold strategy: {strategy!r}")


def pr_curve_points(y_true: np.ndarray, scores: np.ndarray) -> dict[str, list[float]]:
    """Serialisable PR curve, for the report figures and the /metrics endpoint."""
    precision, recall, thresholds = precision_recall_curve(
        np.asarray(y_true).astype(int), np.asarray(scores, dtype=float)
    )
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
    }


def downsample_curve(
    curve: dict[str, list[float]], max_points: int = 200
) -> dict[str, list[float]]:
    """Thin a PR curve for transport to the browser, by recall.

    precision_recall_curve emits one point per distinct score, about 57,000
    on the ULB test split, for a 400 px polyline. Keeping every k-th point by
    position looked right on a smooth curve and wrong on a good model: the
    95 frauds sit in the last 96 points, so position-based thinning kept one
    of them and drew the model at half its precision. The points kept here
    are, for each of a set of evenly spaced recall values, the first and the
    last curve point at the nearest recall, plus both endpoints, so the drawn
    polyline passes through what the model does at each recall and keeps the
    vertical runs. Thresholds are dropped: the browser draws x against y.
    """
    precision = [float(v) for v in curve["precision"]]
    recall = [float(v) for v in curve["recall"]]
    n = len(recall)
    if n <= max_points:
        return {"precision": precision, "recall": recall}

    recalls = np.asarray(recall)
    distinct, first_at = np.unique(recalls, return_index=True)
    _, last_from_end = np.unique(recalls[::-1], return_index=True)
    last_at = n - 1 - last_from_end
    targets = np.linspace(0.0, 1.0, max(2, max_points // 2 - 1))
    nearest = np.abs(distinct[None, :] - targets[:, None]).argmin(axis=1)
    chosen = {0, n - 1}
    for k in set(nearest.tolist()):
        chosen.update((int(first_at[k]), int(last_at[k])))
    indices = sorted(chosen)
    return {
        "precision": [precision[i] for i in indices],
        "recall": [recall[i] for i in indices],
    }


def budget_threshold(negative_scores: np.ndarray, fpr_budget_per_10k: float) -> float:
    """The most sensitive cutoff that flags no more than the budget's share of the rows
    believed legitimate. For a window in which no fraud is known (labels that have not
    matured yet), there is no recall to trade against, and the budget alone sets the cutoff."""
    ordered = np.sort(np.asarray(negative_scores, dtype=float))[::-1]
    if len(ordered) == 0:
        return 1.0
    allowed = int(np.floor(float(fpr_budget_per_10k) / 10_000.0 * len(ordered)))
    if allowed >= len(ordered):
        return float(ordered[-1])
    # The (allowed + 1)-th highest score must stay below the cutoff; the next float above
    # it is the lowest cutoff that keeps it there. Every risk score is strictly below 1.0.
    return float(min(np.nextafter(ordered[allowed], 1.0), 1.0))
