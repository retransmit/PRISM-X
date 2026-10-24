"""Comprehensive Forensic Audit on Fraud Data Realness.

Evaluates the statistical authenticity, kinematic integrity, protocol realism,
and machine-learning separability of generated synthetic fraud data against
empirical banking and cybercrime benchmarks.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Any, Dict, List, Tuple

import numpy as np

# Reconfigure stdout for UTF-8 in Windows environments
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from prism_generator.engine import DiscreteEventEngine


def benford_mad(amounts: List[float]) -> Tuple[float, Dict[int, float]]:
    """Calculates Mean Absolute Deviation (MAD) from Benford's Law for first digits."""
    first_digits: List[int] = []
    for a in amounts:
        if a <= 0:
            continue
        s = f"{a:.4f}".lstrip("0.")
        if s and s[0].isdigit() and s[0] != "0":
            first_digits.append(int(s[0]))

    if not first_digits:
        return 0.0, {}

    n = len(first_digits)
    counts = Counter(first_digits)
    observed = {d: counts.get(d, 0) / n for d in range(1, 10)}
    expected = {d: math.log10(1.0 + 1.0 / d) for d in range(1, 10)}

    mad = sum(abs(observed[d] - expected[d]) for d in range(1, 10)) / 9.0
    return float(mad), observed


def train_eval_pure_numpy_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 150,
    lr: float = 0.05,
) -> Tuple[float, float, Dict[str, float]]:
    """Trains a regularized Logistic Regression classifier using pure NumPy gradient descent.
    
    Returns (ROC_AUC, PR_AUC, metrics_dict).
    """
    # Feature standardization
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std < 1e-6] = 1.0

    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    # Add bias column
    N_tr, D = X_train_norm.shape
    X_tr_b = np.hstack([np.ones((N_tr, 1)), X_train_norm])
    N_te = len(X_test_norm)
    X_te_b = np.hstack([np.ones((N_te, 1)), X_test_norm])

    # Class weighting for imbalance
    pos_weight = (len(y_train) - np.sum(y_train)) / max(1.0, np.sum(y_train))
    sample_weights = np.where(y_train == 1, pos_weight, 1.0)

    # Initialize weights
    w = np.zeros(D + 1)

    # Gradient descent with L2 regularization
    l2_reg = 0.01
    for _ in range(epochs):
        z = np.clip(X_tr_b @ w, -25.0, 25.0)
        p = 1.0 / (1.0 + np.exp(-z))
        err = (p - y_train) * sample_weights
        grad = (X_tr_b.T @ err) / N_tr + l2_reg * w
        grad[0] -= l2_reg * w[0]  # don't regularize bias
        w -= lr * grad

    # Predict on test set
    z_test = np.clip(X_te_b @ w, -25.0, 25.0)
    y_probs = 1.0 / (1.0 + np.exp(-z_test))

    # Calculate ROC-AUC using vectorized pair comparison
    pos_indices = np.where(y_test == 1)[0]
    neg_indices = np.where(y_test == 0)[0]
    n_pos = len(pos_indices)
    n_neg = len(neg_indices)

    if n_pos == 0 or n_neg == 0:
        roc_auc = 0.5
        pr_auc = 0.0
    else:
        pos_scores = y_probs[pos_indices]
        neg_scores = y_probs[neg_indices]
        roc_auc = float(np.mean(pos_scores[:, None] > neg_scores[None, :]) + 0.5 * np.mean(pos_scores[:, None] == neg_scores[None, :]))

        # Precision-Recall curve & Average Precision (PR-AUC)
        sorted_indices = np.argsort(-y_probs)
        sorted_labels = y_test[sorted_indices]
        cum_tp = np.cumsum(sorted_labels)
        cum_fp = np.cumsum(1 - sorted_labels)
        recalls = cum_tp / n_pos
        precisions = cum_tp / (cum_tp + cum_fp)

        # Average Precision (PR-AUC)
        recalls_diff = np.diff(np.concatenate(([0.0], recalls)))
        pr_auc = float(np.sum(precisions * recalls_diff))

    # Threshold evaluation at 0.50
    preds = (y_probs >= 0.50).astype(int)
    tp = int(np.sum((preds == 1) & (y_test == 1)))
    fp = int(np.sum((preds == 1) & (y_test == 0)))
    tn = int(np.sum((preds == 0) & (y_test == 0)))
    fn = int(np.sum((preds == 0) & (y_test == 1)))

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)

    metrics = {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
    return roc_auc, pr_auc, metrics


def audit_dataset_realness(region: str, n_transactions: int = 10000, seed: int = 42) -> Dict[str, Any]:
    print("\n" + "=" * 90)
    print(f"   FORENSIC DATA AUDIT: REGION = {region} ({n_transactions:,} TRANSACTIONS)")
    print("=" * 90)

    # 1. Initialize Engine & Generate Batch
    engine = DiscreteEventEngine(n_cards=1000, n_merchants=150, region=region, seed=seed)
    records = engine.generate_batch(
        n_transactions=n_transactions,
        fraud_prevalence=0.025,  # Realistic 2.5% fraud base rate
        start_time_seconds=1730419200.0,  # Nov 1, 2024
    )

    frauds = [r for r in records if r["is_fraud"] == 1]
    legit = [r for r in records if r["is_fraud"] == 0]
    hard_negatives = [r for r in legit if "HARD_NEGATIVE" in r.get("scenario_tag", "")]

    print(f"Generated Total: {len(records)}")
    print(f"  - Legitimate Transactions: {len(legit):>5} ({len(legit)/len(records)*100:.2f}%)")
    print(f"  - Fraudulent Transactions: {len(frauds):>5} ({len(frauds)/len(records)*100:.2f}%)")
    print(f"  - Hard Negatives (Legit):  {len(hard_negatives):>5} ({len(hard_negatives)/len(records)*100:.2f}%)")

    # -------------------------------------------------------------------------
    # Audit 1: Benford's Law First-Digit Forensic Anomaly
    # -------------------------------------------------------------------------
    print("\n[1] Benford's Law First-Digit Conformity:")
    legit_amounts = [float(r["amount"]) for r in legit]
    fraud_amounts = [float(r["amount"]) for r in frauds]
    organic_amounts = [float(r["amount"]) for r in legit if "DHANTERAS" not in r.get("scenario_tag", "")]

    mad_legit, obs_legit = benford_mad(legit_amounts)
    mad_org, _ = benford_mad(organic_amounts)
    mad_fraud, obs_fraud = benford_mad(fraud_amounts)

    if region == "IN" and any("DHANTERAS" in r.get("scenario_tag", "") for r in legit):
        print(f"  Legitimate Overall MAD:       {mad_legit:.4f} (Elevated by authentic Dhanteras ₹1.65L-₹1.95L cluster)")
        print(f"  Legitimate Baseline MAD:      {mad_org:.4f} (Threshold < 0.012 -> {'CONFORMING (PASS)' if mad_org < 0.012 else 'VIOLATION'})")
    else:
        print(f"  Legitimate Spend MAD:         {mad_legit:.4f} (Threshold < 0.012 -> {'CONFORMING (PASS)' if mad_legit < 0.012 else 'VIOLATION'})")
    print(f"  Fraudulent Spend MAD:         {mad_fraud:.4f} (Expected > 0.015 -> {'ANOMALOUS MANIPULATION (PASS)' if mad_fraud > 0.015 else 'UNEXPECTED CONFORMITY'})")

    # -------------------------------------------------------------------------
    # Audit 2: Circadian & Nocturnal Diurnal Distributions
    # -------------------------------------------------------------------------
    print("\n[2] Circadian Nocturnal Dynamics (00:00 - 05:00 Window):")
    legit_hours = [int(r["hour_of_day"]) for r in legit]
    fraud_hours = [int(r["hour_of_day"]) for r in frauds]

    legit_night_pct = sum(1 for h in legit_hours if 0 <= h < 5) / len(legit) * 100
    fraud_night_pct = sum(1 for h in fraud_hours if 0 <= h < 5) / len(frauds) * 100
    legit_day_pct = sum(1 for h in legit_hours if 10 <= h < 20) / len(legit) * 100

    print(f"  Legitimate Night Trough (0-5h): {legit_night_pct:.2f}% (Empirical Central Bank Baseline: < 4.5%)")
    print(f"  Legitimate Daytime Peak (10-20h): {legit_day_pct:.2f}%")
    print(f"  Fraud Nocturnal Targeting (0-5h): {fraud_night_pct:.2f}% (Adversarial Target: Elevated 15%-30%)")

    # -------------------------------------------------------------------------
    # Audit 3: Attack Scenario Breakdown & Parameter Realism
    # -------------------------------------------------------------------------
    print("\n[3] Adversarial Playbook Breakdown:")
    scenario_counts = Counter(r["scenario_tag"] for r in frauds)
    for scen, count in scenario_counts.most_common():
        pct = count / len(frauds) * 100
        sample_amts = [float(r["amount"]) for r in frauds if r["scenario_tag"] == scen]
        min_a, med_a, max_a = min(sample_amts), float(np.median(sample_amts)), max(sample_amts)
        curr = records[0]["currency"]
        print(f"  {scen:<35}: {count:>3} ({pct:>4.1f}%) | Amounts: [{min_a:.2f} to {max_a:.2f}] (med: {med_a:.2f} {curr})")

    # -------------------------------------------------------------------------
    # Audit 4: Kinematic Velocity & Impossible Travel Check
    # -------------------------------------------------------------------------
    print("\n[4] Kinematic Integrity & Physical Constraints:")
    cp_legit_velocities = [float(r["haversine_velocity_kph"]) for r in legit if r["channel_type"].startswith("CP")]
    supersonic_cp_legit = [v for v in cp_legit_velocities if v > 900.0]
    print(f"  Card-Present Legitimate Transactions with Velocity > 900 km/h: {len(supersonic_cp_legit)} (Must be 0)")
    max_cp_vel = max(cp_legit_velocities) if cp_legit_velocities else 0.0
    print(f"  Max Legitimate Card-Present Velocity Observed: {max_cp_vel:.1f} km/h")

    # -------------------------------------------------------------------------
    # Audit 5: Banking Network Plumbing & Authentication Signals
    # -------------------------------------------------------------------------
    print("\n[5] Institutional Banking Rails & Authorization Outcomes:")
    iso_counts_legit = Counter(r["response_code"] for r in legit)
    iso_counts_fraud = Counter(r["response_code"] for r in frauds)
    print("  Legitimate ISO 8583 Distribution:")
    for code, count in iso_counts_legit.most_common(5):
        print(f"    ISO {code}: {count:>5} ({count/len(legit)*100:.1f}%)")
    print("  Fraudulent ISO 8583 Distribution:")
    for code, count in iso_counts_fraud.most_common(5):
        print(f"    ISO {code}: {count:>5} ({count/len(frauds)*100:.1f}%)")

    # -------------------------------------------------------------------------
    # Audit 6: Dispute & Post-Authorization Adjudication
    # -------------------------------------------------------------------------
    print("\n[6] Post-Authorization Dispute Lifecycle:")
    disputes = [r for r in frauds if r.get("dispute_status", "NONE") != "NONE"]
    print(f"  Total Disputes Triggered: {len(disputes)} / {len(frauds)} fraud transactions")
    disp_status_counts = Counter(r.get("dispute_status", "") for r in frauds)
    for st, count in disp_status_counts.most_common():
        print(f"    {st:<32}: {count}")

    if region == "US":
        ce3_count = sum(1 for r in frauds if r.get("ce3_qualified") == True)
        arb_fees = sum(float(r.get("arbitration_fee_usd", 0.0)) for r in frauds)
        print(f"    Visa CE 3.0 Pre-Dispute Deflections: {ce3_count}")
        print(f"    Total Card Network Arbitration Fees: ${arb_fees:,.2f}")
    else:
        rbi_tiers = Counter(r.get("rbi_liability_tier", "") for r in disputes)
        print("    RBI Limited Liability Customer Tiers:")
        for tier, count in rbi_tiers.most_common():
            print(f"      {tier:<30}: {count}")
        lien_outcomes = Counter(r.get("cfcfrms_1930_lien_status", "") for r in disputes)
        print("    CFCFRMS / 1930 Golden Hour Cyber Lien Outcomes:")
        for lien, count in lien_outcomes.most_common():
            print(f"      {lien:<30}: {count}")

    # -------------------------------------------------------------------------
    # Audit 7: Criminal Syndicate & Graph Infrastructure
    # -------------------------------------------------------------------------
    print("\n[7] Criminal Syndicate & Graph Infrastructure Telemetry:")
    syndicates = Counter(r.get("syndicate_id", "") for r in frauds if r.get("syndicate_id"))
    botnets = Counter(r.get("botnet_cluster_id", "") for r in frauds if r.get("botnet_cluster_id"))
    mule_rings = Counter(r.get("mule_ring_id", "") for r in frauds if r.get("mule_ring_id"))
    mules = Counter(r.get("beneficiary_account_id", "") for r in frauds if r.get("beneficiary_account_id"))
    print(f"  Active Syndicates Identified: {len(syndicates)}")
    print(f"  Botnet Proxy Clusters:        {len(botnets)}")
    print(f"  Mule Rings:                   {len(mule_rings)}")
    print(f"  Distinct Mule Accounts:       {len(mules)}")
    # Multi-card fan-in test
    mule_to_cards: Dict[str, set] = {}
    for r in frauds:
        b_acc = r.get("beneficiary_account_id")
        if b_acc:
            mule_to_cards.setdefault(b_acc, set()).add(r["card_id"])
    fan_in_count = sum(1 for cards in mule_to_cards.values() if len(cards) > 1)
    print(f"  Mule Accounts with Multi-Card Convergence Fan-In: {fan_in_count}")

    # -------------------------------------------------------------------------
    # Audit 8: Machine Learning Separability & Hard Negative Tension
    # -------------------------------------------------------------------------
    print("\n[8] Machine Learning Separability & Hard Negative Tension:")
    X = []
    y = []
    scenarios = []

    for r in records:
        row = [
            float(r["amount"]),
            float(r["hour_of_day"]),
            float(r["day_of_week"]),
            float(r.get("z_score_amount_30d", 0.0)),
            float(r.get("tx_count_1h", 0)),
            float(r.get("tx_count_24h", 0)),
            float(r.get("tx_amount_sum_24h", 0.0)),
            float(r.get("haversine_velocity_kph", 0.0)),
            float(r.get("ip_distance_from_home_km", 0.0)),
            float(r.get("geo_risk_score", 0)),
            1.0 if r.get("is_cross_border") else 0.0,
            float(r.get("billing_shipping_match", 1)),
        ]
        X.append(row)
        y.append(int(r["is_fraud"]))
        scenarios.append(r.get("scenario_tag", ""))

    X_mat = np.array(X, dtype=np.float64)
    y_vec = np.array(y, dtype=np.int32)

    # 70/30 Train/Test split chronologically (no future leak)
    split_idx = int(0.70 * len(X_mat))
    X_train, y_train = X_mat[:split_idx], y_vec[:split_idx]
    X_test, y_test = X_mat[split_idx:], y_vec[split_idx:]

    roc_auc, pr_auc, metrics = train_eval_pure_numpy_classifier(
        X_train, y_train, X_test, y_test, epochs=250, lr=0.08
    )

    print(f"  Test Set Size: {len(y_test)} (Fraud count: {int(np.sum(y_test))})")
    print(f"  Logistic Regression ROC-AUC: {roc_auc:.4f}")
    print(f"  Logistic Regression PR-AUC:  {pr_auc:.4f}")
    print(f"  Classifier Precision:        {metrics['precision']:.4f}")
    print(f"  Classifier Recall:           {metrics['recall']:.4f}")
    print(f"  Classifier F1 Score:         {metrics['f1']:.4f}")
    print(f"  Confusion Matrix: TP={metrics['tp']}, FP={metrics['fp']}, TN={metrics['tn']}, FN={metrics['fn']}")

    if pr_auc > 0.98:
        print("  WARNING: PR-AUC > 0.98 suggests features are overly separable / synthetic leak!")
    elif pr_auc < 0.20:
        print("  WARNING: PR-AUC < 0.20 suggests model has almost zero predictive signal.")
    else:
        print(f"  REALISTIC ML FIDELITY: PR-AUC of {pr_auc:.4f} reflects true operational complexity (hard negatives create false alarms, stealthy fraud slips through).")

    return {
        "region": region,
        "n_records": len(records),
        "n_legit": len(legit),
        "n_frauds": len(frauds),
        "benford_mad_legit": mad_legit,
        "benford_mad_fraud": mad_fraud,
        "legit_night_pct": legit_night_pct,
        "fraud_night_pct": fraud_night_pct,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "metrics": metrics,
    }


if __name__ == "__main__":
    us_results = audit_dataset_realness("US", n_transactions=10000, seed=42)
    in_results = audit_dataset_realness("IN", n_transactions=10000, seed=42)
