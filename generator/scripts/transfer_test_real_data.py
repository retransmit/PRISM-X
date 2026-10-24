"""Transfer test: does a model trained on Prism Generator detect fraud in real transactions?

This is the realism claim worth making, and it is a different claim from distributional
similarity. Distributional gates in `spec/18` measure distance to a reference this
project chose, and `reports/pillar1_external_corpus_scorecard.md` already showed that
gate is gameable: a sampler of the reference scores 0.0000 while real card transactions
score worse than this generator does. So "our numbers look like the Fed's numbers"
proves nothing about realism.

Transfer does. If a model fitted on synthetic traffic ranks fraud correctly on
transactions nobody here generated, then the behavioural structure being simulated --
velocity, recency, amount relative to the card's own history -- carries real signal in
the same places it does in production.

## What is and is not tested

The two corpora share no column names. This test restricts both to a six-feature space
each can derive independently:

    amount, hours_since_last_card_tx, tx_count_1h, tx_count_24h,
    amount_vs_card_mean, hour_of_day

So it tests whether the *behavioural* structure transfers. It does NOT test anything
overlapping only with Prism Generator: ISO 8583 response codes, ASN/ISP purity, device
fingerprints, mule rings, interbank routing. Those are claimed on structural grounds
against standards documents, not here.

## Corpora

    real    IEEE-CIS / Vesta Features, Kaggle competition `ieee-fraud-detection`,
            train_transaction.csv: 590,540 rows, 3.499% fraud, 182 days.
            Obtain with: kaggle competitions download ieee-fraud-detection -f train_transaction.csv
    synthetic  this repo's engine, seed 42

## Protocol

Both splits are time-ordered, never shuffled: real traffic evaluated on its own future.
The reference model is trained on the real train split and evaluated on the real test
split -- that is the ceiling a synthetic-trained model is trying to approach. Because
prevalence differs between corpora (3.499% real vs Prism Generator's demonstration rate), the
primary metric is ROC-AUC, which is prevalence-independent; PR-AUC is reported too and
reweighted to the real test prevalence so the two are comparable.

Usage:
    python scripts/transfer_test_real_data.py --ieee /path/to/train_transaction.csv
    python scripts/transfer_test_real_data.py --ieee ... --synthetic /path/to/prism_generator.csv.gz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

FEATURES = [
    "amount",
    "hours_since_last_card_tx",
    "tx_count_1h",
    "tx_count_24h",
    "amount_vs_card_mean",
    "hour_of_day",
]

IEEE_COLS = ["isFraud", "TransactionDT", "TransactionAmt", "card1"]
PRISM GENERATOR_COLS = ["is_fraud", "tx_time_seconds", "amount", "card_id", "timestamp_utc"]


def _shared_features(
    df: pd.DataFrame, time_col: str, card_col: str, amount_col: str, label_col: str
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Derive the six shared features from (time, card, amount) for either corpus.

    The same arithmetic runs on both, so any difference in the result is a property of
    the data rather than of the extraction.

    Returns features, labels and timestamps, all in the same post-sort order. The sort
    below is internal, so returning `df[label_col]` untouched would silently misalign
    every label against its own features. That mistake scores a real-data model at
    ROC 0.50 -- an impossible number that reads like a finding about the data rather
    than a bug in this script.
    """
    df = df.copy()
    df["ts"] = pd.to_datetime(df[time_col], unit="s", origin="2019-01-01")
    df = df.sort_values([card_col, "ts"], kind="mergesort").set_index("ts")

    grouped = df.groupby(card_col, sort=False)[time_col]
    out = pd.DataFrame(index=df.index)
    out["amount"] = df[amount_col].astype("float64")
    out["hours_since_last_card_tx"] = (
        (grouped.diff() / 3600.0).clip(lower=0).fillna(999.0)
    )
    out["tx_count_1h"] = (
        df.groupby(card_col, sort=False)[time_col]
        .rolling("1h")
        .count()
        .reset_index(level=0, drop=True)
        .values
    )
    out["tx_count_24h"] = (
        df.groupby(card_col, sort=False)[time_col]
        .rolling("24h")
        .count()
        .reset_index(level=0, drop=True)
        .values
    )
    card_mean = df.groupby(card_col)[amount_col].transform("mean")
    out["amount_vs_card_mean"] = (out["amount"] / card_mean.replace(0, np.nan)).fillna(1.0)
    out["hour_of_day"] = ((df[time_col] // 3600) % 24).astype("int64")
    return (
        out.reset_index(drop=True),
        df[label_col].to_numpy(),
        df[time_col].to_numpy(),
    )


def load_ieee(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    raw = pd.read_csv(path, usecols=IEEE_COLS)
    return _shared_features(raw, "TransactionDT", "card1", "TransactionAmt", "isFraud")


def load_prism_generator(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    raw = pd.read_csv(path, usecols=PRISM GENERATOR_COLS, low_memory=False)
    # tx_time_seconds is already seconds-from-epoch; rebase so both corpora share an origin.
    raw = raw.assign(TransactionDT=raw["tx_time_seconds"] - raw["tx_time_seconds"].min())
    return _shared_features(raw, "TransactionDT", "card_id", "amount", "is_fraud")


def _model() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=120, learning_rate=0.05, random_state=42, n_jobs=2, verbose=-1
    )


def pr_at_real_prevalence(y_true, scores, base_prevalence: float, target_prevalence: float) -> float:
    """PR-AUC rescaled so a model trained at one prevalence is comparable to one trained at another.

    Precision at fixed recall scales linearly with the positive-class prior; averaging
    precision across the recall sweep and multiplying by (target/base) is the standard
    first-order correction, exact when the class prior is the only thing that differs.
    """
    if base_prevalence <= 0:
        return float("nan")
    ap = average_precision_score(y_true, scores)
    return float(min(1.0, ap * (target_prevalence / base_prevalence)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ieee", type=Path, required=True, help="train_transaction.csv from ieee-fraud-detection")
    ap.add_argument(
        "--synthetic",
        type=Path,
        default=None,
        help="Prism Generator CSV(.gz). Omit to use the published US 100k batch if it is staged locally.",
    )
    ap.add_argument("--test-frac", type=float, default=0.2)
    args = ap.parse_args()

    if not args.ieee.exists():
        print(f"real corpus not found: {args.ieee}")
        print("obtain it with: kaggle competitions download ieee-fraud-detection -f train_transaction.csv")
        return 2

    real_x, real_y, real_t = load_ieee(args.ieee)
    # _shared_features sorts by (card, ts) so the per-card windows are correct, which
    # leaves the frame grouped by card. The time-ordered split must re-sort by time
    # first: splitting the card-grouped frame at 80% would test on the last 20% of card
    # ids rather than the most recent 20% of transactions.
    order = np.argsort(real_t, kind="mergesort")
    real_x = real_x.iloc[order].reset_index(drop=True)
    real_y = real_y[order]

    split = int(len(real_x) * (1 - args.test_frac))
    rx_tr, rx_te = real_x.iloc[:split], real_x.iloc[split:]
    ry_tr, ry_te = real_y[:split], real_y[split:]
    real_prev = float(ry_te.mean())

    print(f"real corpus      : {len(real_x):,} rows, fraud {real_y.mean()*100:.3f}%")
    print(f"  time split     : {len(rx_tr):,} train / {len(rx_te):,} test "
          f"(last {args.test_frac:.0%} by time, never shuffled)")
    print(f"  test prevalence: {real_prev*100:.3f}%")

    ceiling = _model().fit(rx_tr[FEATURES], ry_tr)
    c_scores = ceiling.predict_proba(rx_te[FEATURES])[:, 1]
    c_roc = roc_auc_score(ry_te, c_scores)
    c_ap = average_precision_score(ry_te, c_scores)

    print("\nCEILING  train on real -> test on real")
    print(f"  ROC-AUC {c_roc:.4f}   PR-AUC {c_ap:.4f}")

    if args.synthetic is None:
        print("\nno --synthetic given, so the transfer leg was not run.")
        return 0

    syn_x, syn_y, _syn_t = load_prism_generator(args.synthetic)
    print(f"\nsynthetic corpus: {len(syn_x):,} rows, fraud {syn_y.mean()*100:.3f}%")

    transfer = _model().fit(syn_x[FEATURES], syn_y)
    t_scores = transfer.predict_proba(rx_te[FEATURES])[:, 1]
    t_roc = roc_auc_score(ry_te, t_scores)
    t_ap = average_precision_score(ry_te, t_scores)
    t_ap_adj = pr_at_real_prevalence(ry_te, t_scores, float(syn_y.mean()), real_prev)

    print("TRANSFER  train on synthetic -> test on real")
    print(f"  ROC-AUC {t_roc:.4f}   PR-AUC {t_ap:.4f}   PR-AUC@real-prevalence {t_ap_adj:.4f}")

    retention = t_roc / c_roc if c_roc else float("nan")
    print(f"\nRETENTION (transfer ROC / ceiling ROC) = {retention*100:.1f}%")
    print("\nThis measures behavioural-structure transfer only. Columns that exist solely in")
    print("Prism Generator (ISO 8583 codes, ASN purity, device fingerprints, mule rings) are not")
    print("represented in this shared space and are not tested here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())