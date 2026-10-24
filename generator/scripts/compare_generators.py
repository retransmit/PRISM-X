"""Generator comparison against real fraud detection, grouped by what each corpus can express.

Extends scripts/compare_generators.py with more simulators and with a second feature
group, because the corpora differ in what they can derive:

    Group A -- has a wall clock (can derive recency, velocity, hour-of-day)
        Prism Generator, e-commerce-fraud-detection-dataset
        Full six-feature test.

    Group B -- discrete step index only (no clock, no time unit published)
        BankSim (Lopez-Rojas et al., calibrated on a real Spanish bank), PaySim
        Restricted to the two features all four corpora share: amount and
        amount per-customer mean.

Comparing across groups would be meaningless, so each generator is only ever compared
against the ceiling measured on its own group's feature set.

PROVENANCE WARNING: the e-commerce corpus carries no published data dictionary or README,
so whether it is synthetic or real is unverified. It is included because it is the only
other Kaggle corpus in this set with a wall clock and card-shaped fields, not because its
origin is established.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

FULL = ["amount", "hours_since_last", "tx_count_1h", "tx_count_24h",
        "amount_over_card_mean", "hour_of_day"]
BASIC = ["amount", "amount_over_card_mean"]


def _ratio(df, amount_col, card_col):
    m = df.groupby(card_col)[amount_col].transform("mean")
    return (df[amount_col] / m.replace(0, np.nan)).fillna(1.0)


def clock_features(df, time_col, amount_col, card_col, label_col):
    """Six shared features for corpora that carry a real timestamp.

    Returns features, labels and timestamps in the SAME post-sort order. The sort below is
    internal, so a caller that re-reads the label off its own un-sorted frame misaligns
    every label from its features and scores a real-data model at ROC ~0.50. That mistake
    has now been made twice in this project, so the label is returned from here rather
    than re-read by the caller.
    """
    df = df.copy()
    # IEEE-CIS ships integer seconds since a reference; the e-commerce corpus ships ISO
    # strings. Normalise to a datetime index either way.
    raw_t = df[time_col]
    numeric_t = pd.api.types.is_numeric_dtype(raw_t)
    if numeric_t:
        df["ts"] = pd.to_datetime(raw_t.astype("int64"), unit="s", utc=True)
    else:
        df["ts"] = pd.to_datetime(raw_t, format="mixed", utc=True)
    df = df.sort_values([card_col, "ts"], kind="mergesort").set_index("ts")
    # set_index moves the column out of the frame; groupby/rolling below need it back.
    df["ts"] = df.index
    g = df.groupby(card_col, sort=False)["ts"]
    out = pd.DataFrame(index=df.index)
    out["amount"] = df[amount_col].astype("float64")
    out["hours_since_last"] = (g.diff().dt.total_seconds() / 3600.0).clip(lower=0).fillna(999.0)
    out["tx_count_1h"] = (df.groupby(card_col, sort=False)["ts"]
                          .rolling("1h").count().reset_index(level=0, drop=True).values)
    out["tx_count_24h"] = (df.groupby(card_col, sort=False)["ts"]
                           .rolling("24h").count().reset_index(level=0, drop=True).values)
    out["amount_over_card_mean"] = _ratio(df, amount_col, card_col)
    out["hour_of_day"] = df["ts"].dt.hour.astype("int64")
    stamps = df[time_col].to_numpy() if numeric_t else df["ts"].astype("int64").to_numpy()
    return out.reset_index(drop=True), df[label_col].to_numpy(), stamps


def basic_features(df, amount_col, card_col, label_col):
    out = pd.DataFrame(index=df.index)
    out["amount"] = df[amount_col].astype("float64")
    out["amount_over_card_mean"] = _ratio(df, amount_col, card_col)
    return out, df[label_col].to_numpy(), df[card_col].to_numpy()


def load_ieee(path):
    raw = pd.read_csv(path, usecols=["isFraud", "TransactionDT", "TransactionAmt", "card1"])
    return clock_features(raw, "TransactionDT", "TransactionAmt", "card1", "isFraud")


def load_prism_generator(path):
    raw = pd.read_csv(path, usecols=["is_fraud", "tx_time_seconds", "amount", "card_id"],
                      low_memory=False)
    return clock_features(raw, "tx_time_seconds", "amount", "card_id", "is_fraud")


def load_ecom(path):
    raw = pd.read_csv(path, usecols=["is_fraud", "transaction_time", "amount", "user_id"])
    epoch = raw["transaction_time"].map(
        lambda s: int(pd.Timestamp(s).timestamp())).astype("int64")
    raw = raw.assign(epoch=epoch)
    return clock_features(raw, "epoch", "amount", "user_id", "is_fraud")


def load_banksim(path):
    raw = pd.read_csv(path)
    return basic_features(raw, "amount", "customer", "fraud")


def load_paysim(path):
    raw = pd.read_csv(path, usecols=["isFraud", "step", "amount", "nameOrig"])
    return basic_features(raw, "amount", "nameOrig", "isFraud")


def model():
    return lgb.LGBMClassifier(n_estimators=120, learning_rate=0.05, random_state=42,
                              n_jobs=2, verbose=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ieee", type=Path, required=True)
    ap.add_argument("--prism_generator", type=Path, required=True)
    ap.add_argument("--ecom", type=Path, required=True)
    ap.add_argument("--banksim", type=Path, required=True)
    ap.add_argument("--paysim", type=Path, required=True)
    ap.add_argument("--test-frac", type=float, default=0.2)
    a = ap.parse_args()
    frac = 1 - a.test_frac

    rx, ry, rt = load_ieee(a.ieee)
    order = np.argsort(rt, kind="mergesort")
    rx, ry = rx.iloc[order].reset_index(drop=True), ry[order]
    split = int(len(rx) * frac)

    groups = [
        ("GROUP A  six features (corpora with a wall clock)", FULL,
         [("Prism Generator", load_prism_generator, a.prism_generator), ("e-commerce*", load_ecom, a.ecom)]),
        ("GROUP B  two features (no clock in these corpora)", BASIC,
         [("BankSim", load_banksim, a.banksim), ("PaySim", load_paysim, a.paysim)]),
    ]

    for title, features, gens in groups:
        r_tr, r_te = rx.iloc[:split][features], rx.iloc[split:][features]
        y_tr, y_te = ry[:split], ry[split:]
        prev = float(y_te.mean())
        c = model().fit(r_tr, y_tr).predict_proba(r_te)[:, 1]
        c_roc = roc_auc_score(y_te, c)
        # A real-data model must beat chance comfortably on these features; the measured
        # ceiling is ~0.73. Anything near 0.50 means the labels are misaligned against the
        # features (the internal sort permutes rows), not that the data is hard. Refuse to
        # print generator numbers against an impossible ceiling.
        if not 0.6 < c_roc < 1.0:
            raise SystemExit(
                f"ceiling ROC {c_roc:.4f} is at chance level -- labels and features are "
                "misaligned, so every transfer number below would be meaningless"
            )
        print(f"\n=== {title} ===")
        print(f"  ceiling real->real  ROC {c_roc:.4f}  PR {average_precision_score(y_te,c):.4f} "
              f"(test prevalence {prev*100:.3f}%)")
        for name, loader, path in gens:
            sx, sy, _ = loader(path)
            prior = float(sy.mean())
            s = model().fit(sx[features], sy).predict_proba(r_te)[:, 1]
            roc = roc_auc_score(y_te, s)
            adj = min(1.0, average_precision_score(y_te, s) * (prev / prior)) if prior > 0 else float("nan")
            print(f"  {name:12s} -> real  ROC {roc:.4f}  retention {roc/c_roc*100:5.1f}%  "
                  f"PR@real-priv {adj:.4f}  (train {len(sx):,} rows, {prior*100:.3f}% fraud)")
    print("\n* e-commerce corpus provenance is unverified; see module docstring.")


if __name__ == "__main__":
    raise SystemExit(main())