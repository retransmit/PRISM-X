"""Per-axis realism comparison, with non-comparability reported rather than scored.

Why this exists
---------------
scripts/compare_generators.py ranked four generators by transfer ROC on a feature space
shared by all of them: amount, and amount-per-entity-mean. That feature space is defined
by the WEAKEST corpus in the set. BankSim and PaySim carry only step/type/amount/
nameOrig/nameDest/balances -- no clock, no device, no authentication, no merchant
context. Scoring them on an axis they structurally cannot occupy is a category error,
not a comparison.

Transfer ROC is also a USE test, not a REALISM test. Real data scores only ~0.73 on it.
A generator scoring 0.53 has not been shown to be unreal; it has been shown to not help
for this one task. Realism and downstream utility are different properties.

This script therefore compares axes, and for every axis that a given corpus cannot
express it prints the reason rather than a number. No absence is ever scored as zero.

Every number printed here is a distance between the generator and the real IEEE-CIS
corpus on that axis. Smaller is closer. There is no single ranking column, because the
axes are not commensurable and that is the point.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# Each corpus is declared with what it actually carries. `clock` is the timestamp column
# if the corpus has a real one, else None. Anything an axis needs that is absent is
# reported as non-comparable rather than defaulted.
CORPORA = {
    "IEEE-CIS (real)":   {"amount": "TransactionAmt", "entity": "card1",
                          "label": "isFraud", "clock": "TransactionDT"},
    "Prism Generator":          {"amount": "amount", "entity": "card_id",
                          "label": "is_fraud", "clock": "tx_time_seconds"},
    "BankSim":           {"amount": "amount", "entity": "customer",
                          "label": "fraud", "clock": None},
    "PaySim":            {"amount": "amount", "entity": "nameOrig",
                          "label": "isFraud", "clock": None},
    "e-commerce*":       {"amount": "amount", "entity": "user_id",
                          "label": "is_fraud", "clock": "transaction_time"},
}

QS = [0.10, 0.50, 0.90]


def load(name: str, path: Path) -> pd.DataFrame:
    c = CORPORA[name]
    cols = [c["amount"], c["entity"], c["label"]]
    if c["clock"]:
        cols.append(c["clock"])
    raw = pd.read_csv(path, usecols=cols, low_memory=False)
    raw = raw.rename(columns={c["amount"]: "amount", c["entity"]: "entity",
                              c["label"]: "fraud", c["clock"]: "clock"})
    raw["fraud"] = pd.to_numeric(raw["fraud"], errors="coerce").fillna(0).astype(int)
    raw["amount"] = pd.to_numeric(raw["amount"], errors="coerce")
    raw = raw.dropna(subset=["amount"])
    return raw


def epoch(col: pd.Series) -> pd.Series:
    """Normalise a timestamp column to int64 seconds, whether it is numeric or ISO."""
    if pd.api.types.is_numeric_dtype(col):
        return col.astype("int64")
    return col.map(lambda s: int(pd.Timestamp(s).timestamp())).astype("int64")


def add_common(df: pd.DataFrame, has_clock: bool) -> pd.DataFrame:
    """Derive the scale-free views the amount axes need, and timing if available.

    `has_clock` is a flag rather than a column name because a corpus without a timestamp
    has no such column to name -- that absence is what the NOT COMPARABLE rows report.
    """
    df = df.copy()
    df["log_amount"] = np.log10(df["amount"].clip(lower=0.01))
    # Scale-free: how large this transaction is for THIS entity. Comparing raw amounts
    # across corpora of different currency and scale would measure the units, not the
    # generator, so the amount axes use this ratio instead.
    ent_mean = df.groupby("entity")["amount"].transform("mean")
    df["amt_over_ent"] = (df["amount"] / ent_mean.replace(0, np.nan)).clip(0, 50).fillna(1.0)
    df["log_amt_over_ent"] = np.log10(df["amt_over_ent"].clip(lower=0.01))
    if has_clock:
        df["ts"] = epoch(df["clock"])
        df = df.sort_values(["entity", "ts"], kind="mergesort")
        df["gap_h"] = (df.groupby("entity", sort=False)["ts"].diff() / 3600.0).fillna(np.nan)
    else:
        df["gap_h"] = np.nan
    return df


def ks(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 100 or b.size < 100:
        return float("nan")
    return float(ks_2samp(a, b).statistic)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ieee", type=Path, required=True)
    ap.add_argument("--prism_generator", type=Path, required=True)
    ap.add_argument("--ecom", type=Path, required=True)
    ap.add_argument("--banksim", type=Path, required=True)
    ap.add_argument("--paysim", type=Path, required=True)
    a = ap.parse_args()
    paths = {"IEEE-CIS (real)": a.ieee, "Prism Generator": a.prism_generator, "BankSim": a.banksim,
             "PaySim": a.paysim, "e-commerce*": a.ecom}

    loaded = {}
    for name, p in paths.items():
        df = load(name, p)
        df = add_common(df, CORPORA[name]["clock"] is not None)
        loaded[name] = df
        print(f"loaded {name:16s} {len(df):>9,} rows  {df['fraud'].mean()*100:6.3f}% fraud"
              f"  clock={'yes' if CORPORA[name]['clock'] else 'NONE'}")
    print()

    ref = loaded["IEEE-CIS (real)"]
    names = [n for n in CORPORA if n != "IEEE-CIS (real)"]

    def block(title, note, fn, needs_clock):
        print(f"--- {title} ---")
        print(f"    {note}")
        for n in names:
            if needs_clock and CORPORA[n]["clock"] is None:
                print(f"  {n:16s} NOT COMPARABLE - corpus carries no timestamp column, "
                      f"only a 'step' index of undisclosed unit")
                continue
            try:
                print(f"  {n:16s} {fn(loaded[n], ref)}")
            except Exception as exc:                                  # noqa: BLE001
                print(f"  {n:16s} MEASUREMENT FAILED: {type(exc).__name__}: {exc}")
        print()

    def amt_row(df, ref):
        return ("p10/p50/p90 amt/entity  "
                + "/".join(f"{np.quantile(df['log_amt_over_ent'], q):.2f}" for q in QS)
                + f"   KS vs real {ks(df['log_amt_over_ent'].to_numpy(), ref['log_amt_over_ent'].to_numpy()):.3f}")

    block("AXIS 1  amount relative to the entity's own typical spend",
          "scale-free, so it compares generator behaviour and not currency units", amt_row, False)

    def raw_row(df, ref):
        return ("p10/p50/p90 log10 amount  "
                + "/".join(f"{np.quantile(df['log_amount'], q):.2f}" for q in QS)
                + f"   KS vs real {ks(df['log_amount'].to_numpy(), ref['log_amount'].to_numpy()):.3f}")

    block("AXIS 2  raw amount scale",
          "reported separately because different corpora use different units; "
          "a large KS here may be currency, not realism", raw_row, False)

    def fraud_amt(df, ref):
        f, l = df.loc[df.fraud == 1, "amt_over_ent"], df.loc[df.fraud == 0, "amt_over_ent"]
        if f.size < 50 or l.size < 50:
            return f"too few fraud rows ({f.size}) to measure"
        rf = ref.loc[ref.fraud == 1, "amt_over_ent"]
        return (f"fraud mean amt/entity {f.mean():6.2f}  (real {rf.mean():5.2f})  "
                f"KS {ks(f.to_numpy(), rf.to_numpy()):.3f}")

    block("AXIS 3  does fraud stand out in the amount distribution at all?",
          "if a generator's fraud rows are indistinguishable from its legitimate rows, "
          "no amount-based model can work on it", fraud_amt, False)

    def burst_row(df, ref):
        g = df["gap_h"].to_numpy()
        r = ref["gap_h"].to_numpy()
        q = lambda x, p: np.quantile(x[np.isfinite(x)], p)          # noqa: E731
        return (f"median gap {q(g,.5):8.3f}h  p90 {q(g,.9):9.3f}h  "
                f"(real {q(r,.5):.3f}h / {q(r,.9):.3f}h)  KS {ks(g, r):.3f}")

    block("AXIS 4  inter-transaction timing (dormancy and burstiness)",
          "the axis a simulator with no clock cannot express at all", burst_row, True)

    print("Reading these: smaller KS = closer to real. Axes are NOT combined into a score,")
    print("because they measure different things and a weighted total would hide exactly")
    print("the structural differences these blocks are meant to surface.")
    print("\n* e-commerce corpus provenance is unverified; see compare_generators.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())