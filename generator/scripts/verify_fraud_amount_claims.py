"""Independent re-verification of the probe/harvest fraud-amount claims.

Written because the first diagnostic (scripts/axis3_diagnostic.py) reported a column it
labelled "txn/fraud-ent" that turned out not to mean what the label said. Nothing here
trusts that script; every figure is recomputed from the source CSVs.

Specific things being checked:

  1. "fraud ents" -- the earlier run printed the number of fraud ROWS under that heading,
     not the number of distinct fraud entities. Both are printed here.
  2. Card-size denominators are computed once per entity, not once per fraud row. An
     entity with 500 fraud rows was contributing its transaction count 500 times to the
     earlier mean.
  3. Whether real IEEE fraud is concentrated on a few mega-cards. If it is, the real
     median ratio is itself unrepresentative and the comparison is weakened.
  4. The claimed decomposition arithmetic, recomputed from the scenario table.
  5. The realised mean fraud amount against the spec/09 target, which is ungated.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CORPORA = {
    "IEEE-CIS (real)": {"amount": "TransactionAmt", "entity": "card1", "label": "isFraud"},
    "Prism Generator":        {"amount": "amount", "entity": "card_id", "label": "is_fraud"},
    "BankSim":         {"amount": "amount", "entity": "customer", "label": "fraud"},
    "PaySim":          {"amount": "amount", "entity": "nameOrig", "label": "isFraud"},
    "e-commerce*":     {"amount": "amount", "entity": "user_id", "label": "is_fraud"},
}


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

    store: dict[str, pd.DataFrame] = {}
    for name, p in paths.items():
        c = CORPORA[name]
        df = pd.read_csv(p, usecols=[c["amount"], c["entity"], c["label"]], low_memory=False)
        df = df.rename(columns={c["amount"]: "amount", c["entity"]: "entity",
                                c["label"]: "fraud"})
        df["fraud"] = pd.to_numeric(df["fraud"], errors="coerce").fillna(0).astype(int)
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        store[name] = df.dropna(subset=["amount"])

    # ---- CHECK 1/2: entities vs rows, and card size counted once per entity ----
    print("=" * 80)
    print("CHECK 1+2  distinct fraud ENTITIES vs fraud ROWS, and card size per entity")
    print("=" * 80)
    print(f"{'corpus':16s} {'fraud ROWS':>11s} {'fraud ENTS':>10s} {'all ENTS':>9s} "
          f"{'med card size':>14s} {'mean card size':>15s}")
    for name, df in store.items():
        f = df[df.fraud == 1]
        fents = f.entity.nunique()
        size = df.groupby("entity").size()          # counted ONCE per entity
        fsize = size.reindex(f.entity.unique()).dropna()
        print(f"{name:16s} {len(f):>11,} {fents:>10,} {df.entity.nunique():>9,} "
              f"{fsize.median():>14,.1f} {fsize.mean():>15,.1f}")

    # ---- CHECK 3: is real fraud concentrated on mega-cards? ----
    print()
    print("=" * 80)
    print("CHECK 3  concentration of fraud rows -- is the real median representative?")
    print("=" * 80)
    for name, df in store.items():
        f = df[df.fraud == 1]
        if not len(f):
            continue
        top = f.entity.value_counts(normalize=True)
        c1 = top.cumsum()
        n_cover = int((c1 <= 0.01).sum()) + 1
        share1 = c1.iloc[min(n_cover, len(c1)) - 1]
        print(f"{name:16s} top 1% of fraud-bearing entities hold {share1*100:5.1f}% of "
              f"fraud rows ({n_cover:,} entities)")

    # ---- recompute ratio distribution, per row, with no clip ----
    print()
    print("=" * 80)
    print("RATIO = amount / that entity's mean amount   (no clip applied)")
    print("=" * 80)
    print(f"{'corpus':16s} {'p10':>7s} {'p50':>7s} {'p90':>7s} {'p99':>7s} "
          f"{'mean':>7s} {'>5x%':>7s}  (fraud rows only)")
    ratios: dict[str, pd.Series] = {}
    for name, df in store.items():
        m = df.groupby("entity")["amount"].transform("mean")
        r = (df.amount / m.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        ratios[name] = r[df.fraud == 1].to_numpy()
        v = ratios[name]
        print(f"{name:16s} {np.quantile(v,.1):>7.3f} {np.quantile(v,.5):>7.3f} "
              f"{np.quantile(v,.9):>7.3f} {np.quantile(v,.99):>7.3f} {v.mean():>7.3f} "
              f"{(v>5).mean()*100:>6.2f}%")

    # ---- CHECK 4/5: Prism Generator scenario decomposition ----
    print()
    print("=" * 80)
    print("CHECK 4+5  Prism Generator scenario decomposition, recomputed")
    print("=" * 80)
    fx = pd.read_csv(a.prism_generator, low_memory=False)
    mf = fx.groupby("card_id")["amount"].transform("mean")
    fxf = fx[fx.is_fraud == 1].copy()
    fxf["ratio"] = (fxf.amount / mf[fxf.index]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    tot = len(fxf)
    g = fxf.groupby("scenario_tag")["ratio"].agg(
        n="count", median="median", mean="mean", pct=lambda s: (s > 5).mean() * 100)
    g = g.sort_values("n", ascending=False)
    g["share%"] = g.n / tot * 100
    print(g[["n", "share%", "median", "mean", "pct"]].to_string(
        float_format=lambda x: f"{x:.3f}"))
    contrib = float((g.n / tot * g.pct).sum())
    print(f"\nsum of (share x pct_gt5x) = {contrib:.2f}%")
    print(f"direct measurement of fraud rows with ratio > 5 "
          f"= {(fxf.ratio > 5).mean()*100:.2f}%")

    print()
    print("=" * 80)
    print("CHECK 5b  realised mean fraud amount vs the ungated spec/09 target of 185.0 USD")
    print("=" * 80)
    for name in ("IEEE-CIS (real)", "Prism Generator", "BankSim"):
        f = store[name][store[name].fraud == 1]
        mu = f.amount.mean()
        print(f"{name:16s} mean fraud amount {mu:10,.2f}   median {f.amount.median():9,.2f}"
              f"   (Prism Generator USD only; other corpora are not USD)")
    print()
    print("spec/09 tolerance_relative 0.45 on target 185.0 => acceptable band "
          f"{185*0.55:,.1f} to {185*1.45:,.1f} USD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())