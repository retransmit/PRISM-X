"""Diagnostic for the Axis 3 'fraud mean amt/entity' figure.

The fidelity run reported Prism Generator fraud mean amt/entity = 2.82 against real 1.18, and
BankSim 5.95. Two things need checking before either number is quoted:

  1. HOW MANY ROWS IS IT? A mean over a handful of fraud rows is not a measurement.
     Every corpus prints its own fraud count, its entity count and its transactions
     per fraud entity.

  2. THE DENOMINATOR INCLUDES THE ROW ITSELF. amt/entity is scaled by that entity's mean
     computed over all its rows, so on a fraud row the denominator already contains the
     numerator. That mechanically pulls the ratio toward 1 and is worst exactly where
     fraud entities have few legitimate rows. So a leave-one-out denominator is reported
     alongside, and the two are never mixed.

Bootstrap intervals are over rows, so they reflect sampling noise in the corpus and not
confidence in anything about the generator.
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

RNG = np.random.default_rng(42)
B = 2000


def boot_mean(x: np.ndarray) -> tuple[float, float, float]:
    """Mean with a percentile bootstrap 95% interval, resampling rows."""
    x = x[np.isfinite(x)]
    if x.size < 30:
        return float(x.mean()) if x.size else float("nan"), float("nan"), float("nan")
    idx = RNG.integers(0, x.size, size=(B, x.size))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load(name: str, path: Path) -> pd.DataFrame:
    c = CORPORA[name]
    raw = pd.read_csv(path, usecols=[c["amount"], c["entity"], c["label"]], low_memory=False)
    raw = raw.rename(columns={c["amount"]: "amount", c["entity"]: "entity",
                              c["label"]: "fraud"})
    raw["fraud"] = pd.to_numeric(raw["fraud"], errors="coerce").fillna(0).astype(int)
    raw["amount"] = pd.to_numeric(raw["amount"], errors="coerce")
    return raw.dropna(subset=["amount"])


def ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Both denominators: entity mean including the row, and leaving the row out."""
    g = df.groupby("entity")["amount"]
    incl = g.transform("mean")
    cnt = g.transform("count")
    # Leave-one-out denominator, guarded against a zero or negative result when an
    # entity has exactly one row (count-1 == 0, so fall back to the inclusive mean).
    loo = (g.transform("sum") - df["amount"]) / (cnt - 1).replace(0, np.nan)
    out = pd.DataFrame({
        "fraud": df["fraud"].to_numpy(),
        "incl": (df["amount"] / incl.replace(0, np.nan)).clip(0, 50).fillna(1.0),
        "loo": (df["amount"] / loo).clip(0, 50).fillna(1.0),
        "ent_n": cnt.to_numpy(),
    })
    return out


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

    print("=" * 78)
    print("SAMPLE SIZES -- a mean over a handful of rows is not a measurement")
    print("=" * 78)
    print(f"{'corpus':16s} {'rows':>10s} {'fraud':>8s} {'fraud%':>7s} "
          f"{'entities':>9s} {'fraud ents':>11s} {'txn/fraud-ent':>14s}")
    stats = {}
    for name, p in paths.items():
        df = load(name, p)
        r = ratios(df)
        fr = df["fraud"] == 1
        fe = r.loc[fr, "ent_n"]
        stats[name] = (r, int(fr.sum()))
        print(f"{name:16s} {len(df):>10,} {int(fr.sum()):>8,} {fr.mean()*100:>6.3f}% "
              f"{df['entity'].nunique():>9,} {fe.size:>11,} "
              f"{fe.mean():>14.2f}")
    print()
    print("A corpus whose fraud entities average ~1 transaction has no legitimate history")
    print("to be unusual relative to, so its ratio is close to 1 by construction.")
    print()

    ref = stats["IEEE-CIS (real)"][0].loc[stats["IEEE-CIS (real)"][0].fraud == 1, "incl"].to_numpy()
    rm, rlo, rhi = boot_mean(ref)

    print("=" * 78)
    print("AXIS 3 VALUE, both denominators, with bootstrap 95% intervals")
    print("=" * 78)
    print(f"real fraud mean amt/entity = {rm:.3f}  [{rlo:.3f}, {rhi:.3f}]   <- reference\n")
    print(f"{'corpus':16s} {'incl mean':>10s} {'incl 95% CI':>18s} "
          f"{'loo mean':>9s} {'loo 95% CI':>18s} {'x real':>7s}")
    for name in paths:
        if name == "IEEE-CIS (real)":
            continue
        r, _ = stats[name]
        f = r[r.fraud == 1]
        m1, l1, h1 = boot_mean(f["incl"].to_numpy())
        m2, l2, h2 = boot_mean(f["loo"].to_numpy())
        print(f"{name:16s} {m1:>10.3f} [{l1:.3f}, {h1:.3f}] "
              f"{m2:>9.3f} [{l2:.3f}, {h2:.3f}] {m1/rm:>6.2f}x")

    print()
    print("Reading the x real column against the reference interval: an 'x real' whose CI")
    print("spans 1.00 is not distinguishable from the real ratio.")
    print()
    print("=" * 78)
    print("TAIL SHAPE -- a few very large rows can carry a mean on their own")
    print("=" * 78)
    print(f"{'corpus':16s} {'fraud p50':>9s} {'p90':>8s} {'p99':>8s} {'max':>9s} "
          f"{'>5x share':>10s}")
    for name in paths:
        r, _ = stats[name]
        v = r.loc[r.fraud == 1, "incl"].to_numpy()
        v = v[np.isfinite(v)]
        q = lambda p: np.quantile(v, p)                            # noqa: E731
        print(f"{name:16s} {q(.5):>9.3f} {q(.9):>8.3f} {q(.99):>8.3f} {v.max():>9.3f} "
              f"{(v > 5).mean()*100:>9.2f}%")
    print()
    print("If a corpus's median is near 1 while its mean is several times real, the mean")
    print("is being carried by a tail and does not describe typical fraud in it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())