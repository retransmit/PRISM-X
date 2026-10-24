"""Bootstrap confidence intervals on every transfer ROC, plus pairwise differences.

Why this exists
---------------
A transfer ROC quoted as a bare number is not a result. Prism Generator 0.4757, BankSim 0.4564
and PaySim 0.5269 were all reported without intervals, so three questions were
unanswerable:

  1. Is PaySim's 0.5269 actually above chance, or is it noise around 0.5000? PaySim's
     per-entity amount ratio is identically 1.000 by construction (one transaction per
     fraud entity), so a model trained on it carries almost no amount signal and should
     not reliably clear chance.
  2. Does Prism Generator genuinely beat BankSim, or is a ~0.02 gap inside the noise?
  3. Is the e-commerce corpus's apparent deficit real?

Every ROC is bootstrapped over test ROWS, resampling the same rows for every generator so
that paired differences share the resample and the interval on the difference is
meaningfully narrower than an unpaired one would be.

A generator whose interval contains 0.5000 is reported as indistinguishable from chance,
and NOT ranked. That is the point of running this.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("cg", Path(__file__).parent / "compare_generators.py")
cg = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(cg)

RNG = np.random.default_rng(2026)
B = 2000


def model():
    return lgb.LGBMClassifier(n_estimators=120, learning_rate=0.05, random_state=42,
                              n_jobs=2, verbose=-1)


def paired_bootstrap(y, preds, B, rng):
    """Same resample applied to every predictor, so differences are paired.

    Drawn one iteration at a time rather than as a single (B, n) index matrix: at
    B=2000 and n=118k that matrix alone would be ~1.9 GB, and this machine has 14 GB
    with most of it already taken.
    """
    n = len(y)
    out = np.empty((len(preds), B))
    preds = [np.asarray(p) for p in preds]
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if np.unique(yb).size < 2:
            out[:, b] = 0.5
            continue
        for j, p in enumerate(preds):
            out[j, b] = roc_auc_score(yb, p[idx])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ieee", type=Path, required=True)
    ap.add_argument("--prism_generator", type=Path, required=True)
    ap.add_argument("--banksim", type=Path, required=True)
    ap.add_argument("--paysim", type=Path, required=True)
    a = ap.parse_args()

    rx, ry, rt = cg.load_ieee(a.ieee)
    order = np.argsort(rt, kind="mergesort")
    rx, ry = rx.iloc[order].reset_index(drop=True), ry[order]
    split = int(len(rx) * 0.8)
    feats = cg.BASIC

    r_tr, r_te = rx.iloc[:split][feats], rx.iloc[split:][feats]
    y_tr, y_te = ry[:split], ry[split:]
    ce = model().fit(r_tr, y_tr).predict_proba(r_te)[:, 1]

    names, preds = ["ceiling real->real"], [ce]
    for nm, path in [("BankSim", a.banksim), ("PaySim", a.paysim), ("Prism Generator", a.prism_generator)]:
        sx, sy, _ = cg.load_banksim(path) if nm == "BankSim" else (
            cg.load_paysim(path) if nm == "PaySim" else cg.load_prism_generator(path))
        names.append(nm)
        preds.append(model().fit(sx[feats], sy).predict_proba(r_te)[:, 1])

    draws = paired_bootstrap(np.asarray(y_te), preds, B, RNG)
    c_roc = roc_auc_score(y_te, ce)

    print(f"paired bootstrap over {B} resamples of {len(y_te):,} test rows\n")
    print(f"{'model':18s} {'ROC':>7s}  {'95% CI':>18s}   vs chance 0.5000")
    for i, nm in enumerate(names):
        d = draws[i]
        lo, hi = np.percentile(d, [2.5, 97.5])
        above = lo > 0.5
        verdict = "ABOVE chance" if above else ("BELOW chance" if hi < 0.5 else
                                                "indistinguishable from chance")
        print(f"{nm:18s} {d.mean():>7.4f}  [{lo:.4f}, {hi:.4f}]   {verdict}")

    print(f"\nretention vs ceiling ({c_roc:.4f}):")
    for i, nm in enumerate(names[1:], start=1):
        d = draws[i]
        print(f"  {nm:12s} {d.mean()/c_roc*100:5.1f}%   "
              f"[{np.percentile(d,2.5)/c_roc*100:.1f}%, {np.percentile(d,97.5)/c_roc*100:.1f}%]")

    print("\npaired differences (same resample, so the interval is on the difference itself):")
    # names: 0 ceiling, 1 BankSim, 2 PaySim, 3 Prism Generator
    for x, y in [(3, 1), (3, 2), (1, 2)]:
        diff = draws[x] - draws[y]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        sig = "DIFFERENT" if (lo > 0 or hi < 0) else "not separable at 95%"
        print(f"  {names[x]:16s} - {names[y]:16s} = {diff.mean():+.4f}  "
              f"[{lo:+.4f}, {hi:+.4f}]   {sig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())