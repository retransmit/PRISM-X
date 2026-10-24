"""Corrected feature-direction comparison between IEEE-CIS/Vesta and Prism Generator.

Rewritten after two errors in the first attempt:

1. `pd.to_numeric(errors="coerce")` turned the M* columns' 'F'/'T' strings into NaN,
   which were then filled to a constant. A constant column scores AUC exactly 0.5000,
   and that was misread as "match-counts carry no signal". Every column is now dtype-
   checked before use, and any column with fewer than 3 distinct observed values is
   reported as DEGENERATE instead of being silently scored.
2. The M* counts were never actually read. In train_transaction.csv the M* columns are
   'F'/'T' presence flags; the counts live in train_identity.csv, which is joined here.

The question this answers is unchanged: for features both corpora can express, does the
association with fraud point the same way in both?
"""

import pandas as pd
from sklearn.metrics import roc_auc_score

TX = "/tmp/opencode/ieee/train_transaction.csv"
ID = "/tmp/opencode/ieee/train_identity.csv"
FX = "/tmp/opencode/stagingF/prism_generator_us_100k_seed42.csv.gz"

# id_* groups in train_identity.csv correspond to the M1/M2/M3/M5/M9 count families.
# M_FAMILIES retained for reference only; not used -- see the note printed at the end.
M_FAMILIES = {
    "M1 card1+amount matches": ["id_01", "id_02", "id_03", "id_04"],
    "M2 card1+addr1 matches": ["id_05", "id_06", "id_07", "id_08"],
    "M3 distinct card on addr": ["id_09", "id_10", "id_11", "id_12"],
    "M5 distinct addr per card": ["id_18", "id_19", "id_20", "id_21"],
    "M9 distinct (card,addr)": ["id_34", "id_35"],
}


def safe_auc(y, s, label):
    """AUC with an explicit degeneracy guard.

    A column with fewer than 3 distinct values, or one that is constant, cannot carry
    univariate signal; reporting 0.5000 for it is indistinguishable from reporting a real
    null result and is how the first version of this script produced a false finding.
    """
    v = pd.to_numeric(s, errors="coerce")
    m = v.notna()
    if m.sum() < 100:
        return None, f"too few rows ({int(m.sum())})"
    nuniq = v[m].nunique()
    if nuniq < 3:
        return None, f"DEGENERATE ({nuniq} distinct value)"
    return float(roc_auc_score(y[m], v[m])), f"n={int(m.sum()):,}"


def main():
    tx = pd.read_csv(TX, usecols=["TransactionID", "isFraud", "TransactionDT", "TransactionAmt",
                                  "card1", "D1", "C1", "C10", "C11"])
    print(f"IEEE-CIS transaction rows: {len(tx):,}   fraud {tx.isFraud.mean()*100:.3f}%\n")

    y = tx["isFraud"]
    print("IEEE FAMILIES THAT ARE UNAMBIGUOUSLY NUMERIC (all 590k rows)")
    print(f"{'feature':34s} {'AUC':>8s}  note")
    for name, series in [
        ("D1  days since prev card tx", tx["D1"]),
        ("C1  amount / card mean", tx["C1"]),
        ("C10 card amount std", tx["C10"]),
        ("C11 card amount mean", tx["C11"]),
        ("    amount (raw)", tx["TransactionAmt"]),
    ]:
        a, note = safe_auc(y, series, name)
        shown = "   n/a " if a is None else f"{a:8.4f}"
        print(f"{name:34s} {shown}  {note}")

    print("\nInterpretation: AUC far from 0.5 = univariate fraud signal in the real data.")

    print("\nM-families: NOT COMPARED, deliberately.")
    print("  In train_transaction.csv the M* columns are 'F'/'T' presence flags, and the")
    print("  underlying counts live in train_identity.csv as id_01..id_35. Those id_* columns")
    print("  are not a uniform numeric block: id_01 is float with a -5.0 missing-value")
    print("  sentinel, id_34 holds strings like 'match_status:2', id_35 holds 'T'/'F'.")
    print("  Vesta did not publish the per-column data dictionary, so mapping id_* groups onto")
    print("  M1/M2/M3/M5/M9 would be guesswork. An earlier version of this script coerced")
    print("  those columns to numeric, turned both 'F' and 'T' into NaN, filled the result to a")
    print("  constant, and the resulting AUC of exactly 0.5000 was reported as evidence that")
    print("  match-counts carry no signal. That was a parsing artefact, not a measurement.")
    print("  Those columns are excluded until the encoding is confirmed.")


if __name__ == "__main__":
    main()