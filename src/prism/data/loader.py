"""Loading and schema validation for the ULB credit card dataset.

Expected shape: 284,807 rows x 31 columns: Time, V1..V28, Amount, Class.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from prism import config

PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]
EXPECTED_COLUMNS = ["Time", *PCA_COLUMNS, "Amount", "Class"]


def load_raw(path: Path | None = None, *, validate: bool = True) -> pd.DataFrame:
    """Read creditcard.csv and assert it is the dataset we think it is."""
    csv_path = path or config.get_path("tracks.ulb.raw_csv")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}.\n"
            "Download 'Credit Card Fraud Detection' (ULB) from Kaggle and place "
            "creditcard.csv in data/raw/."
        )

    df = pd.read_csv(csv_path)
    if validate:
        _validate(df)
    return df


def _validate(df: pd.DataFrame) -> None:
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    # The proposal (§7 Module 1) commits to a null-value assertion. This dataset
    # ships clean; if that stops being true, the scaler and SMOTE both silently
    # misbehave, so fail loudly here rather than debug it three modules later.
    null_counts = df[EXPECTED_COLUMNS].isna().sum()
    offenders = null_counts[null_counts > 0]
    if not offenders.empty:
        raise ValueError(f"Null values present in: {offenders.to_dict()}")

    labels = set(df["Class"].unique().tolist())
    if not labels <= {0, 1}:
        raise ValueError(f"Class column must be binary 0/1, found: {sorted(labels)}")


def describe_balance(df: pd.DataFrame, target: str = "Class") -> dict[str, float | int]:
    """Class-distribution summary for the EDA section of the report."""
    counts = df[target].value_counts()
    n_total = int(len(df))
    n_fraud = int(counts.get(1, 0))
    n_normal = int(counts.get(0, 0))
    return {
        "n_total": n_total,
        "n_normal": n_normal,
        "n_fraud": n_fraud,
        "fraud_rate": n_fraud / n_total if n_total else 0.0,
        # The value XGBoost's scale_pos_weight defaults to.
        "imbalance_ratio": n_normal / n_fraud if n_fraud else float("inf"),
    }


if __name__ == "__main__":
    frame = load_raw()
    stats = describe_balance(frame)
    print(f"rows      : {stats['n_total']:,}")
    print(f"normal    : {stats['n_normal']:,}")
    print(f"fraud     : {stats['n_fraud']:,}")
    print(f"fraud rate: {stats['fraud_rate']:.5%}")
    print(f"imbalance : 1 : {stats['imbalance_ratio']:.1f}")
