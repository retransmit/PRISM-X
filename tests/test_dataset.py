"""Dataset validation tests for the ULB credit card fraud detection dataset."""

import pandas as pd
import pytest

from prism import config


# Resolved through config.yaml against the project root, the same way
# prism.data.loader finds it. A path relative to the current directory
# would make the skip below depend on where pytest was invoked from.
DATASET_PATH = config.get_path("tracks.ulb.raw_csv")

# The ULB download is gitignored, so it is absent on a fresh clone. These
# checks are about the real dataset and cannot be meaningfully faked, so
# they skip rather than fail, and the rest of the suite stays runnable
# without a 150MB download. They still run in full wherever it is present.
pytestmark = pytest.mark.skipif(
    not DATASET_PATH.exists(),
    reason="data/raw/creditcard.csv not present (Kaggle ULB download, gitignored)",
)

EXPECTED_COLUMNS = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
    "Class",
]


def load_dataset() -> pd.DataFrame:
    assert DATASET_PATH.exists(), f"Dataset not found at: {DATASET_PATH}"
    return pd.read_csv(DATASET_PATH)


def test_dataset_exists():
    """Verify that the raw CSV is present in data/raw/."""
    assert DATASET_PATH.exists(), "creditcard.csv is missing from data/raw/"


def test_dataset_shape():
    """Verify dataset dimensions: exactly 284,807 rows and 31 columns."""
    df = load_dataset()
    assert df.shape == (284807, 31), f"Unexpected dataset shape: {df.shape}"


def test_dataset_columns():
    """Verify column names match Time, V1-V28, Amount, Class."""
    df = load_dataset()
    assert list(df.columns) == EXPECTED_COLUMNS


def test_no_missing_values():
    """Verify zero NaN/null values across the dataset."""
    df = load_dataset()
    assert df.isnull().sum().sum() == 0, "Dataset contains unexpected null values"


def test_target_values():
    """Verify binary targets {0, 1}."""
    df = load_dataset()
    assert set(df["Class"].unique()) == {0, 1}


def test_fraud_count():
    """Verify exact fraud count in the ULB benchmark dataset."""
    df = load_dataset()
    fraud_count = int(df["Class"].sum())
    assert fraud_count == 492, f"Expected 492 fraud cases, found {fraud_count}"


def test_normal_count():
    """Verify exact legitimate transaction count."""
    df = load_dataset()
    normal_count = int((df["Class"] == 0).sum())
    assert normal_count == 284315, f"Expected 284,315 normal cases, found {normal_count}"


def test_numeric_columns():
    """Verify all features are numeric float/int types."""
    df = load_dataset()
    assert all(
        pd.api.types.is_numeric_dtype(df[col]) for col in EXPECTED_COLUMNS
    ), "All columns must have numeric dtypes"


def test_duplicate_report(capsys):
    """Report duplicate rows informatively rather than failing.
    
    In financial transaction logs, identical timestamps, amounts, and PCA features 
    can occur naturally or due to logging redundancy, so they should be examined 
    before arbitrary deletion.
    """
    df = load_dataset()
    duplicate_count = int(df.duplicated().sum())
    print(f"\n[EDA Note] Duplicate rows detected: {duplicate_count:,} out of {len(df):,}")
    assert duplicate_count >= 0