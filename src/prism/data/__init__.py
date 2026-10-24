"""Module 1: data loading, exploratory analysis and leakage-free preprocessing."""

from prism.data.eda import EDA_FIGURES, generate_eda_report
from prism.data.loader import describe_balance, load_raw
from prism.data.preprocess import (
    SplitData,
    load_scaler,
    prepare_splits,
    save_scaler,
    transform_features,
)

__all__ = [
    "load_raw",
    "describe_balance",
    "EDA_FIGURES",
    "generate_eda_report",
    "SplitData",
    "prepare_splits",
    "transform_features",
    "load_scaler",
    "save_scaler",
]
