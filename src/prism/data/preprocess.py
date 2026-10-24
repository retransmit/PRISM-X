"""Module 1: leakage-free splitting and scaling.

READ BEFORE CHANGING THE ORDER OF OPERATIONS.

The proposal (§3.2, §5.3) stakes its evaluation validity on two rules:

  1. Split BEFORE fitting anything. A StandardScaler fitted on the full dataset
     has already seen the test set's mean and variance. Every metric downstream
     is then optimistic, and the examiner's first question is why.

  2. Never resample the test set. SMOTE belongs inside the cross-validation
     folds (see ``prism.training.resampling``), not here, and never on data
     you intend to report metrics against.

This module therefore does, strictly in order:
    drop exact duplicates -> stratified split -> fit scaler on TRAIN ONLY
    -> transform train/val/test.

The validation split exists so the decision threshold can be tuned without
touching the test set. The test set is read exactly once, at final evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from prism import config, tracks

# The ordinal code a categorical value unseen in training receives.
UNKNOWN_CODE = -1


@dataclass(frozen=True)
class SplitData:
    """The output of :func:`prepare_splits`."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    scaler: StandardScaler
    scale_columns: list[str]
    duplicates_dropped: int = 0
    encoder: OrdinalEncoder | None = None
    categorical_columns: tuple[str, ...] = ()
    track: str = "ulb"

    @property
    def feature_names(self) -> list[str]:
        return list(self.X_train.columns)

    def summary(self) -> str:
        def line(name: str, y: pd.Series) -> str:
            n, pos = len(y), int(y.sum())
            return f"  {name:<6} n={n:>7,}  fraud={pos:>4,}  rate={pos / n:.5%}"

        return "\n".join(
            [
                f"Split summary ({'time-ordered' if self.track != 'ulb' else 'stratified, fraud rate preserved in each'}):",
                line("train", self.y_train),
                line("val", self.y_val),
                line("test", self.y_test),
                f"  duplicates dropped before splitting: {self.duplicates_dropped:,}",
            ]
        )

    def as_dict(self) -> dict[str, Any]:
        """The split summary as the results generator reads it: what summary()
        prints, as numbers rather than a log line."""

        def window(y: pd.Series) -> dict[str, int]:
            return {"rows": int(len(y)), "fraud": int(y.sum())}

        return {
            "track": self.track,
            "rule": "time-ordered" if tracks.get(self.track).split == "time_ordered" else "stratified",
            "train": window(self.y_train),
            "val": window(self.y_val),
            "test": window(self.y_test),
            "duplicates_dropped": int(self.duplicates_dropped),
            "n_features": len(self.feature_names),
            "categorical_columns": list(self.categorical_columns),
        }


def prepare_splits(
    df: pd.DataFrame,
    *,
    schema: tracks.TrackSchema | None = None,
    test_size: float | None = None,
    val_size: float | None = None,
    random_state: int | None = None,
) -> SplitData:
    """Split, then encode and scale using train-only statistics.

    ULB: stratified, as the proposal describes. India: by time (design D4),
    because fraud campaigns cluster and a shuffled split leaks a campaign's
    own rows into training.
    """
    schema = schema or tracks.active()
    target = schema.target_column
    scale_columns = list(schema.scale_columns)
    random_state = random_state if random_state is not None else config.seed()

    # Exact duplicate rows must go before the split, or one copy can land in
    # train and another in test and the model is scored on a row it has seen.
    # The real ULB file carries 1,081 of them, 19 among the frauds; left in,
    # 298 test rows were byte-for-byte copies of training rows.
    n_before = len(df)
    df = df.drop_duplicates()
    duplicates_dropped = n_before - len(df)

    absent = [c for c in [*schema.feature_columns, target] if c not in df.columns]
    if absent:
        raise ValueError(f"Frame is missing columns the {schema.name} track needs: {absent}")
    X = df[schema.feature_columns].copy()
    y = df[target].astype(int)

    if schema.split == "time_ordered":
        order = np.argsort(df[schema.time_column].to_numpy(dtype=float), kind="stable")
        X, y = X.iloc[order], y.iloc[order]
        n_train = int(len(X) * float(schema.setting("train_fraction")))
        n_val = int(len(X) * float(schema.setting("val_fraction")))
        X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
        X_val, y_val = X.iloc[n_train:n_train + n_val], y.iloc[n_train:n_train + n_val]
        X_test, y_test = X.iloc[n_train + n_val:], y.iloc[n_train + n_val:]
        for name, labels in (("training window", y_train), ("validation window", y_val), ("test window", y_test)):
            if int(labels.sum()) == 0:
                raise ValueError(
                    f"The {name} holds no fraud: a time-ordered split of this export cannot tune or "
                    "score a threshold. Use a longer export or a higher stated boost."
                )
    else:
        test_size = test_size if test_size is not None else schema.setting("test_size")
        val_size = val_size if val_size is not None else schema.setting("val_size")
        # --- Step 1: hold out the test set. Nothing below ever refits on it. ---
        X_fit, X_test, y_fit, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        # --- Step 2: carve validation out of what remains (for threshold tuning). ---
        X_train, X_val, y_train, y_val = train_test_split(
            X_fit, y_fit, test_size=val_size, stratify=y_fit, random_state=random_state
        )

    # --- Codes and scaling statistics come from TRAIN ONLY; validation and
    # test are transformed with them, never fitted on. V1-V28 are already PCA
    # outputs and are left untouched on the ULB track; only the columns the
    # schema names are standardised. ---
    missing = [c for c in scale_columns if c not in X.columns]
    if missing:
        raise ValueError(f"scale_columns not present in data: {missing}")
    X_train, encoder, scaler = fit_preprocessing(X_train, schema)
    X_val = apply_preprocessing(X_val, schema, encoder, scaler)
    X_test = apply_preprocessing(X_test, schema, encoder, scaler)
    categorical = tuple(schema.categorical_columns)

    return SplitData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        scaler=scaler,
        scale_columns=scale_columns,
        duplicates_dropped=duplicates_dropped,
        encoder=encoder,
        categorical_columns=categorical,
        track=schema.name,
    )


def _apply_encoder(
    X: pd.DataFrame, encoder: OrdinalEncoder, columns: tuple[str, ...]
) -> pd.DataFrame:
    out = X.copy()
    out[list(columns)] = encoder.transform(X[list(columns)].astype(str)).astype(np.int64)
    return out


def _apply_scaler(
    X: pd.DataFrame, scaler: StandardScaler, columns: list[str]
) -> pd.DataFrame:
    out = X.copy()
    out[columns] = scaler.transform(X[columns])
    return out


def fit_preprocessing(
    X: pd.DataFrame, schema: tracks.TrackSchema
) -> tuple[pd.DataFrame, OrdinalEncoder | None, StandardScaler]:
    """Fit the encoder and the scaler on ``X`` alone and return it transformed."""
    categorical = tuple(schema.categorical_columns)
    encoder = None
    out = X
    if categorical:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=UNKNOWN_CODE, dtype=np.int64
        ).fit(X[list(categorical)].astype(str))
        out = _apply_encoder(out, encoder, categorical)
    scale_columns = list(schema.scale_columns)
    scaler = StandardScaler().fit(out[scale_columns])
    return _apply_scaler(out, scaler, scale_columns), encoder, scaler


def apply_preprocessing(
    X: pd.DataFrame, schema: tracks.TrackSchema, encoder: OrdinalEncoder | None, scaler: StandardScaler
) -> pd.DataFrame:
    """Apply preprocessing fitted elsewhere to rows it was not fitted on."""
    out = X
    if schema.categorical_columns:
        out = _apply_encoder(out, encoder, tuple(schema.categorical_columns))
    return _apply_scaler(out, scaler, list(schema.scale_columns))


def transform_features(
    row: dict[str, float | str] | pd.DataFrame,
    scaler: StandardScaler,
    feature_names: list[str],
    scale_columns: list[str],
    encoder: OrdinalEncoder | None = None,
    categorical_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Apply the persisted training encoder and scaler to live inference input.

    The single preprocessing path for the API, so a streamed transaction is
    transformed byte-identically to the training data. Column order is forced
    to match the training frame: XGBoost and the autoencoder are both order-
    sensitive, and a reordered vector produces a wrong score silently rather
    than raising. A categorical value unseen in training becomes UNKNOWN_CODE.

    Returns a DataFrame rather than an array because SHAP needs column names to
    attribute contributions to features.
    """
    frame = pd.DataFrame([row]) if isinstance(row, dict) else row.copy()

    missing = [c for c in feature_names if c not in frame.columns]
    if missing:
        raise ValueError(f"Payload is missing required features: {missing}")

    categorical = list(categorical_columns)
    if categorical:
        if encoder is None:
            raise ValueError("categorical_columns given without the fitted encoder")
        frame[categorical] = encoder.transform(frame[categorical].astype(str)).astype(np.int64)
    numeric = [c for c in feature_names if c not in categorical]
    frame[numeric] = frame[numeric].astype(float)
    frame = frame[feature_names]
    frame[scale_columns] = scaler.transform(frame[scale_columns])
    return frame


def save_scaler(scaler: StandardScaler, path: Path | None = None) -> Path:
    target = path or tracks.active().model_dir / "scaler.joblib"
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, target)
    return target


def load_scaler(path: Path | None = None) -> StandardScaler:
    # joblib unpickles, which is arbitrary code execution on a hostile file.
    # Safe here: models/ contains only artifacts this project wrote itself and
    # is gitignored, never fetched from a remote or supplied by a user. If model
    # artifacts ever become downloadable, replace this with a checksum check.
    target = path or tracks.active().model_dir / "scaler.joblib"
    if not target.exists():
        raise FileNotFoundError(
            f"No fitted scaler at {target}. Run `python -m prism.training.train` first."
        )
    return joblib.load(target)


def save_encoder(encoder: OrdinalEncoder, path: Path | None = None) -> Path:
    target = path or tracks.active().model_dir / "encoder.joblib"
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, target)
    return target


def load_encoder(path: Path | None = None) -> OrdinalEncoder | None:
    """None when absent: the ULB track has no categoricals and no encoder file.

    joblib unpickles; the same trust model as load_scaler applies: models/ holds
    only artifacts this project wrote itself and is never fetched or uploaded.
    """
    target = path or tracks.active().model_dir / "encoder.joblib"
    if not target.exists():
        return None
    return joblib.load(target)
