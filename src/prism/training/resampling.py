"""Module 2/3: SMOTE, applied only where it is valid to apply it (§5.3).

The failure mode this module exists to prevent:

    BAD:   X, y = SMOTE().fit_resample(X, y)          # then split
           -> synthetic rows interpolated from a fraud case end up in BOTH the
              train and test sets. The model has effectively memorised test
              data. Reported recall can look near-perfect and means nothing.

    GOOD:  split first, then resample the training fold only, then score
           against the untouched validation fold.

``cross_val_scores`` below implements the good version explicitly rather than
hiding it inside a pipeline object, because at defence you want to be able to
point at the loop and show where the resampling happens.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, train_test_split

from prism import config, tracks
from prism.data.preprocess import apply_preprocessing, fit_preprocessing
from prism.evaluation.metrics import evaluate
from prism.evaluation.threshold import tune_threshold
from prism.models.base import FraudModel


def make_smote(categorical_indices: Sequence[int] = ()) -> SMOTE:
    """SMOTE, or SMOTENC when the frame carries categorical codes: SMOTE would
    interpolate a code halfway between two unrelated channels."""
    common = dict(
        sampling_strategy=config.get("resampling.sampling_strategy", 0.1),
        k_neighbors=config.get("resampling.k_neighbors", 5),
        random_state=config.seed(),
    )
    if categorical_indices:
        return SMOTENC(categorical_features=list(categorical_indices), **common)
    return SMOTE(**common)


def resample_training_fold(
    X: pd.DataFrame, y: pd.Series | np.ndarray, categorical_columns: Sequence[str] = ()
) -> tuple[pd.DataFrame, np.ndarray]:
    """Oversample the minority class.

    ONLY EVER CALL THIS ON A TRAINING FOLD. Passing validation or test data
    here invalidates every metric computed downstream.
    """
    if not config.get("resampling.enabled", True):
        return X, np.asarray(y).astype(int)

    indices = [list(X.columns).index(c) for c in categorical_columns] if isinstance(X, pd.DataFrame) else []
    X_resampled, y_resampled = make_smote(indices).fit_resample(X, np.asarray(y).astype(int))
    if isinstance(X_resampled, pd.DataFrame) and categorical_columns:
        X_resampled[list(categorical_columns)] = X_resampled[list(categorical_columns)].astype(np.int64)
    return X_resampled, np.asarray(y_resampled).astype(int)


def imbalance_overrides(resample_applied: bool) -> dict:
    """Model parameters that must accompany a resampled training set.

    SMOTE and scale_pos_weight answer the same question. XGBoostModel derives
    scale_pos_weight from the class ratio of whatever it is fitted on, so on a
    set already oversampled to a 1:10 ratio it would pass a further 10x weight
    and count the minority class twice. Pin the weight to 1.0 whenever the
    training rows were resampled; the weighted arm keeps the natural ratio.
    """
    return {"scale_pos_weight": 1.0} if resample_applied else {}


def model_factory(name: str, *, resample: bool, categorical_columns: Sequence[str] = ()) -> Callable[[], FraudModel]:
    """A factory whose models carry the arm's imbalance treatment: the class
    weight is pinned to one when the fold is resampled, so SMOTE and cost
    weighting are never stacked."""
    from prism.models import build

    overrides = imbalance_overrides(resample)
    if categorical_columns:
        overrides["categorical_columns"] = tuple(categorical_columns)
    return lambda: build(name, **overrides)


def cross_val_scores(
    model_factory: Callable[[], FraudModel],
    frame: pd.DataFrame,
    schema: tracks.TrackSchema,
    n_splits: int | None = None,
    *,
    resample: bool = True,
    verbose: bool = True,
) -> list[dict]:
    """K-fold CV over the raw training rows, each fold self-contained.

    Folds are forward-chaining on a time-ordered track and stratified
    otherwise. Inside each fold: the encoder and scaler are fitted on the
    fitting rows alone, an inner holdout tunes the threshold, the fitting
    rows are resampled if asked, and the fold's held-out rows are scored once.
    A fold whose holdout or tuning rows hold no fraud is skipped and named.
    """
    n_splits = n_splits or config.get("evaluation.cv_folds", 5)
    X_raw = frame[schema.feature_columns]
    y = frame[schema.target_column].astype(int)
    time_ordered = schema.split == "time_ordered"
    if time_ordered:
        order = np.argsort(frame[schema.time_column].to_numpy(dtype=float), kind="stable")
        X_raw, y = X_raw.iloc[order], y.iloc[order]
        splitter = TimeSeriesSplit(n_splits=n_splits)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.seed())
    holdout = float(config.get("evaluation.cv_threshold_holdout", 0.2))

    results: list[dict] = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X_raw, y), start=1):
        X_tr, y_tr = X_raw.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X_raw.iloc[val_idx], y.iloc[val_idx]
        # The threshold needs rows of its own: tuning it on the fold's held-out
        # portion and then scoring that portion would grade the model on the
        # cutoff it just chose for those rows. On a time-ordered track the
        # tuning rows are the latest of the fitting window.
        if time_ordered:
            cut = int(len(X_tr) * (1.0 - holdout))
            X_fit, X_tune, y_fit, y_tune = X_tr.iloc[:cut], X_tr.iloc[cut:], y_tr.iloc[:cut], y_tr.iloc[cut:]
        else:
            X_fit, X_tune, y_fit, y_tune = train_test_split(
                X_tr, y_tr, test_size=holdout, stratify=y_tr, random_state=config.seed()
            )
        if int(y_tune.sum()) == 0 or int(y_va.sum()) == 0:
            if verbose:
                print(f"  fold {fold}/{n_splits}: skipped, no fraud to tune or score on")
            continue

        # Preprocessing is fitted on the fitting rows alone and applied to the rest of the fold.
        X_fit, encoder, scaler = fit_preprocessing(X_fit, schema)
        X_tune = apply_preprocessing(X_tune, schema, encoder, scaler)
        X_va = apply_preprocessing(X_va, schema, encoder, scaler)
        if resample:
            X_fit, y_fit = resample_training_fold(X_fit, y_fit, categorical_columns=schema.categorical_columns)

        model = model_factory()
        model.fit(X_fit, y_fit)
        threshold = tune_threshold(
            y_tune, model.risk_score(X_tune), strategy=schema.setting("threshold_strategy", None)
        )
        result = evaluate(y_va, model.risk_score(X_va), threshold, model=f"{model.name}-fold{fold}")
        results.append(result.as_dict())
        if verbose:
            print(f"  fold {fold}/{n_splits}: {result}")
    return results
