"""XGBoost: the supervised, cost-sensitive detector (§5.2.3).

Class imbalance is handled here by ``scale_pos_weight`` rather than by
oversampling: it reweights the gradient contribution of the minority class
instead of fabricating synthetic rows. Left unset in config, it is computed at
fit time as n_negative / n_positive (~577.9 on the full ULB dataset).

This is usually the strongest of the three models, and the one whose SHAP
explanations are cheapest to compute (TreeExplainer is exact and fast on tree
ensembles, no sampling approximation needed).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from prism import config
from prism.models.base import ArrayLike, FraudModel


class XGBoostModel(FraudModel):
    name = "xgboost"

    def __init__(self, **overrides) -> None:
        params = dict(config.get("models.xgboost"))
        params.update(overrides)
        params.setdefault("random_state", config.seed())
        # None means "derive from the data at fit time".
        self._explicit_pos_weight = params.pop("scale_pos_weight", None)
        self.categorical_columns = tuple(params.pop("categorical_columns", ()))
        if self.categorical_columns:
            params["enable_categorical"] = True
        self.fitted_pos_weight: float | None = None
        self.params = params
        self._model: XGBClassifier | None = None
        self.feature_names: list[str] = []
        # Codes seen in training, per column. Fixed categories keep the pandas
        # category codes identical between a training frame and a single
        # served row; an unseen code (-1) becomes NaN, which XGBoost treats as
        # missing rather than as a category of its own.
        self.categories: dict[str, list[int]] = {}

    def with_categories(self, X: ArrayLike) -> ArrayLike:
        """The frame XGBoost sees: ordinal codes as pandas categoricals with the
        training categories, for the classifier and the explainer alike."""
        if not self.categorical_columns or not isinstance(X, pd.DataFrame):
            return X
        frame = X.copy()
        for column in self.categorical_columns:
            known = self.categories[column]
            codes = frame[column].astype(int)
            # A code unseen in training is missing to XGBoost. It is masked
            # here rather than left for pandas to reject: pandas 3 deprecates
            # a Categorical built from values outside its categories.
            frame[column] = pd.Categorical(codes.where(codes.isin(known)), categories=known)
        return frame

    def fit(self, X: ArrayLike, y: pd.Series | np.ndarray | None = None):
        if y is None:
            raise ValueError("XGBoostModel is supervised: y is required.")

        y_arr = np.asarray(y).astype(int)
        pos = int(y_arr.sum())
        neg = int(len(y_arr) - pos)
        if pos == 0:
            raise ValueError("Training set contains no fraud cases.")

        weight = self._explicit_pos_weight or (neg / pos)
        self.fitted_pos_weight = float(weight)
        self.feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else []
        if self.categorical_columns:
            self.categories = {
                c: sorted(int(v) for v in pd.unique(X[c]) if int(v) >= 0) for c in self.categorical_columns
            }
        self._model = XGBClassifier(scale_pos_weight=weight, **self.params).fit(self.with_categories(X), y_arr)
        return self

    def risk_score(self, X: ArrayLike) -> np.ndarray:
        assert self._model is not None, "call fit() first"
        return self._model.predict_proba(self.with_categories(X))[:, 1]

    @property
    def booster(self):
        """The raw booster, for shap.TreeExplainer."""
        assert self._model is not None, "call fit() first"
        return self._model

    def save(self, model_dir: Path) -> Path:
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / f"{self.name}.json"
        # XGBoost's native JSON format: portable and pickle-free.
        assert self._model is not None, "call fit() first"
        self._model.save_model(path)
        (model_dir / f"{self.name}.meta.json").write_text(
            json.dumps(
                {
                    "feature_names": self.feature_names,
                    "categorical_columns": list(self.categorical_columns),
                    "categories": self.categories,
                }
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, model_dir: Path) -> "XGBoostModel":
        path = model_dir / f"{cls.name}.json"
        if not path.exists():
            raise FileNotFoundError(f"No trained XGBoost model at {path}")
        instance = cls()
        model = XGBClassifier()
        model.load_model(path)
        instance._model = model

        meta_path = model_dir / f"{cls.name}.meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            instance.feature_names = meta["feature_names"]
            instance.categorical_columns = tuple(meta.get("categorical_columns", ()))
            instance.categories = {k: [int(c) for c in v] for k, v in meta.get("categories", {}).items()}
            if instance.categorical_columns:
                instance.params["enable_categorical"] = True
        return instance
