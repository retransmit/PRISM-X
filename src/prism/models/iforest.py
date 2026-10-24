"""Isolation Forest: the unsupervised baseline (§5.2.1).

Isolates points by randomly partitioning the feature space; anomalies need
fewer splits to isolate and so sit at shallower average depth. It never sees a
fraud label during training, which makes it the honest baseline for "what could
we detect with no labelled fraud history at all?"
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from prism import config
from prism.models.base import ArrayLike, CategoricalExpander, FraudModel, ScoreCalibrator


class IsolationForestModel(FraudModel):
    name = "isolation_forest"

    def __init__(self, **overrides) -> None:
        params = dict(config.get("models.isolation_forest"))
        params.update(overrides)
        params.setdefault("random_state", config.seed())
        self.categorical_columns = tuple(params.pop("categorical_columns", ()))
        self.params = params
        self._model: IsolationForest | None = None
        self._calibrator = ScoreCalibrator()
        self._expander = CategoricalExpander(self.categorical_columns)

    def fit(self, X: ArrayLike, y: pd.Series | np.ndarray | None = None):
        # y is accepted for interface symmetry and deliberately unused; this
        # model is unsupervised by design, not by oversight.
        del y
        if isinstance(X, pd.DataFrame):
            self._expander.fit(X)
        matrix = self._expander.transform(X)
        self._model = IsolationForest(**self.params).fit(matrix)
        self._calibrator.fit(self._raw_scores(matrix))
        return self

    @property
    def forest(self) -> IsolationForest:
        """The fitted forest, for shap.TreeExplainer."""
        assert self._model is not None, "call fit() first"
        return self._model

    @property
    def expander(self) -> CategoricalExpander:
        return self._expander

    def _raw_scores(self, matrix: np.ndarray) -> np.ndarray:
        assert self._model is not None, "call fit() first"
        # score_samples: lower = more anomalous. Negate so higher = worse,
        # matching the risk_score contract.
        return -self._model.score_samples(matrix)

    def risk_score(self, X: ArrayLike) -> np.ndarray:
        return self._calibrator.transform(self._raw_scores(self._expander.transform(X)))

    def save(self, model_dir: Path) -> Path:
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / f"{self.name}.joblib"
        # Self-produced artifact only; see note in prism.data.preprocess.
        joblib.dump(
            {"model": self._model, "calibrator": self._calibrator.state(), "expander": self._expander.state()},
            path,
        )
        return path

    @classmethod
    def load(cls, model_dir: Path) -> "IsolationForestModel":
        path = model_dir / f"{cls.name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"No trained Isolation Forest at {path}")
        payload = joblib.load(path)
        instance = cls()
        instance._model = payload["model"]
        instance._calibrator = ScoreCalibrator.from_state(payload["calibrator"])
        instance._expander = CategoricalExpander.from_state(payload.get("expander", {}))
        instance.categorical_columns = tuple(instance._expander.columns)
        return instance
