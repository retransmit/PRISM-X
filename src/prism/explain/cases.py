"""Case-based evidence for an alert: the nearest training transactions.

"Of the 50 most similar transactions the system learned from, 45 were
confirmed fraud" is the one clause of the explanation that needs no feature
names at all, and it is how a fraud investigator reasons about a case. The
index holds the scaled training rows exactly as the service scores them, so
a query at serving time is compared in the same representation the models
were fitted on.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from prism import config
from prism.models.base import CategoricalExpander


class CaseIndex:
    def __init__(
        self,
        neighbours: NearestNeighbors,
        labels: np.ndarray,
        k: int,
        columns: list[str],
        expander: CategoricalExpander | None = None,
    ) -> None:
        self._neighbours = neighbours
        self._labels = labels
        self._k = k
        self._columns = columns
        self._expander = expander or CategoricalExpander()

    @classmethod
    def fit(
        cls,
        X_train: pd.DataFrame,
        y_train,
        k: int | None = None,
        *,
        exclude: tuple[str, ...] = ("Time",),
        categorical_columns: tuple[str, ...] = (),
    ) -> "CaseIndex":
        """Index the scaled training rows. Brute force: one query against
        ~180k rows of 30 floats is a few milliseconds, and it runs only for
        flagged transactions.

        The clock column is excluded: a live transaction carries a clock far
        outside the training range, and if it counted toward distance it would
        dominate every query and match nothing. Similarity is about what a
        transaction looks like, not when it was. Categoricals enter one-hot,
        so the distance between two channels is the same whatever codes the
        encoder gave them.
        """
        k = int(k if k is not None else config.get("explain.similar_cases", 50))
        features = [c for c in X_train.columns if c not in exclude]
        expander = CategoricalExpander([c for c in categorical_columns if c in features]).fit(X_train[features])
        matrix = expander.transform(X_train[features])
        neighbours = NearestNeighbors(n_neighbors=k, algorithm="brute").fit(matrix)
        labels = np.asarray(y_train, dtype=np.int8)
        return cls(neighbours, labels, k, features, expander)

    def __setstate__(self, state: dict) -> None:
        """An index pickled before the categorical expander existed (a deployment's
        models/ can predate it) loads with an empty expander instead of crashing
        the service in its warm-up."""
        self.__dict__.update(state)
        self.__dict__.setdefault("_expander", CategoricalExpander())

    @property
    def n_rows(self) -> int:
        return len(self._labels)

    @property
    def k(self) -> int:
        return self._k

    @property
    def features(self) -> list[str]:
        """The columns distance is measured over. Never includes the clock column."""
        return list(self._columns)

    def query(self, frame: pd.DataFrame) -> dict:
        """Fraud count among the k nearest training rows to one scaled row."""
        matrix = self._expander.transform(frame[self._columns])
        _, index = self._neighbours.kneighbors(matrix)
        return {
            "similar_fraud": int(self._labels[index[0]].sum()),
            "similar_total": self._k,
            "base_rate": float(self._labels.mean()),
        }

    # joblib is a pickle. The only file ever loaded here is the one this class
    # wrote into models/ during training, the same trust model as scaler.joblib
    # and isolation_forest.joblib beside it; nothing is loaded from a request.
    def save(self, path: Path) -> Path:
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path) -> "CaseIndex":
        return joblib.load(path)
