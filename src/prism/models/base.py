"""A single interface over three very different detectors.

Isolation Forest returns a signed isolation score, the autoencoder returns a
reconstruction error in arbitrary units, and XGBoost returns a calibrated
probability. The API and the benchmark both need to treat them uniformly, so
every model exposes:

    fit(X, y)          -> self          (y ignored by the unsupervised models)
    risk_score(X)      -> float in [0, 1], higher = more suspicious
    save(dir) / load(dir)

``risk_score`` is the contract that makes threshold tuning comparable across
models. For XGBoost it is the raw positive-class probability; for the other two
it is a logistic squash of the native score, calibrated on the training
distribution at fit time; ``ScoreCalibrator`` below says why min-max was abandoned.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

ArrayLike = np.ndarray | pd.DataFrame



class FraudModel(ABC):
    """Common interface for every detector in the comparison."""

    name: str = "base"

    @abstractmethod
    def fit(self, X: ArrayLike, y: pd.Series | np.ndarray | None = None) -> "FraudModel":
        ...

    @abstractmethod
    def risk_score(self, X: ArrayLike) -> np.ndarray:
        """Return per-row risk in [0, 1]; higher means more likely fraud."""

    def predict(self, X: ArrayLike, threshold: float = 0.5) -> np.ndarray:
        """Binary labels at a given decision threshold."""
        return (self.risk_score(X) >= threshold).astype(int)

    @abstractmethod
    def save(self, model_dir: Path) -> Path:
        ...

    @classmethod
    @abstractmethod
    def load(cls, model_dir: Path) -> "FraudModel":
        ...

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


class ScoreCalibrator:
    """Maps an unbounded anomaly score into the open interval (0, 1).

    The transform is strictly monotonic, so it changes no ranking: PR-AUC,
    ROC-AUC and the precision-recall curve are identical before and after.
    Calibration exists to make the number interpretable and comparable across
    models, not to improve them.

    A min-max clip between the 1st and 99th training percentiles was tried
    first and rejected. Fraud is ~0.2% of the data, so the 99th percentile of
    training scores is still a *normal* transaction, every genuine anomaly
    clipped to exactly 1.0. Two things broke: the dashboard could no longer
    rank alerts by severity (all maximal), and the mass of tied scores depressed
    PR-AUC, which resolves ties pessimistically.

    An algebraic sigmoid centred on the upper tail keeps the ordering intact
    and never saturates or ties: the logistic form used before was clipped at
    30 standard units and tied every score beyond it, which left the strongest
    anomalies as one unordered block.
    """

    def __init__(self) -> None:
        self.center: float = 0.0
        self.scale: float = 1.0

    def fit(self, scores: np.ndarray) -> "ScoreCalibrator":
        scores = np.asarray(scores, dtype=float)
        # Centre where "unusual" begins, so ordinary traffic sits well below 0.5
        # and the tail, where fraud lives, gets the resolution.
        self.center = float(np.percentile(scores, 99))
        spread = self.center - float(np.median(scores))
        # Degenerate case: a constant score distribution.
        self.scale = float(spread / 2.0) if spread > 1e-12 else 1e-6
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        z = (np.asarray(scores, dtype=float) - self.center) / self.scale
        # An algebraic sigmoid: strictly increasing everywhere, inside (0, 1)
        # for every finite z, never flat. The logistic form it replaces was
        # clipped at |z| = 30 and tied every score beyond it.
        return 0.5 * (1.0 + z / (1.0 + np.abs(z)))

    def state(self) -> dict[str, float | str]:
        return {"center": self.center, "scale": self.scale, "form": "algebraic"}

    @classmethod
    def from_state(cls, state: dict[str, float | str]) -> "ScoreCalibrator":
        if state.get("form") != "algebraic":
            raise ValueError(
                "This model's scores were calibrated with the earlier logistic squash; its tuned "
                "threshold does not apply to the current form. Retrain the model."
            )
        calibrator = cls()
        calibrator.center = float(state["center"])
        calibrator.scale = float(state["scale"])
        return calibrator


class CategoricalExpander:
    """One-hot expansion of ordinal-coded categorical columns, fitted on training rows.

    The Isolation Forest and the autoencoder read every input as a number, and
    an ordinal code would place one channel numerically between two unrelated
    ones. Each code seen in training becomes an indicator column. A code
    unseen at serving (the encoder's -1) lights no indicator, which is what
    "unknown" should mean to a distance or a reconstruction.
    """

    def __init__(self, columns: Sequence[str] = ()) -> None:
        self.columns = list(columns)
        self.categories: dict[str, list[int]] = {}
        self.numeric_width = 0

    def fit(self, X: pd.DataFrame) -> "CategoricalExpander":
        self.numeric_width = int(X.shape[1] - len(self.columns))
        for column in self.columns:
            self.categories[column] = sorted(int(v) for v in pd.unique(X[column]) if int(v) >= 0)
        return self

    @property
    def output_dim(self) -> int:
        return self.numeric_width + sum(len(v) for v in self.categories.values())

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            if self.columns:
                raise ValueError("categorical columns need a DataFrame with column names")
            return np.asarray(X, dtype=np.float32)
        blocks = [X.drop(columns=self.columns).to_numpy(dtype=np.float32)]
        for column in self.columns:
            codes = X[column].to_numpy()
            known = self.categories[column]
            block = np.zeros((len(X), len(known)), dtype=np.float32)
            for j, code in enumerate(known):
                block[:, j] = codes == code
            blocks.append(block)
        return np.hstack(blocks)

    def state(self) -> dict:
        return {
            "columns": list(self.columns),
            "numeric_width": self.numeric_width,
            "categories": {k: list(v) for k, v in self.categories.items()},
        }

    @classmethod
    def from_state(cls, state: dict) -> "CategoricalExpander":
        expander = cls(state.get("columns", ()))
        expander.numeric_width = int(state.get("numeric_width", 0))
        expander.categories = {k: [int(c) for c in v] for k, v in state.get("categories", {}).items()}
        return expander
