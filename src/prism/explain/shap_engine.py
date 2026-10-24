"""Module 4: SHAP explanations. This is the module the project is named for.

A Shapley value answers: how much did this feature push this prediction away
from the model's average prediction? Summed over all features, the attributions
plus the base value reconstruct the model output exactly, which is what makes
them defensible to a regulator rather than merely suggestive.

Explainer per model, each exact and additive in the model's own space:

  xgboost             TreeExplainer, or XGBoost's own TreeSHAP when the model
                      has categorical splits (log-odds)
  isolation_forest    scikit-learn TreeSHAP on the average path length, sign
                      flipped so a positive push means more suspicious
  autoencoder         per-feature reconstruction error (base zero)

One-hot blocks are summed back onto their column, so every explanation is keyed
by the track's feature names. Every explanation ends in reason codes
(prism.explain.reasons).

--------------------------------------------------------------------------
CAVEAT THAT MUST REACH THE REPORT (§3.3 promises human-readable explanations):
V1-V28 are anonymised PCA components. SHAP will correctly report "V14 drove
this alert", a true statement that explains nothing to a fraud analyst. Only
Time and Amount carry real-world meaning. Decide before Phase 5 how you are
handling this; see the README section "Two things that will come up at defence".
--------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import shap

from prism import tracks
from prism.explain.reasons import reason_codes
from prism.models.base import FraudModel

# Features whose names mean something to a human. Everything else is a latent
# PCA component and should be presented as such in the UI.
INTERPRETABLE_FEATURES = {"Time", "Amount"}


def group_expanded(values: np.ndarray, expander, feature_names: list[str]) -> np.ndarray:
    """Attributions over the expanded (one-hot) matrix summed back per feature: the
    numerics keep their place, each one-hot block collapses onto its column's name, so an
    explanation speaks in the track's feature names whichever code was lit."""
    values = np.atleast_2d(np.asarray(values, dtype=float))
    categorical = list(expander.columns)
    numeric = [name for name in feature_names if name not in categorical]
    out = np.zeros((values.shape[0], len(feature_names)), dtype=float)
    for j, name in enumerate(numeric):
        out[:, feature_names.index(name)] = values[:, j]
    offset = len(numeric)
    for column in categorical:
        width = len(expander.categories[column])
        out[:, feature_names.index(column)] = values[:, offset:offset + width].sum(axis=1)
        offset += width
    return out


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    value: float
    shap_value: float          # signed: >0 pushes toward fraud
    interpretable: bool        # False => anonymised PCA component
    display: str | None = None # the raw categorical value as text, None for numerics
    label: str | None = None   # the track's display name for the feature, None without a schema

    def as_dict(self) -> dict:
        return asdict(self)


class NativeXGBoostExplainer:
    """Exact TreeSHAP computed by XGBoost itself, for models with categorical splits.

    shap.TreeExplainer walks the trees on its own and does not follow
    categorical splits; XGBoost's pred_contribs does, and is the same
    algorithm. The last column of its output is the bias, which is the
    expected value the Explanation reports as base_value.
    """

    def __init__(self, model) -> None:
        self._model = model
        self.expected_value = 0.0

    def shap_values(self, frame: pd.DataFrame) -> np.ndarray:
        import xgboost

        matrix = xgboost.DMatrix(self._model.with_categories(frame), enable_categorical=True)
        contributions = self._model.booster.get_booster().predict(matrix, pred_contribs=True)
        self.expected_value = float(contributions[0, -1])
        return contributions[:, :-1]


class IsolationExplainer:
    """scikit-learn TreeSHAP on the Isolation Forest's average path length, which falls
    as a row looks more anomalous; the sign is flipped so a positive push means more
    suspicious, and the explainer's space is the negative path length."""

    def __init__(self, model, feature_names: list[str]) -> None:
        self._model = model
        self._feature_names = feature_names
        self._explainer = shap.TreeExplainer(model.forest)
        self.expected_value = -float(np.ravel(self._explainer.expected_value)[0])

    def shap_values(self, frame: pd.DataFrame) -> np.ndarray:
        values = np.asarray(self._explainer.shap_values(self._model.expander.transform(frame)))
        return -group_expanded(values, self._model.expander, self._feature_names)


class ReconstructionExplainer:
    """The autoencoder's per-feature reconstruction error: the share of the row's error
    each column carries, summed back from the one-hot blocks; the base is zero and the
    prediction is the error the score was calibrated from."""

    expected_value = 0.0

    def __init__(self, model, feature_names: list[str]) -> None:
        self._model = model
        self._feature_names = feature_names

    def shap_values(self, frame: pd.DataFrame) -> np.ndarray:
        return group_expanded(self._model.squared_errors(frame), self._model.expander, self._feature_names)


@dataclass(frozen=True)
class Explanation:
    model: str
    base_value: float          # the model's average output
    prediction: float          # base_value + sum(shap_values)
    contributions: list[FeatureContribution]

    def top(self, k: int = 8) -> list[FeatureContribution]:
        """The k features with the largest absolute influence."""
        return sorted(self.contributions, key=lambda c: abs(c.shap_value), reverse=True)[:k]

    def concentration(self, k: int = 3) -> float:
        """Share of total absolute attribution held by the top k features.

        Lets the UI say "3 of 30 factors carry 71% of this score", a true and
        useful statement even when the factors themselves cannot be named.
        """
        total = sum(abs(c.shap_value) for c in self.contributions)
        if total == 0:
            return 0.0
        return float(sum(abs(c.shap_value) for c in self.top(k)) / total)

    def narrative(self, k: int = 3) -> str:
        """The attribution-only fallback, served when the reference artifacts
        are absent; with them the service writes the measured-facts narrative.

        Named features (Time, Amount) are described directly. V1-V28 are
        anonymised PCA components with no recoverable business meaning, so they
        are counted, never named: "V14 drove this alert" is true and useless,
        and the same policy holds for every sentence the service can serve.
        """
        drivers = self.top(k)
        if not drivers or all(c.shap_value == 0 for c in drivers):
            return "No feature moved this transaction away from the baseline score."

        direction = "higher" if sum(c.shap_value for c in drivers) >= 0 else "lower"
        named = [c for c in drivers if c.interpretable]
        latent = [c for c in drivers if not c.interpretable]

        parts = [
            f"{c.feature} = {c.value:,.2f} ({c.shap_value:+.3f})" for c in named
        ]
        if latent:
            noun = "factor" if len(latent) == 1 else "factors"
            parts.append(f"{len(latent)} anonymised {noun}")

        return (
            f"Risk pushed {direction} mainly by {len(drivers)} of "
            f"{len(self.contributions)} factors, carrying "
            f"{self.concentration(k):.0%} of the total attribution: "
            + "; ".join(parts)
            + "."
        )

    def as_dict(self, k: int = 8) -> dict:
        top = self.top(k)
        return {
            "model": self.model,
            "base_value": self.base_value,
            "prediction": self.prediction,
            "top_features": [c.as_dict() for c in top],
            "narrative": self.narrative(k=min(3, k)),
            # Lets the UI say "3 of 28 latent factors account for 71% of this
            # score" when the driving features have no human-readable name.
            "explained_fraction": self._explained_fraction(top),
            "interpretable_fraction": self._interpretable_fraction(top),
            # Short stable codes per driver family, whichever model produced the score.
            "reason_codes": reason_codes(self.contributions),
        }

    def _explained_fraction(self, top: list[FeatureContribution]) -> float:
        total = sum(abs(c.shap_value) for c in self.contributions)
        return float(sum(abs(c.shap_value) for c in top) / total) if total else 0.0

    def _interpretable_fraction(self, top: list[FeatureContribution]) -> float:
        total = sum(abs(c.shap_value) for c in top)
        named = sum(abs(c.shap_value) for c in top if c.interpretable)
        return float(named / total) if total else 0.0


class ShapEngine:
    """Wraps a shap explainer and formats output for the API and the dashboard."""

    def __init__(
        self,
        model: FraudModel,
        feature_names: list[str],
        explainer,
        schema: tracks.TrackSchema | None = None,
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.explainer = explainer
        # Which features carry a name a person can read. None keeps the ULB
        # rule (Time and Amount only) for callers that predate the tracks.
        self.schema = schema

    @classmethod
    def for_model(
        cls,
        model: FraudModel,
        feature_names: list[str],
        background: pd.DataFrame | None = None,
        schema: tracks.TrackSchema | None = None,
    ) -> "ShapEngine":
        # Dispatched on the registry name, so no model's dependency is imported here;
        # ``background`` is accepted for callers that predate the tree explainers.
        del background
        name = getattr(model, "name", "")
        if name == "xgboost":
            if model.categorical_columns:
                return cls(model, feature_names, NativeXGBoostExplainer(model), schema)
            return cls(model, feature_names, shap.TreeExplainer(model.booster), schema)
        if name == "isolation_forest":
            return cls(model, feature_names, IsolationExplainer(model, feature_names), schema)
        if name == "autoencoder":
            return cls(model, feature_names, ReconstructionExplainer(model, feature_names), schema)
        raise NotImplementedError(
            f"No explainer wired for {type(model).__name__}. See module docstring."
        )

    def explain(self, X: pd.DataFrame | np.ndarray, raw: dict | None = None) -> Explanation:
        """Explain a single transaction. ``raw`` is the unscaled input; when it
        is given, categorical contributions carry the raw value as text."""
        frame = (
            X if isinstance(X, pd.DataFrame)
            else pd.DataFrame(np.atleast_2d(X), columns=self.feature_names)
        )
        if len(frame) != 1:
            raise ValueError(f"explain() takes one row, got {len(frame)}.")

        values = np.asarray(self.explainer.shap_values(frame))
        # Binary classifiers may return (1, n_features) or (1, n_features, 2).
        if values.ndim == 3:
            values = values[..., -1]
        values = values.reshape(-1)

        base = self.explainer.expected_value
        base_value = float(np.ravel(base)[-1]) if np.ndim(base) else float(base)

        row = frame.iloc[0]
        categorical = set(getattr(self.model, "categorical_columns", ()))
        contributions = [
            FeatureContribution(
                feature=name,
                value=float(row[name]),
                shap_value=float(values[i]),
                interpretable=(
                    self.schema.is_interpretable(name) if self.schema else name in INTERPRETABLE_FEATURES
                ),
                display=(str(raw[name]) if raw is not None and name in categorical and name in raw else None),
                label=(self.schema.label(name) if self.schema else None),
            )
            for i, name in enumerate(self.feature_names)
        ]

        return Explanation(
            model=self.model.name,
            base_value=base_value,
            prediction=base_value + float(values.sum()),
            contributions=contributions,
        )
