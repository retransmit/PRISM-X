"""Module 4: SHAP explainability engine."""

from prism.explain.shap_engine import (
    Explanation,
    FeatureContribution,
    ShapEngine,
    INTERPRETABLE_FEATURES,
)

__all__ = ["ShapEngine", "Explanation", "FeatureContribution", "INTERPRETABLE_FEATURES"]
