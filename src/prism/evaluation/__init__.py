"""Module 3: evaluation, benchmarking and the Phase 3 experiment matrix."""

from prism.evaluation.experiments import (
    DEFAULT_ARMS,
    RESAMPLEABLE,
    ExperimentArm,
    comparison_table,
    run_arm,
    run_matrix,
)
from prism.evaluation.metrics import EvalResult, evaluate
from prism.evaluation.threshold import pr_curve_points, tune_threshold

__all__ = [
    "EvalResult",
    "evaluate",
    "tune_threshold",
    "pr_curve_points",
    "ExperimentArm",
    "DEFAULT_ARMS",
    "RESAMPLEABLE",
    "run_arm",
    "run_matrix",
    "comparison_table",
]
