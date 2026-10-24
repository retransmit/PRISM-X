"""Module 2: training orchestration and resampling.

Submodules are imported lazily: the service imports ``prism.training.manifest``
at startup and must not pay for the resampling stack to do it.
"""

from importlib import import_module

_LAZY = {
    "cross_val_scores": "prism.training.resampling",
    "make_smote": "prism.training.resampling",
    "resample_training_fold": "prism.training.resampling",
}


def __getattr__(name: str):
    if name in _LAZY:
        return getattr(import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["cross_val_scores", "make_smote", "resample_training_fold"]
