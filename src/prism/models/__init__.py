"""Module 2: the detectors under comparison.

Model classes are imported lazily. Eagerly importing all of them here would mean
that touching ``FraudModel``, or running the preprocessing tests, pulls torch
and xgboost into the process. Consequences that matter:

  * a deployment serving only XGBoost does not need a deep-learning framework;
  * a missing optional dependency degrades one model instead of taking down
    the whole service;
  * the leakage tests run without a 200MB install.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from prism.models.base import FraudModel

# name -> (module, class). Import happens on first use, not at package import.
_MODEL_MODULES: dict[str, tuple[str, str]] = {
    "isolation_forest": ("prism.models.iforest", "IsolationForestModel"),
    "autoencoder": ("prism.models.autoencoder", "AutoencoderModel"),
    "xgboost": ("prism.models.xgb", "XGBoostModel"),
}

MODEL_NAMES: tuple[str, ...] = tuple(_MODEL_MODULES)

# The file each model's save() writes its fitted state to, by registry name;
# the manifest records a digest of it and serving checks the digest.
ARTIFACT_FILES: dict[str, str] = {
    "isolation_forest": "isolation_forest.joblib",
    "autoencoder": "autoencoder.pt",
    "xgboost": "xgboost.json",
}


def get_model_class(name: str) -> type[FraudModel]:
    """Resolve a registry name to its class, importing the module on demand."""
    if name not in _MODEL_MODULES:
        raise KeyError(f"Unknown model {name!r}. Available: {sorted(_MODEL_MODULES)}")
    module_name, class_name = _MODEL_MODULES[name]
    try:
        return getattr(import_module(module_name), class_name)
    except ImportError as exc:
        raise ImportError(
            f"Model {name!r} needs a dependency that is not installed: {exc}"
        ) from exc


def build(name: str, **overrides) -> FraudModel:
    """Instantiate a model by its registry name."""
    return get_model_class(name)(**overrides)


def load_all(model_dir: Path) -> dict[str, FraudModel]:
    """Load every persisted model.

    A model is skipped, not fatal, when its artifact is missing (not trained
    yet) or its dependency is absent. The service should serve whatever is
    actually available.
    """
    loaded: dict[str, FraudModel] = {}
    for name in MODEL_NAMES:
        try:
            loaded[name] = get_model_class(name).load(model_dir)
        except FileNotFoundError:
            continue
        except ImportError as exc:
            print(f"[models] skipping {name}: {exc}")
            continue
    return loaded


def __getattr__(name: str):
    """Lazy re-export, so ``from prism.models import XGBoostModel`` still works."""
    for module_name, class_name in _MODEL_MODULES.values():
        if name == class_name:
            return getattr(import_module(module_name), class_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FraudModel",
    "MODEL_NAMES",
    "get_model_class",
    "build",
    "load_all",
    "IsolationForestModel",
    "AutoencoderModel",
    "XGBoostModel",
]
