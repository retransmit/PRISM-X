"""Drift: how far the service's recent traffic has moved from what the serving bundle learned.

Each bundle keeps a reference from its own training and validation rows (plan Decision
9): the decile edges and shares of every numeric input (a numeric input with at most
``drift_discrete_max`` distinct values, a flag or a small count, is binned by value with
an ``other`` bin instead, since deciles would put nearly all of it in one bin), the shares of every categorical
input with categories under ``category_floor`` pooled as ``other``, and the decile edges
and shares of the champion's validation scores. The population stability index compares
the recent rows' shares with the reference's, bin by bin, with every share floored at
``drift_epsilon`` so an empty bin is finite. At or above ``drift_warn`` a feature is
``warn``, at or above ``drift_alert`` it is ``alert`` (the conventional 0.10 and 0.25).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from prism import tracks

FILENAME = "drift_reference.json"
OTHER = "other"
DECILES = np.linspace(0.1, 0.9, 9)


def _edges(values: np.ndarray) -> list[float]:
    """The inner decile edges, de-duplicated; the first and last bins are open."""
    return [float(edge) for edge in np.unique(np.quantile(values, DECILES))]


def _binned_shares(values: np.ndarray, edges: Sequence[float]) -> list[float]:
    """Shares over right-closed bins: (-inf, e0], (e0, e1], ..., (e_last, inf). A value equal
    to an edge falls in the bin below it, so a mass at a decile edge (a column that is mostly
    zero) stays in a bin of its own instead of swallowing every value above it."""
    if len(values) == 0:
        return [0.0] * (len(edges) + 1)
    bins = np.searchsorted(np.asarray(edges, dtype=float), values, side="left")
    counts = np.bincount(bins, minlength=len(edges) + 1)
    return [float(c) / len(values) for c in counts]


def _numeric(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)


def _value_shares(values: np.ndarray, known: Sequence[float]) -> list[float]:
    """Shares over the known values, then one ``other`` bin for anything else."""
    if len(values) == 0:
        return [0.0] * (len(known) + 1)
    counts = [float(np.count_nonzero(values == value)) for value in known]
    other = float(len(values)) - sum(counts)
    return [c / len(values) for c in (*counts, other)]


def _category_shares(values: pd.Series, known: Sequence[str]) -> list[float]:
    """Shares over ``known`` (which ends with ``other``); anything else counts as other."""
    named = [k for k in known if k != OTHER]
    text = values.astype(str)
    text = text.where(text.isin(named), OTHER)
    counts = text.value_counts()
    total = float(len(text)) or 1.0
    return [float(counts.get(k, 0)) / total for k in known]


def reference(frame: pd.DataFrame, scores: np.ndarray, schema: tracks.TrackSchema, s: dict) -> dict:
    """The reference of one bundle, from its training and validation rows and its
    champion's validation scores."""
    floor = float(s["category_floor"])
    discrete_max = int(s["drift_discrete_max"])
    numeric = {}
    for column in schema.numeric_columns:
        values = _numeric(frame[column])
        distinct = np.unique(values)
        if len(distinct) <= discrete_max:
            known = [float(v) for v in distinct]
            numeric[column] = {"values": known, "shares": _value_shares(values, known)}
            continue
        edges = _edges(values)
        numeric[column] = {"edges": edges, "shares": _binned_shares(values, edges)}
    categorical = {}
    for column in schema.categorical_columns:
        observed = frame[column].astype(str).value_counts(normalize=True)
        kept = {str(k): float(v) for k, v in observed.items() if v >= floor and str(k) != OTHER}
        kept[OTHER] = float(max(0.0, 1.0 - sum(kept.values())))
        categorical[column] = {"shares": kept}
    score_values = np.asarray(scores, dtype=float)
    score_edges = _edges(score_values)
    return {
        "track": schema.name,
        "rows": int(len(frame)),
        "numeric": numeric,
        "categorical": categorical,
        "score": {"edges": score_edges, "shares": _binned_shares(score_values, score_edges)},
    }


def save(ref: dict, model_dir: Path) -> Path:
    target = Path(model_dir) / FILENAME
    target.write_text(json.dumps(ref, indent=2), encoding="utf-8")
    return target


def load(model_dir: Path) -> dict | None:
    path = Path(model_dir) / FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def psi(expected: Sequence[float], actual: Sequence[float], epsilon: float) -> float:
    """The population stability index: the sum over bins of (a - e) ln(a / e)."""
    total = 0.0
    for e, a in zip(expected, actual, strict=True):
        e, a = max(float(e), epsilon), max(float(a), epsilon)
        total += (a - e) * math.log(a / e)
    return float(total)


def state(value: float, s: dict) -> str:
    if value >= float(s["drift_alert"]):
        return "alert"
    if value >= float(s["drift_warn"]):
        return "warn"
    return "ok"


SEVERITY = {"ok": 0, "warn": 1, "alert": 2}


def report(ref: dict, frame: pd.DataFrame, scores: np.ndarray | None, s: dict) -> dict:
    """Every input's PSI and state, highest first, the score's (None when no scores are
    given), and the worst state."""
    epsilon = float(s["drift_epsilon"])
    features = []
    for column, entry in ref["numeric"].items():
        if column not in frame.columns:
            continue
        observed = _numeric(frame[column])
        actual = (
            _value_shares(observed, entry["values"]) if "values" in entry else _binned_shares(observed, entry["edges"])
        )
        value = psi(entry["shares"], actual, epsilon)
        features.append({"feature": column, "psi": value, "state": state(value, s)})
    for column, entry in ref["categorical"].items():
        if column not in frame.columns:
            continue
        known = list(entry["shares"])
        value = psi([entry["shares"][k] for k in known], _category_shares(frame[column], known), epsilon)
        features.append({"feature": column, "psi": value, "state": state(value, s)})
    features.sort(key=lambda row: row["psi"], reverse=True)
    score = None
    if scores is not None:
        score_value = psi(
            ref["score"]["shares"], _binned_shares(np.asarray(scores, dtype=float), ref["score"]["edges"]), epsilon
        )
        score = {"psi": score_value, "state": state(score_value, s)}
    states = [row["state"] for row in features] + ([score["state"]] if score else [])
    worst = max(states or ["ok"], key=SEVERITY.get)
    return {"features": features, "score": score, "state": worst, "rows": int(len(frame))}
