"""What a single column gives away about the label.

A detector trained on a generated stream can learn the generator's habits
instead of fraud: a device seen only on fraud rows, a channel fraud never
uses, a card identifier. The audit measures, per feature, how well that one
column alone ranks the training labels and which values appear on one class
only. Training writes the result beside the metrics and refuses a feature
above the track's ceiling unless the configuration allows it with a reason.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from prism import config, tracks


def one_sided_values(column: pd.Series, y: pd.Series, min_support: int) -> tuple[list[str], list[str]]:
    """Values seen on fraud rows only, and on legitimate rows only, each with
    at least ``min_support`` rows."""
    counts = pd.crosstab(column.astype(str), y.astype(int)).reindex(columns=[0, 1], fill_value=0)
    fraud_only = counts.index[(counts[0] == 0) & (counts[1] >= min_support)].tolist()
    legit_only = counts.index[(counts[1] == 0) & (counts[0] >= min_support)].tolist()
    return [str(v) for v in fraud_only], [str(v) for v in legit_only]


def audit(
    frame: pd.DataFrame, y: pd.Series, schema: tracks.TrackSchema, *, min_support: int | None = None
) -> list[dict[str, Any]]:
    """One row per feature of ``schema`` over ``frame`` (the training window,
    before encoding and scaling), strongest single column first. A
    categorical column is scored by its per-value fraud rate on these same
    rows, which is optimistic on purpose: this is a leak detector."""
    min_support = int(
        min_support if min_support is not None else config.get("evaluation.shortcut_min_support", 25)
    )
    y = y.astype(int)
    rows: list[dict[str, Any]] = []
    for feature in schema.feature_columns:
        column = frame[feature]
        if feature in schema.categorical_columns:
            rate = y.groupby(column.astype(str)).mean()
            score = column.astype(str).map(rate).to_numpy(dtype=float)
            fraud_only, legit_only = one_sided_values(column, y, min_support)
            kind = "categorical"
        else:
            score = column.to_numpy(dtype=float)
            fraud_only, legit_only = [], []
            kind = "numeric"
        measurable = y.nunique() == 2 and bool(np.isfinite(score).all())
        auc = float(roc_auc_score(y, score)) if measurable else None
        rows.append({
            "feature": feature,
            "kind": kind,
            "auc": max(auc, 1.0 - auc) if auc is not None else None,
            "fraud_only_values": fraud_only,
            "legitimate_only_values": legit_only,
        })
    return sorted(rows, key=lambda r: -(r["auc"] or 0.0))


def offenders(rows: list[dict[str, Any]], *, ceiling: float, allowed: dict[str, str]) -> list[dict[str, Any]]:
    return [r for r in rows if r["auc"] is not None and r["auc"] > ceiling and r["feature"] not in allowed]


def write(rows: list[dict[str, Any]], path: Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return path
