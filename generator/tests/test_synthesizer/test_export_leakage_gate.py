"""Export-level label leakage gate (spec/07_export_leakage_gate.yaml).

Scores what a bank's detector would see: the authorisation feed joined to the gateway
telemetry on transaction_id, written and read back exactly as the CLI exports them. The
label is recovered from the engine records for evaluation only.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from prism_generator import SimulationEngine
from prism_generator.cli import AUTH_STREAM_COLUMNS, GATEWAY_TELEMETRY_COLUMNS, _export_csv_view
from prism_generator.spec_loader import _find_spec_dir

SPEC = yaml.safe_load((_find_spec_dir() / "07_export_leakage_gate.yaml").read_text(encoding="utf-8"))
GATE = SPEC["export_leakage_gate"]
GEN = GATE["generation"]
EVAL = GATE["evaluation"]
LIMITS = GATE["thresholds"]
EXCLUDED = set(GATE["identifier_columns_excluded"])
CODE_COLUMNS = set(GATE["code_columns_treated_as_categorical"])
# The issuer's own decision codes: excluded from the single-column ceiling, still present in
# the export, still in the learner's feature matrix and the fraud-only-value check (spec/07).
DECISION_COLUMNS = set(GATE["decision_columns_outside_single_column_audit"])


def _read_view(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def export_and_join(records: List[Dict[str, Any]], out_dir: Path) -> List[Dict[str, str]]:
    """Writes the two views the way cmd_generate does and joins them on transaction_id."""
    _export_csv_view(out_dir / "gate_auth_stream.csv", records, AUTH_STREAM_COLUMNS)
    _export_csv_view(out_dir / "gate_gateway_telemetry.csv", records, GATEWAY_TELEMETRY_COLUMNS)
    auth = _read_view(out_dir / "gate_auth_stream.csv")
    gateway = {r["transaction_id"]: r for r in _read_view(out_dir / "gate_gateway_telemetry.csv")}
    joined = []
    for row in auth:
        extra = {k: v for k, v in gateway[row["transaction_id"]].items() if k not in row}
        joined.append({**row, **extra})
    return joined


def _is_numeric_column(values: List[str]) -> bool:
    for v in values:
        if v == "":
            continue
        try:
            float(v)
        except ValueError:
            return False
    return True


def build_matrix(rows: List[Dict[str, str]], labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, int]:
    """Time-ordered feature matrix. Categorical columns become ordinal codes fitted on the
    training window; values unseen in training become missing (NaN)."""
    order = np.argsort([float(r["tx_time_seconds"]) for r in rows], kind="stable")
    rows = [rows[i] for i in order]
    y = labels[order]
    cut = int(len(rows) * EVAL["train_fraction"])

    names = [c for c in rows[0].keys() if c not in EXCLUDED] + ["hour_of_day"]
    columns: List[np.ndarray] = []
    is_categorical: List[bool] = []
    for name in names:
        if name == "hour_of_day":
            values = [str(int((float(r["tx_time_seconds"]) % 86400.0) // 3600.0)) for r in rows]
        else:
            values = [r[name] for r in rows]
        if name not in CODE_COLUMNS and _is_numeric_column(values):
            columns.append(np.array([float(v) if v != "" else np.nan for v in values], dtype=float))
            is_categorical.append(False)
        else:
            seen = {v: i for i, v in enumerate(sorted(set(values[:cut])))}
            columns.append(np.array([seen.get(v, np.nan) for v in values], dtype=float))
            is_categorical.append(True)
    X = np.column_stack(columns)
    return X, np.array(is_categorical), names, y, cut


def single_column_scores(X: np.ndarray, is_categorical: np.ndarray, names: List[str], y: np.ndarray, cut: int) -> List[Tuple[float, str]]:
    """ROC-AUC of each column alone on the test window, direction-free."""
    scores = []
    prevalence = float(y[:cut].mean())
    for j, name in enumerate(names):
        col = X[:, j]
        if is_categorical[j]:
            train_codes = col[:cut]
            rate = {}
            for code in np.unique(train_codes[~np.isnan(train_codes)]):
                mask = train_codes == code
                rate[code] = float(y[:cut][mask].mean())
            score = np.array([rate.get(c, prevalence) if not np.isnan(c) else prevalence for c in col[cut:]])
        else:
            score = np.nan_to_num(col[cut:], nan=float(np.nanmedian(col[:cut])) if np.isfinite(np.nanmedian(col[:cut])) else 0.0)
        if len(np.unique(score)) < 2 or len(np.unique(y[cut:])) < 2:
            auc = 0.5
        else:
            auc = float(roc_auc_score(y[cut:], score))
        scores.append((max(auc, 1.0 - auc), name))
    return sorted(scores, reverse=True)


def audited_single_column_scores(ranked: List[Tuple[float, str]]) -> List[Tuple[float, str]]:
    """`ranked` restricted to the columns the single-column ceiling audits (spec/07).

    The authorisation response codes are the decision system's own output. A detector cannot
    consume its own decision as an input, so the ceiling - which asks whether an observable
    *input* separates the classes - does not audit them. They remain in the export and in the
    learner's feature matrix, where the learner ceiling still covers them.
    """
    return [(auc, name) for auc, name in ranked if name not in DECISION_COLUMNS]


def fraud_only_values(rows: List[Dict[str, str]], labels: np.ndarray) -> List[Tuple[str, str, int]]:
    """Categorical values with enough support that are never legitimate."""
    found = []
    for name in rows[0].keys():
        if name in EXCLUDED:
            continue
        values = [r[name] for r in rows]
        if name not in CODE_COLUMNS and _is_numeric_column(values):
            continue
        counts: Dict[str, List[int]] = {}
        for v, label in zip(values, labels):
            counts.setdefault(v, [0, 0])[int(label)] += 1
        for v, (legit, fraud) in counts.items():
            if legit == 0 and fraud >= LIMITS["fraud_only_value_min_support"]:
                found.append((name, v, fraud))
    return found


@pytest.mark.parametrize("region", GEN["regions"])
@pytest.mark.parametrize("adversary_mode", GEN["adversary_modes"])
def test_exported_views_carry_no_label_shortcut(tmp_path: Path, region: str, adversary_mode: str) -> None:
    engine = SimulationEngine(
        n_cards=GEN["n_cards"],
        n_merchants=GEN["n_merchants"],
        region=region,
        adversary_mode=adversary_mode,
        seed=GEN["seed"],
    )
    records = engine.generate_batch(
        n_transactions=GEN["n_transactions"],
        fraud_prevalence=GEN["fraud_prevalence"],
        time_span_days=GEN["time_span_days"],
    )
    label_of = {r["transaction_id"]: int(r["is_fraud"]) for r in records}

    rows = export_and_join(records, tmp_path)
    assert len(rows) == len(records)
    for banned in ("is_fraud", "scenario_tag", "syndicate_id", "risk_score"):
        assert banned not in rows[0], f"{banned} rode along in the exported views"
    labels = np.array([label_of[r["transaction_id"]] for r in rows], dtype=int)
    assert labels.sum() >= 30, "too few fraud rows to score the gate"

    leaks = fraud_only_values(rows, labels)
    assert not leaks, f"{region}/{adversary_mode}: values that are never legitimate: {leaks}"

    X, is_categorical, names, y, cut = build_matrix(rows, labels)
    assert y[cut:].sum() >= 10, "too few fraud rows in the test window"

    ranked = single_column_scores(X, is_categorical, names, y, cut)
    audited = audited_single_column_scores(ranked)
    top = [(name, round(auc, 3)) for auc, name in audited[:6]]
    decision = {name: round(auc, 3) for auc, name in ranked if name in DECISION_COLUMNS}
    assert audited[0][0] <= LIMITS["max_single_column_roc_auc"], (
        f"{region}/{adversary_mode}: a single exported input separates the classes: {top}; "
        f"decision codes (learner-covered, outside this ceiling): {decision}"
    )

    clf = HistGradientBoostingClassifier(categorical_features=is_categorical, **EVAL["learner_params"])
    clf.fit(X[:cut], y[:cut])
    probs = clf.predict_proba(X[cut:])[:, 1]
    pr_auc = float(average_precision_score(y[cut:], probs))
    assert pr_auc <= LIMITS["model_pr_auc_ceiling"], (
        f"{region}/{adversary_mode}: PR-AUC {pr_auc:.4f} on the exported views is above the ceiling; "
        f"strongest columns {top}"
    )
    assert pr_auc >= LIMITS["model_pr_auc_floor"], (
        f"{region}/{adversary_mode}: PR-AUC {pr_auc:.4f} is below the floor; the fraud carries no signal"
    )
