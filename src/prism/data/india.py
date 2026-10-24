"""The Indian track's source: a generator export (design D2 and D3).

Reads the authorisation feed joined to the gateway telemetry on transaction_id
and the label from the master file, and keeps nothing an issuer would not have
before answering: response codes, the authorisation code and the 3-D Secure
status are outcomes and never become features. Rows come back in time order
with a fresh RangeIndex, so a time-ordered split is a slice.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from prism import config, tracks
from prism.features import ratios

OUTCOME_COLUMNS: tuple[str, ...] = ("response_code", "auth_response_code", "auth_code", "trans_status_3ds")
CATEGORICAL_RAW: tuple[str, ...] = tracks.INDIA.categorical_columns
NUMERIC_RAW: tuple[str, ...] = (
    *(c for c in tracks.INDIA.numeric_columns if c not in tracks.INDIA.derived_columns),
    *tracks.INDIA.extra_inputs,
)
# An absent code (a card-present payment has no ECI) is the empty string on the wire
# (generator/spec/19), and the models learn that value, so a live request encodes as the
# training rows did.
EMPTY_CATEGORY = ""
BOOLEAN_COLUMNS = ("is_cross_border",)

GENERATE_HINT = (
    "Generate it with the generator's CLI, for example:\n"
    "  <generator> generate -n 90000 -c 2000 -m 300 --region IN "
    "--calibration rbi-psi-2026-07 --fraud-rate 0.005 --days 60 --seed 42 "
    "-o <prism>/data/raw/india/train_in.csv --export-institutional-views"
)


def export_paths(stem: Path) -> dict[str, Path]:
    """The four files the generator's CLI writes for one output stem."""
    stem = Path(stem)
    return {
        "master": stem.with_name(stem.name + ".csv"),
        "auth": stem.with_name(stem.name + "_auth_stream.csv"),
        "gateway": stem.with_name(stem.name + "_gateway_telemetry.csv"),
        "calibration": stem.with_name(stem.name + "_calibration.json"),
    }


def generator_revision() -> str | None:
    """The revision of the generator that ships in this repository (generator/SOURCE),
    or None when no copy is present."""
    source = config.PROJECT_ROOT / "generator" / "SOURCE"
    if not source.exists():
        return None
    for line in source.read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("revision "):
            return line.split(" ", 1)[1].strip()
    return None


def _project_relative(path: Path) -> str:
    """A path as the report quotes it: relative to the project root when inside it,
    so a committed artifact never carries one machine's drive letter."""
    try:
        return path.resolve().relative_to(config.PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def export_facts(stem: Path) -> dict[str, Any]:
    """What the report quotes about one export: its size, its cards and
    merchants, the days it spans, and the calibration report's verdict.

    Read from the files the CLI wrote rather than from the loaded frame, so
    a demo export the pipeline never trains on can be described the same way.
    """
    paths = export_paths(stem)
    auth = pd.read_csv(paths["auth"], usecols=["card_id", "merchant_id", "tx_time_seconds"])
    master = pd.read_csv(paths["master"], usecols=["is_fraud"])
    times = auth["tx_time_seconds"].astype(float)

    calibration = None
    if paths["calibration"].exists():
        report = json.loads(paths["calibration"].read_text(encoding="utf-8"))
        calibration = {
            "profile": report.get("profile"),
            "overall": report.get("overall"),
            "generated": report.get("generated", {}),
            "gated": [
                {
                    key: target.get(key)
                    for key in ("id", "statistic", "target", "observed", "tolerance_relative", "status")
                }
                for target in report.get("targets", [])
                if target.get("gate")
            ],
        }
    return {
        "stem": _project_relative(Path(stem)),
        "generator_revision": generator_revision(),
        "rows": int(len(auth)),
        "fraud_rows": int(master["is_fraud"].astype(int).sum()),
        "cards": int(auth["card_id"].nunique()),
        "merchants": int(auth["merchant_id"].nunique()),
        "span_days": float((times.max() - times.min()) / 86_400.0),
        "calibration": calibration,
    }


def load_india(stem: Path | None = None) -> pd.DataFrame:
    """Join the export's views, keep the pre-authorisation columns, label, sort."""
    stem = Path(stem) if stem is not None else config.get_path("tracks.india.export_stem")
    paths = export_paths(stem)
    missing = [str(p) for key, p in paths.items() if key != "calibration" and not p.exists()]
    if missing:
        raise FileNotFoundError(f"Indian export incomplete, missing: {missing}\n{GENERATE_HINT}")

    # Everything as text first: entry modes carry leading zeros and an empty
    # ECI is a value, not a null. Numbers are converted explicitly below.
    auth = pd.read_csv(paths["auth"], dtype=str, keep_default_na=False)
    gateway = pd.read_csv(paths["gateway"], dtype=str, keep_default_na=False)
    master = pd.read_csv(
        paths["master"], dtype=str, keep_default_na=False, usecols=["transaction_id", "is_fraud"]
    )
    if not (len(auth) == len(gateway) == len(master)):
        raise ValueError(
            f"Export views disagree on row count: auth {len(auth)}, gateway {len(gateway)}, master {len(master)}"
        )

    # Equal counts with different ids would inner-join to fewer rows without
    # a word; compare the id sets before joining.
    auth_ids = set(auth["transaction_id"])
    for name, view in (("gateway_telemetry", gateway), ("master", master)):
        foreign = set(view["transaction_id"]) ^ auth_ids
        if foreign:
            sample = sorted(foreign)[:5]
            raise ValueError(
                f"transaction_id sets differ between auth_stream and {name}: {len(foreign)} ids "
                f"appear in one view only, for example {sample}"
            )
    frame = auth.merge(
        gateway.drop(columns=["timestamp_utc", "merchant_id"]), on="transaction_id", validate="one_to_one"
    )
    frame = frame.merge(master, on="transaction_id", validate="one_to_one")

    keep = [*tracks.INDIA.identifier_columns, *NUMERIC_RAW, *CATEGORICAL_RAW, tracks.INDIA.target_column]
    absent = [c for c in keep if c not in frame.columns]
    if absent:
        raise ValueError(f"Export is missing columns the Indian track needs: {absent}")
    frame = frame[keep].copy()

    for column in (*NUMERIC_RAW, tracks.INDIA.time_column):
        if column in BOOLEAN_COLUMNS:
            frame[column] = frame[column].str.strip().str.lower().map(
                {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0}
            )
        else:
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    frame[tracks.INDIA.target_column] = pd.to_numeric(frame[tracks.INDIA.target_column]).astype(int)
    for column in CATEGORICAL_RAW:
        frame[column] = frame[column].replace("", EMPTY_CATEGORY).astype(str)

    nulls = frame[[*NUMERIC_RAW, tracks.INDIA.time_column]].isna().sum()
    offenders = nulls[nulls > 0]
    if not offenders.empty:
        raise ValueError(f"Null values present in: {offenders.to_dict()}")
    labels = set(frame[tracks.INDIA.target_column].unique().tolist())
    if not labels <= {0, 1}:
        raise ValueError(f"is_fraud must be binary 0/1, found: {sorted(labels)}")

    # The limit enters the features as ratios; the raw columns stay for the
    # service's own derivation and for the reference statistics.
    frame = ratios.add_limit_ratios(frame)
    return frame.sort_values(tracks.INDIA.time_column, kind="stable").reset_index(drop=True)
