"""Fit, persist and reload the reference material behind the explanation.

    python -m prism.explain.artifacts

Two objects live beside the models in the model directory:

  reference.joblib   population statistics on the raw training rows
  cases.joblib       nearest-neighbour index over the scaled training rows

Both are fitted on the training split only. The reference sees raw amounts and
times because the sentence quotes them in currency and daily-cycle terms; the
case index sees the scaled rows because that is the representation the
service scores, so a query at serving time is compared like with like.

Training persists them automatically. This module's entry point rebuilds them
for a model directory trained before they existed, from the same deterministic
split, so no retraining is needed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import argparse

from prism import tracks
from prism.data.frames import load_frame
from prism.data.preprocess import SplitData, prepare_splits
from prism.explain.cases import CaseIndex
from prism.explain.reference import ReferenceStats

REFERENCE_FILE = "reference.joblib"
CASES_FILE = "cases.joblib"


def build(
    raw_frame: pd.DataFrame, splits: SplitData, schema: tracks.TrackSchema | None = None
) -> tuple[ReferenceStats, CaseIndex]:
    """Fit both artifacts on the training rows of ``splits`` only."""
    schema = schema or tracks.get(splits.track)
    raw_train = raw_frame.loc[splits.X_train.index].drop(columns=[schema.target_column])
    reference = ReferenceStats.from_training(raw_train, splits.y_train, schema=schema)
    cases = CaseIndex.fit(
        splits.X_train,
        splits.y_train,
        exclude=(schema.time_column,),
        categorical_columns=splits.categorical_columns,
    )
    return reference, cases


def save(reference: ReferenceStats, cases: CaseIndex, model_dir: Path) -> None:
    reference.save(Path(model_dir) / REFERENCE_FILE)
    cases.save(Path(model_dir) / CASES_FILE)


def load(model_dir: Path) -> tuple[ReferenceStats, CaseIndex] | None:
    """Both artifacts, or None when either is missing so the caller can degrade."""
    reference_path = Path(model_dir) / REFERENCE_FILE
    cases_path = Path(model_dir) / CASES_FILE
    if not (reference_path.exists() and cases_path.exists()):
        return None
    return ReferenceStats.load(reference_path), CaseIndex.load(cases_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the explanation reference material.")
    tracks.add_track_argument(parser)
    schema = tracks.activate_from_args(parser.parse_args())
    model_dir = schema.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    frame = load_frame(schema)
    splits = prepare_splits(frame, schema=schema)
    reference, cases = build(frame, splits, schema=schema)
    save(reference, cases, model_dir)
    print(
        f"reference on {reference.n_rows:,} training rows, "
        f"{cases.k} nearest cases per query -> {model_dir}"
    )


if __name__ == "__main__":
    main()
