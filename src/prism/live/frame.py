"""The training frame of the live loop, rebuilt from the audit log.

The rows are the requests the service scored, as it scored them; the label is
operational: a row is fraud once a fraud label for it is known at the cut-off and
legitimate otherwise. The columns and dtypes are the offline frame's
(``prism.data.frames.load_frame``), so every training and preprocessing step reads
either one the same way.
"""

from __future__ import annotations

import pandas as pd

from prism import tracks
from prism.api import db
from prism.data.india import BOOLEAN_COLUMNS, CATEGORICAL_RAW, EMPTY_CATEGORY, NUMERIC_RAW
from prism.features import ratios
from prism.features.velocity import add_velocity_features


def labelled_frame(as_of: float, schema: tracks.TrackSchema) -> pd.DataFrame:
    """Every logged row before ``as_of`` on the stream clock, labelled with what was known then."""
    if schema.name != tracks.INDIA.name:
        raise ValueError(f"The live loop runs on the Indian track only, not {schema.name}")
    logged = db.logged_requests(until=as_of)
    target = schema.target_column
    keep = [*schema.identifier_columns, *NUMERIC_RAW, *CATEGORICAL_RAW, target]
    inputs = [c for c in keep if c not in ("merchant_id", target)]
    frame = pd.DataFrame(logged, columns=[*inputs, "merchant"]) if not logged else pd.DataFrame(logged)
    frame["merchant_id"] = frame.get("merchant", pd.Series(index=frame.index, dtype=object)).fillna("").astype(str)
    known = db.known_labels(as_of)
    frame[target] = frame["transaction_id"].map(known).fillna(0).astype(int)
    frame = frame[keep].copy()
    for column in ("transaction_id", schema.card_column, "merchant_id"):
        frame[column] = frame[column].astype(str)
    for column in (*NUMERIC_RAW, schema.time_column):
        frame[column] = frame[column].astype(float)
    for column in BOOLEAN_COLUMNS:
        frame[column] = frame[column].astype(float)
    for column in CATEGORICAL_RAW:
        frame[column] = frame[column].fillna(EMPTY_CATEGORY).astype(str)

    frame = ratios.add_limit_ratios(frame)
    frame = frame.sort_values(schema.time_column, kind="stable").reset_index(drop=True)
    return add_velocity_features(frame)
