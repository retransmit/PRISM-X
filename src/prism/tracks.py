"""One pipeline, two data tracks (design D1).

A track names its source, its columns and its split rule. Every stage reads the
schema from the track instead of assuming ``Time, V1..V28, Amount, Class``, and
each track keeps its own artifacts under ``models/<track>/`` and
``reports/metrics/<track>/`` so a run on one never overwrites the other.

Column lists are code, not configuration: they are the shape of a source, and
a typo in YAML would surface as a KeyError three modules later. The knobs
(paths, split fractions, threshold policy, prevalence target) live under
``tracks.<name>`` in ``config.yaml``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prism import config

PCA_COLUMNS: tuple[str, ...] = tuple(f"V{i}" for i in range(1, 29))

_MISSING = object()


@dataclass(frozen=True)
class TrackSchema:
    name: str
    target_column: str
    amount_column: str
    time_column: str
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    scale_columns: tuple[str, ...]
    identifier_columns: tuple[str, ...]
    derived_columns: tuple[str, ...]
    split: str
    card_column: str | None = None
    currency: str | None = None
    # Inputs the service needs that are not features: on the Indian track the
    # limit and the balance feed the ratios the model reads, and never the
    # model itself, because every card's limit identifies the card.
    extra_inputs: tuple[str, ...] = ()
    # None means every feature has a name a person can read. The ULB track
    # names the two that do; the rest are anonymised components.
    interpretable: tuple[str, ...] | None = None
    display_names: dict[str, str] = field(default_factory=dict)
    # The feed of the shipped generator's contract (generator/spec/19) a request on this
    # track is checked against; None for a track whose inputs the contract does not describe.
    contract_feed: str | None = None

    @property
    def feature_columns(self) -> list[str]:
        """Model input order: numerics first, then categoricals."""
        return [*self.numeric_columns, *self.categorical_columns]

    @property
    def input_columns(self) -> list[str]:
        """What a stream must send: every model input the service does not
        derive itself, plus the clock and the card when derivation needs them."""
        columns = [c for c in self.feature_columns if c not in self.derived_columns]
        if self.derived_columns:
            for extra in (self.time_column, self.card_column):
                if extra and extra not in columns:
                    columns.append(extra)
        for extra in self.extra_inputs:
            if extra not in columns:
                columns.append(extra)
        return columns

    def is_interpretable(self, feature: str) -> bool:
        return self.interpretable is None or feature in self.interpretable

    def label(self, feature: str) -> str:
        return self.display_names.get(feature, feature)

    @property
    def model_dir(self) -> Path:
        return config.get_path("api.model_dir") / self.name

    @property
    def metrics_dir(self) -> Path:
        return config.PROJECT_ROOT / "reports" / "metrics" / self.name

    @property
    def figures_dir(self) -> Path:
        return config.PROJECT_ROOT / "reports" / "figures" / self.name

    def setting(self, key: str, default: Any = _MISSING) -> Any:
        """A knob under tracks.<name>.<key> in config.yaml."""
        if default is _MISSING:
            return config.get(f"tracks.{self.name}.{key}")
        return config.get(f"tracks.{self.name}.{key}", default)


ULB = TrackSchema(
    name="ulb",
    target_column="Class",
    amount_column="Amount",
    time_column="Time",
    numeric_columns=("Time", *PCA_COLUMNS, "Amount"),
    categorical_columns=(),
    scale_columns=("Time", "Amount"),
    identifier_columns=(),
    derived_columns=(),
    split="stratified",
    interpretable=("Time", "Amount"),
)

# The limit as ratios (prism.features.ratios), never as the raw value.
INDIA_RATIOS: tuple[str, ...] = ("utilization", "amount_to_limit")

# Computed by prism.features.velocity from the card's earlier rows, offline
# over the export and online over the audit log, by the same function.
INDIA_VELOCITY: tuple[str, ...] = (
    "hour_of_day",
    "card_tx_1h",
    "card_tx_24h",
    "card_amount_24h",
    "seconds_since_last",
    "card_distinct_mcc_30m",
    "first_mcc_30d",
    "first_cnp_30d",
)

# Everything the service derives itself before scoring.
INDIA_DERIVED: tuple[str, ...] = (*INDIA_RATIOS, *INDIA_VELOCITY)

INDIA = TrackSchema(
    name="india",
    target_column="is_fraud",
    amount_column="amount",
    time_column="tx_time_seconds",
    numeric_columns=(
        "amount",
        "vaai_score",
        "ip_distance_from_home_km",
        "geo_risk_score",
        "cvv_match_flag",
        "is_cross_border",
        *INDIA_DERIVED,
    ),
    categorical_columns=(
        "product_id",
        "mcc",
        "pos_entry_mode",
        "pos_condition_code",
        "eci",
        "channel_type",
        "asn_type",
        "avs_match_code",
        "acquirer_bin",
        "gateway_provider",
    ),
    scale_columns=(
        "amount",
        "utilization",
        "amount_to_limit",
        "vaai_score",
        "ip_distance_from_home_km",
        "geo_risk_score",
        "card_tx_1h",
        "card_tx_24h",
        "card_amount_24h",
        "seconds_since_last",
        "card_distinct_mcc_30m",
        "hour_of_day",
    ),
    identifier_columns=("transaction_id", "card_id", "merchant_id", "tx_time_seconds"),
    derived_columns=INDIA_DERIVED,
    split="time_ordered",
    card_column="card_id",
    currency="INR",
    extra_inputs=("credit_limit", "available_balance"),
    contract_feed="authorisation_request",
    interpretable=None,
    display_names={
        "amount": "amount",
        "utilization": "share of the limit in use",
        "amount_to_limit": "amount as a share of the limit",
        "vaai_score": "network risk score",
        "ip_distance_from_home_km": "distance between the IP address and home, in km",
        "geo_risk_score": "geolocation mismatch score",
        "cvv_match_flag": "CVV match",
        "is_cross_border": "cross-border flag",
        "hour_of_day": "hour of day",
        "card_tx_1h": "transactions on this card in the last hour",
        "card_tx_24h": "transactions on this card in the last day",
        "card_amount_24h": "amount spent on this card in the last day",
        "seconds_since_last": "seconds since the card's previous transaction",
        "card_distinct_mcc_30m": "merchant categories used in the last 30 minutes",
        "first_mcc_30d": "first purchase in this merchant category in 30 days",
        "first_cnp_30d": "first online purchase on this card in 30 days",
        "product_id": "card product",
        "mcc": "merchant category",
        "pos_entry_mode": "entry mode",
        "pos_condition_code": "point-of-sale condition",
        "eci": "3-D Secure indicator",
        "channel_type": "channel",
        "asn_type": "network type",
        "avs_match_code": "address verification",
        "acquirer_bin": "acquirer",
        "gateway_provider": "gateway",
    },
)

TRACKS: dict[str, TrackSchema] = {ULB.name: ULB, INDIA.name: INDIA}

_override: str | None = None


def get(name: str) -> TrackSchema:
    if name not in TRACKS:
        raise KeyError(f"Unknown track {name!r}. Available: {sorted(TRACKS)}")
    return TRACKS[name]


def active() -> TrackSchema:
    """The track this process works on: a --track flag if one was given, else config."""
    return get(_override or str(config.get("track", "ulb")))


def set_active(name: str | None) -> None:
    global _override
    _override = None if name is None else get(name).name


def add_track_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--track",
        default=None,
        choices=sorted(TRACKS),
        help="data track to work on (default: the `track` key in config.yaml)",
    )


def activate_from_args(args: argparse.Namespace) -> TrackSchema:
    set_active(getattr(args, "track", None))
    return active()
