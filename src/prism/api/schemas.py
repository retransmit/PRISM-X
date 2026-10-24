"""Module 5: request/response contracts.

Pydantic validates every incoming payload before it reaches a model. A missing
or mistyped field becomes a 422 with a precise message rather than a silent
NaN propagating into the feature vector.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from prism import contract, tracks

_active: tracks.TrackSchema | None = None

# Longest categorical or card value accepted; the audit log's text columns
# are this wide, and Postgres would reject a longer one after scoring.
TEXT_INPUT_MAX = 64


def set_active_schema(schema: tracks.TrackSchema | None) -> None:
    """The service sets this once its engine has loaded; tests set it per fixture."""
    global _active
    _active = schema


def active_schema() -> tracks.TrackSchema:
    return _active or tracks.active()


class TransactionIn(BaseModel):
    """One transaction as the stream sends it.

    The model features arrive as extra keys named by the active track's schema
    (``Time, V1..V28, Amount`` on ULB; the pre-authorisation columns plus the
    clock and the card on India), so the same class serves both tracks and a
    missing column is a 422 naming the track rather than a KeyError deeper in.
    """

    model_config = ConfigDict(extra="allow")

    transaction_id: str | None = Field(default=None, max_length=64)

    # Context the stream may supply. Stored in the audit log for the boards.
    # On the Indian track card_id is also a model input (velocity features).
    card_id: str | None = Field(default=None, max_length=64)
    merchant: str | None = Field(default=None, max_length=120)
    channel: str | None = Field(default=None, max_length=32)
    currency: str | None = Field(default=None, max_length=8)

    def _supplied(self) -> dict[str, Any]:
        supplied = dict(self.model_extra or {})
        if self.card_id is not None:
            supplied.setdefault("card_id", self.card_id)
        return supplied

    @model_validator(mode="after")
    def _require_track_inputs(self) -> "TransactionIn":
        schema = active_schema()
        supplied = self._supplied()
        missing = [name for name in schema.input_columns if name not in supplied]
        if missing:
            raise ValueError(f"Missing features for the {schema.name} track: {missing}")
        # Every numeric input is checked here so a mistyped value is a 422
        # naming the column, not a 500 from the float() deeper in.
        text_columns = set(schema.categorical_columns) | {schema.card_column}
        for name in schema.input_columns:
            if name in text_columns:
                if len(str(supplied[name])) > TEXT_INPUT_MAX:
                    raise ValueError(f"{name} must be at most {TEXT_INPUT_MAX} characters")
                continue
            try:
                value = float(supplied[name])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a number") from exc
            # float("NaN") and float("Infinity") succeed; a NaN would score as
            # a missing value and then fail the audit-log insert as NULL.
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            if name == schema.amount_column and value < 0:
                raise ValueError(f"{schema.amount_column} must be at least 0")
        # The shipped generator's contract: a wrong type or a value outside a closed code
        # set is refused here, naming the field and the value.
        if schema.contract_feed is not None:
            problems = contract.violations(schema.contract_feed, supplied)
            if problems:
                raise ValueError("The request breaks the contract: " + "; ".join(problems))
        return self

    def to_feature_dict(self) -> dict[str, float | str]:
        schema = active_schema()
        supplied = self._supplied()
        features: dict[str, float | str] = {}
        for name in schema.input_columns:
            if name in schema.categorical_columns or name == schema.card_column:
                features[name] = str(supplied[name])
            else:
                features[name] = float(supplied[name])
        return features

    def context(self) -> dict[str, str | None]:
        return {
            "card_id": self.card_id,
            "merchant": self.merchant,
            "channel": self.channel,
            "currency": self.currency,
        }


DecisionActor = Literal["analyst", "customer"]
DecisionVerdict = Literal["fraud", "legitimate"]


class DecisionIn(BaseModel):
    """An analyst's verdict or a cardholder's response to a scored transaction.
    The actor is the credential the request carries, so the body does not name it."""

    transaction_id: str = Field(..., max_length=64)
    verdict: DecisionVerdict
    note: str | None = Field(default=None, max_length=280)


class DecisionOut(BaseModel):
    transaction_id: str
    actor: DecisionActor
    verdict: DecisionVerdict
    note: str | None
    decided_at: str


FeedLabelSource = Literal["INVESTIGATOR_ALERT", "CHARGEBACK_DISPUTE", "UNLABELLED"]


class LabelIn(BaseModel):
    """A label as the bank learns it, in the generator's label-event shape (contract
    spec/19 ``label_event``). The other fields of the event are accepted and ignored."""

    model_config = ConfigDict(extra="ignore")

    transaction_id: str = Field(..., max_length=64)
    discovered_label: Literal[0, 1]
    label_source: FeedLabelSource
    discovery_time_seconds: float = Field(..., ge=0, allow_inf_nan=False)


class LabelOut(BaseModel):
    transaction_id: str
    label: int
    source: str
    known_at: float
    recorded_at: str


DriftState = Literal["ok", "warn", "alert"]


class FeatureDrift(BaseModel):
    feature: str
    psi: float
    state: DriftState


class ScoreDrift(BaseModel):
    psi: float
    state: DriftState


class DriftOut(BaseModel):
    """The serving bundle's reference against the service's most recent rows."""

    bundle_id: str
    rows: int
    state: DriftState
    # The score is compared only over rows the serving bundle scored; until it has scored
    # live.drift_min_rows of them there is no score drift to report.
    score: ScoreDrift | None
    score_rows: int
    features: list[FeatureDrift]


class LiveOut(BaseModel):
    """The live loop for the board: what serves, what the bank has taught it, how retrains went."""

    bundle_id: str
    champion: str
    labels: dict[str, int]
    events: list[dict[str, Any]]


class ModelScore(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    risk_score: float = Field(..., ge=0, le=1)
    threshold: float
    flagged: bool


class FeatureAttribution(BaseModel):
    feature: str
    value: float
    shap_value: float
    interpretable: bool
    # The raw categorical value as text; absent for numerics and on the ULB track.
    display: str | None = None
    # The track's display name for the feature, so the diagram and the sentence agree.
    label: str | None = None


class ExplanationFacts(BaseModel):
    """The measured facts the plain-English narrative is built from."""

    amount: float
    amount_pct_below: float
    cycle_share: float
    typical_cycle_share: float
    unusual_count: int
    n_characteristics: int
    band: float
    similar_fraud: int
    similar_total: int
    base_rate: float
    clearing_amount: float | None = None
    # Whether every amount below clearing_amount clears too; the sentence
    # claims a range only when this is true.
    clearing_is_monotone: bool | None = None


class ReasonCode(BaseModel):
    """One driver family's share of the push towards fraud: a code a bank can show a
    cardholder and log for audit, whichever model produced the score."""

    code: str
    label: str
    share: float
    features: list[str]


class ExplanationOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    base_value: float
    prediction: float
    top_features: list[FeatureAttribution]
    # Must be declared here or pydantic strips it from the /predict response:
    # undeclared keys are dropped silently, not passed through.
    narrative: str
    explained_fraction: float
    interpretable_fraction: float
    # Declared for the same reason as narrative: an undeclared key is
    # stripped silently and never reaches the client.
    facts: ExplanationFacts | None = None
    reason_codes: list[ReasonCode] = []


class PredictionOut(BaseModel):
    transaction_id: str
    flagged: bool
    primary_model: str
    risk_score: float
    scores: list[ModelScore]
    explanation: ExplanationOut | None = None
    latency_ms: float
    received_at: str


class MetricsOut(BaseModel):
    track: str
    models: list[str]
    thresholds: dict[str, float]
    processed: int
    flagged: int
    mean_latency_ms: float
    benchmark: list[dict[str, Any]] | None = None
    # Curve geometry for the dashboard's comparison panel. Undeclared keys are
    # stripped by pydantic, so this must exist here to reach the client.
    pr_curves: dict[str, dict[str, list[float]]] | None = None


class LatencyPercentiles(BaseModel):
    p50: float
    p95: float
    p99: float


class CardSummary(BaseModel):
    card_id: str
    transactions: int
    held: int


class StatsOut(BaseModel):
    """Counters for the boards, computed from the audit log (spec 4.5)."""

    processed: int
    flagged: int
    flag_rate: float
    per_minute: int
    latency_ms: LatencyPercentiles
    decided: int
    confirmed_fraud: int
    marked_legitimate: int
    customer_responses: int
    first_received_at: str | None
    last_received_at: str | None
    cards: list[CardSummary]
