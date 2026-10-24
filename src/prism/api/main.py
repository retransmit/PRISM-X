"""Module 5: the FastAPI inference service (§6.2.2).

    uvicorn prism.api.main:app --reload

Endpoints (all under /api/v1):
    GET  /health        readiness + which models loaded
    POST /predict       score one transaction, explain it if flagged
    POST /explain       force a SHAP explanation regardless of risk
    GET  /metrics       throughput, latency, benchmark table
    GET  /transactions  recent audit-log rows, for the dashboard feed
    POST /decisions     an analyst verdict or a cardholder response, append-only
    GET  /stats         counters from the audit log for the boards
    POST /labels        a label as the bank learns it, from the label feed's credential
    GET  /drift         input and score drift of the serving bundle (Indian track)
    GET  /live          the serving bundle, the labels received and the last retrains
"""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from prism import config, tracks
from prism.api import db, schemas
from prism.api.inference import InferenceEngine
from prism.api.schemas import (
    DecisionIn,
    DecisionOut,
    DriftOut,
    LiveOut,
    LabelIn,
    LabelOut,
    MetricsOut,
    PredictionOut,
    StatsOut,
    TransactionIn,
)
from prism.features import velocity
from prism.live import bundles, drift

engine: InferenceEngine | None = None
# Why no engine is served, for /health, when loading refused or failed.
engine_error: str | None = None
# The bundle the engine was loaded from ("offline" for models/<track>/), and why the
# bundle the pointer names is not the one serving, when it could not be loaded.
serving_bundle: str = bundles.OFFLINE
bundle_error: str | None = None
_poll = {"at": float("-inf"), "pointer": None}
# The last drift report, reused for live.drift_refresh_seconds.
_drift_cache: dict = {}
_poll_lock = threading.Lock()


def _load_serving_engine() -> InferenceEngine:
    """The engine of the bundle the pointer names; the offline directory when that
    bundle cannot be loaded, with the reason kept for /health."""
    global serving_bundle, bundle_error
    schema = tracks.active()
    bundle_id, directory = bundles.OFFLINE, schema.model_dir
    if bundles.live_enabled():
        try:
            bundle_id, directory = bundles.serving_dir(schema)
        except ValueError as exc:
            bundle_error = str(exc)
            print(f"[api] {bundle_error}; serving the offline directory")
    if bundle_id != bundles.OFFLINE:
        try:
            loaded = InferenceEngine(model_dir=directory, track=schema.name).load()
            serving_bundle, bundle_error = bundle_id, None
            return loaded
        except Exception as exc:  # noqa: BLE001  a broken bundle must not keep the service down
            bundle_error = f"Bundle {bundle_id} at {directory} could not be loaded: {exc}"
            print(f"[api] {bundle_error}; serving the offline directory")
    serving_bundle = bundles.OFFLINE
    return InferenceEngine().load()


def _poll_bundle() -> None:
    """Swap to the bundle the pointer names, at most once per live.bundle_poll_seconds.

    The check is a stat of the pointer file; only a changed pointer reads it. The new
    engine is loaded in full (warm-up included) before it replaces the serving one, and
    a bundle that fails to load leaves the serving engine in place with the reason on
    /health. A refused bundle is tried again whenever the pointer is rewritten, so a bundle
    repaired and promoted again under the same id is served.
    """
    global engine, engine_error, serving_bundle, bundle_error
    if not bundles.live_enabled():
        return
    interval = float(config.get("live.bundle_poll_seconds", 5))
    now = time.monotonic()
    if now - _poll["at"] < interval or not _poll_lock.acquire(blocking=False):
        return
    try:
        _poll["at"] = now
        schema = engine.schema if engine is not None else tracks.active()
        pointer = bundles.pointer_path(schema)
        try:
            stamp = pointer.stat().st_mtime_ns
        except FileNotFoundError:
            stamp = None
        if stamp == _poll["pointer"]:
            return
        _poll["pointer"] = stamp
        try:
            bundle_id, directory = bundles.serving_dir(schema)
        except ValueError as exc:
            bundle_error = str(exc)
            print(f"[api] {bundle_error}; still serving {serving_bundle}")
            return
        if bundle_id == serving_bundle:
            return
        try:
            loaded = InferenceEngine(model_dir=directory, track=schema.name).load()
        except Exception as exc:  # noqa: BLE001
            bundle_error = f"Bundle {bundle_id} at {directory} could not be loaded: {exc}"
            print(f"[api] {bundle_error}; still serving {serving_bundle}")
            return
        engine, engine_error = loaded, None
        serving_bundle, bundle_error = bundle_id, None
        schemas.set_active_schema(loaded.schema)
        print(f"[api] serving bundle {bundle_id} (primary {loaded.primary})")
    finally:
        _poll_lock.release()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, engine_error
    global bundle_error
    db.init_db()
    _poll.update({"at": float("-inf"), "pointer": None})
    _drift_cache.clear()
    bundle_error = None
    try:
        engine = _load_serving_engine()
        engine_error = None
        try:
            _poll["pointer"] = bundles.pointer_path(engine.schema).stat().st_mtime_ns
        except FileNotFoundError:
            _poll["pointer"] = None
        # Requests are validated against the track the engine serves, so a
        # payload from the other track is a 422 naming the track, not a 500.
        schemas.set_active_schema(engine.schema)
        print(f"[api] track: {engine.schema.name}")
        print(f"[api] loaded models: {', '.join(engine.models)}")
        print(f"[api] primary model: {engine.primary}")
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        # Start anyway so /health can explain what is wrong (no models, an
        # unvouched directory, a model calibrated under an earlier form),
        # rather than the server refusing to boot with a stack trace.
        print(f"[api] STARTING WITHOUT MODELS: {exc}")
        engine = None
        engine_error = str(exc)
    yield


app = FastAPI(
    title="Prism: Explainable Fraud Detection",
    description="Real-time scoring with per-transaction SHAP attribution.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get("api.cors_origins", ["http://localhost:5173"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_engine() -> InferenceEngine:
    """FastAPI dependency yielding the loaded engine.

    A dependency rather than a module-global lookup so tests can substitute a
    fixture-built engine via ``app.dependency_overrides``. Read directly from
    the global, the service tier is untestable without a trained model sitting
    on disk at the configured path.
    """
    _poll_bundle()
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="No models loaded. Run `python -m prism.training.train` and restart.",
        )
    return engine


def get_engine_optional() -> InferenceEngine | None:
    """Non-raising counterpart to get_engine.

    /health must be able to report the *absence* of an engine rather than 503,
    so it cannot depend on get_engine, but it still needs to be overridable in
    tests, which a bare global read is not.
    """
    _poll_bundle()
    return engine


@app.get("/api/v1/health")
def health(active: InferenceEngine | None = Depends(get_engine_optional)) -> dict:
    return {
        "status": "ready" if active else "no_models",
        "track": active.schema.name if active else tracks.active().name,
        "models": sorted(active.models) if active else [],
        "primary_model": active.primary if active else None,
        "explainer": bool(active and active.explainer),
        "detail": None if active else engine_error,
        "bundle_id": serving_bundle if active else None,
        "bundle_error": bundle_error,
    }


def _history_for(active: InferenceEngine, features: dict) -> list:
    """The card's earlier transactions from the audit log, for the velocity
    features (design D3). Empty on a track that derives nothing."""
    schema = active.schema
    if not schema.derived_columns or not schema.card_column:
        return []
    return db.card_history(
        str(features[schema.card_column]),
        before=float(features[schema.time_column]),
        window_seconds=velocity.longest_window(),
    )


def _request_hash(transaction: TransactionIn) -> str:
    """A digest of the request as received, for idempotent retries."""
    canonical = json.dumps(transaction.model_dump(mode="json", exclude_none=True), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _prediction_from_row(row: db.TransactionLog) -> PredictionOut:
    return PredictionOut(
        transaction_id=row.transaction_id,
        flagged=row.flagged,
        primary_model=row.primary_model,
        risk_score=row.risk_score,
        scores=json.loads(row.scores_json),
        explanation=json.loads(row.explanation_json) if row.explanation_json else None,
        latency_ms=row.latency_ms,
        received_at=_iso_utc(row.received_at),
    )


def _stored_or_conflict(transaction_id: str, request_hash: str) -> PredictionOut | None:
    """The stored decision for a retry of the same request; a 409 for a
    different request under the same id; None when the id is new. A row
    without a hash predates the hash and answers any retry."""
    existing = db.find_transaction(transaction_id)
    if existing is None:
        return None
    if existing.request_hash in (None, request_hash):
        return _prediction_from_row(existing)
    raise HTTPException(
        status_code=409,
        detail=f"Transaction {transaction_id} was already scored from a different request; ids are unique.",
    )


@app.post("/api/v1/predict", response_model=PredictionOut)
def predict(
    transaction: TransactionIn,
    background: BackgroundTasks,
    active: InferenceEngine = Depends(get_engine),
) -> PredictionOut:
    started = time.perf_counter()
    features = transaction.to_feature_dict()
    transaction_id = transaction.transaction_id or str(uuid.uuid4())
    request_hash = _request_hash(transaction)
    stored = _stored_or_conflict(transaction_id, request_hash)
    if stored is not None:
        return stored

    try:
        result = active.score(features, history=_history_for(active, features))
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # The log keeps the track's own amount and clock, the merchant category
    # and the channel type, so it can answer the next history query for this card.
    schema = active.schema
    scored = result["features"]
    # One instant for the answer and the audit row, so a retry answered from
    # the row reports the same time the first answer did.
    received_at = datetime.now(timezone.utc)
    try:
        db.log_transaction(
            transaction_id=transaction_id,
            features=scored,
            primary_model=result["primary_model"],
            risk_score=result["risk_score"],
            flagged=result["flagged"],
            latency_ms=result["latency_ms"],
            scores=result["scores"],
            explanation=None,
            context=transaction.context(),
            amount=float(scored[schema.amount_column]),
            time_feature=float(scored[schema.time_column]),
            mcc=str(scored[velocity.MCC_COLUMN]) if velocity.MCC_COLUMN in scored else None,
            channel_type=str(scored[velocity.CHANNEL_COLUMN]) if velocity.CHANNEL_COLUMN in scored else None,
            request_hash=request_hash,
            received_at=received_at,
            request={**scored, "transaction_id": transaction_id, "merchant": transaction.merchant},
            bundle_id=serving_bundle,
        )
    except db.DuplicateTransaction:
        # Two identical requests raced; the first one's row is the answer.
        return _stored_or_conflict(transaction_id, request_hash)

    service_ms = (time.perf_counter() - started) * 1000
    # The answer goes out now; the challengers and the explanation follow it.
    background.add_task(_complete, active, transaction_id, result, service_ms)
    return PredictionOut(
        transaction_id=transaction_id,
        flagged=result["flagged"],
        primary_model=result["primary_model"],
        risk_score=result["risk_score"],
        scores=result["scores"],
        explanation=None,
        latency_ms=service_ms,
        received_at=received_at.isoformat(),
    )


def _complete(active: InferenceEngine, transaction_id: str, result: dict, service_ms: float) -> None:
    """After the answer: the challengers' scores and, for an alert, the
    explanation, onto the audit row. A failure here is printed and the row
    keeps what the request wrote; the answer was already given."""
    scores = list(result["scores"])
    explanation = None
    # Each part on its own: a challenger that fails must not cost the alert
    # its explanation, and the other way round.
    try:
        scores += active.score_challengers(result["frame"])
    except Exception as exc:  # noqa: BLE001  the row is the record; nothing here may hide it
        print(f"[api] challengers for {transaction_id} failed: {exc}")
    try:
        explain_only_when_flagged = config.get("api.explain_when_flagged_only", True)
        if result["flagged"] or not explain_only_when_flagged:
            explanation = active.explain(result["frame"], raw=result["features"])
    except Exception as exc:  # noqa: BLE001
        print(f"[api] explanation of {transaction_id} failed: {exc}")
    db.complete_transaction(transaction_id, scores=scores, explanation=explanation, latency_ms=service_ms)


@app.post("/api/v1/explain")
def explain(
    transaction: TransactionIn, active: InferenceEngine = Depends(get_engine)
) -> dict:
    """SHAP attribution for any transaction, flagged or not."""
    if active.explainer is None:
        raise HTTPException(
            status_code=501,
            detail=f"No SHAP explainer wired for model {active.primary!r}.",
        )
    features = transaction.to_feature_dict()
    try:
        result = active.score(features, history=_history_for(active, features))
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return active.explain(result["frame"], raw=result["features"]) or {}


@app.get("/api/v1/metrics", response_model=MetricsOut)
def metrics(active: InferenceEngine = Depends(get_engine)) -> MetricsOut:
    processed, flagged = db.counts()

    benchmark_path = active.schema.metrics_dir / "benchmark.json"
    benchmark = (
        json.loads(benchmark_path.read_text(encoding="utf-8"))
        if benchmark_path.exists()
        else None
    )

    curves_path = active.schema.metrics_dir / "pr_curves.json"
    pr_curves = (
        json.loads(curves_path.read_text(encoding="utf-8"))
        if curves_path.exists()
        else None
    )

    return MetricsOut(
        track=active.schema.name,
        models=sorted(active.models),
        thresholds={name: active.threshold_for(name) for name in active.models},
        processed=processed,
        flagged=flagged,
        mean_latency_ms=active.mean_latency_ms,
        benchmark=benchmark,
        pr_curves=pr_curves,
    )


@app.get("/api/v1/stats", response_model=StatsOut)
def stats() -> StatsOut:
    """Counters from the audit log. Needs no models, so it never 503s."""
    summary = db.stats(window=int(config.get("api.stats_window", 1000)))
    for key in ("first_received_at", "last_received_at"):
        if summary[key] is not None:
            summary[key] = _iso_utc(datetime.fromisoformat(summary[key]))
    return StatsOut(**summary)


def _iso_utc(moment: datetime) -> str:
    """ISO 8601 with an explicit offset.

    Every timestamp here is written in UTC, but SQLite hands them back naive,
    and a browser parses a naive ISO string as local time. Saying +00:00
    keeps a clock in Mumbai from showing the UTC hour as the local one.
    """
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.isoformat()


def _decision_dict(decision: db.Decision | None) -> dict | None:
    if decision is None:
        return None
    return {
        "transaction_id": decision.transaction_id,
        "actor": decision.actor,
        "verdict": decision.verdict,
        "note": decision.note,
        "decided_at": _iso_utc(decision.decided_at),
    }


@app.get("/api/v1/transactions")
def transactions(
    limit: int = 50, flagged: bool | None = None, card_id: str | None = None
) -> list[dict]:
    """Recent audit-log rows: the feed (no filter), the alert queue
    (flagged=true) and one cardholder's history (card_id)."""
    rows = db.recent(limit=min(limit, 500), flagged=flagged, card_id=card_id)
    decisions = db.latest_decisions([row.transaction_id for row in rows])
    return [
        {
            "transaction_id": row.transaction_id,
            "received_at": _iso_utc(row.received_at),
            "amount": row.amount,
            "risk_score": row.risk_score,
            "flagged": row.flagged,
            "primary_model": row.primary_model,
            "latency_ms": row.latency_ms,
            "card_id": row.card_id,
            "merchant": row.merchant,
            "channel": row.channel,
            "currency": row.currency,
            "scores": json.loads(row.scores_json),
            "explanation": json.loads(row.explanation_json)
            if row.explanation_json
            else None,
            "decisions": {
                "analyst": _decision_dict(decisions.get(row.transaction_id, {}).get("analyst")),
                "customer": _decision_dict(decisions.get(row.transaction_id, {}).get("customer")),
            },
        }
        for row in rows
    ]


# The tokens config.local.yaml.example ships with; a deployment that kept
# them would accept verdicts from anyone who read the example.
PLACEHOLDER_CREDENTIALS = {
    "replace-with-a-long-random-string",
    "replace-with-another-long-random-string",
    "replace-with-a-third-long-random-string",
}


# Every role a credential can name. Each needs its own token, so the label feed can
# never post a verdict and the analyst console can never post a label.
CREDENTIAL_ROLES = ("analyst", "customer", "label_feed")


def _role_from_credential(allowed: tuple[str, ...], what: str):
    """A dependency answering the role whose credential the request carries, refusing
    any role outside ``allowed``.

    Verdicts and labels both teach the model, so whoever can post one can steer it. The
    role is read from a bearer token, never from the body.
    """

    def role(authorization: str | None = Header(default=None)) -> str:
        configured = config.get("api.credentials", {}) or {}
        credentials = {
            name: str(token) for name, token in configured.items() if name in CREDENTIAL_ROLES and token
        }
        if not any(name in credentials for name in allowed):
            raise HTTPException(
                status_code=503,
                detail=f"No {what} credentials are configured: set api.credentials "
                f"({', '.join(allowed)}) in config.local.yaml.",
            )
        if any(token in PLACEHOLDER_CREDENTIALS for token in credentials.values()):
            raise HTTPException(
                status_code=503,
                detail="api.credentials still holds a placeholder from config.local.yaml.example; set real tokens.",
            )
        if len(set(credentials.values())) < len(credentials):
            raise HTTPException(
                status_code=503,
                detail="api.credentials gives the same token to more than one role; each role needs its own.",
            )
        scheme, _, token = (authorization or "").partition(" ")
        token = token.strip()
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(
                status_code=401,
                detail=f"A {what} needs a bearer credential ({', '.join(allowed)}).",
                headers={"WWW-Authenticate": "Bearer"},
            )
        # Compared as bytes: compare_digest refuses non-ASCII text, and a stray
        # character in a header must be a 401, not a 500.
        presented = token.encode("utf-8")
        for name, expected in credentials.items():
            if name in allowed and secrets.compare_digest(presented, expected.encode("utf-8")):
                return name
        raise HTTPException(
            status_code=401,
            detail=f"The credential is not one this service accepts a {what} from.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return role


_actor_from_credential = _role_from_credential(("analyst", "customer"), "verdict")
_label_feed_from_credential = _role_from_credential(("label_feed",), "label")


def _live_track_only() -> tracks.TrackSchema:
    """The live loop needs a stream and delayed labels; only the Indian track has them."""
    schema = schemas.active_schema()
    if schema.contract_feed is None:
        raise HTTPException(
            status_code=404,
            detail=f"The live loop runs on the Indian track only; this service serves {schema.name}.",
        )
    return schema


def _label_dict(row: db.Label) -> dict:
    return {
        "transaction_id": row.transaction_id,
        "label": row.label,
        "source": row.source,
        "known_at": row.known_at,
        "recorded_at": _iso_utc(row.recorded_at),
    }


@app.post("/api/v1/decisions", response_model=DecisionOut, status_code=201)
def decide(decision: DecisionIn, actor: str = Depends(_actor_from_credential)) -> DecisionOut:
    """Record an analyst verdict or a cardholder response, as the role the
    credential names. Append-only."""
    stored = db.record_decision(
        transaction_id=decision.transaction_id,
        actor=actor,
        verdict=decision.verdict,
        note=decision.note,
    )
    if stored is None:
        raise HTTPException(
            status_code=404,
            detail=f"Transaction {decision.transaction_id} is not in the audit log.",
        )
    # A verdict is a label the bank now knows, from the moment the stream has reached.
    clock = db.stream_clock()
    db.record_label(
        transaction_id=decision.transaction_id,
        label=1 if decision.verdict == "fraud" else 0,
        source=actor,
        known_at=clock if clock is not None else 0.0,
    )
    return DecisionOut(**_decision_dict(stored))


@app.get("/api/v1/live", response_model=LiveOut)
def live_state(
    _schema: tracks.TrackSchema = Depends(_live_track_only),
    active: InferenceEngine = Depends(get_engine),
) -> LiveOut:
    """The loop at a glance: the serving bundle and its champion, the labels received by
    source and the last retrains with their verdicts."""
    if not bundles.live_enabled():
        raise HTTPException(
            status_code=404,
            detail="The live loop is not switched on for this service; start it with PRISM_LIVE=1.",
        )
    return LiveOut(
        bundle_id=serving_bundle,
        champion=active.primary,
        labels=db.label_counts(),
        events=bundles.events(active.schema)[-int(config.get("live.events_shown", 20)):],
    )


@app.get("/api/v1/drift", response_model=DriftOut)
def drift_report(
    _schema: tracks.TrackSchema = Depends(_live_track_only),
    active: InferenceEngine = Depends(get_engine),
) -> DriftOut:
    """The serving bundle's drift: its reference against the service's most recent rows."""
    reference = drift.load(active.model_dir)
    if reference is None:
        raise HTTPException(
            status_code=409,
            detail=f"The serving models have no drift reference in {active.model_dir}; retrain them "
            f"(python -m prism.training.train --track {active.schema.name}) or promote a bundle.",
        )
    s = dict(config.get("live"))
    key = (str(active.model_dir), serving_bundle, db.database_identity())
    cached = _drift_cache.get("report")
    if (
        cached is not None
        and _drift_cache.get("key") == key
        and time.monotonic() - _drift_cache["at"] < float(s["drift_refresh_seconds"])
    ):
        return cached
    rows, scores, scored_by = db.recent_scored(int(s["drift_window_rows"]))
    needed = int(s["drift_min_rows"])
    if len(rows) < needed:
        raise HTTPException(
            status_code=409,
            detail=f"Drift needs at least {needed} scored rows in the audit log; it holds {len(rows)}.",
        )
    # Score drift compares the serving bundle's own scores with its reference; rows another
    # bundle scored would measure the change of model, not of traffic.
    own = [score for score, bundle in zip(scores, scored_by) if bundle == serving_bundle]
    report = drift.report(
        reference, pd.DataFrame(rows), np.asarray(own, dtype=float) if len(own) >= needed else None, s
    )
    answer = DriftOut(**report, score_rows=len(own), bundle_id=serving_bundle)
    _drift_cache.update({"key": key, "at": time.monotonic(), "report": answer})
    return answer


@app.post("/api/v1/labels", response_model=LabelOut, status_code=201)
def post_label(
    label: LabelIn,
    response: Response,
    _role: str = Depends(_label_feed_from_credential),
    _schema: tracks.TrackSchema = Depends(_live_track_only),
) -> LabelOut:
    """A label as the bank learns it: an investigator's verdict, a chargeback or a clean
    maturity. 201 when stored, 200 with the stored row for a repeat."""
    stored, written = db.record_label(
        transaction_id=label.transaction_id,
        label=label.discovered_label,
        source=label.label_source,
        known_at=label.discovery_time_seconds,
    )
    if stored is None:
        raise HTTPException(
            status_code=404,
            detail=f"Transaction {label.transaction_id} is not in the audit log; a label needs a scored transaction.",
        )
    if not written:
        response.status_code = 200
    return LabelOut(**_label_dict(stored))
