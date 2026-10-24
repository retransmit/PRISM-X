"""Persistence tier: the audit log (§6.2.4).

Every scored transaction is recorded with its prediction, per-model scores,
SHAP attribution and latency. This is not telemetry: §3.3 of the proposal
justifies the whole project on regulatory grounds, and "we froze this account
because the model said so" is only defensible if the reasoning was retained at
decision time. The log is also the raw material for later retraining.

SQLite by default so the project runs with no external services; point
``api.database_url`` at Postgres for the deployed configuration.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    case,
    create_engine,
    func,
    inspect,
    text,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from prism import config


class Base(DeclarativeBase):
    pass


class TransactionLog(Base):
    __tablename__ = "transaction_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transaction_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)

    amount: Mapped[float] = mapped_column(Float)
    time_feature: Mapped[float] = mapped_column(Float)

    primary_model: Mapped[str] = mapped_column(String(32))
    risk_score: Mapped[float] = mapped_column(Float, index=True)
    flagged: Mapped[bool] = mapped_column(Boolean, index=True)
    latency_ms: Mapped[float] = mapped_column(Float)

    # JSON blobs: per-model scores and the SHAP attribution as served to the UI.
    scores_json: Mapped[str] = mapped_column(Text)
    explanation_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Context supplied by the stream and never seen by the model: who paid,
    # where, on which channel, in which currency. Nullable because the ULB
    # replay has none of them, and rows logged before these columns existed
    # are null too (see _ensure_columns).
    card_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    merchant: Mapped[str | None] = mapped_column(String(120), nullable=True)
    channel: Mapped[str | None] = mapped_column(String(32), nullable=True)
    currency: Mapped[str | None] = mapped_column(String(8), nullable=True)
    # Merchant category of the transaction, kept so the velocity features can
    # ask "first purchase in this category in 30 days" at serving time.
    mcc: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # The channel type that was scored (a model input), separate from the
    # display field `channel` the boards show. The velocity features read this.
    channel_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # A digest of the request that produced this row, so a retry of the same
    # request is answered from the log and a different request under the
    # same id is refused. Null on rows logged before it existed.
    request_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # The inputs that were scored, as JSON, so the log can be learned from: the live
    # loop rebuilds its training frame from these. Null on rows logged before it existed.
    request_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # The model bundle that scored the row ("offline" for models/<track>/), so a decision
    # can be traced to its model and score drift compares like with like.
    bundle_id: Mapped[str | None] = mapped_column(String(40), nullable=True)


# Column name -> DDL type, for logs created before the context columns existed.
CONTEXT_COLUMNS = {
    "card_id": "VARCHAR(64)",
    "merchant": "VARCHAR(120)",
    "channel": "VARCHAR(32)",
    "currency": "VARCHAR(8)",
    "mcc": "VARCHAR(64)",
    "channel_type": "VARCHAR(32)",
    "request_hash": "VARCHAR(64)",
    "request_json": "TEXT",
    "bundle_id": "VARCHAR(40)",
}


def _ensure_columns(engine) -> list[str]:
    """Add the context columns to a transaction_log that predates them.

    create_all never alters an existing table, so an audit log written by an
    older service would make every read fail on the first missing column.
    ALTER TABLE ... ADD COLUMN is supported by SQLite and Postgres alike and
    leaves existing rows untouched with nulls. Returns the columns it added.
    """
    existing = {column["name"] for column in inspect(engine).get_columns("transaction_log")}
    added: list[str] = []
    with engine.begin() as conn:
        for name, ddl in CONTEXT_COLUMNS.items():
            if name not in existing:
                conn.execute(text(f"ALTER TABLE transaction_log ADD COLUMN {name} {ddl}"))
                added.append(name)
        conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_transaction_log_card_id ON transaction_log (card_id)")
        )
        if "channel_type" in added and "channel" in existing:
            # Rows logged before channel_type existed carried the channel type
            # in the display field; the velocity features keep their history.
            conn.execute(text(
                "UPDATE transaction_log SET channel_type = channel "
                "WHERE channel_type IS NULL AND channel IS NOT NULL"
            ))
    return added


class DuplicateTransaction(RuntimeError):
    """A row with this transaction id is already in the log."""


def _ensure_unique_ids(engine) -> None:
    """Refuse a log whose ids are not unique, then make them so.

    A log written by an older service may hold one id on two cardholders'
    payments; a verdict on that id would attach to both. The upgrade stops
    and names them rather than guess which row the verdicts meant.
    """
    unique_already = any(
        index["unique"] and list(index["column_names"]) == ["transaction_id"]
        for index in inspect(engine).get_indexes("transaction_log")
    )
    with engine.begin() as conn:
        duplicates = conn.execute(text(
            "SELECT transaction_id, COUNT(*) FROM transaction_log "
            "GROUP BY transaction_id HAVING COUNT(*) > 1 ORDER BY COUNT(*) DESC"
        )).all()
        if duplicates:
            rows = sum(int(row[1]) for row in duplicates)
            named = ", ".join(f"{row[0]} (x{row[1]})" for row in duplicates[:10])
            more = f", and {len(duplicates) - 10} more" if len(duplicates) > 10 else ""
            raise RuntimeError(
                f"The audit log holds duplicate transaction ids and cannot be upgraded: {len(duplicates)} ids "
                f"on {rows} rows ({named}{more}). Point api.database_url at a new log, or remove the "
                "duplicate rows from this one first."
            )
        if not unique_already:
            conn.execute(text(
                "CREATE UNIQUE INDEX ux_transaction_log_transaction_id ON transaction_log (transaction_id)"
            ))


class Decision(Base):
    """A human's response to a scored transaction.

    Append-only: the latest row per (transaction, actor) is the current
    decision and earlier rows are history. An audit log that can be edited is
    not an audit log, so nothing here is ever updated or deleted.
    """

    __tablename__ = "decision"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transaction_id: Mapped[str] = mapped_column(String(64), index=True)
    actor: Mapped[str] = mapped_column(String(16))
    verdict: Mapped[str] = mapped_column(String(16))
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


# Who taught the bank a label, strongest first (live-loop plan, Decision 7): a chargeback
# is not overruled by a verdict, an analyst's verdict outranks the investigator queue,
# a cardholder's word outranks only a clean maturity.
LABEL_SOURCES = ("CHARGEBACK_DISPUTE", "analyst", "INVESTIGATOR_ALERT", "customer", "UNLABELLED")


class Label(Base):
    """What the bank has learned about a scored transaction, and when.

    Append-only like the decisions: a later label of one source is a new row and the
    latest of that source is the one in force. ``known_at`` is on the stream clock
    (seconds, the same clock as ``time_feature``), so a training frame can ask which
    labels were known at any instant of the stream.
    """

    __tablename__ = "label"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transaction_id: Mapped[str] = mapped_column(String(64), index=True)
    label: Mapped[int] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(24))
    known_at: Mapped[float] = mapped_column(Float, index=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


_engine = None
_SessionFactory: sessionmaker[Session] | None = None


def resolve_database_url(url: str) -> str:
    """A relative SQLite path is taken from the project root, not the working
    directory, so a service started from elsewhere opens the same log."""
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        return url
    path = url[len(prefix):]
    if path == ":memory:" or Path(path).is_absolute():
        return url
    return prefix + (config.PROJECT_ROOT / path).as_posix()


def init_db(database_url: str | None = None):
    """Create the engine and tables. Idempotent, safe to call on every start."""
    global _engine, _SessionFactory
    url = resolve_database_url(database_url or config.get("api.database_url"))

    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    _engine = create_engine(url, future=True, connect_args=connect_args)
    Base.metadata.create_all(_engine)
    _ensure_columns(_engine)
    _ensure_unique_ids(_engine)
    _SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def dispose() -> None:
    """Close the engine's connections; a scratch database can then be removed."""
    global _engine, _SessionFactory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionFactory = None


def get_session() -> Session:
    if _SessionFactory is None:
        init_db()
    assert _SessionFactory is not None
    return _SessionFactory()


def find_transaction(transaction_id: str) -> TransactionLog | None:
    """The row for an id, or None. Sessions do not expire on commit, so the
    row is readable after the session closes."""
    with get_session() as session:
        return (
            session.query(TransactionLog)
            .filter(TransactionLog.transaction_id == transaction_id)
            .first()
        )


def log_transaction(
    *,
    transaction_id: str,
    features: dict[str, Any],
    primary_model: str,
    risk_score: float,
    flagged: bool,
    latency_ms: float,
    scores: list[dict[str, Any]],
    explanation: dict[str, Any] | None,
    context: dict[str, str | None] | None = None,
    amount: float | None = None,
    time_feature: float | None = None,
    request_hash: str | None = None,
    received_at: datetime | None = None,
    mcc: str | None = None,
    channel_type: str | None = None,
    request: dict[str, Any] | None = None,
    bundle_id: str | None = None,
) -> None:
    """Record one scored transaction. ``amount`` and ``time_feature`` are the
    track's own values when given; the ULB column names remain the fallback."""
    given = context or {}
    with get_session() as session:
        session.add(
            TransactionLog(
                transaction_id=transaction_id,
                received_at=received_at or datetime.now(timezone.utc),
                amount=float(amount if amount is not None else features.get("Amount", 0.0)),
                time_feature=float(
                    time_feature if time_feature is not None else features.get("Time", 0.0)
                ),
                primary_model=primary_model,
                risk_score=risk_score,
                flagged=flagged,
                latency_ms=latency_ms,
                scores_json=json.dumps(scores),
                explanation_json=json.dumps(explanation) if explanation else None,
                card_id=given.get("card_id"),
                merchant=given.get("merchant"),
                channel=given.get("channel"),
                currency=given.get("currency"),
                mcc=mcc,
                channel_type=channel_type,
                request_hash=request_hash,
                request_json=json.dumps(request) if request is not None else None,
                bundle_id=bundle_id,
            )
        )
        try:
            session.commit()
        except IntegrityError as exc:
            session.rollback()
            raise DuplicateTransaction(transaction_id) from exc


def card_history(
    card_id: str, before: float, window_seconds: float
) -> list[tuple[float, float, str, str]]:
    """The card's transactions strictly before ``before`` and inside the window,
    oldest first, in the shape prism.features.velocity.velocity_row takes.

    The window is measured on the transaction's own clock (time_feature), not
    on received_at: a replay compresses days into minutes. Rows count as
    logged: ids are unique, one row per payment, so two genuine payments with
    identical values both count and the velocity features match what
    training computed offline.
    """
    with get_session() as session:
        rows = (
            session.query(
                TransactionLog.time_feature,
                TransactionLog.amount,
                TransactionLog.mcc,
                TransactionLog.channel_type,
            )
            .filter(TransactionLog.card_id == card_id)
            .filter(TransactionLog.time_feature < before)
            .filter(TransactionLog.time_feature >= before - window_seconds)
            .order_by(TransactionLog.time_feature.asc())
            .all()
        )
    return [(float(t), float(a), str(m or ""), str(c or "")) for t, a, m, c in rows]


def logged_requests(until: float | None = None) -> list[dict[str, Any]]:
    """Every logged row that kept its inputs, as the stored dict plus ``received_at``,
    oldest stream time first; only rows before ``until`` on the stream clock when given."""
    with get_session() as session:
        query = session.query(
            TransactionLog.request_json, TransactionLog.received_at
        ).filter(TransactionLog.request_json.isnot(None))
        if until is not None:
            query = query.filter(TransactionLog.time_feature < until)
        rows = query.order_by(TransactionLog.time_feature.asc(), TransactionLog.id.asc()).all()
    return [{**json.loads(body), "received_at": received} for body, received in rows]


def recent_scored(limit: int) -> tuple[list[dict[str, Any]], list[float], list[str | None]]:
    """The newest ``limit`` rows that kept their inputs, oldest first, the risk score each
    was given and the bundle that gave it: the window the drift report compares with the
    serving bundle's reference."""
    with get_session() as session:
        rows = (
            session.query(TransactionLog.request_json, TransactionLog.risk_score, TransactionLog.bundle_id)
            .filter(TransactionLog.request_json.isnot(None))
            .order_by(TransactionLog.id.desc())
            .limit(limit)
            .all()
        )
    rows = rows[::-1]
    return (
        [json.loads(body) for body, _, _ in rows],
        [float(score) for _, score, _ in rows],
        [bundle for _, _, bundle in rows],
    )


def database_identity() -> str:
    """The log this process writes, for keying anything cached per log."""
    return str(_engine.url) if _engine is not None else ""


def stream_clock() -> float | None:
    """The latest transaction time in the log: where the stream has reached."""
    with get_session() as session:
        latest = session.query(func.max(TransactionLog.time_feature)).scalar()
    return float(latest) if latest is not None else None


def complete_transaction(
    transaction_id: str, *, scores: list[dict[str, Any]], explanation: dict[str, Any] | None, latency_ms: float
) -> None:
    """Attach what was computed after the answer to the row the request wrote."""
    with get_session() as session:
        row = session.query(TransactionLog).filter(TransactionLog.transaction_id == transaction_id).first()
        if row is None:
            return
        row.scores_json = json.dumps(scores)
        row.explanation_json = json.dumps(explanation) if explanation else None
        row.latency_ms = float(latency_ms)
        session.commit()


def recent(
    limit: int = 50, flagged: bool | None = None, card_id: str | None = None
) -> list[TransactionLog]:
    """Newest rows first. `flagged` is a three-way switch: None means both."""
    with get_session() as session:
        query = session.query(TransactionLog)
        if flagged is not None:
            query = query.filter(TransactionLog.flagged.is_(flagged))
        if card_id is not None:
            query = query.filter(TransactionLog.card_id == card_id)
        return list(query.order_by(TransactionLog.id.desc()).limit(limit).all())


def counts() -> tuple[int, int]:
    """(processed, flagged) for the /metrics endpoint."""
    with get_session() as session:
        total = session.query(TransactionLog).count()
        flagged = (
            session.query(TransactionLog).filter(TransactionLog.flagged.is_(True)).count()
        )
        return total, flagged


def record_decision(
    *, transaction_id: str, actor: str, verdict: str, note: str | None
) -> Decision | None:
    """Append a decision. Returns None when the transaction was never scored."""
    with get_session() as session:
        known = (
            session.query(TransactionLog.id)
            .filter(TransactionLog.transaction_id == transaction_id)
            .first()
        )
        if known is None:
            return None
        decision = Decision(
            transaction_id=transaction_id,
            actor=actor,
            verdict=verdict,
            note=note,
            decided_at=datetime.now(timezone.utc),
        )
        session.add(decision)
        session.commit()
        session.refresh(decision)
        return decision


def record_label(
    *, transaction_id: str, label: int, source: str, known_at: float
) -> tuple[Label | None, bool]:
    """Append a label. (None, False) when the transaction was never scored; the stored
    row and False when it repeats that source's latest label exactly (a retry)."""
    if source not in LABEL_SOURCES:
        raise ValueError(f"Unknown label source {source!r}; expected one of {LABEL_SOURCES}")
    with get_session() as session:
        known = (
            session.query(TransactionLog.id)
            .filter(TransactionLog.transaction_id == transaction_id)
            .first()
        )
        if known is None:
            return None, False
        latest = (
            session.query(Label)
            .filter(Label.transaction_id == transaction_id, Label.source == source)
            .order_by(Label.known_at.desc(), Label.id.desc())
            .first()
        )
        if latest is not None and latest.label == int(label) and latest.known_at == float(known_at):
            return latest, False
        row = Label(
            transaction_id=transaction_id,
            label=int(label),
            source=source,
            known_at=float(known_at),
            recorded_at=datetime.now(timezone.utc),
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return row, True


def known_labels(as_of: float) -> dict[str, int]:
    """The label in force for every transaction with a label known at ``as_of``: the
    strongest source wins, and within a source the latest."""
    rank = {source: i for i, source in enumerate(LABEL_SOURCES)}
    with get_session() as session:
        rows = (
            session.query(Label.transaction_id, Label.label, Label.source, Label.known_at, Label.id)
            .filter(Label.known_at <= as_of)
            .all()
        )
    best: dict[str, tuple[tuple[int, float, int], int]] = {}
    for transaction_id, label, source, known_at, row_id in rows:
        key = (-rank[source], float(known_at), int(row_id))
        if transaction_id not in best or key > best[transaction_id][0]:
            best[transaction_id] = (key, int(label))
    return {transaction_id: label for transaction_id, (_, label) in best.items()}


def label_counts() -> dict[str, int]:
    """Label rows per source."""
    with get_session() as session:
        rows = session.query(Label.source, func.count(Label.id)).group_by(Label.source).all()
    return {source: int(n) for source, n in rows}


def latest_decisions(transaction_ids: list[str]) -> dict[str, dict[str, Decision]]:
    """{transaction_id: {actor: latest Decision}} for the given transactions.

    One query for the whole listing. Rows come back oldest first, so a later
    row for the same (transaction, actor) simply overwrites the earlier one.
    """
    if not transaction_ids:
        return {}
    with get_session() as session:
        rows = (
            session.query(Decision)
            .filter(Decision.transaction_id.in_(transaction_ids))
            .order_by(Decision.id.asc())
            .all()
        )
    latest: dict[str, dict[str, Decision]] = {}
    for row in rows:
        latest.setdefault(row.transaction_id, {})[row.actor] = row
    return latest


def stats(window: int = 1000) -> dict[str, Any]:
    """Everything the boards' counters show, computed from the log alone.

    Decisions are counted latest-per-actor over flagged rows, in Python: the
    decision table holds human actions, so it stays small even when the log
    does not.
    """
    now = datetime.now(timezone.utc)
    with get_session() as session:
        processed = session.query(TransactionLog).count()
        flagged = session.query(TransactionLog).filter(TransactionLog.flagged.is_(True)).count()
        per_minute = (
            session.query(TransactionLog)
            .filter(TransactionLog.received_at >= now - timedelta(seconds=60))
            .count()
        )
        latencies = [
            row[0]
            for row in session.query(TransactionLog.latency_ms)
            .order_by(TransactionLog.id.desc())
            .limit(window)
            .all()
        ]
        first, last = session.query(
            func.min(TransactionLog.received_at), func.max(TransactionLog.received_at)
        ).one()
        flagged_ids = {
            row[0]
            for row in session.query(TransactionLog.transaction_id)
            .filter(TransactionLog.flagged.is_(True))
            .all()
        }
        decisions = session.query(Decision).order_by(Decision.id.asc()).all()
        held = func.sum(case((TransactionLog.flagged.is_(True), 1), else_=0))
        cards = (
            session.query(
                TransactionLog.card_id,
                func.count(TransactionLog.id),
                held,
                func.max(TransactionLog.id),
            )
            .filter(TransactionLog.card_id.isnot(None))
            .group_by(TransactionLog.card_id)
            .order_by(func.max(TransactionLog.id).desc())
            .limit(50)
            .all()
        )

    latest: dict[tuple[str, str], str] = {}
    for decision in decisions:
        latest[(decision.transaction_id, decision.actor)] = decision.verdict
    analyst = {t: v for (t, a), v in latest.items() if a == "analyst" and t in flagged_ids}
    customer = {t: v for (t, a), v in latest.items() if a == "customer" and t in flagged_ids}

    def percentile(q: float) -> float:
        return float(np.percentile(latencies, q)) if latencies else 0.0

    return {
        "processed": processed,
        "flagged": flagged,
        "flag_rate": (flagged / processed) if processed else 0.0,
        "per_minute": per_minute,
        "latency_ms": {"p50": percentile(50), "p95": percentile(95), "p99": percentile(99)},
        "decided": len(analyst),
        "confirmed_fraud": sum(1 for v in analyst.values() if v == "fraud"),
        "marked_legitimate": sum(1 for v in analyst.values() if v == "legitimate"),
        "customer_responses": len(customer),
        "first_received_at": first.isoformat() if first else None,
        "last_received_at": last.isoformat() if last else None,
        "cards": [
            {"card_id": card_id, "transactions": int(n), "held": int(h or 0)}
            for card_id, n, h, _ in cards
        ],
    }
