"""The audit log must survive being upgraded in place (spec section 4.2)."""

from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("sqlalchemy")

from prism.api import db  # noqa: E402

OLD_SCHEMA = """
CREATE TABLE transaction_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id VARCHAR(64),
    received_at DATETIME,
    amount FLOAT,
    time_feature FLOAT,
    primary_model VARCHAR(32),
    risk_score FLOAT,
    flagged BOOLEAN,
    latency_ms FLOAT,
    scores_json TEXT,
    explanation_json TEXT
)
"""


def _old_log(path):
    with sqlite3.connect(path) as conn:
        conn.execute(OLD_SCHEMA)
        conn.execute(
            "INSERT INTO transaction_log (transaction_id, received_at, amount, time_feature,"
            " primary_model, risk_score, flagged, latency_ms, scores_json)"
            " VALUES ('txn-old', '2026-09-12 06:00:00.000000', 12.5, 100.0, 'xgboost',"
            " 0.01, 0, 20.0, '[]')"
        )


def test_init_db_adds_context_columns_to_an_old_audit_log(tmp_path):
    path = tmp_path / "old.db"
    _old_log(path)

    db.init_db(f"sqlite:///{path.as_posix()}")

    with sqlite3.connect(path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(transaction_log)")}
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"card_id", "merchant", "channel", "currency"} <= columns
    assert "decision" in tables, "an upgraded log must also gain the decision table"

    rows = db.recent(limit=5)
    assert rows[0].transaction_id == "txn-old"
    assert rows[0].card_id is None


def test_init_db_is_idempotent_on_an_upgraded_log(tmp_path):
    path = tmp_path / "twice.db"
    _old_log(path)
    url = f"sqlite:///{path.as_posix()}"

    db.init_db(url)
    db.init_db(url)  # must not raise on columns that already exist

    with sqlite3.connect(path) as conn:
        names = [row[1] for row in conn.execute("PRAGMA table_info(transaction_log)")]
    assert names.count("card_id") == 1


def _fresh(tmp_path):
    url = f"sqlite:///{(tmp_path / 'history.db').as_posix()}"
    db.init_db(url)
    return url


def _log(txn, card, t, amount, mcc, channel):
    db.log_transaction(
        transaction_id=txn, features={"x": 1.0}, primary_model="xgboost", risk_score=0.1, flagged=False,
        latency_ms=1.0, scores=[], explanation=None,
        context={"card_id": card, "merchant": None, "channel": None, "currency": "INR"},
        amount=amount, time_feature=t, mcc=mcc, channel_type=channel,
    )


def test_the_scored_amount_clock_category_and_channel_are_stored(tmp_path):
    _fresh(tmp_path)
    _log("t1", "CARD_1", 1_000.0, 250.0, "5411", "CP_POS_CHIP")
    row = db.recent(limit=1)[0]
    assert row.amount == 250.0 and row.time_feature == 1_000.0
    assert row.mcc == "5411" and row.channel_type == "CP_POS_CHIP" and row.channel is None


def test_card_history_is_strictly_earlier_and_inside_the_window(tmp_path):
    _fresh(tmp_path)
    _log("a", "CARD_1", 1_000.0, 10.0, "5411", "CP_POS_CHIP")
    _log("b", "CARD_1", 2_000.0, 20.0, "5812", "CNP_WEB")
    _log("c", "CARD_1", 3_000.0, 30.0, "5732", "CNP_WEB")   # the row being scored: same clock, excluded
    _log("d", "CARD_1", 4_000.0, 40.0, "5732", "CNP_WEB")   # the future: excluded
    _log("e", "CARD_2", 2_500.0, 50.0, "5411", "CP_POS_CHIP")  # another card
    history = db.card_history("CARD_1", before=3_000.0, window_seconds=1_500.0)
    assert history == [(2_000.0, 20.0, "5812", "CNP_WEB")]
    assert db.card_history("CARD_1", before=3_000.0, window_seconds=10_000.0) == [
        (1_000.0, 10.0, "5411", "CP_POS_CHIP"), (2_000.0, 20.0, "5812", "CNP_WEB"),
    ]
    assert db.card_history("CARD_9", before=3_000.0, window_seconds=10_000.0) == []


def test_an_old_log_gains_the_mcc_column(tmp_path):
    url = _fresh(tmp_path)
    import sqlalchemy

    engine = sqlalchemy.create_engine(url)
    with engine.begin() as conn:
        conn.execute(sqlalchemy.text("ALTER TABLE transaction_log DROP COLUMN mcc"))
    assert "mcc" in db._ensure_columns(engine)
    _log("t1", "CARD_1", 1_000.0, 250.0, "5411", "CP_POS_CHIP")
    assert db.recent(limit=1)[0].mcc == "5411"


def test_two_identical_payments_both_count_in_history(tmp_path):
    """Ids are unique, so two rows are two payments: a burst of identical
    micro-authorisations must count as the burst it is."""
    _fresh(tmp_path)
    _log("a", "CARD_1", 1_000.0, 10.0, "5411", "CP_POS_CHIP")
    _log("b", "CARD_1", 1_000.0, 10.0, "5411", "CP_POS_CHIP")
    assert db.card_history("CARD_1", before=2_000.0, window_seconds=10_000.0) == [
        (1_000.0, 10.0, "5411", "CP_POS_CHIP"), (1_000.0, 10.0, "5411", "CP_POS_CHIP"),
    ]


def test_init_db_adds_the_history_columns_to_an_old_log(tmp_path):
    import sqlalchemy

    path = tmp_path / "old.db"
    _old_log(path)
    url = f"sqlite:///{path.as_posix()}"
    db.init_db(url)
    columns = {c["name"] for c in sqlalchemy.inspect(sqlalchemy.create_engine(url)).get_columns("transaction_log")}
    assert {"channel_type", "request_hash", "mcc"} <= columns


def test_a_relative_sqlite_url_is_resolved_against_the_project_root(tmp_path, monkeypatch):
    """config.yaml says sqlite:///prism_audit.db; a service started from another
    directory must not open a second, empty log there."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setattr(db.config, "PROJECT_ROOT", tmp_path)
    monkeypatch.chdir(elsewhere)

    db.init_db("sqlite:///relative_audit.db")
    _log("t1", "CARD_1", 1_000.0, 250.0, "5411", "CP_POS_CHIP")

    assert (tmp_path / "relative_audit.db").exists()
    assert not (elsewhere / "relative_audit.db").exists()
    assert db.resolve_database_url("sqlite:///:memory:") == "sqlite:///:memory:"
    assert db.resolve_database_url("postgresql://u:p@host/prism") == "postgresql://u:p@host/prism"
    absolute = f"sqlite:///{(tmp_path / 'abs.db').as_posix()}"
    assert db.resolve_database_url(absolute) == absolute


def test_init_db_refuses_an_old_log_with_duplicate_ids(tmp_path):
    """A verdict on a shared id would attach to two cardholders' payments;
    the upgrade stops and names the ids rather than guess."""
    path = tmp_path / "dups.db"
    with sqlite3.connect(path) as conn:
        conn.execute(OLD_SCHEMA)
        for _ in range(2):
            conn.execute(
                "INSERT INTO transaction_log (transaction_id, received_at, amount, time_feature,"
                " primary_model, risk_score, flagged, latency_ms, scores_json)"
                " VALUES ('txn-dup', '2026-09-12 06:00:00.000000', 12.5, 100.0, 'xgboost', 0.01, 0, 20.0, '[]')"
            )
    with pytest.raises(RuntimeError, match="txn-dup"):
        db.init_db(f"sqlite:///{path.as_posix()}")


def test_a_fresh_log_rejects_a_second_row_with_the_same_id(tmp_path):
    _fresh(tmp_path)
    _log("t1", "CARD_1", 1_000.0, 250.0, "5411", "CP_POS_CHIP")
    with pytest.raises(db.DuplicateTransaction):
        _log("t1", "CARD_1", 2_000.0, 250.0, "5411", "CP_POS_CHIP")


def test_duplicate_id_refusal_counts_them_and_names_a_remedy(tmp_path):
    path = tmp_path / "dups.db"
    with sqlite3.connect(path) as conn:
        conn.execute(OLD_SCHEMA)
        for txn in ("txn-a", "txn-a", "txn-b", "txn-b", "txn-b"):
            conn.execute(
                "INSERT INTO transaction_log (transaction_id, received_at, amount, time_feature,"
                " primary_model, risk_score, flagged, latency_ms, scores_json)"
                f" VALUES ('{txn}', '2026-09-12 06:00:00.000000', 12.5, 100.0, 'xgboost', 0.01, 0, 20.0, '[]')"
            )
    with pytest.raises(RuntimeError) as caught:
        db.init_db(f"sqlite:///{path.as_posix()}")
    message = str(caught.value)
    assert "2 ids" in message and "5 rows" in message and "api.database_url" in message


def test_an_old_log_backfills_the_channel_type_from_the_display_channel(tmp_path):
    """Rows logged before channel_type existed carried the channel type in the
    display field (the Indian replay sends the same value in both); the
    upgrade copies it so the velocity features keep their history."""
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:
        conn.execute(OLD_SCHEMA)
        conn.execute("ALTER TABLE transaction_log ADD COLUMN channel VARCHAR(32)")
        conn.execute(
            "INSERT INTO transaction_log (transaction_id, received_at, amount, time_feature,"
            " primary_model, risk_score, flagged, latency_ms, scores_json, channel)"
            " VALUES ('txn-old', '2026-09-12 06:00:00.000000', 12.5, 100.0, 'xgboost', 0.01, 0, 20.0, '[]', 'CNP_WEB')"
        )
    db.init_db(f"sqlite:///{path.as_posix()}")
    assert db.recent(limit=1)[0].channel_type == "CNP_WEB"
