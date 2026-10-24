"""The replay must spread rows over a small pool of plainly synthetic cards."""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import pytest

pytest.importorskip("httpx")

SPEC = importlib.util.spec_from_file_location(
    "stream", Path(__file__).resolve().parents[1] / "simulator" / "stream.py"
)
stream = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(stream)


def test_card_ids_are_plainly_synthetic():
    assert stream.card_pool(12) == [f"card-{i:02d}" for i in range(1, 13)]


def test_a_small_pool_is_fully_used():
    rng = random.Random(0)
    pool = stream.card_pool(12)
    seen = {rng.choice(pool) for _ in range(500)}
    assert seen == set(pool)


def test_replay_attaches_a_card_id_from_the_pool(monkeypatch):
    """Drive run() against a fake client: every posted payload carries a pool id."""
    import asyncio

    import pandas as pd

    rows = pd.DataFrame({name: [0.0, 1.0] for name in stream.FEATURES} | {"Class": [0, 1]})
    monkeypatch.setattr(
        stream, "load_unseen_rows", lambda: (rows[rows.Class == 1], rows[rows.Class == 0])
    )
    posted: list[dict] = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"flagged": False, "transaction_id": "x", "risk_score": 0.0, "latency_ms": 1.0}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url):
            return FakeResponse()

        async def post(self, url, json):
            posted.append(json)
            return FakeResponse()

    monkeypatch.setattr(stream.httpx, "AsyncClient", FakeClient)

    asyncio.run(
        stream.run("http://test/api/v1/predict", rate=0.0, jitter=0.0, fraud_boost=0.0, limit=5, cards=3)
    )

    assert len(posted) == 5
    assert {p["card_id"] for p in posted} <= set(stream.card_pool(3))
    assert all(name in posted[0] for name in stream.FEATURES)


def test_transaction_ids_are_unique_across_runs(monkeypatch):
    """Two replays must never reuse an id: decisions attach by transaction id,
    and a restarted simulator would otherwise pin them to last week's rows."""
    import asyncio
    import re

    import pandas as pd

    rows = pd.DataFrame({name: [0.0, 1.0] for name in stream.FEATURES} | {"Class": [0, 1]})
    monkeypatch.setattr(
        stream, "load_unseen_rows", lambda: (rows[rows.Class == 1], rows[rows.Class == 0])
    )
    posted: list[dict] = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"flagged": False, "transaction_id": "x", "risk_score": 0.0, "latency_ms": 1.0}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url):
            return FakeResponse()

        async def post(self, url, json):
            posted.append(json)
            return FakeResponse()

    monkeypatch.setattr(stream.httpx, "AsyncClient", FakeClient)
    for _ in range(2):
        asyncio.run(
            stream.run("http://test/api/v1/predict", rate=0.0, jitter=0.0, fraud_boost=0.0, limit=3, cards=3)
        )

    ids = [p["transaction_id"] for p in posted]
    assert len(ids) == 6
    assert len(set(ids)) == 6, ids
    assert all(re.fullmatch(r"txn-[0-9a-f]{6}-\d{6}", i) for i in ids), ids


def test_alert_line_never_names_a_component(monkeypatch, capsys):
    """The demo terminal followed the boards' policy on the Indian track but
    still printed V14, V4, V10 as drivers on ULB; a component is counted, not named."""
    import asyncio

    import pandas as pd

    rows = pd.DataFrame({name: [0.0, 1.0] for name in stream.FEATURES} | {"Class": [0, 1]})
    monkeypatch.setattr(
        stream, "load_unseen_rows", lambda: (rows[rows.Class == 1], rows[rows.Class == 0])
    )

    class Alert:
        def raise_for_status(self):
            pass

        def json(self):
            return {"flagged": True, "transaction_id": "x", "risk_score": 0.99, "latency_ms": 1.0}

    class Explained:
        """What /explain answers for the alert; the answer itself carries no explanation."""

        status_code = 200

        def json(self):
            return {"top_features": [
                {"feature": "V14", "interpretable": False},
                {"feature": "Amount", "interpretable": True},
                {"feature": "V4", "interpretable": False},
            ]}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url):
            return Alert()

        async def post(self, url, json):
            return Explained() if url.endswith("/explain") else Alert()

    monkeypatch.setattr(stream.httpx, "AsyncClient", FakeClient)
    asyncio.run(
        stream.run("http://test/api/v1/predict", rate=0.0, jitter=0.0, fraud_boost=0.0, limit=1, cards=3)
    )
    out = capsys.readouterr().out
    assert "drivers=Amount + 2 anonymised factors" in out
    assert "V14" not in out and "V4" not in out
