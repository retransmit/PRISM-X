"""The Indian replay posts the export's own rows, in time order, with the boards' context."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("httpx")

from prism import tracks  # noqa: E402
from prism.data import india  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "stream", Path(__file__).resolve().parents[1] / "simulator" / "stream.py"
)
stream = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(stream)


def test_replay_posts_rows_in_time_order_with_context(india_stem, monkeypatch):
    rows = india.load_india(india_stem).head(6)
    monkeypatch.setattr(stream, "load_india_rows", lambda source=None: rows)
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

    asyncio.run(stream.run_india("http://x/api/v1/predict", rate=0.0, jitter=0.0, limit=6, source=None, client_factory=FakeClient))

    assert len(posted) == 6
    assert [p["tx_time_seconds"] for p in posted] == sorted(p["tx_time_seconds"] for p in posted)
    first = posted[0]
    for column in tracks.INDIA.input_columns:
        assert column in first, column
    assert isinstance(first["channel_type"], str) and isinstance(first["amount"], float)
    assert first["card_id"] == rows.iloc[0]["card_id"]
    assert first["currency"] == "INR" and first["merchant"] == rows.iloc[0]["merchant_id"]
    assert first["channel"] == rows.iloc[0]["channel_type"]
    assert first["transaction_id"] == rows.iloc[0]["transaction_id"]
    assert "is_fraud" not in first


class _Health:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


def _client(posted, health, post_response=None, explain_response=None):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url):
            return _Health(health)

        async def post(self, url, json):
            # The replay asks /explain for an alert's drivers; the answer itself carries none.
            if url.endswith("/explain"):
                return explain_response or _Health({})
            posted.append(json)
            return post_response or _Health({"flagged": False, "transaction_id": "x", "risk_score": 0.0, "latency_ms": 1.0})

    return FakeClient


def test_replay_stops_when_the_service_serves_another_track(india_stem, monkeypatch, capsys):
    """A unit started with --track india against a service still on ulb would
    otherwise print thousands of opaque failures; /health carries the track."""
    rows = india.load_india(india_stem).head(3)
    monkeypatch.setattr(stream, "load_india_rows", lambda source=None: rows)
    posted: list[dict] = []
    client = _client(posted, {"status": "ready", "track": "ulb", "models": ["xgboost"]})

    asyncio.run(stream.run_india("http://x/api/v1/predict", rate=0.0, jitter=0.0, limit=3, source=None, client_factory=client))

    assert posted == []
    out = capsys.readouterr().out
    assert "ulb" in out and "config.local.yaml" in out


def test_a_rejected_request_prints_the_service_detail(india_stem, monkeypatch, capsys):
    """httpx's own message for a 422 names the status and the url, never the
    reason the service gave."""
    import httpx

    rows = india.load_india(india_stem).head(2)
    monkeypatch.setattr(stream, "load_india_rows", lambda source=None: rows)

    class Rejected:
        def raise_for_status(self):
            request = httpx.Request("POST", "http://x/api/v1/predict")
            response = httpx.Response(
                422, request=request,
                json={"detail": [{"msg": "Value error, Missing features for the india track: ['channel_type']"}]},
            )
            raise httpx.HTTPStatusError(
                "Client error '422 Unprocessable Entity' for url 'http://x/api/v1/predict'",
                request=request, response=response,
            )

    posted: list[dict] = []
    client = _client(posted, {"status": "ready", "track": "india"}, post_response=Rejected())
    asyncio.run(stream.run_india("http://x/api/v1/predict", rate=0.0, jitter=0.0, limit=2, source=None, client_factory=client))

    assert len(posted) == 2
    assert "channel_type" in capsys.readouterr().out


def test_alert_line_names_drivers_by_their_labels(india_stem, monkeypatch, capsys):
    """The boards print the track's display names; the demo terminal read
    card_tx_1h beside a board saying "transactions on this card in the last hour"."""
    rows = india.load_india(india_stem).head(1)
    monkeypatch.setattr(stream, "load_india_rows", lambda source=None: rows)
    alert = _Health({"flagged": True, "transaction_id": "x", "risk_score": 0.99, "latency_ms": 1.0})
    explained = _Health({"top_features": [
        {"feature": "card_tx_1h", "label": "transactions on this card in the last hour", "display": None},
        {"feature": "pos_entry_mode", "label": "entry mode", "display": "012"},
    ]})
    posted: list[dict] = []
    client = _client(posted, {"status": "ready", "track": "india"}, post_response=alert, explain_response=explained)
    asyncio.run(stream.run_india("http://x/api/v1/predict", rate=0.0, jitter=0.0, limit=1, source=None, client_factory=client))
    out = capsys.readouterr().out
    assert "transactions on this card in the last hour, entry mode 012" in out
    assert "card_tx_1h" not in out



def test_a_failing_explain_call_does_not_end_the_replay(india_stem, monkeypatch, capsys):
    """The terminal asks /explain for an alert's drivers; a timeout there must
    cost one line of drivers, not the whole replay."""
    import httpx

    rows = india.load_india(india_stem).head(2)
    monkeypatch.setattr(stream, "load_india_rows", lambda source=None: rows)
    alert = _Health({"flagged": True, "transaction_id": "x", "risk_score": 0.99, "latency_ms": 1.0})
    posted: list[dict] = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url):
            return _Health({"status": "ready", "track": "india"})

        async def post(self, url, json):
            if url.endswith("/explain"):
                raise httpx.ReadTimeout("explain timed out")
            posted.append(json)
            return alert

    asyncio.run(stream.run_india("http://x/api/v1/predict", rate=0.0, jitter=0.0, limit=2, source=None, client_factory=FakeClient))
    assert len(posted) == 2
    assert "Sent 2, flagged 2" in capsys.readouterr().out
