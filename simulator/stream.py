"""Module 6: transaction streaming simulator (§6.2.1).

    python simulator/stream.py --rate 1.5

Replays rows the models have never seen, the held-out test split, selected by
reusing ``prepare_splits`` so the partition is byte-identical to training, as
JSON POSTs to /api/v1/predict at human-watchable intervals.

Raw (unscaled) values are sent deliberately. Scaling belongs to the service, so
the simulator exercises the same preprocessing path a real bank feed would.

Real fraud prevalence is 0.17%, which at 1.5s per transaction means roughly one
alert every ten minutes, which is unwatchable in a demo. ``--fraud-boost``
oversamples fraud rows for presentation only. It changes nothing about the
model or the reported metrics, but say so out loud when demonstrating.

``--track india`` replays a whole Indian demo export in time order instead
(``--source`` picks the export stem; the default is ``tracks.india.demo_stem``).
Its prevalence is the export's own, which its calibration report records, so
``--fraud-boost`` and ``--cards`` do not apply: the rows carry real card ids and
the service derives the velocity features from the order they arrive in.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
import uuid
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prism import config, tracks  # noqa: E402
from prism.data.india import load_india  # noqa: E402
from prism.data.loader import load_raw  # noqa: E402
from prism.data.preprocess import prepare_splits  # noqa: E402
from prism.live.run import india_payload  # noqa: E402,F401  the runner and the replay send one payload

FEATURES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]


def _failure_text(exc: httpx.HTTPError) -> str:
    """The service's own reason when it sent one; httpx's message names only
    the status and the url."""
    response = getattr(exc, "response", None)
    if response is None:
        return str(exc)
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        return str(exc)
    if isinstance(detail, list):
        detail = "; ".join(str(item.get("msg", item)) if isinstance(item, dict) else str(item) for item in detail)
    return f"{response.status_code}: {detail}" if detail else str(exc)


async def _served_track(client, endpoint: str, expected: str) -> bool:
    """Probe /health before the replay. One sentence and False when the
    service is down, has no models, or serves the other track: a unit that
    posts the wrong track's rows prints thousands of failures and never ends."""
    try:
        response = await client.get(endpoint.replace("/predict", "/health"))
    except httpx.ConnectError:
        print(f"Cannot reach {endpoint}. Is uvicorn running?")
        return False
    try:
        health = response.json()
    except ValueError:
        return True
    if not isinstance(health, dict):
        return True
    track = health.get("track")
    if track is not None and track != expected:
        print(
            f"The service is serving the {track} track, not {expected}; "
            f"set `track: {expected}` in config.local.yaml and restart it."
        )
        return False
    if health.get("status") == "no_models":
        print(f"The service has no models loaded; train the {expected} track and restart it.")
        return False
    return True


def _driver_text(top: list[dict]) -> str:
    """The alert line's drivers as the boards would show them: a named feature
    by its label, an anonymised component counted and never named."""
    named = [
        f"{f.get('label') or f['feature']}{' ' + f['display'] if f.get('display') else ''}"
        for f in top if f.get("interpretable", True)
    ]
    anonymised = sum(1 for f in top if not f.get("interpretable", True))
    parts = []
    if named:
        parts.append(", ".join(named))
    if anonymised:
        parts.append(f"{anonymised} anonymised {'factor' if anonymised == 1 else 'factors'}")
    return " + ".join(parts) or "n/a"


def card_pool(size: int) -> list[str]:
    """Synthetic cardholder ids. Visibly synthetic: no fake last-four digits."""
    return [f"card-{i:02d}" for i in range(1, size + 1)]


def load_unseen_rows():
    """The held-out test split, with original unscaled values."""
    df = load_raw()
    splits = prepare_splits(df)
    raw_test = df.loc[splits.X_test.index]
    frauds = raw_test[raw_test["Class"] == 1]
    normals = raw_test[raw_test["Class"] == 0]
    return frauds, normals


async def run(
    endpoint: str, rate: float, jitter: float, fraud_boost: float, limit: int, cards: int
):
    frauds, normals = load_unseen_rows()
    print(
        f"Replaying held-out data: {len(normals):,} normal / {len(frauds):,} fraud "
        f"-> {endpoint}"
    )
    if fraud_boost > 0:
        print(f"Fraud oversampled to ~{fraud_boost:.0%} for demonstration.")

    rng = random.Random(config.seed())
    # Not `pool`: the loop below already uses that name for the row source.
    card_ids = card_pool(cards)
    # Ids carry a per-run token: decisions attach by transaction id, so a
    # restarted replay must never reuse the ids of an earlier one.
    run_token = uuid.uuid4().hex[:6]
    print(f"Spreading the replay over {len(card_ids)} synthetic cards; run token {run_token}.")
    sent = flagged_count = 0

    async with httpx.AsyncClient(timeout=10.0) as client:
        if not await _served_track(client, endpoint, "ulb"):
            return

        while limit <= 0 or sent < limit:
            pool = frauds if (rng.random() < fraud_boost and len(frauds)) else normals
            row = pool.iloc[rng.randrange(len(pool))]

            payload = {name: float(row[name]) for name in FEATURES}
            payload["transaction_id"] = f"txn-{run_token}-{sent:06d}"
            payload["card_id"] = rng.choice(card_ids)

            try:
                response = await client.post(endpoint, json=payload)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                print(f"  request failed: {_failure_text(exc)}")
                await asyncio.sleep(rate)
                continue

            body = response.json()
            sent += 1
            actual = "FRAUD" if row["Class"] == 1 else "normal"

            if body["flagged"]:
                flagged_count += 1
                # The answer carries no explanation; it lands on the audit row after
                # the answer. The terminal asks /explain for the drivers of an alert.
                try:
                    explained = await client.post(endpoint.replace("/predict", "/explain"), json=payload)
                    top = (explained.json() if getattr(explained, "status_code", 200) == 200 else {}).get("top_features", [])
                except httpx.HTTPError as exc:
                    # One line of drivers lost, not the replay.
                    print(f"  drivers unavailable: {_failure_text(exc)}")
                    top = []
                drivers = _driver_text(top[:3])
                print(
                    f"  [ALERT] {body['transaction_id']}  risk={body['risk_score']:.4f}  "
                    f"actual={actual}  amount={row['Amount']:.2f}  "
                    f"drivers={drivers}  {body['latency_ms']:.1f}ms"
                )
            else:
                print(
                    f"          {body['transaction_id']}  risk={body['risk_score']:.4f}  "
                    f"actual={actual}  {body['latency_ms']:.1f}ms"
                )

            await asyncio.sleep(max(0.0, rate + rng.uniform(-jitter, jitter)))

    print(f"\nSent {sent}, flagged {flagged_count}.")


def load_india_rows(source: Path | None = None):
    """A whole demo export, in time order. Unseen by construction: it was
    generated with a different seed and prevalence from the training export,
    and the calibration report next to it says which."""
    stem = Path(source) if source is not None else config.get_path("tracks.india.demo_stem")
    return load_india(stem)


async def run_india(
    endpoint: str,
    rate: float,
    jitter: float,
    limit: int,
    source: Path | None,
    client_factory=httpx.AsyncClient,
):
    rows = load_india_rows(source)
    rng = random.Random(config.seed())
    print(f"Replaying {len(rows):,} Indian rows in time order -> {endpoint}; ids are the export's own.")
    sent = flagged_count = 0

    async with client_factory(timeout=10.0) as client:
        if not await _served_track(client, endpoint, "india"):
            return

        for _, row in rows.iterrows():
            if limit > 0 and sent >= limit:
                break
            payload = india_payload(row)
            try:
                response = await client.post(endpoint, json=payload)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                print(f"  request failed: {_failure_text(exc)}")
                await asyncio.sleep(rate)
                continue
            body = response.json()
            sent += 1
            actual = "FRAUD" if int(row["is_fraud"]) == 1 else "normal"
            if body["flagged"]:
                flagged_count += 1
                # The answer carries no explanation; it lands on the audit row after
                # the answer. The terminal asks /explain for the drivers of an alert.
                try:
                    explained = await client.post(endpoint.replace("/predict", "/explain"), json=payload)
                    top = (explained.json() if getattr(explained, "status_code", 200) == 200 else {}).get("top_features", [])
                except httpx.HTTPError as exc:
                    # One line of drivers lost, not the replay.
                    print(f"  drivers unavailable: {_failure_text(exc)}")
                    top = []
                # The boards show the track's display names; the terminal says the same.
                drivers = _driver_text(top[:3])
                print(
                    f"  [ALERT] {body['transaction_id']}  risk={body['risk_score']:.4f}  "
                    f"actual={actual}  amount={float(row['amount']):.2f} INR  "
                    f"drivers={drivers}  {body['latency_ms']:.1f}ms"
                )
            else:
                print(
                    f"          {body['transaction_id']}  risk={body['risk_score']:.4f}  "
                    f"actual={actual}  {body['latency_ms']:.1f}ms"
                )
            await asyncio.sleep(max(0.0, rate + rng.uniform(-jitter, jitter)))

    print(f"\nSent {sent}, flagged {flagged_count}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream unseen transactions to Prism.")
    tracks.add_track_argument(parser)
    parser.add_argument("--endpoint", default=config.get("simulator.endpoint"))
    parser.add_argument(
        "--source", default=None,
        help="export stem to replay on the Indian track (default: tracks.india.demo_stem)",
    )
    parser.add_argument(
        "--rate", type=float, default=config.get("simulator.interval_seconds", 1.5),
        help="seconds between transactions",
    )
    parser.add_argument(
        "--jitter", type=float, default=config.get("simulator.jitter_seconds", 0.5)
    )
    parser.add_argument(
        "--fraud-boost", type=float, default=config.get("simulator.fraud_boost", 0.05),
        help="probability of forcing a fraud row (demo only, ULB track only)",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="stop after N transactions (0 = forever)"
    )
    parser.add_argument(
        "--cards", type=int, default=config.get("simulator.cards", 12),
        help="size of the synthetic card pool the replay is spread over (ULB track only)",
    )
    args = parser.parse_args()
    schema = tracks.activate_from_args(args)

    try:
        if schema.name == "india":
            asyncio.run(run_india(args.endpoint, args.rate, args.jitter, args.limit, args.source))
        else:
            asyncio.run(
                run(args.endpoint, args.rate, args.jitter, args.fraud_boost, args.limit, args.cards)
            )
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
