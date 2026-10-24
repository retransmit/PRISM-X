"""The runner: a generated stream through the service, with the runner playing the bank.

    PRISM_LABEL_TOKEN=<label_feed token> python -m prism.live.run --endpoint http://127.0.0.1:8000

The service it drives runs with the live switch on (PRISM_LIVE=1).

Rows go to ``/api/v1/predict`` in time order, each as soon as the previous one is
answered. The bank is the generator's supervision model (plan Decision 1): its
investigator queue reviews Prism's alerts, up to ``k_daily`` a day, and nothing else;
chargebacks arrive at their delays; unreported fraud never does; legitimate rows mature
clean. Each label goes to ``/api/v1/labels`` once the stream clock reaches its discovery
time. At each retrain boundary the stream pauses with no request in flight while the
retrain job runs (Decision 3), then the service is asked for its health so a promoted
bundle is loaded before the next payment.

The truth stays here: the runner scores the served model against it for the report and
sends the service only the request fields and the labels the bank learned.

A run starts clean or not at all: the live model root must hold no earlier run and the
service's audit log no payments, so the run log, the event log and the audit log describe
one run. A payment the service answers with an error, a body that is not JSON or no answer
in time counts as a failed request and the stream goes on. The run stops, with its log
written and marked incomplete, when a retrain read no rows (the job's audit log is not the
service's) or when the service does not serve a promoted bundle.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from prism import config, tracks
from prism.data import india
from prism.live import bundles

DAY = 86_400.0
# Prism's answer is the queue's score: 1.0 for an alert, 0.0 otherwise, so a threshold
# between them sends exactly Prism's alerts to the investigators.
ALERT_SCORE, CLEAR_SCORE, QUEUE_THRESHOLD = 1.0, 0.0, 0.5
API = "/api/v1"


def check_schedule(s: dict) -> None:
    """Refuse a schedule whose first retrain has no training window: the backtest ends
    ``label_lag_days`` before the clock, and with the backtest and validation windows
    before it that takes the first ``label_lag_days + backtest_days + val_days`` days."""
    needed = float(s["label_lag_days"]) + float(s["backtest_days"]) + float(s["val_days"])
    first = s["first_retrain_day"]
    if float(first) <= needed:
        raise ValueError(
            f"live.first_retrain_day {first} leaves no training window: the label lag, the backtest and the "
            f"validation windows take the first {needed:g} days, so the first retrain must come after day {needed:g}"
        )


def _call(client, method: str, path: str, **kwargs) -> tuple[int, dict]:
    """One request: its status and its JSON body, or status 0 and the error when the service
    did not answer, and the start of the text when the body is not JSON."""
    import httpx

    try:
        answer = getattr(client, method)(path, **kwargs)
    except httpx.HTTPError as exc:
        return 0, {"detail": f"{type(exc).__name__}: {exc}"}
    try:
        body = answer.json()
    except ValueError:
        body = {"detail": str(getattr(answer, "text", ""))[:200]}
    return int(answer.status_code), body if isinstance(body, dict) else {"detail": body}


def india_payload(row) -> dict:
    """One export row as the service takes it: the inputs the schema names,
    categoricals and the card as text, plus the context the boards show."""
    schema = tracks.INDIA
    payload: dict = {}
    for column in schema.input_columns:
        value = row[column]
        payload[column] = (
            str(value)
            if (column in schema.categorical_columns or column == schema.card_column)
            else float(value)
        )
    # The export's own id. A restarted replay repeats the same ids and the
    # service answers each repeat from its log instead of logging it again.
    payload["transaction_id"] = str(row["transaction_id"])
    payload["merchant"] = str(row["merchant_id"])
    payload["channel"] = str(row["channel_type"])
    payload["currency"] = schema.currency
    return payload


def load_stream(stem: Path) -> pd.DataFrame:
    """The export in time order, with the master columns the bank needs beside the inputs."""
    frame = india.load_india(stem)
    master = pd.read_csv(
        india.export_paths(stem)["master"], dtype=str, keep_default_na=False, usecols=["transaction_id", "scenario_tag"]
    ).set_index("transaction_id")
    frame["scenario_tag"] = master.loc[frame["transaction_id"].astype(str), "scenario_tag"].astype(str).to_numpy()
    return frame


def stream_origin(stem: Path) -> float:
    """The stream's first transaction time, read from the authorisation view alone."""
    times = pd.read_csv(india.export_paths(stem)["auth"], usecols=[tracks.INDIA.time_column])
    return float(times[tracks.INDIA.time_column].min())


def subprocess_retrain(database_url: str, origin: float) -> Callable[[float], dict]:
    """The retrain job in its own process, with this interpreter; its event is read back
    from the bundles' event log. A job that fails is recorded there too, with its stream
    day and the end of its error, so the board and the report show it."""

    def failed(as_of: float, reason: str) -> dict:
        event = {"as_of": float(as_of), "stream_day": (float(as_of) - origin) / DAY, "status": "failed", "reason": reason}
        bundles.record_event(tracks.INDIA, event)
        return event

    def retrain(as_of: float) -> dict:
        command = [sys.executable, "-m", "prism.live.retrain", "--as-of", repr(float(as_of)), "--database-url", database_url]
        done = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if done.returncode != 0:
            tail = (done.stderr or done.stdout).strip().splitlines()[-5:]
            return failed(as_of, " | ".join(tail))
        recorded = bundles.events(tracks.INDIA)
        if not recorded or float(recorded[-1].get("as_of", -1.0)) != float(as_of):
            return failed(as_of, "the retrain job recorded no event")
        return recorded[-1]

    return retrain


def truth_backtest(bundle_dir: Path, truth: dict[str, int]) -> dict | None:
    """Both models' PR-AUC on a retrain's backtest rows against the truth, for the report:
    the operational labels the promotion read favour the incumbent, whose alerts the bank
    investigated within hours while its misses wait for a chargeback or are never reported."""
    path = Path(bundle_dir) / "backtest.json"
    if not path.exists():
        return None
    backtest = json.loads(path.read_text(encoding="utf-8"))
    y = np.asarray([int(truth[t]) for t in backtest["transaction_id"]], dtype=int)
    if not y.any():
        return {"fraud": 0, "challenger_pr_auc": None, "champion_pr_auc": None}
    return {
        "fraud": int(y.sum()),
        "challenger_pr_auc": float(average_precision_score(y, backtest["challenger"])),
        "champion_pr_auc": float(average_precision_score(y, backtest["champion"])),
    }


def _windows(origin: float, served: pd.DataFrame, days: float) -> list[dict]:
    """The served champion against the truth, per window of ``days`` stream days."""
    out = []
    span = days * DAY
    index = ((served["time"] - origin) // span).astype(int)
    for k, rows in served.groupby(index):
        fraud = rows["truth"] == 1
        out.append({
            "window": int(k),
            "start_day": float(k * days),
            "end_day": float((k + 1) * days),
            "rows": int(len(rows)),
            "fraud": int(fraud.sum()),
            "pr_auc": float(average_precision_score(rows["truth"], rows["risk"])) if fraud.any() else None,
            "recall": float(rows.loc[fraud, "flagged"].mean()) if fraud.any() else None,
            "alerts": int(rows["flagged"].sum()),
            "bundles": sorted(rows["bundle"].unique().tolist()),
        })
    return out


def run(
    client,
    stem: Path,
    s: dict,
    *,
    label_token: str,
    retrain_fn: Callable[[float], dict],
    out: Path,
    limit: int = 0,
) -> dict:
    """Play the stream through ``client`` (an httpx-style client on the service's base URL)
    and write the run log to ``out`` after every stream day. Returns the totals."""
    from prism_generator.stream import LabelFeed, SupervisionEngine

    check_schedule(s)
    schema = tracks.INDIA
    leftovers = [
        path for path in (bundles.pointer_path(schema), bundles.root(schema) / bundles.EVENTS) if path.exists()
    ]
    if leftovers:
        raise RuntimeError(
            f"The live model root {bundles.live_dir(schema)} already holds a run "
            f"({', '.join(p.name for p in leftovers)}); move it aside so this run's events are its own."
        )
    status, health = _call(client, "get", f"{API}/health")
    if status != 200 or health.get("track") != schema.name or health.get("status") != "ready":
        raise RuntimeError(f"The service is not ready on the Indian track: {health}")
    status, stats = _call(client, "get", f"{API}/stats")
    if status != 200:
        raise RuntimeError(f"The service's stats could not be read: {stats.get('detail')}")
    if int(stats.get("processed", 0)) > 0:
        raise RuntimeError(
            f"The service's audit log already holds {int(stats['processed']):,} payment(s); start the service on a "
            "fresh live.database_url so the run's labels and retrains read this run alone."
        )
    bundle_id = str(health.get("bundle_id") or bundles.OFFLINE)

    stream = load_stream(stem)
    if limit > 0:
        stream = stream.iloc[:limit]
    truth_of = dict(zip(stream["transaction_id"].astype(str), stream[tracks.INDIA.target_column].astype(int)))
    time_column, target = schema.time_column, schema.target_column
    origin = float(stream[time_column].iloc[0])
    supervision = SupervisionEngine(
        k_daily=int(s["supervision"]["k_daily"]), alert_threshold=QUEUE_THRESHOLD, seed=int(s["supervision"]["seed"])
    )
    feed = LabelFeed()
    auth = {"Authorization": f"Bearer {label_token}"}

    next_retrain = origin + float(s["first_retrain_day"]) * DAY
    every = float(s["retrain_every_days"]) * DAY
    days: list[dict] = []
    retrains: list[dict] = []
    drift: list[dict] = []
    posted: Counter[str] = Counter()
    refused = failed = 0
    served = {"time": [], "truth": [], "risk": [], "flagged": [], "bundle": []}

    def day_entry(day: int) -> dict:
        while len(days) <= day:
            days.append({
                "day": len(days), "requests": 0, "flagged": 0, "fraud": 0, "caught": 0,
                "failed": 0, "labels": {}, "bundle_id": bundle_id,
            })
        return days[day]

    def release(clock: float) -> None:
        nonlocal refused
        for label in feed.due(clock):
            status, _ = _call(client, "post", f"{API}/labels", json=label, headers=auth)
            if status in (200, 201):
                posted[label["label_source"]] += 1
                entry = day_entry(int((clock - origin) // DAY))
                entry["labels"][label["label_source"]] = entry["labels"].get(label["label_source"], 0) + 1
            else:
                refused += 1

    finished = False

    def log() -> dict:
        frame = pd.DataFrame(served)
        return {
            # A limited run never stands for the stream: the results refuse it.
            "complete": finished and limit == 0,
            "limit": int(limit),
            "settings": {key: value for key, value in s.items()},
            "stem": india._project_relative(Path(stem)),
            "generator_revision": india.generator_revision(),
            "origin": origin,
            "rows": int(len(stream)),
            "days": days,
            "labels": {
                "posted": dict(posted),
                "refused": refused,
                "pending_at_end": feed.pending,
                "never_reported": feed.dark,
            },
            "failed_requests": failed,
            "retrains": retrains,
            "drift": drift,
            "windows": _windows(origin, frame, float(s["retrain_every_days"])) if len(frame) else [],
        }

    def write() -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        staged = out.with_name(out.name + ".tmp")
        staged.write_text(json.dumps(log(), indent=2), encoding="utf-8")
        os.replace(staged, out)

    def stop(message: str) -> None:
        write()
        raise RuntimeError(message)

    current_day = 0
    for _, row in stream.iterrows():
        clock = float(row[time_column])
        while clock >= next_retrain:
            # The stream pauses: every earlier row has been answered and the labels known
            # by the boundary have been posted before the job reads the log. Drift is read
            # first, while the bundle that scored the window is still the one serving.
            release(next_retrain)
            status, body = _call(client, "get", f"{API}/drift")
            drift.append({"as_of": next_retrain, "status": status, "bundle_id": bundle_id, **(
                body if status == 200 else {"detail": body.get("detail")}
            )})
            event = dict(retrain_fn(next_retrain))
            if event.get("challenger_bundle"):
                scored = truth_backtest(bundles.root(schema) / event["challenger_bundle"], truth_of)
                if scored is not None:
                    event["truth"] = scored
            retrains.append(event)
            day_at = (next_retrain - origin) / DAY
            rows = event.get("rows")
            if isinstance(rows, dict) and served["time"] and sum(int(n) for n in rows.values()) == 0:
                stop(
                    f"The retrain at stream day {day_at:g} read no rows while this run has sent "
                    f"{len(served['time']):,} payments: its --database-url does not name the service's audit log "
                    "(the service's PRISM_DATABASE_URL)."
                )
            wanted = event.get("challenger_bundle") if event.get("status") == "promoted" else None
            checks = int(s["swap_checks"]) if wanted else 1
            for attempt in range(checks):
                status, health = _call(client, "get", f"{API}/health")
                if status == 200:
                    bundle_id = str(health.get("bundle_id") or bundle_id)
                if wanted is None or bundle_id == wanted:
                    break
                if attempt + 1 < checks:
                    time.sleep(float(s["bundle_poll_seconds"]))
            else:
                stop(
                    f"The service did not swap to the promoted bundle {wanted} after {checks} health checks "
                    f"{float(s['bundle_poll_seconds']):g} s apart; it serves {bundle_id}. It must run with "
                    "PRISM_LIVE=1 and the same live.model_dir as the retrain job."
                )
            next_retrain += every
        release(clock)

        day = int((clock - origin) // DAY)
        if day != current_day:
            write()
            current_day = day
        entry = day_entry(day)
        truth = int(row[target])
        status, result = _call(client, "post", f"{API}/predict", json=india_payload(row))
        if status != 200:
            failed += 1
            entry["failed"] += 1
            continue
        flagged = bool(result["flagged"])
        entry["requests"] += 1
        entry["flagged"] += int(flagged)
        entry["fraud"] += truth
        entry["caught"] += int(flagged and truth == 1)
        entry["bundle_id"] = bundle_id
        for key, value in (("time", clock), ("truth", truth), ("risk", float(result["risk_score"])),
                           ("flagged", int(flagged)), ("bundle", bundle_id)):
            served[key].append(value)
        record = {
            "transaction_id": str(row["transaction_id"]),
            "card_id": str(row[schema.card_column]),
            "tx_time_seconds": clock,
            "amount": float(row[schema.amount_column]),
            "currency": schema.currency,
            "is_fraud": truth,
            "scenario_tag": str(row["scenario_tag"]),
        }
        feed.push(supervision.process_record(record, risk_score=ALERT_SCORE if flagged else CLEAR_SCORE))

    # Labels the bank had not learned by the end of the stream stay unknown.
    finished = True
    write()
    totals = np.asarray(served["truth"], dtype=int)
    return {
        "requests": int(len(totals)),
        "flagged": int(np.sum(served["flagged"])),
        "fraud": int(totals.sum()),
        "caught": int(np.sum(np.asarray(served["flagged"]) * totals)),
        "retrains": len(retrains),
        "labels_posted": int(sum(posted.values())),
        "labels_refused": refused,
        "failed_requests": failed,
    }


def main() -> None:
    import httpx

    parser = argparse.ArgumentParser(description="Replay the live stream through the service, playing the bank.")
    parser.add_argument("--endpoint", default=f"http://{config.get('api.host')}:{config.get('api.port')}")
    parser.add_argument("--limit", type=int, default=0, help="stop after this many rows (0: the whole stream)")
    parser.add_argument("--stem", default=None, help="the export stem (default: live.stem)")
    parser.add_argument("--database-url", default=None, help="the service's audit log (default: live.database_url)")
    parser.add_argument("--out", default=None, help="the run log (default: reports/metrics/india/live/run.json)")
    args = parser.parse_args()
    token = os.environ.get("PRISM_LABEL_TOKEN", "")
    if not token:
        raise SystemExit("Set PRISM_LABEL_TOKEN to the service's api.credentials.label_feed token.")
    tracks.set_active(tracks.INDIA.name)
    s = dict(config.get("live"))
    stem = Path(args.stem) if args.stem else config.get_path("live.stem")
    out = Path(args.out) if args.out else tracks.INDIA.metrics_dir / "live" / "run.json"
    with httpx.Client(base_url=args.endpoint, timeout=60.0) as client:
        totals = run(
            client, stem, s, label_token=token,
            retrain_fn=subprocess_retrain(args.database_url or str(config.get("live.database_url")), stream_origin(stem)),
            out=out, limit=args.limit,
        )
    (out.parent / "events.json").write_text(json.dumps(bundles.events(tracks.INDIA), indent=2), encoding="utf-8")
    print(json.dumps(totals, indent=2))
    print(f"Run log -> {out}")


if __name__ == "__main__":
    main()
