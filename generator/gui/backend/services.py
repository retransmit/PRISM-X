"""Backend services bridging the GUI API to Prism Generator's core simulation and benchmark kernels."""

from __future__ import annotations

import io
import csv
import time
import asyncio
from typing import Any, Dict, List, Optional

from prism_generator.engine import SimulationEngine
from prism_generator.benchmark import XAIBenchmarkHarness
from prism_generator.visualizer import compile_simulation_data_bundle
from prism_generator.graph_transformer import ForensicGraphTransformer


class SimulationState:
    """In-memory cache and real-time streaming engine of the currently active simulation run."""

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {
            "region": "IN",
            "n_transactions": 0,
            "fraud_count": 0,
            "legit_count": 0,
            "seed": 42,
            "generated_at": None,
        }
        self.is_generating: bool = False
        self.generation_progress: Dict[str, Any] = {
            "current": 0,
            "total": 0,
            "pct": 0.0,
            "tps": 0.0,
            "status": "idle",
        }
        self.active_websockets: set = set()
        self.is_streaming: bool = False
        self.streaming_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.active_engine: Optional[SimulationEngine] = None

    async def handle_websocket(self, websocket: Any) -> None:
        await websocket.accept()
        self.active_websockets.add(websocket)
        self._loop = asyncio.get_running_loop()
        try:
            # Immediate handshake state
            await websocket.send_json({
                "type": "init",
                "metadata": self.metadata,
                "is_generating": self.is_generating,
                "progress": self.generation_progress,
                "is_streaming": self.is_streaming,
            })
            while True:
                data = await websocket.receive_json()
                action = data.get("action")
                if action == "ping":
                    await websocket.send_json({"type": "pong"})
                elif action == "generate":
                    asyncio.create_task(self.generate_async(
                        region=data.get("region", "IN"),
                        n_transactions=int(data.get("n_transactions", 1000)),
                        fraud_prevalence=float(data.get("fraud_prevalence", 0.04)),
                        adversary_mode=data.get("adversary_mode", "intent"),
                        seed=int(data.get("seed", 42)),
                    ))
                elif action == "start_stream":
                    asyncio.create_task(self.start_live_stream_async(
                        region=data.get("region", "IN"),
                        target_tps=float(data.get("target_tps", 20.0)),
                        fraud_prevalence=float(data.get("fraud_prevalence", 0.04)),
                        adversary_mode=data.get("adversary_mode", "intent"),
                        seed=int(data.get("seed", 42)),
                    ))
                elif action == "stop_stream":
                    await self.stop_live_stream_async()
        except Exception:
            pass
        finally:
            self.active_websockets.discard(websocket)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        dead = set()
        for ws in list(self.active_websockets):
            try:
                await ws.send_json(message)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.active_websockets.discard(ws)

    def broadcast_sync(self, message: Dict[str, Any]) -> None:
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self.broadcast(message), self._loop)

    async def generate_async(
        self,
        region: str = "IN",
        n_transactions: int = 1000,
        fraud_prevalence: float = 0.05,
        adversary_mode: str = "intent",
        seed: int = 42,
        simulation_mode: str = "transactions",
        time_span_days: Optional[int] = 14,
    ) -> Dict[str, Any]:
        if self.is_generating:
            return self.metadata

        self._loop = asyncio.get_running_loop()
        self.is_generating = True
        start_time = time.time()
        initial_total = n_transactions if simulation_mode == "transactions" else int((time_span_days or 14) * max(100, min(2000, (time_span_days or 14) * 35)) * 2.13)
        self.generation_progress = {
            "current": 0,
            "total": initial_total,
            "pct": 0.0,
            "tps": 0.0,
            "status": "simulating",
            "simulation_mode": simulation_mode,
            "time_span_days": time_span_days,
        }
        await self.broadcast({
            "type": "progress",
            "progress": self.generation_progress,
        })

        transformer = ForensicGraphTransformer(canvas_width=1400, canvas_height=900)
        last_progressive_pct = 0.0

        def on_progress(
            cur: int,
            tot: int,
            sample_tx: Optional[Dict[str, Any]] = None,
            records_so_far: Optional[List[Dict[str, Any]]] = None,
        ):
            nonlocal last_progressive_pct
            elapsed = max(0.001, time.time() - start_time)
            tps = round(cur / elapsed, 1)
            pct = round(min(100.0, (cur / max(1, tot)) * 100.0), 1)
            self.generation_progress = {
                "current": cur,
                "total": tot,
                "pct": pct,
                "tps": tps,
                "status": "simulating",
                "simulation_mode": simulation_mode,
                "time_span_days": time_span_days,
            }
            msg: Dict[str, Any] = {
                "type": "progress",
                "progress": self.generation_progress,
            }
            if sample_tx:
                msg["sample_tx"] = {
                    "transaction_id": sample_tx.get("transaction_id"),
                    "card_id": sample_tx.get("card_id"),
                    "merchant_id": sample_tx.get("merchant_id"),
                    "amount": float(sample_tx.get("amount", 0.0)),
                    "currency": str(sample_tx.get("currency", "USD" if region == "US" else "INR")),
                    "is_fraud": int(float(sample_tx.get("is_fraud", 0))),
                    "scenario_tag": sample_tx.get("scenario_tag", ""),
                    "response_code": sample_tx.get("response_code", "00"),
                    "hop_origin": sample_tx.get("hop_origin", ""),
                    "syndicate_id": sample_tx.get("syndicate_id", ""),
                    "botnet_cluster_id": sample_tx.get("botnet_cluster_id", ""),
                    "beneficiary_account_id": sample_tx.get("beneficiary_account_id", ""),
                }

            # Compile and attach progressive threat graph at milestones (every >= 15% progress)
            if records_so_far and (pct - last_progressive_pct >= 15.0 or pct >= 95.0):
                last_progressive_pct = pct
                fraud_so_far = [r for r in records_so_far if int(float(r.get("is_fraud", 0))) == 1]
                if len(fraud_so_far) >= 4:
                    try:
                        partial_graph = transformer.transform(fraud_so_far)
                        msg["threat_graph"] = partial_graph
                    except Exception:
                        pass

            self.broadcast_sync(msg)

        def worker():
            if simulation_mode == "days":
                days = time_span_days if time_span_days is not None else 14
                cards_count = max(100, min(2000, days * 35))
                merchants_count = max(50, min(1000, cards_count // 4))
                engine = SimulationEngine(
                    n_cards=cards_count,
                    n_merchants=merchants_count,
                    region=region,
                    adversary_mode=adversary_mode,
                    seed=seed,
                )
                self.active_engine = engine
                interval = max(50, min(500, (cards_count * days) // 25))
                return engine.generate_batch(
                    simulation_mode="days",
                    time_span_days=days,
                    fraud_prevalence=fraud_prevalence,
                    pace_to_sample_budget=False,
                    progress_callback=on_progress,
                    progress_interval=interval,
                )
            else:
                target_txs = n_transactions if n_transactions is not None else 1000
                engine = SimulationEngine(
                    n_cards=max(50, target_txs // 10),
                    n_merchants=max(20, target_txs // 25),
                    region=region,
                    adversary_mode=adversary_mode,
                    seed=seed,
                )
                self.active_engine = engine
                interval = max(50, min(500, target_txs // 20))
                return engine.generate_batch(
                    n_transactions=target_txs,
                    fraud_prevalence=fraud_prevalence,
                    time_span_days=max(3, target_txs // 100),
                    simulation_mode="transactions",
                    progress_callback=on_progress,
                    progress_interval=interval,
                )

        try:
            records = await asyncio.to_thread(worker)
            self.records = records
            fraud_count = sum(1 for r in self.records if int(float(r.get("is_fraud", 0))) == 1)
            legit_count = len(self.records) - fraud_count

            times = [float(r.get("tx_time_seconds", 0.0)) for r in self.records]
            elapsed_days = round((max(times) - min(times)) / 86400.0, 2) if len(times) > 1 else 0.0
            n_cards_active = len(self.active_engine.cards) if self.active_engine else 0
            daily_tx_rate = round(len(self.records) / max(1, n_cards_active) / max(0.1, elapsed_days), 2) if elapsed_days > 0 else 0.0

            self.metadata = {
                "region": region,
                "simulation_mode": simulation_mode,
                "time_span_days": time_span_days if simulation_mode == "days" else round(elapsed_days, 1),
                "elapsed_days": elapsed_days,
                "n_cards": n_cards_active,
                "avg_daily_tx_per_card": daily_tx_rate,
                "n_transactions": len(self.records),
                "fraud_count": fraud_count,
                "legit_count": legit_count,
                "fraud_prevalence": round(fraud_count / max(1, len(self.records)), 4),
                "adversary_mode": adversary_mode,
                "seed": seed,
            }
            self.is_generating = False
            self.generation_progress = {
                "current": len(self.records),
                "total": len(self.records),
                "pct": 100.0,
                "tps": round(len(self.records) / max(0.001, time.time() - start_time), 1),
                "status": "complete",
            }
            bundle = compile_simulation_data_bundle(records=self.records, region=region, seed=seed)
            await self.broadcast({
                "type": "complete",
                "metadata": self.metadata,
                "bundle": bundle,
            })
            return self.metadata
        except Exception as e:
            self.is_generating = False
            self.generation_progress["status"] = "error"
            await self.broadcast({
                "type": "error",
                "error": str(e),
            })
            raise

    async def start_live_stream_async(
        self,
        region: str = "IN",
        target_tps: float = 15.0,
        fraud_prevalence: float = 0.05,
        adversary_mode: str = "intent",
        seed: int = 42,
    ) -> None:
        if self.is_streaming:
            return
        self.is_streaming = True
        self._loop = asyncio.get_running_loop()
        await self.broadcast({"type": "stream_started", "target_tps": target_tps})

        async def stream_loop():
            if (
                self.active_engine is not None
                and getattr(self.active_engine, "region", "") == region
                and getattr(self.active_engine, "adversary_mode", "") == adversary_mode
            ):
                engine = self.active_engine
            else:
                engine = SimulationEngine(
                    n_cards=100,
                    n_merchants=40,
                    region=region,
                    adversary_mode=adversary_mode,
                    seed=seed,
                )
                self.active_engine = engine

            interval_sec = 1.0 / max(1.0, min(100.0, target_tps))
            streamed_count = 0
            fraud_streamed = 0
            approved_count = 0
            declined_count = 0

            while self.is_streaming:
                micro_batch = engine.generate_batch(
                    n_transactions=25,
                    fraud_prevalence=fraud_prevalence,
                    time_span_days=1,
                )
                for tx in micro_batch:
                    if not self.is_streaming:
                        break
                    streamed_count += 1
                    is_fr = int(float(tx.get("is_fraud", 0))) == 1
                    if is_fr:
                        fraud_streamed += 1
                    resp = str(tx.get("response_code", "00"))
                    if resp in ("00", "10"):
                        approved_count += 1
                    else:
                        declined_count += 1

                    self.records.append(tx)
                    if len(self.records) > 10000:
                        self.records.pop(0)

                    clean_tx = {
                        "transaction_id": str(tx.get("transaction_id", "")),
                        "timestamp_utc": str(tx.get("timestamp_utc", "")),
                        "card_id": str(tx.get("card_id", "")),
                        "merchant_id": str(tx.get("merchant_id", "")),
                        "amount": float(tx.get("amount", 0.0)),
                        "currency": str(tx.get("currency", "INR" if region == "IN" else "USD")),
                        "is_fraud": 1 if is_fr else 0,
                        "scenario_tag": str(tx.get("scenario_tag", "")),
                        "response_code": resp,
                        "hop_origin": str(tx.get("hop_origin", "")),
                        "syndicate_id": str(tx.get("syndicate_id", "")),
                        "botnet_cluster_id": str(tx.get("botnet_cluster_id", "")),
                        "beneficiary_account_id": str(tx.get("beneficiary_account_id", "")),
                    }

                    await self.broadcast({
                        "type": "live_tx",
                        "tx": clean_tx,
                        "stats": {
                            "streamed_count": streamed_count,
                            "fraud_count": fraud_streamed,
                            "approved_count": approved_count,
                            "declined_count": declined_count,
                            "current_tps": target_tps,
                        },
                    })
                    await asyncio.sleep(interval_sec)

            self.is_streaming = False
            await self.broadcast({"type": "stream_stopped"})

        self.streaming_task = asyncio.create_task(stream_loop())

    async def stop_live_stream_async(self) -> None:
        self.is_streaming = False
        if self.streaming_task:
            self.streaming_task.cancel()
            self.streaming_task = None
        await self.broadcast({"type": "stream_stopped"})

    def generate(
        self,
        region: str = "IN",
        n_transactions: int = 1000,
        fraud_prevalence: float = 0.05,
        adversary_mode: str = "intent",
        seed: int = 42,
    ) -> Dict[str, Any]:
        engine = SimulationEngine(
            n_cards=max(50, n_transactions // 10),
            n_merchants=max(20, n_transactions // 25),
            region=region,
            adversary_mode=adversary_mode,
            seed=seed,
        )
        self.active_engine = engine
        self.records = engine.generate_batch(
            n_transactions=n_transactions,
            fraud_prevalence=fraud_prevalence,
            time_span_days=max(3, n_transactions // 100),
        )

        fraud_count = sum(1 for r in self.records if int(float(r.get("is_fraud", 0))) == 1)
        legit_count = len(self.records) - fraud_count

        self.metadata = {
            "region": region,
            "n_transactions": len(self.records),
            "fraud_count": fraud_count,
            "legit_count": legit_count,
            "fraud_prevalence": round(fraud_count / max(1, len(self.records)), 4),
            "adversary_mode": adversary_mode,
            "seed": seed,
        }
        return self.metadata

    def get_paginated(
        self,
        page: int = 1,
        page_size: int = 50,
        filter_status: str = "all",
        search: str = "",
    ) -> Dict[str, Any]:
        items = self.records
        if filter_status == "fraud":
            items = [r for r in items if int(float(r.get("is_fraud", 0))) == 1]
        elif filter_status == "legit":
            items = [r for r in items if int(float(r.get("is_fraud", 0))) == 0]

        if search:
            q = search.lower().strip()
            items = [
                r for r in items
                if q in str(r.get("transaction_id", "")).lower()
                or q in str(r.get("card_id", "")).lower()
                or q in str(r.get("merchant_id", "")).lower()
                or q in str(r.get("scenario_tag", "")).lower()
            ]

        total = len(items)
        start = (page - 1) * page_size
        end = start + page_size
        page_items = items[start:end]

        # Clean serialized items
        cleaned = []
        for r in page_items:
            cleaned.append({
                "transaction_id": r.get("transaction_id"),
                "timestamp_utc": r.get("timestamp_utc"),
                "card_id": r.get("card_id"),
                "amount": float(r.get("amount", 0.0)),
                "currency": r.get("currency", "USD"),
                "channel_type": r.get("channel_type", ""),
                "mcc": r.get("mcc"),
                "merchant_id": r.get("merchant_id"),
                "response_code": r.get("response_code", "00"),
                "is_fraud": int(float(r.get("is_fraud", 0))),
                "scenario_tag": r.get("scenario_tag", ""),
                "dominant_causal_driver": r.get("dominant_causal_driver", ""),
            })

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "items": cleaned,
            "metadata": self.metadata,
        }

    def get_by_id(self, tx_id: str) -> Optional[Dict[str, Any]]:
        for r in self.records:
            if r.get("transaction_id") == tx_id:
                # Return deep dictionary with formatted XAI fields
                res = dict(r)
                # Ensure Shapley attributions are clean floats
                for key in ("analytical_shapley_probability", "analytical_shapley_log_odds"):
                    if key in res and isinstance(res[key], dict):
                        res[key] = {k: round(float(v), 6) for k, v in res[key].items()}
                return res
        return None

    def export_csv(self) -> str:
        if not self.records:
            return ""
        output = io.StringIO()
        fieldnames_set = set()
        fieldnames = []
        for r in self.records:
            for k in r.keys():
                if k not in fieldnames_set:
                    fieldnames_set.add(k)
                    fieldnames.append(k)

        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in self.records:
            row = {}
            for k, v in r.items():
                if isinstance(v, (dict, list)):
                    row[k] = str(v)
                else:
                    row[k] = v
            writer.writerow(row)
        return output.getvalue()


# Global state singleton
sim_state = SimulationState()


def run_xai_benchmark(
    model_type: str = "lightgbm",
    n_transactions: int = 500,
    fraud_prevalence: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    harness = XAIBenchmarkHarness(
        n_transactions=n_transactions,
        fraud_prevalence=fraud_prevalence,
        seed=seed,
    )
    summary = harness.run_benchmark(model_type=model_type)

    return {
        "model_name": summary.model_name,
        "explainer_name": summary.explainer_name,
        "n_evaluated_samples": summary.n_evaluated_samples,
        "auc_roc": round(float(summary.auc_roc), 4),
        "pr_auc": round(float(summary.pr_auc), 4),
        "mean_kendall_tau": round(float(summary.mean_kendall_tau), 4),
        "mean_spearman_rho": round(float(summary.mean_spearman_rho), 4),
        "mean_precision_at_3": round(float(summary.mean_precision_at_3), 4),
        "mean_intervention_precision_at_3": round(float(summary.mean_intervention_precision_at_3), 4),
        "mean_intervention_recall_at_3": round(float(summary.mean_intervention_recall_at_3), 4),
        "mean_relative_attribution_error": round(float(summary.mean_relative_attribution_error), 2),
        "mean_normalized_l2_distance": round(float(summary.mean_normalized_l2_distance), 4),
    }
