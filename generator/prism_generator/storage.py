"""High-capacity streaming dataset storage, parallel simulation, and aggregations."""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl

from .contract import load_contract

# Institutional schema definitions
# spec/19: the two views a detector is trained from come from the contract
AUTH_STREAM_COLUMNS = list(load_contract().export_view("auth_stream"))

THREAT_INTEL_GRAPH_COLUMNS = [
    "transaction_id",
    "card_id",
    "merchant_id",
    "syndicate_id",
    "botnet_cluster_id",
    "mule_ring_id",
    "beneficiary_account_id",
    "ip_subnet_prefix",
    "device_fingerprint_id",
    "mule_bank_id",
    "layering_hop_latency_seconds",
]


GATEWAY_TELEMETRY_COLUMNS = list(load_contract().export_view("gateway_telemetry"))

CLEARING_SETTLEMENT_COLUMNS = [
    "transaction_id",
    "clearing_mti",
    "clearing_delay_hours",
    "amount",
    "amount_minor",
    "settled_amount",
    "settled_amount_minor",
    "currency",
    "interchange_fee_minor",
    "mcc",
    "merchant_id",
    "mid",
    "acquirer_bin",
]

DISPUTE_RECOVERY_COLUMNS = [
    "transaction_id",
    "card_id",
    "amount",
    "currency",
    "is_fraud",
    "scenario_tag",
    "dispute_status",
    "dispute_reason_code",
    "ce3_qualified",
    "arbitration_fee_usd",
    "rbi_liability_tier",
    "rbi_provisional_credit_mandate_days",
    "cfcfrms_1930_lien_status",
]


@dataclass
class SimulationAggregates:
    """Running global telemetry across arbitrary-scale transaction streams."""
    total_transactions: int = 0
    approved_count: int = 0
    declined_count: int = 0
    total_fraud_count: int = 0
    total_volume: float = 0.0
    total_fraud_volume: float = 0.0
    
    # 4-hop switch funnels
    hop_drops: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    iso_codes: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    macro_options: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Temporal distributions (24-hour diurnal bins)
    hourly_legit: List[int] = field(default_factory=lambda: [0] * 24)
    hourly_fraud: List[int] = field(default_factory=lambda: [0] * 24)
    
    # Threat graph clusters
    syndicates: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    botnets: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    merchants_fraud: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    mules_fraud: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    
    # Top sampled threat interactions for graph visualization (capped to avoid memory blowup)
    threat_edges: List[Dict[str, Any]] = field(default_factory=list)
    threat_nodes_map: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    threat_sample_records: List[Dict[str, Any]] = field(default_factory=list)
    max_retained_nodes: int = 5000
    max_sample_records: int = 1500

    def update(self, records: List[Dict[str, Any]]) -> None:
        """Incrementally updates running aggregates from a batch of transaction records."""
        for r in records:
            self.total_transactions += 1
            amt = float(r.get("amount", 0.0))
            is_fraud = int(r.get("is_fraud", 0))
            resp = str(r.get("response_code", "00"))
            is_app = resp in ("00", "10", "ISO_00", "ISO_10") or resp.endswith("00") or resp.endswith("10")
            
            if is_app:
                self.approved_count += 1
            else:
                self.declined_count += 1
                
            self.total_volume += amt
            self.iso_codes[resp] += 1
            
            hop = str(r.get("hop_origin", "ISSUER_HOST"))
            self.hop_drops[hop] += 1
            
            macro = str(r.get("scenario_tag", "UNKNOWN"))
            self.macro_options[macro] += 1
            
            hr = int(r.get("hour_of_day", 0)) % 24
            if is_fraud:
                self.total_fraud_count += 1
                self.total_fraud_volume += amt
                self.hourly_fraud[hr] += 1
                
                # Threat network updates
                syn = str(r.get("syndicate_id", "SYN_UNASSIGNED"))
                bot = str(r.get("botnet_cluster_id", "BOT_UNASSIGNED"))
                mer = str(r.get("merchant_id", "MER_UNASSIGNED"))
                mule = str(r.get("mule_ring_id", "MULE_UNASSIGNED"))
                
                self.syndicates[syn] += 1
                self.botnets[bot] += 1
                self.merchants_fraud[mer] += amt
                self.mules_fraud[mule] += amt

                # Store sample fraud records for forensic graph transformation
                if len(self.threat_sample_records) < self.max_sample_records:
                    self.threat_sample_records.append(r)
                
                # Graph sampling for visualizer
                if len(self.threat_nodes_map) < self.max_retained_nodes:
                    card_id = str(r.get("card_id", "CARD_UNKNOWN"))
                    # Register nodes
                    if syn not in self.threat_nodes_map:
                        self.threat_nodes_map[syn] = {"id": syn, "type": "SYNDICATE", "label": syn, "volume": 0}
                    self.threat_nodes_map[syn]["volume"] += 1

                    if bot not in self.threat_nodes_map:
                        self.threat_nodes_map[bot] = {"id": bot, "type": "BOTNET", "label": bot, "volume": 0}
                    self.threat_nodes_map[bot]["volume"] += 1

                    if card_id not in self.threat_nodes_map:
                        self.threat_nodes_map[card_id] = {
                            "id": card_id,
                            "type": "CARD",
                            "label": card_id,
                            "tier": str(r.get("product_id", "CREDIT")),
                            "volume": 0,
                        }
                    self.threat_nodes_map[card_id]["volume"] += 1

                    if mer not in self.threat_nodes_map:
                        self.threat_nodes_map[mer] = {"id": mer, "type": "MERCHANT", "label": mer, "volume": 0}
                    self.threat_nodes_map[mer]["volume"] += 1

                    if mule not in self.threat_nodes_map:
                        self.threat_nodes_map[mule] = {"id": mule, "type": "MULE_RING", "label": mule, "volume": 0}
                    self.threat_nodes_map[mule]["volume"] += 1

                    # Register representative edges
                    if len(self.threat_edges) < self.max_retained_nodes * 2:
                        self.threat_edges.append({"source": syn, "target": bot, "type": "CONTROLS"})
                        self.threat_edges.append({"source": bot, "target": card_id, "type": "TARGETS"})
                        self.threat_edges.append({"source": card_id, "target": mer, "type": "PURCHASES"})
                        self.threat_edges.append({"source": mer, "target": mule, "type": "FUNNELS"})
            else:
                self.hourly_legit[hr] += 1

    def to_dict(self) -> Dict[str, Any]:
        """Serializes aggregates into a JSON-serializable dictionary."""
        return {
            "total_transactions": self.total_transactions,
            "approved_count": self.approved_count,
            "declined_count": self.declined_count,
            "approval_rate": self.approved_count / max(1, self.total_transactions),
            "total_fraud_count": self.total_fraud_count,
            "fraud_prevalence": self.total_fraud_count / max(1, self.total_transactions),
            "total_volume": round(self.total_volume, 2),
            "total_fraud_volume": round(self.total_fraud_volume, 2),
            "hop_drops": dict(self.hop_drops),
            "iso_codes": dict(self.iso_codes),
            "macro_options": dict(self.macro_options),
            "hourly_distribution": {
                "hours": list(range(24)),
                "legitimate": self.hourly_legit,
                "fraud": self.hourly_fraud,
            },
            "top_syndicates": dict(sorted(self.syndicates.items(), key=lambda x: x[1], reverse=True)[:20]),
            "top_botnets": dict(sorted(self.botnets.items(), key=lambda x: x[1], reverse=True)[:20]),
            "top_target_merchants": dict(sorted(self.merchants_fraud.items(), key=lambda x: x[1], reverse=True)[:20]),
            "top_mule_rings": dict(sorted(self.mules_fraud.items(), key=lambda x: x[1], reverse=True)[:20]),
            "graph_summary": {
                "nodes_count": len(self.threat_nodes_map),
                "edges_count": len(self.threat_edges),
            }
        }


class StreamingDatasetWriter:
    """Streams transactions to partitioned Parquet and/or CSV files with zero OOM risk."""

    def __init__(
        self,
        output_dir: str | Path,
        export_parquet: bool = True,
        export_csv: bool = False,
        chunk_size: int = 10000,
        max_retained_nodes: int = 5000,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.export_parquet = export_parquet
        self.export_csv = export_csv
        self.chunk_size = chunk_size
        
        self.aggregates = SimulationAggregates(max_retained_nodes=max_retained_nodes)
        self.chunk_idx = 0

        # Create table subdirectories
        self.tables = {
            "master": (self.output_dir / "master", None),
            "auth_stream": (self.output_dir / "auth_stream", AUTH_STREAM_COLUMNS),
            "threat_intel_graph": (self.output_dir / "threat_intel_graph", THREAT_INTEL_GRAPH_COLUMNS),
            "gateway_telemetry": (self.output_dir / "gateway_telemetry", GATEWAY_TELEMETRY_COLUMNS),
            "clearing_settlement": (self.output_dir / "clearing_settlement", CLEARING_SETTLEMENT_COLUMNS),
            "dispute_recovery": (self.output_dir / "dispute_recovery", DISPUTE_RECOVERY_COLUMNS),
        }
        for table_dir, _ in self.tables.values():
            table_dir.mkdir(parents=True, exist_ok=True)

    def write_chunk(self, records: List[Dict[str, Any]]) -> None:
        """Flushes a batch of transactions across all tables and institutional views."""
        if not records:
            return

        # 1. Update running aggregates
        self.aggregates.update(records)

        # 2. Convert batch to Polars DataFrame (very fast Rust backing)
        df_batch = pl.DataFrame(records, strict=False)

        part_name = f"part_{self.chunk_idx:05d}"

        # 3. Write master table and each institutional view table
        for table_name, (table_dir, cols) in self.tables.items():
            if cols is None:
                table_df = df_batch
            else:
                avail_cols = [c for c in cols if c in df_batch.columns]
                table_df = df_batch.select(avail_cols)

            if self.export_parquet:
                pq_path = table_dir / f"{part_name}.parquet"
                table_df.write_parquet(pq_path, compression="zstd")

            if self.export_csv:
                csv_path = table_dir / f"{part_name}.csv"
                table_df.write_csv(csv_path)

        self.chunk_idx += 1

    def finalize(self) -> Dict[str, Any]:
        """Saves metadata summary and returns consolidated telemetry."""
        meta = self.aggregates.to_dict()
        meta_path = self.output_dir / "simulation_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Write forensic threat graph for direct visualization consumption
        if self.aggregates.threat_sample_records:
            from prism_generator.graph_transformer import ForensicGraphTransformer
            transformer = ForensicGraphTransformer()
            graph_data = transformer.transform(self.aggregates.threat_sample_records)
        else:
            graph_data = {
                "nodes": list(self.aggregates.threat_nodes_map.values()),
                "links": self.aggregates.threat_edges,
            }
        graph_path = self.output_dir / "threat_graph_sample.json"
        with open(graph_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2)

        return meta


def _worker_simulation_task(
    worker_id: int,
    n_tx_target: int,
    region: str,
    output_dir: str,
    chunk_size: int,
    adversary_mode: str,
    seed: int,
) -> Dict[str, Any]:
    """Isolated worker simulation task generating a disjoint shard of transactions."""
    from prism_generator.engine import DiscreteEventEngine

    worker_dir = Path(output_dir) / f"worker_{worker_id}"
    writer = StreamingDatasetWriter(
        output_dir=worker_dir,
        export_parquet=True,
        export_csv=False,
        chunk_size=chunk_size,
    )


    # Scale population proportionally with safe per-process memory caps (<150MB RSS per worker)
    # 5,000 cards per worker * 4 workers = 20,000 cards total (1M tx / 20k cards = 50 tx/month = 1.67 tx/day DCPC benchmark)
    n_cards = min(5000, max(250, n_tx_target // 50))
    n_merchants = min(1000, max(40, n_tx_target // 250))

    engine = DiscreteEventEngine(
        n_cards=n_cards,
        n_merchants=n_merchants,
        region=region,
        adversary_mode=adversary_mode,
        seed=seed,
    )

    engine.generate_batch(
        n_transactions=n_tx_target,
        time_span_days=30,
        enforce_invariants=False,
        chunk_callback=writer.write_chunk,
        chunk_size=chunk_size,
    )

    summary = writer.finalize()
    return {
        "worker_id": worker_id,
        "generated": n_tx_target,
        "summary": summary,
        "worker_dir": str(worker_dir),
    }


class ParallelSimulationCoordinator:
    """Coordinates high-throughput simulation across CPU cores writing directly to disk."""

    def __init__(
        self,
        total_transactions: int = 1000000,
        num_workers: Optional[int] = None,
        region: str = "US",
        output_dir: str | Path = "data/simulation_run",
        chunk_size: int = 10000,
        adversary_mode: str = "intent",
        base_seed: int = 42,
    ):
        self.total_transactions = total_transactions
        # Safe default of max 4 workers to guarantee total memory stays < 1.5 GB on laptops/workstations
        self.num_workers = num_workers or max(1, min(4, (os.cpu_count() or 4) // 2))
        self.region = region.upper()
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.adversary_mode = adversary_mode
        self.base_seed = base_seed

    def run(self) -> Dict[str, Any]:
        """Executes parallel simulation pool and merges shard metadata."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tx_per_worker = math.ceil(self.total_transactions / self.num_workers)

        t0 = time.perf_counter()
        print(f"[*] Starting Parallel Simulation: {self.total_transactions:,} transactions")
        print(f"   Workers: {self.num_workers} | Tx per worker: {tx_per_worker:,} | Chunk size: {self.chunk_size:,}")
        print(f"   Region: {self.region} | Mode: {self.adversary_mode} | Output: {self.output_dir}")

        worker_results = []
        mp_ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=self.num_workers, mp_context=mp_ctx) as executor:
            futures = [
                executor.submit(
                    _worker_simulation_task,
                    worker_id=w,
                    n_tx_target=tx_per_worker if w < self.num_workers - 1 else (self.total_transactions - tx_per_worker * (self.num_workers - 1)),
                    region=self.region,
                    output_dir=str(self.output_dir),
                    chunk_size=self.chunk_size,
                    adversary_mode=self.adversary_mode,
                    seed=self.base_seed + w * 1000,
                )
                for w in range(self.num_workers)
            ]

            completed = 0
            for future in as_completed(futures):
                res = future.result()
                worker_results.append(res)
                completed += res["generated"]
                elapsed = time.perf_counter() - t0
                rate = completed / max(0.001, elapsed)
                pct = (completed / self.total_transactions) * 100.0
                print(f"   -> [{completed:,} / {self.total_transactions:,}] ({pct:.1f}%) - {rate:,.0f} tx/sec - Elapsed: {elapsed:.1f}s")

        total_elapsed = time.perf_counter() - t0
        overall_throughput = completed / max(0.001, total_elapsed)
        print("\n[+] Simulation Complete!")
        print(f"   Total Transactions: {completed:,}")
        print(f"   Total Wall Time: {total_elapsed:.2f}s ({total_elapsed/60:.2f} minutes)")
        print(f"   Overall Throughput: {overall_throughput:,.1f} tx/sec")

        # Consolidate metadata
        consolidated = self._consolidate_worker_summaries(worker_results, total_elapsed, overall_throughput)
        
        # Save master metadata
        master_meta_path = self.output_dir / "master_simulation_metadata.json"
        with open(master_meta_path, "w", encoding="utf-8") as f:
            json.dump(consolidated, f, indent=2)

        return consolidated

    def _consolidate_worker_summaries(
        self,
        worker_results: List[Dict[str, Any]],
        wall_time_sec: float,
        throughput_tps: float,
    ) -> Dict[str, Any]:
        """Merges sharded worker summaries into a master institutional summary."""
        master = {
            "total_transactions": sum(r["generated"] for r in worker_results),
            "wall_time_seconds": round(wall_time_sec, 2),
            "throughput_tps": round(throughput_tps, 1),
            "num_workers": self.num_workers,
            "region": self.region,
            "adversary_mode": self.adversary_mode,
            "approved_count": 0,
            "declined_count": 0,
            "total_fraud_count": 0,
            "total_volume": 0.0,
            "total_fraud_volume": 0.0,
            "hop_drops": defaultdict(int),
            "iso_codes": defaultdict(int),
            "macro_options": defaultdict(int),
            "hourly_legitimate": [0] * 24,
            "hourly_fraud": [0] * 24,
            "top_syndicates": defaultdict(int),
            "top_botnets": defaultdict(int),
            "top_target_merchants": defaultdict(float),
            "top_mule_rings": defaultdict(float),
            "worker_shards": [r["worker_dir"] for r in worker_results],
        }

        for r in worker_results:
            s = r["summary"]
            master["approved_count"] += s.get("approved_count", 0)
            master["declined_count"] += s.get("declined_count", 0)
            master["total_fraud_count"] += s.get("total_fraud_count", 0)
            master["total_volume"] += s.get("total_volume", 0.0)
            master["total_fraud_volume"] += s.get("total_fraud_volume", 0.0)

            for k, v in s.get("hop_drops", {}).items():
                master["hop_drops"][k] += v
            for k, v in s.get("iso_codes", {}).items():
                master["iso_codes"][k] += v
            for k, v in s.get("macro_options", {}).items():
                master["macro_options"][k] += v

            h_dist = s.get("hourly_distribution", {})
            leg = h_dist.get("legitimate", [0]*24)
            frd = h_dist.get("fraud", [0]*24)
            for h in range(24):
                master["hourly_legitimate"][h] += leg[h]
                master["hourly_fraud"][h] += frd[h]

            for k, v in s.get("top_syndicates", {}).items():
                master["top_syndicates"][k] += v
            for k, v in s.get("top_botnets", {}).items():
                master["top_botnets"][k] += v
            for k, v in s.get("top_target_merchants", {}).items():
                master["top_target_merchants"][k] += v
            for k, v in s.get("top_mule_rings", {}).items():
                master["top_mule_rings"][k] += v

        master["approval_rate"] = master["approved_count"] / max(1, master["total_transactions"])
        master["fraud_prevalence"] = master["total_fraud_count"] / max(1, master["total_transactions"])
        master["total_volume"] = round(master["total_volume"], 2)
        master["total_fraud_volume"] = round(master["total_fraud_volume"], 2)
        master["hop_drops"] = dict(master["hop_drops"])
        master["iso_codes"] = dict(master["iso_codes"])
        master["macro_options"] = dict(master["macro_options"])
        master["top_syndicates"] = dict(sorted(master["top_syndicates"].items(), key=lambda x: x[1], reverse=True)[:25])
        master["top_botnets"] = dict(sorted(master["top_botnets"].items(), key=lambda x: x[1], reverse=True)[:25])
        master["top_target_merchants"] = dict(sorted(master["top_target_merchants"].items(), key=lambda x: x[1], reverse=True)[:25])
        master["top_mule_rings"] = dict(sorted(master["top_mule_rings"].items(), key=lambda x: x[1], reverse=True)[:25])

        # Consolidate forensic threat graph across worker shards
        try:
            from prism_generator.graph_transformer import ForensicGraphTransformer
            consolidated_fraud: List[Dict[str, Any]] = []
            for r in worker_results:
                w_dir = Path(r["worker_dir"])
                pq_files = sorted(list((w_dir / "threat_intel_graph").glob("*.parquet")))
                for pq in pq_files:
                    df = pl.read_parquet(pq)
                    if "syndicate_id" in df.columns:
                        df_f = df.filter(pl.col("syndicate_id").is_not_null() & (pl.col("syndicate_id") != "") & (pl.col("syndicate_id") != "SYN_UNASSIGNED")).head(350)
                    elif "is_fraud" in df.columns:
                        df_f = df.filter(pl.col("is_fraud") == 1).head(350)
                    else:
                        df_f = df.head(350)
                    consolidated_fraud.extend(df_f.to_dicts())
                    if len(consolidated_fraud) >= 4000:
                        break
                if len(consolidated_fraud) >= 4000:
                    break

            if consolidated_fraud:
                transformer = ForensicGraphTransformer()
                master_graph = transformer.transform(consolidated_fraud)
                master_graph_path = self.output_dir / "threat_graph_sample.json"
                with open(master_graph_path, "w", encoding="utf-8") as f:
                    json.dump(master_graph, f, indent=2)
        except Exception:
            pass

        return master


def _entity_worker_simulation_task(
    worker_id: int,
    n_cards: int,
    n_merchants: int,
    time_span_days: int,
    region: str,
    output_dir: str,
    chunk_size: int,
    adversary_mode: str,
    base_seed: int,
    annual_seasonality: bool,
    p2p_enabled: bool,
    start_time_seconds: float,
) -> Dict[str, Any]:
    """Isolated worker process simulating 100% of the lifecycle for an assigned card partition."""
    from prism_generator.engine import DiscreteEventEngine
    from prism_generator.world import WorldEnvironment
    from prism_generator.syndicates import SyndicateRegistry

    worker_dir = Path(output_dir) / f"worker_{worker_id}"
    writer = StreamingDatasetWriter(
        output_dir=worker_dir,
        export_parquet=True,
        export_csv=False,
        chunk_size=chunk_size,
    )

    # Deterministic shared merchant world and syndicate infrastructure
    shared_world = WorldEnvironment(
        n_merchants=n_merchants,
        region=region,
        seed=base_seed + 99999,
    )
    shared_syndicates = SyndicateRegistry(
        region=region,
        seed=base_seed + 88888,
    )

    engine = DiscreteEventEngine(
        n_cards=n_cards,
        n_merchants=n_merchants,
        region=region,
        adversary_mode=adversary_mode,
        seed=base_seed + worker_id * 1000,
        annual_seasonality=annual_seasonality,
        p2p_enabled=p2p_enabled,
        id_prefix=f"W{worker_id}",
        world=shared_world,
        syndicates=shared_syndicates,
    )

    engine.generate_batch(
        time_span_days=time_span_days,
        start_time_seconds=start_time_seconds,
        simulation_mode="days",
        enforce_invariants=False,
        chunk_callback=writer.write_chunk,
        chunk_size=chunk_size,
    )

    summary = writer.finalize()
    return {
        "worker_id": worker_id,
        "n_cards": n_cards,
        "generated": summary["total_transactions"],
        "summary": summary,
        "worker_dir": str(worker_dir),
    }


class EntityPartitionedParallelCoordinator:
    """Coordinates high-throughput entity-partitioned parallel simulations across CPU cores.

    Key Architectural Invariants:
    1. Zero Clock Overlaps: All workers advance the identical 365-day physical calendar timeline.
    2. Zero ID Collisions: Card IDs ('CARD_W{w}_...') and Tx IDs ('TX_W{w}_...') are namespaced by worker.
    3. Causal Continuity: 100% of transactions for a card occur on a single worker core.
    4. Deterministic Shared Universe: Merchants and syndicates share deterministic seeds without locks.
    5. Streaming Out-of-Core Merge: Shards are merged via Polars LazyFrame and written to master views.
    """

    def __init__(
        self,
        total_cards: int = 100000,
        time_span_days: int = 365,
        num_workers: Optional[int] = None,
        region: str = "US",
        output_dir: str | Path = "data/simulation_annual",
        chunk_size: int = 10000,
        adversary_mode: str = "playbook",
        annual_seasonality: bool = True,
        p2p_enabled: bool = True,
        base_seed: int = 42,
        export_csv: bool = False,
        start_time_seconds: float = 1704067200.0,
    ):
        self.total_cards = total_cards
        self.time_span_days = time_span_days
        # Default to min(10, os.cpu_count() or 4) on modern multi-core systems
        self.num_workers = num_workers or min(10, os.cpu_count() or 4)
        self.region = region.upper()
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.adversary_mode = adversary_mode
        self.annual_seasonality = annual_seasonality
        self.p2p_enabled = p2p_enabled
        self.base_seed = base_seed
        self.export_csv = export_csv
        self.start_time_seconds = start_time_seconds

    def run(self) -> Dict[str, Any]:
        """Executes parallel simulation pool and merges shards into master views."""
        import shutil

        self.output_dir.mkdir(parents=True, exist_ok=True)
        cards_per_worker = math.ceil(self.total_cards / self.num_workers)
        n_merchants = min(1500, max(50, self.total_cards // 100))

        t0 = time.perf_counter()
        print(f"[*] Starting Entity-Partitioned Simulation: {self.total_cards:,} cards across {self.time_span_days} days")
        print(f"   Workers: {self.num_workers} | Cards/worker: {cards_per_worker:,} | Chunk: {self.chunk_size:,}")
        print(f"   Region: {self.region} | Mode: {self.adversary_mode} | Annual: {self.annual_seasonality} | P2P: {self.p2p_enabled}")
        print(f"   Output Directory: {self.output_dir}")

        worker_results = []
        mp_ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=self.num_workers, mp_context=mp_ctx) as executor:
            futures = [
                executor.submit(
                    _entity_worker_simulation_task,
                    worker_id=w,
                    n_cards=cards_per_worker if w < self.num_workers - 1 else (self.total_cards - cards_per_worker * (self.num_workers - 1)),
                    n_merchants=n_merchants,
                    time_span_days=self.time_span_days,
                    region=self.region,
                    output_dir=str(self.output_dir),
                    chunk_size=self.chunk_size,
                    adversary_mode=self.adversary_mode,
                    base_seed=self.base_seed,
                    annual_seasonality=self.annual_seasonality,
                    p2p_enabled=self.p2p_enabled,
                    start_time_seconds=self.start_time_seconds,
                )
                for w in range(self.num_workers)
            ]

            completed_tx = 0
            for future in as_completed(futures):
                res = future.result()
                worker_results.append(res)
                completed_tx += res["generated"]
                elapsed = time.perf_counter() - t0
                rate = completed_tx / max(0.001, elapsed)
                print(f"   -> Worker {res['worker_id']} done ({res['generated']:,} tx) | Total: {completed_tx:,} tx | Rate: {rate:,.0f} tx/sec | Elapsed: {elapsed:.1f}s")

        sim_elapsed = time.perf_counter() - t0
        print(f"\n[*] All {self.num_workers} workers finished in {sim_elapsed:.2f}s ({sim_elapsed/60:.2f} min).")
        print("[*] Merging worker shards into master table and 5 institutional views...")

        # Sequential Polars LazyFrame external merge
        t_merge_start = time.perf_counter()
        view_names = [
            "master",
            "auth_stream",
            "gateway_telemetry",
            "clearing_settlement",
            "dispute_recovery",
            "threat_intel_graph",
        ]

        for view in view_names:
            shard_paths = []
            for w in range(self.num_workers):
                shard_dir = self.output_dir / f"worker_{w}" / view
                if shard_dir.exists():
                    pqs = list(shard_dir.glob("*.parquet"))
                    shard_paths.extend([str(p) for p in pqs])

            if not shard_paths:
                continue

            if len(shard_paths) == 1:
                lf = pl.scan_parquet(shard_paths[0])
            else:
                lf = pl.concat([pl.scan_parquet(p) for p in shard_paths], how="diagonal_relaxed")

            if "tx_time_seconds" in lf.collect_schema().names():
                lf = lf.sort("tx_time_seconds")

            out_view_dir = self.output_dir / view
            out_view_dir.mkdir(parents=True, exist_ok=True)
            final_pq = out_view_dir / f"{view}.parquet"
            lf.sink_parquet(final_pq, compression="zstd")
            if self.export_csv:
                try:
                    final_csv = out_view_dir / f"{view}.csv"
                    schema = lf.collect_schema()
                    flat_cols = [col for col, dt in schema.items() if not dt.is_nested()]
                    lf.select(flat_cols).sink_csv(final_csv)
                except Exception:
                    pass

        merge_elapsed = time.perf_counter() - t_merge_start
        print(f"[+] Out-of-Core Merge complete in {merge_elapsed:.2f}s!")

        # Consolidate master metadata
        total_wall_time = time.perf_counter() - t0
        overall_throughput = completed_tx / max(0.001, total_wall_time)
        coordinator_legacy = ParallelSimulationCoordinator(
            total_transactions=completed_tx,
            num_workers=self.num_workers,
            region=self.region,
            output_dir=self.output_dir,
            adversary_mode=self.adversary_mode,
            base_seed=self.base_seed,
        )
        consolidated = coordinator_legacy._consolidate_worker_summaries(
            worker_results, total_wall_time, overall_throughput
        )
        consolidated["total_cards"] = self.total_cards
        consolidated["time_span_days"] = self.time_span_days
        consolidated["annual_seasonality"] = self.annual_seasonality
        consolidated["p2p_enabled"] = self.p2p_enabled

        # Save master metadata
        master_meta_path = self.output_dir / "master_simulation_metadata.json"
        with open(master_meta_path, "w", encoding="utf-8") as f:
            json.dump(consolidated, f, indent=2)

        # Clean up temporary worker folders
        for w in range(self.num_workers):
            w_dir = self.output_dir / f"worker_{w}"
            if w_dir.exists():
                shutil.rmtree(w_dir, ignore_errors=True)

        print("[+] Cleaned up intermediate worker directories.")
        print(f"[+] Dataset generated successfully at: {self.output_dir}")
        print(f"   Total Transactions: {completed_tx:,}")
        print(f"   Total Time: {total_wall_time:.2f}s ({total_wall_time/60:.2f} min)")
        print(f"   Overall Throughput: {overall_throughput:,.1f} tx/sec")

        return consolidated
