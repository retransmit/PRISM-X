"""FastAPI Application Server for Prism Generator Studio GUI."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Response, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .services import sim_state, run_xai_benchmark
from prism_generator.spec_loader import load_all_specs
from prism_generator.visualizer import compile_simulation_data_bundle


app = FastAPI(
    title="Prism Generator Studio API",
    description="Backend API for payment fraud simulation, causal explainability, and benchmarking.",
    version="0.3.0",
)

# Enable CORS for local Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    region: str = Field(default="IN", description="Payment region: US or IN")
    simulation_mode: str = Field(default="transactions", description="Simulation mode: 'transactions' or 'days'")
    n_transactions: Optional[int] = Field(default=1000, ge=50, le=100000, description="Total transactions to simulate")
    time_span_days: Optional[int] = Field(default=14, ge=1, le=180, description="Number of simulation days if mode is 'days'")
    fraud_prevalence: float = Field(default=0.04, ge=0.001, le=0.5, description="Fraud prevalence ratio")
    adversary_mode: str = Field(default="intent", description="Adversary architecture: intent or playbook")
    seed: int = Field(default=42, description="Deterministic random seed")


class StreamStartRequest(BaseModel):
    region: str = Field(default="IN")
    target_tps: float = Field(default=20.0, ge=1.0, le=100.0)
    fraud_prevalence: float = Field(default=0.04, ge=0.001, le=0.5)
    adversary_mode: str = Field(default="intent")
    seed: int = Field(default=42)


class BenchmarkRequest(BaseModel):
    model_type: str = Field(default="lightgbm", description="Classifier model architecture")
    n_transactions: int = Field(default=500, ge=100, le=5000)
    fraud_prevalence: float = Field(default=0.05, ge=0.01, le=0.3)
    seed: int = Field(default=42)


@app.websocket("/ws/simulation")
async def websocket_simulation(websocket: WebSocket):
    await sim_state.handle_websocket(websocket)


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "version": "0.3.0",
        "loaded_transactions": len(sim_state.records),
        "is_generating": sim_state.is_generating,
        "is_streaming": sim_state.is_streaming,
    }


@app.get("/api/specs/summary")
def get_specs_summary():
    specs = load_all_specs()
    return {
        "regions": ["US", "IN"],
        "adversary_modes": ["intent", "playbook"],
        "calibration_profiles": ["rbi-psi-2026-07", "fed-payments-study-2022"],
        "playbooks": list(specs.playbooks.keys()),
        "models": ["lightgbm", "random_forest", "gradient_boosting"],
    }


@app.post("/api/generate")
async def generate_simulation(req: GenerateRequest):
    try:
        meta = await sim_state.generate_async(
            region=req.region,
            n_transactions=req.n_transactions or 1000,
            fraud_prevalence=req.fraud_prevalence,
            adversary_mode=req.adversary_mode,
            seed=req.seed,
            simulation_mode=req.simulation_mode,
            time_span_days=req.time_span_days or 14,
        )
        return {"success": True, "metadata": meta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stream/start")
async def start_stream(req: StreamStartRequest):
    asyncio.create_task(sim_state.start_live_stream_async(
        region=req.region,
        target_tps=req.target_tps,
        fraud_prevalence=req.fraud_prevalence,
        adversary_mode=req.adversary_mode,
        seed=req.seed,
    ))
    return {"success": True, "status": "streaming_started"}


@app.post("/api/stream/stop")
async def stop_stream():
    await sim_state.stop_live_stream_async()
    return {"success": True, "status": "streaming_stopped"}


@app.get("/api/metadata")
def get_metadata():
    if not sim_state.records:
        sim_state.generate(region="IN", n_transactions=500, fraud_prevalence=0.04, seed=42)
    return sim_state.metadata


@app.get("/api/visualizer/bundle")
def get_visualizer_bundle():
    if not sim_state.records:
        sim_state.generate(region="IN", n_transactions=500, fraud_prevalence=0.04, seed=42)
    bundle = compile_simulation_data_bundle(
        records=sim_state.records,
        region=sim_state.metadata.get("region", "IN"),
        seed=sim_state.metadata.get("seed", 42),
    )
    return bundle


@app.get("/api/transactions")
def list_transactions(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    filter_status: str = Query(default="all"),
    search: str = Query(default=""),
):
    if not sim_state.records:
        # Generate initial default batch if empty
        sim_state.generate(region="IN", n_transactions=500, fraud_prevalence=0.04, seed=42)

    return sim_state.get_paginated(
        page=page,
        page_size=page_size,
        filter_status=filter_status,
        search=search,
    )


@app.get("/api/transactions/{tx_id}")
def get_transaction(tx_id: str):
    tx = sim_state.get_by_id(tx_id)
    if not tx:
        raise HTTPException(status_code=404, detail=f"Transaction {tx_id} not found")
    return tx


@app.post("/api/benchmark")
def run_benchmark(req: BenchmarkRequest):
    try:
        results = run_xai_benchmark(
            model_type=req.model_type,
            n_transactions=req.n_transactions,
            fraud_prevalence=req.fraud_prevalence,
            seed=req.seed,
        )
        return {"success": True, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/export/csv")
def download_csv():
    csv_data = sim_state.export_csv()
    if not csv_data:
        raise HTTPException(status_code=400, detail="No transactions generated yet")
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=prism_generator_export_{sim_state.metadata.get('region', 'ALL')}.csv"},
    )


# Mount static production build if compiled; otherwise serve a placeholder page that
# explains how to build it, so "/" never answers an unexplained 404 (and a fresh clone
# or CI run without `gui/frontend/dist` still passes the API test suite).
dist_dir = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if dist_dir.exists():
    app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="frontend")
else:
    _FRONTEND_BUILD_PLACEHOLDER = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Prism Generator Studio</title>
  <style>
    body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 0; padding: 3rem;
           background: #0f172a; color: #e2e8f0; line-height: 1.6; }
    code { background: #1e293b; padding: .15rem .4rem; border-radius: 4px; }
    pre { background: #1e293b; padding: 1rem; border-radius: 8px; overflow-x: auto; }
    h1 { margin-top: 0; }
    a { color: #7dd3fc; }
  </style>
</head>
<body>
  <h1>Prism Generator Studio</h1>
  <p>The API is running, but the frontend bundle has not been built yet.</p>
  <pre>cd gui/frontend
npm install
npm run build</pre>
  <p>Or run the Vite dev server against this API:</p>
  <pre>cd gui/frontend
npm run dev</pre>
  <p>The API itself is available at <a href="/docs">/docs</a>.</p>
</body>
</html>
"""

    @app.get("/", include_in_schema=False)
    def frontend_placeholder() -> Response:
        return Response(content=_FRONTEND_BUILD_PLACEHOLDER, media_type="text/html")
