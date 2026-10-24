"""Unit tests for Prism Generator Studio GUI backend API endpoints."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="GUI backend needs the 'gui' extra: pip install '.[gui]'")

from fastapi.testclient import TestClient  # noqa: E402

from gui.backend.app import app  # noqa: E402


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "loaded_transactions" in data


def test_specs_summary(client):
    response = client.get("/api/specs/summary")
    assert response.status_code == 200
    data = response.json()
    assert "regions" in data
    assert "US" in data["regions"]
    assert "IN" in data["regions"]
    assert "adversary_modes" in data
    assert "calibration_profiles" in data


def test_generate_and_transactions_flow(client):
    # 1. Trigger generate endpoint
    gen_payload = {
        "region": "IN",
        "n_transactions": 60,
        "fraud_prevalence": 0.10,
        "adversary_mode": "intent",
        "seed": 12345,
    }
    gen_res = client.post("/api/generate", json=gen_payload)
    assert gen_res.status_code == 200
    gen_data = gen_res.json()
    assert gen_data["success"] is True
    assert gen_data["metadata"]["region"] == "IN"
    assert gen_data["metadata"]["n_transactions"] == 60
    assert gen_data["metadata"]["fraud_count"] > 0

    # 2. Query transactions list
    tx_res = client.get("/api/transactions?page=1&page_size=10")
    assert tx_res.status_code == 200
    tx_data = tx_res.json()
    assert tx_data["total"] == 60
    assert len(tx_data["items"]) == 10
    assert tx_data["page"] == 1

    first_tx_id = tx_data["items"][0]["transaction_id"]

    # 3. Query single transaction detail
    detail_res = client.get(f"/api/transactions/{first_tx_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["transaction_id"] == first_tx_id
    assert "amount" in detail
    assert "currency" in detail

    # 4. Filter by fraud
    fraud_res = client.get("/api/transactions?filter_status=fraud")
    assert fraud_res.status_code == 200
    fraud_data = fraud_res.json()
    for item in fraud_data["items"]:
        assert item["is_fraud"] == 1

    # 5. Export CSV
    csv_res = client.get("/api/export/csv")
    assert csv_res.status_code == 200
    assert "text/csv" in csv_res.headers["content-type"]
    assert len(csv_res.text) > 0


def test_static_html_served(client):
    # Root serves index.html when the frontend bundle is built, and a build
    # placeholder page (still HTML, still titled Prism Generator Studio) when it is not.
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Prism Generator Studio" in response.text


def test_benchmark_endpoint(client):
    bench_payload = {
        "model_type": "lightgbm",
        "n_transactions": 100,
        "fraud_prevalence": 0.08,
        "seed": 42,
    }
    bench_res = client.post("/api/benchmark", json=bench_payload)
    assert bench_res.status_code == 200
    bench_data = bench_res.json()
    assert bench_data["success"] is True
    assert "results" in bench_data
    results = bench_data["results"]
    assert "auc_roc" in results
    assert "mean_kendall_tau" in results
    assert "mean_precision_at_3" in results


def test_visualizer_bundle_endpoint(client):
    response = client.get("/api/visualizer/bundle")
    assert response.status_code == 200
    bundle = response.json()
    assert "threat_graph" in bundle
    assert "nodes" in bundle["threat_graph"]
    assert "links" in bundle["threat_graph"]
    assert "metadata" in bundle
    assert "switch_funnel" in bundle
    assert "temporal_series" in bundle
