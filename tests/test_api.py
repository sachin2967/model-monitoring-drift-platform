"""Unit tests for MLOps drift monitoring FastAPI endpoints."""

from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "reference_loaded" in data


def test_benchmark_endpoint():
    response = client.get("/api/benchmark")
    assert response.status_code == 200
    data = response.json()
    assert "benchmark_summary" in data
    assert "simulated_windows" in data
    assert data["benchmark_summary"]["critical_lead_indicator"] is True


def test_monitor_batch_endpoint():
    sample_records = [
        {
            "total_gmv": 1500.0,
            "total_items": 10,
            "avg_item_price": 150.0,
            "avg_delivery_delay": 0.5,
            "mean_review_score": 4.5,
            "orders_per_month": 2.0,
        }
        for _ in range(25)
    ]
    response = client.post("/api/monitor/batch", json={"records": sample_records})
    assert response.status_code == 200
    res = response.json()
    assert "overall_status" in res
    assert "reports" in res
    assert "drift_percentage" in res
    assert "drifted_features_count" in res


def test_invalid_batch_payload():
    response = client.post("/api/monitor/batch", json={"wrong_key": []})
    assert response.status_code == 422
