"""Unit tests for MLOps drift monitoring FastAPI endpoints."""

from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["reference_loaded"] is True


def test_benchmark_endpoint():
    response = client.get("/api/benchmark")
    assert response.status_code == 200
    data = response.json()
    assert "benchmark_summary" in data
    assert "simulated_windows" in data
    assert "inference" in data["benchmark_summary"]
    assert "lead-time claim" in data["benchmark_summary"]["inference"]


def test_monitor_batch_endpoint():
    sample_records = [
        {
            "total_gmv": 0.1,
            "freight_ratio": -0.2,
            "avg_delivery_delay": 0.5,
            "avg_carrier_handoff": 0.1,
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


def test_monitor_rejects_missing_features_and_non_numeric_values():
    rows = [{"avg_delivery_delay": 0.1, "avg_carrier_handoff": 0.2,
             "freight_ratio": 0.3} for _ in range(25)]
    assert client.post("/api/monitor/batch", json={"records": rows}).status_code == 422

    rows = [{"avg_delivery_delay": "bad", "avg_carrier_handoff": 0.2,
             "freight_ratio": 0.3, "total_gmv": 0.4} for _ in range(25)]
    assert client.post("/api/monitor/batch", json={"records": rows}).status_code == 422


def test_monitor_bounds_batch_size():
    row = {"avg_delivery_delay": 0.1, "avg_carrier_handoff": 0.2,
           "freight_ratio": 0.3, "total_gmv": 0.4}
    assert client.post("/api/monitor/batch", json={"records": [row]}).status_code == 422
