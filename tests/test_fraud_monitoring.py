"""Reference identity, cohort integrity, label timing and durable report checks."""

import json
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from api.app import app
from src.drift.fraud import FEATURE_NAMES


@pytest.fixture
def window(tmp_path, monkeypatch):
    path = tmp_path / "reference.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "fraud-reference-v1",
                "feature_version": "ulb-pca-v1",
                "model_version": "a" * 64,
                "feature_names": FEATURE_NAMES,
                "threshold": 0.5,
                "source": "test-only fixture",
                "records": [[float(i)] * 29 for i in range(100)],
                "scores": [i / 100 for i in range(100)],
            }
        )
    )
    monkeypatch.setenv("FRAUD_REFERENCE_PATH", str(path))
    now = datetime.now(UTC) - timedelta(seconds=1)
    created = now - timedelta(days=2)
    return {
        "schema_version": "fraud-events-v1",
        "as_of": now.isoformat(),
        "events": [
            {
                "cursor": i + 1,
                "prediction_id": f"p{i}",
                "transaction_id": f"t{i}",
                "model_version": "a" * 64,
                "feature_version": "ulb-pca-v1",
                "threshold": 0.5,
                "score": 0.8 if i % 2 else 0.2,
                "features": [float(i)] * 29,
                "created_at": created.isoformat(),
                "label": i % 2,
                "label_available_at": (created + timedelta(days=1)).isoformat(),
                "label_source": "test_fixture",
            }
            for i in range(30)
        ],
    }


def test_reference_required_without_synthetic_fallback(window, monkeypatch):
    monkeypatch.delenv("FRAUD_REFERENCE_PATH")
    with TestClient(app) as client:
        assert client.post("/api/monitor/fraud-window", json=window).status_code == 503


def test_real_contract_report_is_persisted_and_retrievable(window):
    with TestClient(app) as client:
        response = client.post("/api/monitor/fraud-window", json=window)
        assert response.status_code == 200, response.text
        report = response.json()
        assert report["distribution_diagnostics"]["total_monitored_features"] == 30
        assert report["label_performance"]["roc_auc"] == 1.0
        assert report["label_performance"]["coverage"] == 1.0
        assert report["label_performance"]["confusion_matrix"]["false_negative"] == 0
        assert client.get(f"/api/reports/{report['report_id']}").json() == report
    # A fresh client sees the same durable result.
    with TestClient(app) as client:
        assert client.get(f"/api/reports/{report['report_id']}").status_code == 200


def test_future_labels_and_immature_cohorts_are_not_used(window):
    for event in window["events"]:
        event["label_available_at"] = (
            datetime.now(UTC) + timedelta(days=1)
        ).isoformat()
    with TestClient(app) as client:
        response = client.post("/api/monitor/fraud-window", json=window)
        assert response.status_code == 200
        assert response.json()["label_performance"]["labeled_rows"] == 0
        assert (
            response.json()["label_performance"]["status"]
            == "insufficient_mature_labels"
        )


@pytest.mark.parametrize(
    "change",
    [
        "duplicate",
        "mixed_model",
        "mixed_policy",
        "future_event",
        "early_label",
        "missing_label_time",
    ],
)
def test_rejects_incoherent_windows(window, change):
    event = window["events"][1]
    if change == "duplicate":
        event["prediction_id"] = window["events"][0]["prediction_id"]
    elif change == "mixed_model":
        event["model_version"] = "b" * 64
    elif change == "mixed_policy":
        event["threshold"] = 0.7
    elif change == "future_event":
        event["created_at"] = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    elif change == "early_label":
        event["label_available_at"] = "2000-01-01T00:00:00+00:00"
    else:
        del event["label_available_at"]
    with TestClient(app) as client:
        assert client.post("/api/monitor/fraud-window", json=window).status_code == 422


def test_single_class_does_not_publish_misleading_ranking_metrics(window):
    for event in window["events"]:
        event["label"] = 0
    with TestClient(app) as client:
        result = client.post("/api/monitor/fraud-window", json=window).json()[
            "label_performance"
        ]
        assert result["roc_auc"] is None
        assert result["average_precision"] is None
        assert result["ranking_metrics_status"] == "undefined_single_class"


def test_bad_reference_checksum_blocks_readiness(window, monkeypatch):
    monkeypatch.setenv("FRAUD_REFERENCE_SHA256", "0" * 64)
    with TestClient(app) as client:
        assert client.get("/live").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.post("/api/monitor/fraud-window", json=window).status_code == 503


def test_production_without_key_fails_closed(window, monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    with TestClient(app) as client:
        assert client.get("/ready").status_code == 503
        assert client.post("/api/monitor/fraud-window", json=window).status_code == 503


def test_local_key_protects_reports_and_schema(window, monkeypatch):
    monkeypatch.setenv("DRIFT_API_KEY", "x" * 32)
    with TestClient(app) as client:
        assert client.get("/docs").status_code == 401
        assert client.get("/api/reports/unknown").status_code == 401
        assert (
            client.get(
                "/api/reports/unknown", headers={"X-API-Key": "x" * 32}
            ).status_code
            == 404
        )
