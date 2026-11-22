"""FastAPI demo service for tabular drift monitoring."""

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.drift.detectors import DriftMonitorEngine
from src.drift.fraud import FraudReference, FraudWindow
from src.drift.history import connect, read_report, record_report
from src.drift.http import MonitoringBoundary, configuration_error
from src.drift.simulator import DriftSimulator
from src.drift.synthetic import FEATURES, make_synthetic_batch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
app = FastAPI(
    title="Model Drift Monitoring Demo API",
    description="Demo API computing tabular feature-shift diagnostics against a synthetic reference sample.",
    version="0.2.0",
)
app.add_middleware(MonitoringBoundary)


def history_path():
    return Path(os.getenv("DRIFT_STATE_PATH", PROJECT_ROOT / "var" / "reports.sqlite3"))


def fraud_reference():
    configured = os.getenv("FRAUD_REFERENCE_PATH")
    if not configured:
        raise HTTPException(
            503,
            "No fraud training reference configured; synthetic fallback is disabled for this endpoint.",
        )
    try:
        path = Path(configured)
        expected = os.getenv("FRAUD_REFERENCE_SHA256")
        if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Reference checksum mismatch")
        return FraudReference(path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(503, "Fraud reference is unavailable or invalid.") from exc


# Initialize engine with baseline
engine = DriftMonitorEngine(psi_threshold=0.20, ks_alpha=0.01)
simulator = DriftSimulator(random_state=42)

# Load a deterministic synthetic reference; no neighboring project data required.
engine.set_reference(make_synthetic_batch(4000, seed=42)[FEATURES])


class BatchDataInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[dict[str, Any]] = Field(min_length=25, max_length=5000)


class SimulationRequest(BaseModel):
    drift_type: str  # 'covariate', 'concept', 'null_spike'
    magnitude: float = 2.0


@app.get("/health")
@app.get("/ready")
def health():
    error = configuration_error()
    if error:
        raise HTTPException(503, error)
    try:
        connection = connect(history_path())
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.rollback()
        finally:
            connection.close()
    except (OSError, sqlite3.Error) as exc:
        raise HTTPException(503, "Report storage unavailable.") from exc
    if os.getenv("FRAUD_REFERENCE_PATH"):
        fraud_reference()
    return {
        "status": "healthy",
        "reference_loaded": engine.reference_df is not None,
        "monitored_features_count": len(engine.numerical_cols)
        + len(engine.categorical_cols),
        "default_reference_scope": "synthetic demonstration",
        "fraud_reference_configured": bool(os.getenv("FRAUD_REFERENCE_PATH")),
    }


@app.get("/live")
def live():
    return {"status": "alive"}


@app.post("/api/monitor/fraud-window")
def evaluate_fraud_window(window: FraudWindow):
    reference = fraud_reference()
    try:
        report = reference.evaluate(window)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    try:
        return record_report(history_path(), report)
    except (OSError, sqlite3.Error) as exc:
        raise HTTPException(
            503, "Report persistence failed; evaluation not acknowledged."
        ) from exc


@app.get("/api/reports/{report_id}")
def get_report(report_id: str):
    try:
        report = read_report(history_path(), report_id)
    except (OSError, sqlite3.Error) as exc:
        raise HTTPException(503, "Report storage unavailable.") from exc
    if report is None:
        raise HTTPException(404, "Report not found.")
    return report


@app.post("/api/monitor/batch")
def evaluate_batch(data: BatchDataInput):
    """Evaluates an incoming demo batch against the synthetic reference."""

    batch_df = pd.DataFrame(data.records)
    required = set(engine.numerical_cols + engine.categorical_cols)
    if any(set(record) != required for record in data.records):
        raise HTTPException(
            422,
            "Every record must explicitly contain the full reference schema; use null for missing values.",
        )
    missing = sorted(required - set(batch_df.columns))
    unexpected = sorted(set(batch_df.columns) - required)
    if missing or unexpected:
        raise HTTPException(
            status_code=422,
            detail={"missing_features": missing, "unexpected_features": unexpected},
        )
    for column in engine.numerical_cols:
        converted = pd.to_numeric(batch_df[column], errors="coerce")
        invalid = batch_df[column].notna() & converted.isna()
        if invalid.any():
            raise HTTPException(
                status_code=422,
                detail=f"Feature {column!r} must contain numbers or nulls.",
            )
        finite = converted.dropna().map(lambda value: abs(value) != float("inf"))
        if not finite.all():
            raise HTTPException(
                status_code=422,
                detail=f"Feature {column!r} must contain finite values.",
            )
        batch_df[column] = converted
    res = engine.evaluate_batch(batch_df)
    return res


@app.get("/api/benchmark")
def get_benchmark_results():
    """Returns the latest 6-window controlled drift benchmark report."""
    rep_file = PROJECT_ROOT / "drift_benchmark_report.json"
    if not rep_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Benchmark not run. Run scripts/run_drift_experiments.py first.",
        )

    with open(rep_file) as f:
        return json.load(f)
