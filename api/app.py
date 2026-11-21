"""FastAPI demo service for tabular drift monitoring."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.drift.detectors import DriftMonitorEngine
from src.drift.simulator import DriftSimulator
from src.drift.synthetic import FEATURES, make_synthetic_batch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
app = FastAPI(
    title="Model Drift Monitoring Demo API",
    description="Demo API computing tabular feature-shift diagnostics against a synthetic reference sample.",
    version="0.2.0",
)

# Initialize engine with baseline
engine = DriftMonitorEngine(psi_threshold=0.20, ks_alpha=0.01)
simulator = DriftSimulator(random_state=42)

# Load a deterministic synthetic reference; no neighboring project data required.
engine.set_reference(make_synthetic_batch(4000, seed=42)[FEATURES])


class BatchDataInput(BaseModel):
    records: list[dict[str, Any]]


class SimulationRequest(BaseModel):
    drift_type: str  # 'covariate', 'concept', 'null_spike'
    magnitude: float = 2.0


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "reference_loaded": engine.reference_df is not None,
        "monitored_features_count": len(engine.numerical_cols)
        + len(engine.categorical_cols),
    }


@app.post("/api/monitor/batch")
def evaluate_batch(data: BatchDataInput):
    """Evaluates an incoming demo batch against the synthetic reference."""

    batch_df = pd.DataFrame(data.records)
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
