"""FastAPI production service for continuous model monitoring and statistical drift detection."""

import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.drift.detectors import DriftMonitorEngine
from src.drift.simulator import DriftSimulator


PROJECT_ROOT = Path(__file__).resolve().parent.parent
app = FastAPI(
    title="MLOps Model Monitoring & Drift Detection API",
    description="Statistical observability microservice computing PSI, KS-test, and Wasserstein distance on production inference streams.",
    version="1.0.0",
)

# Initialize engine with baseline
engine = DriftMonitorEngine(psi_threshold=0.20, ks_alpha=0.01)
simulator = DriftSimulator(random_state=42)

# Load reference dataset if available
data_ref_path = PROJECT_ROOT.parent / "seller-quality-intelligence" / "data" / "processed" / "seller_features.csv"
if data_ref_path.exists():
    ref_df = pd.read_csv(data_ref_path).sample(frac=0.6, random_state=42)
    engine.set_reference(ref_df)
    print("✅ Reference distribution loaded into drift engine.")


class BatchDataInput(BaseModel):
    records: List[Dict[str, Any]]


class SimulationRequest(BaseModel):
    drift_type: str  # 'covariate', 'concept', 'null_spike'
    magnitude: float = 2.0


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "reference_loaded": engine.reference_df is not None,
        "monitored_features_count": len(engine.numerical_cols) + len(engine.categorical_cols),
    }


@app.post("/api/monitor/batch")
def evaluate_batch(data: BatchDataInput):
    """Evaluates incoming live production batch for statistical drift."""
    if engine.reference_df is None:
        raise HTTPException(status_code=503, detail="Reference baseline not loaded.")

    batch_df = pd.DataFrame(data.records)
    res = engine.evaluate_batch(batch_df)
    return res


@app.get("/api/benchmark")
def get_benchmark_results():
    """Returns the latest 6-window controlled drift benchmark report."""
    rep_file = PROJECT_ROOT / "drift_benchmark_report.json"
    if not rep_file.exists():
        raise HTTPException(status_code=404, detail="Benchmark not run. Run scripts/run_drift_experiments.py first.")

    with open(rep_file) as f:
        return json.load(f)
