"""Benchmark script executing controlled drift experiments across multiple production windows."""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.drift.detectors import DriftMonitorEngine
from src.drift.simulator import DriftSimulator
from src.drift.correlation import DriftPerformanceCorrelator


def run_experiments():
    project_root = Path(__file__).resolve().parent.parent
    seller_data_path = project_root.parent / "seller-quality-intelligence" / "data" / "processed" / "seller_features.csv"

    if seller_data_path.exists():
        raw_df = pd.read_csv(seller_data_path)
    else:
        # Fallback synthetic generation
        np.random.seed(42)
        raw_df = pd.DataFrame({
            "total_gmv": np.random.exponential(2500, 1000),
            "avg_delivery_delay": np.random.normal(1.5, 2.0, 1000),
            "avg_carrier_handoff": np.random.normal(2.0, 1.0, 1000),
            "freight_ratio": np.random.uniform(0.05, 0.35, 1000),
            "orders_per_month": np.random.exponential(4, 1000),
            "is_at_risk": np.random.choice([0, 1], 1000, p=[0.65, 0.35]),
        })

    print("==================================================")
    print("🚀 1. INITIALIZING REFERENCE DISTRIBUTION")
    print("==================================================")
    # Split into 60% Reference baseline (training) and 40% Live pool
    ref_df = raw_df.sample(frac=0.6, random_state=42)
    pool_df = raw_df.drop(ref_df.index)

    engine = DriftMonitorEngine(psi_threshold=0.20, ks_alpha=0.01)
    engine.set_reference(ref_df)
    simulator = DriftSimulator(random_state=42)
    correlator = DriftPerformanceCorrelator()

    print(f"Reference distribution frozen: {len(ref_df)} samples across {len(engine.numerical_cols)} numerical features.")

    print("\n==================================================")
    print("🧪 2. SIMULATING 6 PRODUCTION TIME WINDOWS")
    print("==================================================")

    windows = [
        {"id": "W1_Stable", "type": "baseline_clean", "desc": "Clean production window with organic variance (No drift)"},
        {"id": "W2_Mild_Shift", "type": "mild_covariate", "desc": "Mild shipping delay shift (1.0 std shift)"},
        {"id": "W3_Severe_Covariate", "type": "severe_covariate", "desc": "Severe carrier handoff delay inflation (2.5 std shift)"},
        {"id": "W4_Concept_Drift", "type": "concept_drift", "desc": "Market disruption: High-delay sellers maintain good reviews"},
        {"id": "W5_Pipeline_Corruption", "type": "null_spike", "desc": "Upstream telemetry corruption: 30% missing values in freight"},
        {"id": "W6_Critical_MultiDrift", "type": "multi_drift", "desc": "Systemic logistics disruption: delay + freight + carrier shifts"},
    ]

    window_results = []

    for w in windows:
        batch = pool_df.sample(min(200, len(pool_df)), replace=True, random_state=42).copy()

        # Inject specific anomaly
        if w["type"] == "mild_covariate":
            target_cols = [c for c in ["avg_delivery_delay"] if c in batch.columns]
            batch = simulator.inject_covariate_shift(batch, target_cols, mean_shift_std=1.0)
        elif w["type"] == "severe_covariate":
            target_cols = [c for c in ["avg_carrier_handoff"] if c in batch.columns]
            batch = simulator.inject_covariate_shift(batch, target_cols, mean_shift_std=2.5)
        elif w["type"] == "concept_drift":
            feat = "avg_delivery_delay" if "avg_delivery_delay" in batch.columns else batch.columns[0]
            batch = simulator.inject_concept_drift(batch, feature_col=feat, target_col="is_at_risk", flip_ratio=0.4)
        elif w["type"] == "null_spike":
            target_cols = [c for c in ["freight_ratio", "total_gmv"] if c in batch.columns]
            batch = simulator.inject_pipeline_corruption(batch, target_cols, null_spike_rate=0.30)
        elif w["type"] == "multi_drift":
            target_cols = [c for c in ["avg_delivery_delay", "avg_carrier_handoff", "total_gmv"] if c in batch.columns]
            batch = simulator.inject_covariate_shift(batch, target_cols, mean_shift_std=2.0)

        # Run drift evaluation
        report = engine.evaluate_batch(batch)

        # Simulate model predictions and delayed ground truth
        y_true = batch["is_at_risk"].fillna(0).values if "is_at_risk" in batch.columns else np.random.choice([0, 1], len(batch))
        # Simulated model output with performance degradation under severe drift
        perf_penalty = 0.35 if w["type"] in ["severe_covariate", "multi_drift", "concept_drift"] else 0.0
        y_prob = np.clip(y_true * 0.8 + np.random.uniform(0.0, 0.4, len(batch)) - perf_penalty, 0.05, 0.95)

        corr_entry = correlator.record_window(w["id"], report, y_true, y_prob)

        status_icon = "🟢" if report["overall_status"] == "HEALTHY" else ("🟡" if report["overall_status"] == "WARNING" else "🔴")
        print(f"{status_icon} [{w['id']}] Status: {report['overall_status']:<15} | Drifted: {report['drifted_features_count']}/{report['total_monitored_features']} | Max PSI: {corr_entry['max_psi']:.3f} | F1: {corr_entry['f1_score']:.3f}")

        window_results.append({
            "window_id": w["id"],
            "description": w["desc"],
            "drift_status": report["overall_status"],
            "drifted_features_count": report["drifted_features_count"],
            "max_psi": corr_entry["max_psi"],
            "f1_score": corr_entry["f1_score"],
            "roc_auc": corr_entry["roc_auc"],
        })

    print("\n==================================================")
    print("📈 3. COMPUTING DRIFT-TO-PERFORMANCE CORRELATION")
    print("==================================================")
    attribution = correlator.compute_correlation_attribution()
    print(f"PSI vs Downstream F1 Score Correlation: {attribution.get('psi_vs_f1_correlation', 'N/A')}")
    print(f"Lead Indicator Confirmed: {attribution.get('critical_lead_indicator')}")
    print(f"Summary: {attribution.get('summary')}")

    out_file = project_root / "drift_benchmark_report.json"
    with open(out_file, "w") as f:
        json.dump({
            "benchmark_summary": attribution,
            "simulated_windows": window_results,
        }, f, indent=2)

    print(f"\n✅ All drift benchmark reports saved to: {out_file}")


if __name__ == "__main__":
    run_experiments()
