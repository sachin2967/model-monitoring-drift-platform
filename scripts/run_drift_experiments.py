"""Run deterministic controlled drift experiments on a synthetic classifier."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.drift.correlation import DriftPerformanceCorrelator
from src.drift.detectors import DriftMonitorEngine
from src.drift.simulator import DriftSimulator
from src.drift.synthetic import FEATURES, make_synthetic_batch


def run_experiments() -> dict:
    root = Path(__file__).resolve().parent.parent
    training = make_synthetic_batch(4000, seed=42)
    reference = training[FEATURES].copy()
    model = make_pipeline(
        SimpleImputer(strategy="median"), LogisticRegression(max_iter=500)
    )
    model.fit(training[FEATURES], training["is_at_risk"])

    engine = DriftMonitorEngine(psi_threshold=0.20, ks_alpha=0.01)
    engine.set_reference(reference)
    simulator = DriftSimulator(random_state=42)
    correlator = DriftPerformanceCorrelator()

    scenarios = [
        ("W1_Stable", "baseline_clean"),
        ("W2_Mild_Shift", "mild_covariate"),
        ("W3_Severe_Covariate", "severe_covariate"),
        ("W4_Concept_Drift", "concept_drift"),
        ("W5_Pipeline_Corruption", "null_spike"),
        ("W6_Critical_MultiDrift", "multi_drift"),
    ]
    windows = []
    for index, (window_id, scenario) in enumerate(scenarios):
        batch = make_synthetic_batch(1000, seed=100 + index)
        y_true = batch.pop("is_at_risk").to_numpy()
        label_uniform = batch.pop("_label_uniform").to_numpy()

        if scenario == "mild_covariate":
            batch = simulator.inject_covariate_shift(batch, ["avg_delivery_delay"], 0.5)
        elif scenario == "severe_covariate":
            batch = simulator.inject_covariate_shift(
                batch, ["avg_carrier_handoff"], 1.5
            )
        elif scenario == "concept_drift":
            # Deliberately reverse target association for the high-delay cohort.
            mask = batch["avg_delivery_delay"] > batch["avg_delivery_delay"].median()
            y_true[mask.to_numpy()] = 1 - y_true[mask.to_numpy()]
        elif scenario == "null_spike":
            batch = simulator.inject_pipeline_corruption(batch, ["freight_ratio"], 0.30)
        elif scenario == "multi_drift":
            batch = simulator.inject_covariate_shift(
                batch, ["avg_delivery_delay", "avg_carrier_handoff", "total_gmv"], 1.5
            )

        if scenario in {"mild_covariate", "severe_covariate", "multi_drift"}:
            logit = (
                0.7 * batch.avg_delivery_delay
                + 0.6 * batch.avg_carrier_handoff
                - 0.45 * batch.freight_ratio
                + 0.2 * batch.total_gmv
                - 0.2
            )
            probability = 1.0 / (1.0 + np.exp(-logit))
            y_true = (label_uniform < probability).astype(int)

        drift_report = engine.evaluate_batch(batch)
        y_prob = model.predict_proba(batch[FEATURES])[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        max_psi = max(
            (r["psi_score"] or 0.0 for r in drift_report["reports"]), default=0.0
        )
        metrics = {
            "window_id": window_id,
            "scenario": scenario,
            "drift_status": drift_report["overall_status"],
            "drifted_features_count": drift_report["drifted_features_count"],
            "max_psi": round(float(max_psi), 4),
            "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
            "missing_fraction": {
                col: round(float(batch[col].isna().mean()), 4) for col in FEATURES
            },
        }
        corr_report = {
            "overall_status": drift_report["overall_status"],
            "drifted_features_count": drift_report["drifted_features_count"],
            "reports": drift_report["reports"],
        }
        correlator.record_window(window_id, corr_report, y_true, y_prob)
        windows.append(metrics)

    report = {
        "evaluation_type": "controlled synthetic drift sensitivity and model-metric demonstration",
        "seed": 42,
        "training_rows": len(training),
        "validation_rows_per_window": 1000,
        "model": "median imputation + logistic regression fitted on synthetic baseline data",
        "benchmark_summary": correlator.compute_correlation_attribution(),
        "limitations": [
            "Synthetic data and scenarios do not establish production monitoring performance.",
            "Six designed windows are descriptive; correlations have no significance test.",
            "This experiment makes no causal or lead-time claim.",
            "Feature drift can occur without performance degradation, and concept drift may occur without feature drift.",
        ],
        "simulated_windows": windows,
    }
    out = root / "drift_benchmark_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved report to {out}")
    return report


if __name__ == "__main__":
    run_experiments()
