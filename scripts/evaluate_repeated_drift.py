"""Repeated null/shift experiments; quantify sample-size-dependent false alerts."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.drift.detectors import DriftMonitorEngine
from src.drift.synthetic import FEATURES, make_synthetic_batch
from src.drift.validation import wilson_interval


def run(repetitions: int = 100) -> dict:
    if repetitions < 2 or repetitions > 1000:
        raise ValueError("Use between 2 and 1000 independent repetitions.")
    training = make_synthetic_batch(4000, seed=42)
    model = make_pipeline(
        SimpleImputer(strategy="median"), LogisticRegression(max_iter=500)
    )
    model.fit(training[FEATURES], training.is_at_risk)
    engine = DriftMonitorEngine()
    engine.set_reference(training[FEATURES])
    output = []
    for size in (250, 1000):
        observations = {
            name: []
            for name in (
                "stable_null",
                "mean_shift_0.5sd",
                "mean_shift_1.5sd",
                "missing_30pct",
                "label_reversal",
            )
        }
        for replicate in range(repetitions):
            frame = make_synthetic_batch(size, seed=10000 + replicate + size)
            original, labels = frame[FEATURES], frame.is_at_risk.to_numpy()
            for name, scenario_observations in observations.items():
                batch, y = original.copy(), labels.copy()
                if name.startswith("mean_shift"):
                    batch[FEATURES[0]] += 0.5 if name == "mean_shift_0.5sd" else 1.5
                    # Preserve the same conditional outcome rule under covariate shift.
                    logit = (
                        0.7 * batch[FEATURES[0]]
                        + 0.6 * batch[FEATURES[1]]
                        - 0.45 * batch[FEATURES[2]]
                        + 0.2 * batch[FEATURES[3]]
                        - 0.2
                    )
                    y = (
                        frame._label_uniform.to_numpy() < 1 / (1 + np.exp(-logit))
                    ).astype(int)
                elif name == "missing_30pct":
                    positions = np.random.default_rng(20000 + replicate).choice(
                        size, int(size * 0.3), replace=False
                    )
                    batch.iloc[positions, 0] = np.nan
                elif name == "label_reversal":
                    y = 1 - y  # Exact same feature window as stable_null.
                diagnostics = engine.evaluate_batch(batch)
                probability = model.predict_proba(batch)[:, 1]
                scenario_observations.append(
                    {
                        "any_feature_flag": int(
                            diagnostics["drifted_features_count"] > 0
                        ),
                        "window_alert": int(diagnostics["overall_status"] != "HEALTHY"),
                        "roc_auc": float(roc_auc_score(y, probability))
                        if len(np.unique(y)) == 2
                        else None,
                    }
                )
        for name, rows in observations.items():
            feature_flags, alerts = (
                sum(r["any_feature_flag"] for r in rows),
                sum(r["window_alert"] for r in rows),
            )
            auc = [r["roc_auc"] for r in rows if r["roc_auc"] is not None]
            output.append(
                {
                    "scenario": name,
                    "window_rows": size,
                    "repetitions": repetitions,
                    "any_feature_flag_rate": feature_flags / repetitions,
                    "any_feature_flag_95_wilson": wilson_interval(
                        feature_flags, repetitions
                    ),
                    "window_alert_rate": alerts / repetitions,
                    "window_alert_95_wilson": wilson_interval(alerts, repetitions),
                    "mean_model_roc_auc": float(np.mean(auc)) if auc else None,
                }
            )
    return {
        "protocol": "repeated-drift-v1",
        "reference_seed": 42,
        "reference_rows": 4000,
        "fixed_thresholds": {"psi_red": 0.2, "psi_yellow": 0.1, "ks_alpha": 0.01},
        "results": output,
        "limitations": [
            "Synthetic sensitivity/null study, conditional on one frozen reference; not production false-positive rates.",
            "No threshold tuning on these outcomes. Feature tests are not multiplicity-corrected.",
            "Wilson intervals assume independent repetitions within a scenario; scenarios are deliberately paired.",
            "Feature-only diagnostics cannot detect label reversal when features do not change.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=100)
    args = parser.parse_args()
    report = run(args.repetitions)
    path = ROOT / "repeated_drift_report.json"
    path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
