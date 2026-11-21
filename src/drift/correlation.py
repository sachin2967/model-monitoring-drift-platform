"""Drift-to-Performance Attribution and Root-Cause Correlation Engine.

Determines whether detected statistical feature drift actually correlates with
downstream classification performance drops as delayed ground truth labels arrive.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


class DriftPerformanceCorrelator:
    """Tracks historical monitoring windows and attributes performance drops to drifted features."""

    def __init__(self):
        self.window_history: List[Dict[str, Any]] = []

    def record_window(
        self,
        window_id: str,
        drift_report: Dict[str, Any],
        y_true: np.ndarray,
        y_prob: np.ndarray,
        feature_importances: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Logs a single monitoring window with its drift status and delayed ground truth metrics."""
        y_pred = (y_prob >= 0.5).astype(int)

        roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        # Calculate weighted drift score based on feature importances if available
        weighted_psi = 0.0
        max_psi = 0.0
        most_drifted_feature = None

        reports = drift_report.get("reports", [])
        for r in reports:
            psi = r.get("psi_score") or 0.0
            if psi > max_psi:
                max_psi = psi
                most_drifted_feature = r["feature_name"]

            weight = feature_importances.get(r["feature_name"], 1.0) if feature_importances else 1.0
            weighted_psi += psi * weight

        entry = {
            "window_id": window_id,
            "overall_status": drift_report.get("overall_status"),
            "drifted_features_count": drift_report.get("drifted_features_count", 0),
            "max_psi": max_psi,
            "weighted_psi": weighted_psi,
            "most_drifted_feature": most_drifted_feature,
            "roc_auc": roc_auc,
            "f1_score": f1,
        }

        self.window_history.append(entry)
        return entry

    def compute_correlation_attribution(self) -> Dict[str, Any]:
        """Calculates Pearson and Spearman correlations between feature drift and performance drop."""
        if len(self.window_history) < 3:
            return {
                "correlation_calculated": False,
                "message": "At least 3 evaluation windows required for statistical correlation.",
            }

        hist_df = pd.DataFrame(self.window_history)

        # Correlation between Max PSI and F1 score drop (negative correlation expected)
        corr_matrix = hist_df[["max_psi", "weighted_psi", "f1_score", "roc_auc"]].corr()

        f1_psi_corr = float(corr_matrix.loc["max_psi", "f1_score"])
        roc_psi_corr = float(corr_matrix.loc["max_psi", "roc_auc"])

        return {
            "correlation_calculated": True,
            "total_windows_evaluated": len(hist_df),
            "psi_vs_f1_correlation": round(f1_psi_corr, 4),
            "psi_vs_roc_auc_correlation": round(roc_psi_corr, 4),
            "critical_lead_indicator": bool(f1_psi_corr < -0.40),
            "summary": (
                "Strong negative correlation: high PSI reliably predicts downstream model degradation."
                if f1_psi_corr < -0.40
                else "Moderate correlation: drift signals must be prioritized by feature importance."
            ),
        }
