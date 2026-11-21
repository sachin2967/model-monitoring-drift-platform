"""Unit tests for drift-to-performance attribution correlator."""

import numpy as np

from src.drift.correlation import DriftPerformanceCorrelator


def test_correlation_calculator():
    correlator = DriftPerformanceCorrelator()

    # Simulate 4 windows with degrading F1 as PSI increases
    for psi_val, f1_val in [(0.02, 0.95), (0.12, 0.88), (0.45, 0.72), (0.90, 0.55)]:
        mock_report = {
            "overall_status": "OK",
            "reports": [{"feature_name": "f1", "psi_score": psi_val}],
        }
        y_true = np.array([1, 0, 1, 0, 1])
        y_prob = (
            np.array([0.9, 0.1, 0.8, 0.2, 0.7])
            if f1_val > 0.8
            else np.array([0.4, 0.6, 0.3, 0.7, 0.2])
        )
        correlator.record_window(f"W_{psi_val}", mock_report, y_true, y_prob)

    res = correlator.compute_correlation_attribution()
    assert res["correlation_calculated"] is True
    assert res["psi_vs_f1_correlation"] < 0  # Must be negatively correlated
    assert "no p-value" in res["inference"]
    assert "critical_lead_indicator" not in res
