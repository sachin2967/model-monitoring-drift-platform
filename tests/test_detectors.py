"""Unit tests for statistical drift detection algorithms."""

import numpy as np
import pandas as pd
import pytest

from src.drift.detectors import DriftMonitorEngine, calculate_psi


def test_psi_identical_distributions():
    np.random.seed(42)
    sample_a = np.random.normal(0, 1, 1000)
    sample_b = np.random.normal(0, 1, 1000)

    psi = calculate_psi(sample_a, sample_b)
    assert psi < 0.05  # Should be near zero for identical distributions


def test_psi_shifted_distributions():
    np.random.seed(42)
    sample_a = np.random.normal(0, 1, 1000)
    sample_b = np.random.normal(3.0, 1, 1000)  # Major 3-std shift

    psi = calculate_psi(sample_a, sample_b)
    assert psi > 0.50  # Must trigger significant drift


def test_drift_monitor_engine():
    np.random.seed(42)
    ref_df = pd.DataFrame(
        {
            "num_feature": np.random.normal(10, 2, 500),
            "cat_feature": np.random.choice(["A", "B", "C"], 500),
        }
    )

    curr_df = pd.DataFrame(
        {
            "num_feature": np.random.normal(18, 2, 200),  # Heavily shifted
            "cat_feature": np.random.choice(["A", "B", "C"], 200),
        }
    )

    engine = DriftMonitorEngine(psi_threshold=0.20)
    engine.set_reference(ref_df)
    report = engine.evaluate_batch(curr_df)

    assert "overall_status" in report
    assert report["drifted_features_count"] >= 1
    num_report = next(
        r for r in report["reports"] if r["feature_name"] == "num_feature"
    )
    assert num_report["drift_detected"] is True
    assert num_report["severity"] == "RED"


def test_high_missingness_is_reported_and_flagged():
    ref = pd.DataFrame({"num_feature": np.arange(100, dtype=float)})
    current = pd.DataFrame(
        {"num_feature": np.r_[np.full(30, np.nan), np.arange(70, dtype=float)]}
    )
    engine = DriftMonitorEngine()
    engine.set_reference(ref)
    report = engine.evaluate_batch(current)
    feature = report["reports"][0]
    assert feature["missing_fraction"] == 0.30
    assert feature["drift_detected"] is True


def test_categorical_missingness_is_not_reported_as_stable():
    engine = DriftMonitorEngine()
    engine.set_reference(pd.DataFrame({"category": ["a", "b"] * 50}))
    report = engine.evaluate_batch(
        pd.DataFrame({"category": [None] * 30 + ["a", "b"] * 35})
    )
    feature = report["reports"][0]
    assert feature["missing_fraction"] == 0.3
    assert feature["severity"] == "RED"


def test_reference_and_window_completeness_are_enforced():
    engine = DriftMonitorEngine()
    with pytest.raises(ValueError):
        engine.set_reference(pd.DataFrame({"x": [np.nan] * 25}))
    engine.set_reference(pd.DataFrame({"x": np.arange(25)}))
    with pytest.raises(ValueError):
        engine.evaluate_batch(pd.DataFrame({"another_feature": np.arange(25)}))
