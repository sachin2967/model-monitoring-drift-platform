"""Unit tests for synthetic drift simulator."""

import numpy as np
import pandas as pd

from src.drift.simulator import DriftSimulator


def test_covariate_shift_injection():
    sim = DriftSimulator(random_state=42)
    df = pd.DataFrame({"feat": np.random.normal(5, 1, 100)})

    shifted = sim.inject_covariate_shift(df, ["feat"], mean_shift_std=2.0)
    assert shifted["feat"].mean() > df["feat"].mean() + 1.5


def test_pipeline_null_corruption():
    sim = DriftSimulator(random_state=42)
    df = pd.DataFrame({"feat": np.random.normal(5, 1, 100)})

    corrupted = sim.inject_pipeline_corruption(df, ["feat"], null_spike_rate=0.40)
    assert corrupted["feat"].isnull().sum() > 25
