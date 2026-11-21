"""Controlled synthetic drift injection simulator for benchmarking MLOps detectors."""

import numpy as np
import pandas as pd


class DriftSimulator:
    """Injects controlled synthetic drift into tabular test batches."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def inject_covariate_shift(
        self,
        df: pd.DataFrame,
        target_cols: list[str],
        mean_shift_std: float = 1.5,
        variance_scale: float = 1.2,
    ) -> pd.DataFrame:
        """Injects covariate shift by perturbing mean and variance of specific numerical features."""
        shifted_df = df.copy()
        for col in target_cols:
            if col in shifted_df.columns and pd.api.types.is_numeric_dtype(
                shifted_df[col]
            ):
                col_std = shifted_df[col].std()
                col_mean = shifted_df[col].mean()
                shift_amount = mean_shift_std * col_std
                shifted_df[col] = (
                    (shifted_df[col] - col_mean) * variance_scale
                    + col_mean
                    + shift_amount
                )
        return shifted_df

    def inject_concept_drift(
        self,
        df: pd.DataFrame,
        feature_col: str,
        target_col: str,
        flip_ratio: float = 0.35,
    ) -> pd.DataFrame:
        """Simulates concept drift by inverting target outcomes for high-value feature cohorts."""
        corrupted_df = df.copy()
        if feature_col in corrupted_df.columns and target_col in corrupted_df.columns:
            threshold = corrupted_df[feature_col].median()
            subset_mask = corrupted_df[feature_col] > threshold
            # Flip labels in this cohort
            flip_indices = (
                corrupted_df[subset_mask]
                .sample(frac=flip_ratio, random_state=self.random_state)
                .index
            )
            corrupted_df.loc[flip_indices, target_col] = (
                1 - corrupted_df.loc[flip_indices, target_col]
            )
        return corrupted_df

    def inject_pipeline_corruption(
        self, df: pd.DataFrame, target_cols: list[str], null_spike_rate: float = 0.25
    ) -> pd.DataFrame:
        """Simulates upstream telemetry breakage by injecting sudden missing value bursts."""
        corrupted_df = df.copy()
        for col in target_cols:
            if col in corrupted_df.columns:
                null_indices = corrupted_df.sample(
                    frac=null_spike_rate, random_state=self.random_state
                ).index
                corrupted_df.loc[null_indices, col] = np.nan
        return corrupted_df
