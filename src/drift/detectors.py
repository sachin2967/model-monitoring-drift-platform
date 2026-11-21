"""Statistical drift detection algorithms for production machine learning observability.

Implements Population Stability Index (PSI), Two-Sample Kolmogorov-Smirnov test,
Wasserstein distance, and Chi-Square tests for continuous and categorical features.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp, wasserstein_distance


@dataclass
class FeatureDriftReport:
    """Detailed drift status for a single monitored feature."""
    feature_name: str
    feature_type: str  # 'numerical' or 'categorical'
    psi_score: Optional[float]
    ks_statistic: Optional[float]
    ks_p_value: Optional[float]
    wasserstein_dist: Optional[float]
    chi2_stat: Optional[float]
    chi2_p_value: Optional[float]
    drift_detected: bool
    severity: str  # 'GREEN', 'YELLOW', 'RED'
    message: str


def calculate_psi(
    reference: np.ndarray, current: np.ndarray, num_bins: int = 10, epsilon: float = 1e-4
) -> float:
    """Calculates the Population Stability Index (PSI) between reference and current samples.

    Standard Interpretation:
      - PSI < 0.10: No significant shift (stable)
      - 0.10 <= PSI < 0.25: Moderate shift (monitor closely)
      - PSI >= 0.25: Significant distributional shift (action required)
    """
    ref_clean = reference[~np.isnan(reference)]
    curr_clean = current[~np.isnan(current)]

    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return 0.0

    # Determine quantile bins based on reference distribution
    quantiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(ref_clean, quantiles)
    bins[0] = -np.inf
    bins[-1] = np.inf
    # Avoid duplicate bin edges
    bins = np.unique(bins)

    if len(bins) < 2:
        return 0.0

    ref_counts = np.histogram(ref_clean, bins=bins)[0]
    curr_counts = np.histogram(curr_clean, bins=bins)[0]

    # Convert to proportions with smoothing epsilon to prevent division by zero
    ref_pct = np.clip(ref_counts / len(ref_clean), epsilon, 1.0)
    curr_pct = np.clip(curr_counts / len(curr_clean), epsilon, 1.0)

    # Re-normalize
    ref_pct = ref_pct / np.sum(ref_pct)
    curr_pct = curr_pct / np.sum(curr_pct)

    # PSI = sum((Actual% - Expected%) * ln(Actual% / Expected%))
    psi_value = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return float(np.round(psi_value, 4))


class DriftMonitorEngine:
    """Manages baseline reference distribution and scores incoming production windows."""

    def __init__(self, psi_threshold: float = 0.20, ks_alpha: float = 0.01):
        self.psi_threshold = psi_threshold
        self.ks_alpha = ks_alpha
        self.reference_df: Optional[pd.DataFrame] = None
        self.numerical_cols: List[str] = []
        self.categorical_cols: List[str] = []

    def set_reference(self, df: pd.DataFrame):
        """Freezes reference distribution (e.g. training set)."""
        self.reference_df = df.copy()
        # Exclude IDs and target outcomes
        ignore_cols = {"seller_id", "order_id", "customer_id", "product_id", "review_id", "is_at_risk"}
        self.numerical_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ignore_cols]
        self.categorical_cols = [c for c in df.select_dtypes(exclude=[np.number]).columns if c not in ignore_cols]

    def evaluate_batch(self, current_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluates an incoming live batch against the frozen reference distribution."""
        if self.reference_df is None:
            raise ValueError("Reference distribution must be set before evaluating batches.")

        feature_reports: List[FeatureDriftReport] = []
        drift_count = 0

        # 1. Numerical Features
        for col in self.numerical_cols:
            if col not in current_df.columns:
                continue

            ref_vals = self.reference_df[col].dropna().values
            curr_vals = current_df[col].dropna().values

            if len(curr_vals) == 0:
                continue

            psi = calculate_psi(ref_vals, curr_vals)
            ks_res = ks_2samp(ref_vals, curr_vals)
            wass = float(wasserstein_distance(ref_vals, curr_vals))

            # Determine severity based on calibrated PSI and KS test
            if psi >= self.psi_threshold:
                severity = "RED"
                drift_detected = True
                drift_count += 1
                msg = f"Significant numerical drift detected (PSI={psi:.3f}, KS p-val={ks_res.pvalue:.4e})"
            elif psi >= 0.10 or (ks_res.pvalue < self.ks_alpha and wass > (0.2 * np.std(ref_vals))):
                severity = "YELLOW"
                drift_detected = True
                drift_count += 1
                msg = f"Moderate numerical drift detected (PSI={psi:.3f}, KS p-val={ks_res.pvalue:.4e})"
            else:
                severity = "GREEN"
                drift_detected = False
                msg = f"Distribution is stable (PSI={psi:.3f})"

            feature_reports.append(FeatureDriftReport(
                feature_name=col,
                feature_type="numerical",
                psi_score=psi,
                ks_statistic=float(np.round(ks_res.statistic, 4)),
                ks_p_value=float(ks_res.pvalue),
                wasserstein_dist=float(np.round(wass, 4)),
                chi2_stat=None,
                chi2_p_value=None,
                drift_detected=drift_detected,
                severity=severity,
                message=msg,
            ))

        # 2. Categorical Features
        for col in self.categorical_cols:
            if col not in current_df.columns:
                continue

            ref_counts = self.reference_df[col].value_counts()
            curr_counts = current_df[col].value_counts()

            # Align categories
            categories = list(set(ref_counts.index).union(set(curr_counts.index)))
            r_vec = [ref_counts.get(cat, 0) + 1 for cat in categories]
            c_vec = [curr_counts.get(cat, 0) + 1 for cat in categories]

            chi2, p_val, _, _ = chi2_contingency([r_vec, c_vec])

            if p_val < self.ks_alpha:
                severity = "RED"
                drift_detected = True
                drift_count += 1
                msg = f"Significant categorical shift detected (Chi2={chi2:.2f}, p={p_val:.4e})"
            else:
                severity = "GREEN"
                drift_detected = False
                msg = f"Categorical distribution stable (p={p_val:.4f})"

            feature_reports.append(FeatureDriftReport(
                feature_name=col,
                feature_type="categorical",
                psi_score=None,
                ks_statistic=None,
                ks_p_value=None,
                wasserstein_dist=None,
                chi2_stat=float(np.round(chi2, 4)),
                chi2_p_value=float(p_val),
                drift_detected=drift_detected,
                severity=severity,
                message=msg,
            ))

        total_features = len(feature_reports)
        drift_percentage = round((drift_count / max(total_features, 1)) * 100, 1)

        overall_status = "HEALTHY"
        if drift_percentage > 35.0:
            overall_status = "CRITICAL_DRIFT"
        elif drift_percentage > 15.0:
            overall_status = "WARNING"

        return {
            "overall_status": overall_status,
            "total_monitored_features": total_features,
            "drifted_features_count": drift_count,
            "drift_percentage": drift_percentage,
            "reports": [r.__dict__ for r in feature_reports],
        }
