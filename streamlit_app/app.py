"""Interactive Streamlit dashboard for the synthetic drift demo."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.drift.synthetic import FEATURES, make_synthetic_batch

st.set_page_config(
    page_title="Model Monitoring & Drift Detection Platform",
    page_icon="📡",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = PROJECT_ROOT / "drift_benchmark_report.json"
st.title("📡 Model Drift Monitoring Demonstration")
st.markdown(
    "**Synthetic demo:** PSI, KS, Wasserstein, and categorical drift diagnostics against a seeded reference sample. "
    "This dashboard does not monitor live production traffic."
)

st.divider()

if not BENCHMARK_PATH.exists():
    st.warning(
        "Benchmark report not found. Run `scripts/run_drift_experiments.py` first."
    )
    st.stop()

with open(BENCHMARK_PATH) as f:
    report_data = json.load(f)

summary = report_data.get("benchmark_summary", {})
windows = report_data.get("simulated_windows", [])

# --- TOP METRIC CARDS ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Monitoring Protocol", "PSI + KS + Wasserstein")
c2.metric(
    "Descriptive PSI/F1 correlation", str(summary.get("psi_vs_f1_correlation", "N/A"))
)
c3.metric("Inference", "None")
c4.metric("Designed scenarios", f"{len(windows)}")

st.divider()

tab1, tab2, tab3 = st.tabs(
    [
        "📊 Time-Series Window Observability",
        "🔬 Feature Distribution Shift Inspector",
        "📑 Drift Theory & Defensibility",
    ]
)

with tab1:
    st.subheader("Controlled synthetic scenarios")

    win_df = pd.DataFrame(windows)
    st.dataframe(
        win_df.rename(
            columns={
                "window_id": "Window ID",
                "scenario": "Scenario",
                "drift_status": "Status",
                "drifted_features_count": "Drifted Count",
                "max_psi": "Max PSI",
                "f1_score": "Downstream F1",
                "roc_auc": "ROC-AUC",
            }
        ),
        use_container_width=True,
    )

    # Plot PSI vs F1
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()

    x = range(len(win_df))
    ax1.plot(
        x, win_df["max_psi"], "r-o", label="Max PSI (Drift Intensity)", linewidth=2
    )
    ax2.plot(x, win_df["f1_score"], "b--s", label="Downstream F1 Score", linewidth=2)

    ax1.set_xlabel("Designed scenarios")
    ax1.set_ylabel("Max Population Stability Index (PSI)", color="r")
    ax2.set_ylabel("Classification F1-Score", color="b")
    ax1.set_xticks(x)
    ax1.set_xticklabels(win_df["window_id"], rotation=20)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_title("Observed drift and model metrics (descriptive only)")
    st.pyplot(fig)

with tab2:
    st.subheader("Interactive Distribution Comparison")

    raw = make_synthetic_batch(1000, seed=42)
    selected_feature = st.selectbox(
        "Select synthetic numerical feature:", options=FEATURES
    )
    ref_vals = raw[selected_feature].dropna().values
    curr_vals = ref_vals * 1.35 + 2.0
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(
        ref_vals,
        bins=25,
        alpha=0.5,
        label="Synthetic reference",
        color="green",
        density=True,
    )
    ax.hist(
        curr_vals,
        bins=25,
        alpha=0.5,
        label="Illustrative shifted sample",
        color="red",
        density=True,
    )
    ax.set_title(f"Synthetic distribution comparison: {selected_feature}")
    ax.set_ylabel("Probability Density")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)

with tab3:
    st.subheader("Statistical methodology and limitations")
    st.markdown("""
    ### Why Not Rely on Simple p-Values Alone?
    - **The Large Sample Fallacy:** In big data environments ($N > 10,000$), standard hypothesis tests (like KS-test or Chi-Square) return tiny p-values ($p < 0.0001$) even for trivial, practically insignificant variations.
    - **Why Population Stability Index (PSI)?** PSI measures the divergence between two binned probability distributions using symmetrical Kullback-Leibler divergence:
    """)
    st.latex(
        r"PSI = \sum_{i=1}^k (Actual_i - Expected_i) \times \ln\left(\frac{Actual_i}{Expected_i}\right)"
    )
    st.markdown("""
    - PSI bands are heuristics that require calibration for the feature, sample size, and operational cost of alerts.
    - This six-scenario synthetic exercise has no timeline or repeated trajectories and makes no significance, causal, or lead-time claim.
    """)
