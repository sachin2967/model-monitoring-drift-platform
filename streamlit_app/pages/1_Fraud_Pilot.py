"""Read persisted fraud-pilot reports without exposing operator keys to the browser."""

import json
import os
import re
import urllib.error
import urllib.request

import streamlit as st

st.set_page_config(page_title="Fraud monitoring pilot", page_icon="📋", layout="wide")
st.title("Fraud monitoring pilot reports")
st.caption(
    "Historical replay diagnostics. These reports do not validate real payment decisions."
)
base = os.getenv("DRIFT_API_URL", "http://127.0.0.1:8030").rstrip("/")

with st.form("report_lookup"):
    report_id = st.text_input("Persisted report ID", max_chars=32)
    submitted = st.form_submit_button("Load report")

if submitted:
    # Clear old results before lookup so errors cannot leave a stale cohort visible.
    st.session_state.pop("fraud_report", None)
    if not re.fullmatch(r"[a-f0-9]{32}", report_id):
        st.error("Enter the 32-character report ID returned by the monitoring API.")
    else:
        try:
            request = urllib.request.Request(
                f"{base}/api/reports/{report_id}",
                headers={"X-API-Key": os.getenv("DRIFT_API_KEY", "")},
            )
            # Do not send the operator key through ambient HTTP proxy settings.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=10) as response:
                st.session_state["fraud_report"] = json.load(response)
        except (OSError, urllib.error.URLError, ValueError):
            st.error(
                "Report unavailable. Check the report ID, API health and server-side credentials."
            )

report = st.session_state.get("fraud_report")
if report:
    performance = report["label_performance"]
    first, second, third = st.columns(3)
    first.metric("Prediction rows", report["rows"])
    second.metric("Mature labeled rows", performance["labeled_rows"])
    third.metric("Label coverage", f"{performance['coverage']:.1%}")
    st.write("Evaluation cutoff:", report["as_of"])
    st.caption("Reference source: " + report["reference_source"])
    if performance["status"] == "evaluated":
        if performance["roc_auc"] is None:
            st.warning(
                "Only one label class is present; ROC-AUC and average precision are undefined."
            )
        else:
            left, right = st.columns(2)
            left.metric("Labeled-cohort ROC-AUC", f"{performance['roc_auc']:.3f}")
            right.metric(
                "Labeled-cohort average precision",
                f"{performance['average_precision']:.3f}",
            )
        st.json(performance["confusion_matrix"])
    else:
        st.warning(
            "Insufficient mature labels; no model performance estimate is available."
        )
    st.subheader("Feature and score distribution diagnostics")
    st.dataframe(report["distribution_diagnostics"]["reports"], hide_index=True)
    for limitation in report["limitations"]:
        st.info(limitation)
    with st.expander("Report provenance and full result"):
        st.json(report)
