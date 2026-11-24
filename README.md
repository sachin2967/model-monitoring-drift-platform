# Model Drift Monitoring Demonstration

A Python demonstration of tabular feature monitoring using PSI, two-sample KS tests, Wasserstein distance, and a categorical chi-square test. It includes a FastAPI batch endpoint, a Streamlit cockpit, and a reproducible synthetic experiment using a fitted logistic-regression classifier.

This is a **portfolio prototype**, not a production monitoring service. The experiment uses generated data and six deliberately designed scenarios. Its correlations are descriptive only; it does not establish statistical significance, causality, or lead time. Feature drift is not equivalent to model failure, and performance monitoring requires reliable delayed labels and a separately specified evaluation design.

The original API uses a synthetic reference so the demo starts without private data. `/api/monitor/batch` accepts 25–5,000 records with exactly the four monitored numeric features. Malformed, incomplete or oversized batches are rejected. Compose binds API/dashboard to loopback ports 8030/8530. The separate fraud-pilot interface below adds reference identity, operator authentication and persistent reports. Automated alert delivery and model promotion remain unimplemented.

## Fraud-pilot interface

`POST /api/monitor/fraud-window` consumes `fraud-events-v1` exports. It requires
`FRAUD_REFERENCE_PATH`, exported from the matching fraud model's training partition;
there is no synthetic fallback. Duplicate events, incompatible model/feature/policy
versions and invalid label timestamps are rejected. Distribution diagnostics and
available/mature-label metrics are separate; coverage and minimum sample counts are
explicit. Single-class cohorts do not publish ROC-AUC/AP. Reports persist and can be
retrieved with `GET /api/reports/{report_id}`.

Production mode requires a 32+ character `DRIFT_API_KEY`, persistent
`DRIFT_STATE_PATH`, reference path and trusted `FRAUD_REFERENCE_SHA256`. Non-health
routes require `X-API-Key` when configured; bodies are capped at 4 MiB. `/live` checks
the process; `/ready` also checks history and reference configuration. This is an
operator-only pilot, not multi-tenant or highly available hosting. The standalone [operations guide](docs/OPERATIONS.md) describes the service; the
[portfolio runbook](https://github.com/sachin2967/ml-portfolio/blob/main/OPERATIONS_RUNBOOK.md) describes cross-service integration. The default Streamlit page remains synthetic;
the separate **Fraud Pilot** page retrieves persisted reports by ID using server-side
`DRIFT_API_URL` / `DRIFT_API_KEY`. It shows label coverage and insufficient/single-class
states, clears stale results on lookup errors, and never exposes the operator key.

## Experiment

```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
python scripts/run_drift_experiments.py
pytest -q
uvicorn api.app:app --reload
streamlit run streamlit_app/app.py
```

The benchmark trains a median-imputation + logistic-regression classifier on seeded synthetic baseline data and measures each independently generated 1,000-row window. It reports observed drift metrics, F1, ROC-AUC, and missing fractions. The concept-drift scenario reverses labels for one cohort; covariate scenarios shift features and regenerate outcomes from the same synthetic rule. Results are intended to exercise the code paths, not estimate real-world detector sensitivity.

## Interpretation and limits

### Repeated sensitivity and false-alert study

`python scripts/evaluate_repeated_drift.py` runs 100 independently seeded windows
per scenario at each of two sample sizes, for 1,000 evaluations. The
[report](repeated_drift_report.json) separates any-feature flags from aggregate
window alerts and includes 95% Wilson intervals. Conditional on one fixed
synthetic reference, stable-window alerts occurred in 4/100 windows of 250 rows
and 0/100 windows of 1,000 rows. Both designed mean shifts and 30% missingness
were flagged in all repetitions. Reversing labels leaves features unchanged:
the alert rate stays at the stable rate while mean model AUC falls to about
0.25. These are synthetic experiments with fixed thresholds, not estimates of
production sensitivity, a calibrated false-positive guarantee or a retraining rule.

- PSI and distance metrics measure distribution differences, not downstream business harm. Thresholds are operational heuristics and need calibration for the data and sample sizes in use.
- KS and chi-square p-values depend on assumptions and sample size. They are included as diagnostics, not automatic proof of material drift.
- The six windows are designed cases, not independent random repetitions. Their Pearson correlation is exploratory; no inferential test is performed.
- The controlled time windows have no timestamps or repeated trajectories, so the experiment cannot support “lead indicator,” “predicts failure,” or “degradation by days” claims.
- The sample model, feature distributions, labels, and injected shifts are synthetic; no production seller or customer dataset is bundled or required.
- The synthetic UI remains a demonstration. The fraud API adds operator authentication and persistent reports; tenant authorization, automated alert delivery, monitoring of the monitor and cloud availability/load evidence remain open.
- The default API reference is synthetic. Real deployment requires a validated, versioned training/reference dataset, threshold governance, authenticated access, label-aware performance monitoring, data retention controls, and operational runbooks.

## Methods

For numeric features, PSI uses bins derived from the reference distribution; KS and Wasserstein compare non-missing samples. Missingness is reported separately and a high current missing fraction is flagged. Categorical features use a chi-square contingency test. The overall status aggregates feature flags and should be interpreted as a configurable heuristic, not a calibrated risk probability.

## Release verification and operations

The latest 22 November 2026 local verification passed **34 tests** and repository lint.
The API image passed non-root, network-isolated reference, report persistence and serving checks.

See the standalone [operations guide](docs/OPERATIONS.md) for configuration,
failure handling, recovery and rollout limits. Cross-project reports and replay tools
live in the separate [portfolio guide](https://github.com/sachin2967/ml-portfolio).
Hosted CI and cloud deployment are separate checks; a local pass does not imply either.
