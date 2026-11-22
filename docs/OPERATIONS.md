# Monitoring pilot operations

Obtain `fraud_reference.json` from the matching fraud model's training-only reference
exporter. Record its SHA-256 and source/model identity. Set `APP_ENV=production`,
`DRIFT_API_KEY` to a random 32+ character key, `DRIFT_STATE_PATH` to a persistent
SQLite filename, `FRAUD_REFERENCE_PATH` to the reference file, and
`FRAUD_REFERENCE_SHA256` to the trusted digest. Start
`uvicorn api.app:app --host 127.0.0.1 --port 8030 --workers 1` behind TLS/access controls.

`/live` checks liveness; `/ready` checks writable report history and reference validity.
Non-health routes require `X-API-Key`. `POST /api/monitor/fraud-window` accepts the
fraud pilot's `fraud-events-v1` export with 25–5,000 unique events. Each event includes
prediction/transaction identity, service receipt time, model/feature version,
threshold, 29 features and score. Labels require source and availability timestamp.
The append cursor does not track label updates; refresh earlier records for delayed
labels and deduplicate each fixed evaluation cohort.

Optional `as_of` and `minimum_label_age_seconds` define cutoff and prediction age.
Only labels available by the cutoff contribute. Fewer than 25 mature labels produces
an insufficient-data result; single-class cohorts have no ROC-AUC/AP. Coverage does
not prove representativeness and selective labels can bias metrics. Costs are
illustrative 20:1 relative units. Feature/score shift is separate from model quality;
do not automatically promote or retrain a model based on these flags.

Every acknowledged report is persisted and retrieved through
`GET /api/reports/{report_id}`. Back up report storage independently using SQLite's
online backup API, or stop the service before copying a consistent database. Restore
to a new filename while stopped, change `DRIFT_STATE_PATH`, and retrieve a known report.
Retain the corresponding reference and hash with each model release/rollback.

The original `/api/monitor/batch` and default Streamlit page remain explicitly synthetic.
The separate Fraud Pilot page retrieves persisted reports by ID using server-side
`DRIFT_API_URL` and `DRIFT_API_KEY`. It shows coverage, insufficient/single-class
states and provenance; errors clear the previous report. Configuring a reference
does not automatically ingest traffic. Multi-tenant authorization, automated alert delivery, cloud load/availability
testing, retention policy and monitoring of this service remain launch gates.
