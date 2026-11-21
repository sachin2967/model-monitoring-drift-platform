# Model Drift Monitoring Demonstration

A Python demonstration of tabular feature monitoring using PSI, two-sample KS tests, Wasserstein distance, and a categorical chi-square test. It includes a FastAPI batch endpoint, a Streamlit cockpit, and a reproducible synthetic experiment using a fitted logistic-regression classifier.

This is a **portfolio prototype**, not a production monitoring service. The experiment uses generated data and six deliberately designed scenarios. Its correlations are descriptive only; it does not establish statistical significance, causality, or lead time. Feature drift is not equivalent to model failure, and performance monitoring requires reliable delayed labels and a separately specified evaluation design.

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

- PSI and distance metrics measure distribution differences, not downstream business harm. Thresholds are operational heuristics and need calibration for the data and sample sizes in use.
- KS and chi-square p-values depend on assumptions and sample size. They are included as diagnostics, not automatic proof of material drift.
- The six windows are designed cases, not independent random repetitions. Their Pearson correlation is exploratory; no inferential test is performed.
- The controlled time windows have no timestamps or repeated trajectories, so the experiment cannot support “lead indicator,” “predicts failure,” or “degradation by days” claims.
- The sample model, feature distributions, labels, and injected shifts are synthetic; no production seller or customer dataset is bundled or required.
- The API/UI are demonstration surfaces. Authentication, access control, persistence, alert delivery, monitoring of the monitor, and load testing are outside scope.

## Methods

For numeric features, PSI uses bins derived from the reference distribution; KS and Wasserstein compare non-missing samples. Missingness is reported separately and a high current missing fraction is flagged. Categorical features use a chi-square contingency test. The overall status aggregates feature flags and should be interpreted as a configurable heuristic, not a calibrated risk probability.
