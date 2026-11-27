# Limitations and responsible use

## Interpretation

- PSI and distance metrics measure distribution differences, not business harm.
  Thresholds are heuristics and need calibration for the data and sample sizes in use.
- KS and chi-square p-values depend on assumptions and sample size; they are diagnostics,
  not proof of material drift.
- Feature drift can occur without performance loss (covariate scenarios in the
  demonstration), and concept drift can occur without feature drift (W4 raised no PSI alert).

## Demonstration experiment

- Data, model, labels and injected shifts are synthetic.
- The six windows are designed cases, not independent repetitions. Pearson correlations
  are exploratory; no inferential test is performed.
- Windows have no timestamps or repeated trajectories, so the experiment cannot support
  "lead indicator", "predicts failure" or "degradation by days" claims.

## Fraud pilot

- Label coverage does not prove representativeness; selective labels can bias metrics.
- Cost figures use an illustrative 20:1 relative weighting, not measured costs.
- Do not automatically retrain or promote a model based on these flags.

## Not implemented

Tenant authorization, automated alert delivery, model promotion, monitoring of the
monitor, retention policy, and cloud load or availability evidence. Real use needs a
validated versioned reference, threshold governance, authenticated access and runbooks.
