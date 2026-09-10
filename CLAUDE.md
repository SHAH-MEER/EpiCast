# Epicast — Real-Time ILI Forecasting API

## North Star
A deployed, monitored, forecasting API that predicts influenza-like-illness (ILI) rates N weeks ahead, with the full lifecycle (train → track → serve → ship → monitor) actually working end to end, not just described in a README.

## Locked Decisions
- **Data source:** [UKHSA weekly flu/ILI surveillance data | CDC FluView] — pick one as primary, note the other as a future extension.
- **Model strategy:** Build a comparison, not a single model.
  - Prophet as a fast seasonal baseline
  - LightGBM with lag/rolling/seasonal features as the upgrade
  - Both logged to the same MLflow experiment for a genuine "iterated and improved" story

## Definition of Done
- `/predict` endpoint returns a forecast + confidence interval for the next N weeks
- Every model run is versioned in MLflow with MAE/RMSE/MAPE logged
- CI pipeline tests and builds on every push; deploys on merge to main
- A drift report (Evidently) flags when live data diverges from the training distribution
- Whole thing runs with one `docker compose up`
- README explains the architecture, not just "how to run it"

## Phased Build Order
Work through these one at a time. Don't jump ahead to a later phase until the current one is working end to end.

| Phase | Goal |
|---|---|
| 0 | Repo scaffold + data ingestion script producing a clean weekly ILI series |
| 1 | Prophet baseline, MLflow logging working end to end |
| 2 | LightGBM with lag/rolling/seasonal features, logged to the same MLflow experiment |
| 3 | FastAPI service wrapping the best registered model (`/predict`, `/health`) |
| 4 | Dockerfile + docker-compose for a one-command setup |
| 5 | GitHub Actions CI/CD (lint/test on PR, build+deploy on merge) |
| 6 | Evidently drift report comparing incoming vs training data |
| 7 (stretch) | Retraining trigger that fires and logs a new model version when drift crosses a threshold |

Phases 0–6 are the CV-ready core. Phase 7 turns this from "an MLOps demo" into a closed-loop system — worth doing if time allows.

## Non-Goals (explicit — do not build these)
- No custom dashboard frontend — MLflow's and Evidently's own UIs are sufficient
- No Kubernetes or distributed training — this dataset doesn't need it
- No large-scale hyperparameter search — two or three manual iterations is enough to tell the comparison story

## Current Phase
_Update this line as you progress — tell Claude Code which phase you're on at the start of each session._

Status: Phase 1 complete — Prophet seasonal baseline working end to end. Trains on the full
weekly national ILI series, evaluates via a 4-week holdout (MAE/RMSE/MAPE), refits on the full
series, and logs params/metrics/model to MLflow (experiment `epicast-ili-forecast`, local
file+sqlite tracking store). Next: Phase 2, LightGBM with lag/rolling/seasonal features logged
to the same experiment for the baseline-vs-upgrade comparison.
