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

Status: Phase 3 complete — FastAPI service (`/predict`, `/health`) serving the registered
LightGBM forecaster, working end to end over real HTTP. LightGBM was picked over Prophet as the
served model (MAE 0.12 vs 0.79 on the current holdout) even though it has no native confidence
interval, by training two extra quantile regressors (10th/90th percentile) alongside the point
model and applying them to the point model's recursive forecast trajectory. All three models plus
the recursion logic are wrapped in one MLflow pyfunc model (`RecursiveLightGBMForecaster`,
`src/epicast/train/lightgbm_model.py`) registered as `epicast-ili-forecaster`, with the served
version tracked via the `champion` alias (`train ... --register` promotes a new version to it).
`/predict?horizon=N` returns N weeks of `{week_start, yhat, yhat_lower, yhat_upper}` forecast
from the most recent point in the ingested series. One caveat worth revisiting later: interval
coverage on the current 4-observation holdout is only 50% against an 80% nominal interval — too
small a sample to conclude the interval is miscalibrated, but worth rechecking once Phase 6's
drift/monitoring tooling is in place.

Status: Phase 4 complete — `docker compose up --build` verified end to end from a completely
clean state (`docker compose down -v` first, no pre-existing volumes): `mlflow` came up healthy,
`trainer` ingested data, trained both models, and registered LightGBM as `champion` (exit 0), and
`api` served a real forecast at `/predict`. `Dockerfile` is a single image reused across all three
services; `docker-compose.yml` wires `mlflow` (tracking server + registry, sqlite backend +
artifacts on a named volume) → one-shot `trainer` → `api`, with `data/` shared via a second named
volume and both non-mlflow services pointed at `mlflow` over the network via
`MLFLOW_TRACKING_URI`.

Two real bugs only surfaced under actual Docker networking (not guessable from docs alone):

1. MLflow 3.x proxies artifacts through `--artifacts-destination`, not `--default-artifact-root`,
   whenever `--serve-artifacts` is on (the default) — had to switch flags.
2. MLflow 3.x's server validates the request `Host` header against `--allowed-hosts` (default:
   localhost + private IPs) to block DNS-rebinding attacks; the Compose service name `mlflow`
   wasn't in that default list, so the trainer's requests got 403'd until `--allowed-hosts` was
   set explicitly to include `mlflow` and `mlflow:5000`.

Also: host port 8000 collides with a Windows/Hyper-V dynamic port reservation on this dev
machine (confirmed via `netsh interface ipv4 show excludedportrange` and a raw socket bind test,
not Docker-specific) — `api`'s host port is mapped to 8080 instead; the container's internal port
is still 8000. If this shows up on a different machine, `docker compose logs trainer` is still the
first place to look for anything else.

Status: Phase 5 complete — GitHub Actions CI/CD (`.github/workflows/ci.yml`), verified against a
real run, not just YAML-checked: pushed to a live repo
([github.com/SHAH-MEER/EpiCast](https://github.com/SHAH-MEER/EpiCast)) and watched all three jobs
pass. `test` (ruff + pytest) and `build` (Docker build, no push) run on every push/PR; `deploy`
(build + push to GHCR, tagged `latest` and by commit SHA, using the built-in `GITHUB_TOKEN`) runs
only on push to `main`. Deploy target scoped to GHCR rather than a live host — decided with the
user rather than assumed, since Non-Goals rules out complex infra and no hosting target was ever
named. Confirmed the pushed image is real and public by pulling it back down with a plain
`docker pull`, not just trusting the workflow's green checkmark. Added `ruff` with an explicit
rule selection (`E, F, I, UP` in `pyproject.toml`) rather than its undocumented default rule set,
which turned out to include opinionated plugin rules (blind-except, nested-with) not worth having;
fixed the handful of real issues found (import sorting, two lines over 120 chars). Local repo was
renamed from `master` to `main` to match the intended default branch before the first push.

Status: Phase 6 complete — Evidently drift report (`src/epicast/monitor/drift_report.py`),
comparing the most recent weeks of `wili`/`ili` against the _same calendar weeks in prior years_,
not raw history. This mattered in practice, not just in theory: the first version compared against
all of history and flagged "drift" on every single run, purely from ILI's seasonality (a summer
window's mean wili ~1.0 vs. an all-season reference mean ~2.15, up to 8.3 during flu peaks) —
caught by actually running it against the real data before shipping, not assumed. Also excludes
`num_ili`/`num_patients` from monitoring: CDC's ILINet reporting network has roughly doubled in
participants over the dataset's history (~1.15M vs ~2.5M mean patients, same-season comparison),
so those raw counts trend upward regardless of actual disease dynamics — including them made the
report a permanent false alarm, which defeats the point of a flag people are meant to trust.
Exit code 1 signals a genuine drift finding, but that's swallowed (`|| true`) in `docker
compose`'s `trainer` step, since a real finding is a monitoring result, not a training failure,
and shouldn't block `api` from serving a perfectly good model. Verified against real Docker again
(not just locally): clean `docker compose down -v && up --build -d`, drift report generated
inside the container, landed on the host via a bind mount (`./reports:/app/reports`), and `api`
came up regardless of the (correct, season-adjusted, genuinely borderline p=0.048) drift finding
on the current live data. Next: Phase 7 (stretch) — a retraining trigger that fires off this
report's exit code.
