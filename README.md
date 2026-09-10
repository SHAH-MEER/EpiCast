# Epicast

A real-time ILI (influenza-like-illness) forecasting API. The point isn't just a model —
it's the full lifecycle working end to end: **train → track → serve → ship → monitor**.

## Architecture

```text
CDC FluView (Delphi Epidata API)
        │  fetch_ilinet.py
        ▼
data/processed/ilinet_national_weekly.csv   (clean weekly national ILI series)
        │
        ▼
   [Phase 1-2, done] Prophet + LightGBM, both logged to one MLflow experiment
        │              (features: lags, rolling stats, seasonal terms for LightGBM)
        ▼
   MLflow model registry  ── epicast-ili-forecaster@champion (LightGBM: point + 2 quantile models)
        │
        ▼
   [Phase 3, done] FastAPI service  ── /predict, /health
        │
        ▼
   [Phase 4, done] docker compose   ── mlflow + trainer (one-shot) + api
        │
        ▼
   [Phase 5] GitHub Actions   ── lint/test on PR, build+deploy on merge
        │
        ▼
   [Phase 6] Evidently drift report  ── incoming data vs. training distribution
        │
        ▼
   [Phase 7, stretch] drift-triggered retraining
```

Data source: **CDC FluView (ILINet)**, national series, pulled via the public
[Delphi Epidata API](https://cmu-delphi.github.io/delphi-epidata/api/fluview.html) (no API key
required for anonymous, rate-limited access).

## Status

Phase 4 — `docker compose up` brings up MLflow, trains + registers both models, and serves the
API, all in one command. Verified end to end from a completely clean state (no pre-existing
volumes). See `CLAUDE.md` for the full phased build order and definition of done.

## Running the ingestion script

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"

python -m epicast.ingest.fetch_ilinet
# -> data/processed/ilinet_national_weekly.csv
```

Options: `--region` (default `nat`), `--epiweeks` (default `201001-202653`), `--output`.

## Training the Prophet baseline

```bash
python -m epicast.train.prophet_baseline
# holds out the last --horizon weeks (default 4) for MAE/RMSE/MAPE,
# then refits on the full series and logs that model to MLflow

mlflow ui   # -> http://127.0.0.1:5000, experiment "epicast-ili-forecast"
```

Options: `--horizon`, `--target` (`wili` or `ili`), `--data`, `--experiment`.

Note: MAPE on a holdout that lands in flu off-season (ILI rates near their yearly low) will
look large in relative terms even when the absolute error (MAE/RMSE) is small — that's an
artifact of percentage error near a low baseline, not a broken model.

## Training the LightGBM upgrade

```bash
python -m epicast.train.lightgbm_model --register
# same holdout protocol as the Prophet baseline, logged to the same MLflow experiment;
# --register also promotes the model to the "champion" alias the API serves
```

Features: lags at 1/2/3/4/52 weeks, rolling mean/std over 4 and 8 weeks, and sin/cos-of-week-of-year
for seasonality (`src/epicast/features.py`). Three LightGBM regressors share these features: a
point (mean) model plus 10th/90th-percentile quantile models for the confidence interval. Multi-week
forecasts come from recursively feeding the point model's own predictions back in as observed
values, since a lag-feature model can't see past the horizon its lags were built for — the quantile
models are then applied to that same trajectory rather than branching recursively themselves.

All three models plus the recursive-forecast logic are wrapped in one MLflow pyfunc model
(`RecursiveLightGBMForecaster`) and registered as `epicast-ili-forecaster`, so serving code just
calls `.predict(history, params={"horizon": N})` without knowing about features or recursion.

Options: same as the Prophet script, plus `--register`.

## Running the API

```bash
# requires a champion model registered first (see above)
uvicorn epicast.serve.app:app --reload --port 8001
# port 8000 collides with a Windows/Hyper-V dynamic port reservation on some machines --
# if 8000 works fine on yours, feel free to drop --port and use the default

curl http://127.0.0.1:8001/health
curl "http://127.0.0.1:8001/predict?horizon=4"
```

Interactive docs at `http://127.0.0.1:8001/docs`. `/predict` forecasts forward from the most
recent point in `data/processed/ilinet_national_weekly.csv` — re-run the ingestion script to move
that forward — and takes `horizon` (1-12 weeks, default 4) as its only parameter.

## Tests

```bash
pytest
# tests/test_app.py skips if no champion model is registered locally yet
```

## Running everything with Docker

```bash
docker compose up --build
```

This brings up three services:

- `mlflow` — tracking server + model registry at `http://localhost:5000`, backed by a named
  volume (`mlflow-data`) so runs/models survive restarts
- `trainer` — one-shot: ingests data, trains Prophet, trains + registers LightGBM as the
  `champion` alias, then exits (`docker compose` waits for it before starting `api`)
- `api` — the FastAPI service at `http://localhost:8080` (`/docs`, `/health`, `/predict`) — mapped
  to host port 8080 rather than 8000, since 8000 collides with a Windows/Hyper-V dynamic port
  reservation on some machines; the container's internal port is still 8000

Data (`data/`) is shared between `trainer` and `api` via a named volume (`epicast-data`), and
both point at the `mlflow` service over the network via `MLFLOW_TRACKING_URI`.

Verified end to end with a clean `docker compose down -v && docker compose up --build -d` (no
pre-existing volumes): all three services reached the expected state and `/predict` returned a
real forecast. Two things worth knowing if you touch the MLflow service config:

- MLflow 3.x proxies artifacts through `--artifacts-destination` (not `--default-artifact-root`)
  whenever `--serve-artifacts` is on, which it is by default.
- MLflow 3.x's server validates the request `Host` header against `--allowed-hosts` (default:
  localhost + private IPs) to block DNS-rebinding attacks — a Compose service name like `mlflow`
  isn't in that default list, so it needs to be added explicitly or the trainer's requests get a
  403.

First `--build` will take a few minutes (installing prophet/mlflow/lightgbm from scratch);
`docker compose logs trainer` is the first place to look if `api` never comes up.
