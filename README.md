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
   [Phase 1-2] Prophet + LightGBM, both logged to one MLflow experiment
        │              (features: lags, rolling stats, seasonal terms for LightGBM)
        ▼
   MLflow model registry  ── best model promoted
        │
        ▼
   [Phase 3] FastAPI service  ── /predict, /health
        │
        ▼
   [Phase 4] docker compose   ── one-command bring-up of API (+ MLflow, once wired)
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

Phase 1 — Prophet seasonal baseline, logged to MLflow (experiment `epicast-ili-forecast`).
See `CLAUDE.md` for the full phased build order and definition of done.

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

## Tests

```bash
pytest
```
