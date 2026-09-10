# Epicast

**A flu forecasting API with a full, working MLOps lifecycle behind it** — not a notebook, a
system: train → track → serve → ship → monitor, all actually wired together and verified end to
end.

## Why influenza surveillance matters

Every year, the CDC estimates flu causes somewhere between **9.4 and 51 million illnesses**,
**120,000–710,000 hospitalizations**, and **6,300–52,000 deaths** in the United States alone —
the range itself telling you something: some seasons are a bad cold, others overwhelm emergency
rooms, and nobody knows in advance which kind of season it'll be
([CDC, Estimated Flu Burden](https://www.cdc.gov/flu-burden/php/about/index.html)).

That's exactly why the CDC runs [ILINet](https://www.cdc.gov/fluview/overview/index.html):
roughly **4,000 outpatient healthcare providers** across all 50 states report, every single week,
how many patients came in with flu-like symptoms. It's the earliest real signal of whether this
week is a normal week or the start of something that hospitals need to staff up for — vaccine
strain selection, surge planning, and public health messaging all lean on reading that signal
correctly, and reading it *early*.

## Why forecasting it is hard

Not "hard" in the sense of a clever loss function — hard in the sense that most of what breaks a
forecasting system in practice has nothing to do with the model:

- **The signal is aggressively seasonal, and that changes what "normal" means from one month to
  the next.** A flat "is this week different from history" comparison is close to meaningless —
  history includes both flu-season peaks and summer troughs at once. (This bit us directly while
  building this: an early version of the monitoring in this repo compared a summer window against
  a reference mixing every season, and it flagged "drift" on *every single run* — not because
  anything was actually wrong, purely because summer doesn't look like winter.)
- **The ground truth itself isn't stable.** ILINet's reporting network has grown substantially
  over the life of this dataset — the number of participating providers has roughly doubled — so
  raw visit counts trend upward for reasons that have nothing to do with disease activity. A drift
  monitor that doesn't know to ignore that will cry wolf forever.
- **A single number isn't a forecast.** A point estimate with no sense of its own uncertainty is
  close to useless for a real staffing or response decision — but the model family that's best at
  this kind of tabular forecasting (gradient boosting) doesn't hand you a confidence interval for
  free the way classical time-series models do.
- **Shipping the model once is the easy 20%.** The signal shifts as the season turns, which means
  a model that was accurate in October can be quietly wrong by December — and most from-scratch
  forecasting projects never build the part that would actually notice that, let alone respond
  to it.

## Enter Epicast

Epicast is a direct answer to each of those, not a generic ML template:

| The hard part | What Epicast actually does about it |
| --- | --- |
| Seasonality breaks naive comparisons | Every drift check compares the current window against the *same calendar weeks in prior years* — not raw history |
| The reporting network itself isn't stable | Monitoring watches the rate signal (`wili`/`ili`) the model forecasts, not raw participant counts that trend upward regardless of disease activity |
| A point forecast isn't a real answer | The served model is three LightGBM regressors — a point model plus two quantile models — giving `/predict` a genuine forecast *and* confidence interval |
| Models go stale silently | A closed-loop trigger checks for drift and automatically retrains + promotes a new model version when it's crossed |
| "It worked on my machine" | The whole stack — MLflow, training, the API — comes up with one `docker compose up`, verified from a completely clean state, with CI publishing a real image to GHCR on every merge |

And the comparison story is real, not staged: on held-out data, the LightGBM upgrade cuts error
roughly **6x** versus the Prophet baseline (MAE 0.12 vs. 0.79) — both runs logged side by side in
the same MLflow experiment, so that's a claim you can go look at, not take on faith.

## See it in action

Screenshots below pull from [`docs/screenshots/`](docs/screenshots/) — drop a PNG in with the
right filename and it replaces the placeholder automatically. Capture instructions are in
`docs/screenshots/README.md`; the short version is under each image.

### The comparison story is real

![MLflow experiment comparing Prophet and LightGBM runs](docs/screenshots/mlflow-comparison.png)

Open `http://localhost:5000` → the `epicast-ili-forecast` experiment → screenshot the runs table
with MAE/RMSE/MAPE columns visible for both `prophet` and `lightgbm` runs.

### Models are versioned, not just files on disk

![MLflow model registry showing epicast-ili-forecaster versions](docs/screenshots/mlflow-registry.png)

**Models → epicast-ili-forecaster** in the MLflow UI, with the `champion` alias visible on a
version.

### The API is real and self-documenting

![FastAPI Swagger docs showing /predict and /health](docs/screenshots/api-docs.png)

`http://localhost:8080/docs`.

### `/predict` returns a forecast *and* an interval

![Example /predict JSON response with yhat, yhat_lower, yhat_upper](docs/screenshots/predict-response.png)

"Try it out" on `/predict` in `/docs`, or a pretty-printed terminal `curl` call.

### One command brings up the whole stack

![Terminal output of docker compose ps -a showing three services](docs/screenshots/docker-compose-ps.png)

`docker compose ps -a` — `mlflow` healthy, `trainer` exited (0), `api` running.

### Monitoring isn't just a checkbox

![Evidently drift report distribution chart](docs/screenshots/drift-report.png)

Open `reports/drift_report.html` in a browser after running the stack.

### CI/CD is real, not aspirational

![GitHub Actions run with test, build, and deploy all green](docs/screenshots/github-actions.png)

The [Actions tab](https://github.com/SHAH-MEER/EpiCast/actions), any run with all three jobs
green.

## Architecture

```text
CDC FluView (Delphi Epidata API)
        │  fetch_ilinet.py
        ▼
data/processed/ilinet_national_weekly.csv   (clean weekly national ILI series)
        │
        ▼
   Prophet + LightGBM, both logged to one MLflow experiment
        │              (features: lags, rolling stats, seasonal terms for LightGBM)
        ▼
   MLflow model registry  ── epicast-ili-forecaster@champion (LightGBM: point + 2 quantile models)
        │
        ▼
   FastAPI service  ── /predict, /health
        │
        ▼
   docker compose   ── mlflow + trainer (one-shot) + api
        │
        ▼
   GitHub Actions   ── lint/test + build on every push, deploy (GHCR) on main
        │
        ▼
   Evidently drift report  ── current season vs. same weeks in prior years
        │
        ▼
   retrain_trigger.py  ── ingest → check drift → retrain + register if crossed
```

Data source: **CDC FluView (ILINet)**, national series, pulled via the public
[Delphi Epidata API](https://cmu-delphi.github.io/delphi-epidata/api/fluview.html) (no API key
required for anonymous, rate-limited access).

## Quickstart

```bash
docker compose up --build
```

That's the whole setup: it brings up MLflow, ingests data, trains and compares both models,
registers the better one, and starts serving it.

```bash
curl http://localhost:8080/health
curl "http://localhost:8080/predict?horizon=4"
```

MLflow UI: `http://localhost:5000`. API docs: `http://localhost:8080/docs`. See
[Running everything with Docker](#running-everything-with-docker) below for what's actually
happening and a couple of things worth knowing if you touch the compose config.

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

## Drift monitoring

```bash
python -m epicast.monitor.drift_report
# -> reports/drift_report.html, exits 1 if drift is flagged (0 otherwise)
```

Compares the most recent `--current-weeks` (default 12) of `wili`/`ili` against every prior
year's rows on those *same calendar weeks* — not the whole history. ILI is strongly seasonal, so
a naive "current window vs. all of history" comparison flags "drift" on essentially every run: a
summer window (mean wili ~1.0) will always look wildly different from a reference mixing every
season including flu peaks (mean ~2.15, up to 8.3) — confirmed empirically before shipping the
season-aware version, not assumed. The raw participant counts (`num_ili`, `num_patients`) are
excluded from monitoring for the same reason: CDC's reporting network has roughly doubled in
number of participants over the dataset's history, so those columns trend upward regardless of
actual disease dynamics — including them made the report flag drift on every single run,
which is worse than not monitoring at all.

Wired into `docker compose`'s `trainer` step (after training, before `api` starts) — its exit
code is swallowed there (`|| true`) since a genuine drift finding is a monitoring result, not a
training failure, and shouldn't block `api` from serving a perfectly good model. The report
writes to `reports/`, bind-mounted to the host so you can open the HTML directly after
`docker compose up` without reaching into the container.

## Retraining trigger (Phase 7, stretch)

```bash
python -m epicast.monitor.retrain_trigger
# ingests fresh data -> checks drift -> retrains + registers a new champion if drift
# crossed the threshold. Self-contained: nothing needs to be run beforehand.

python -m epicast.monitor.retrain_trigger --skip-ingest   # reuse the existing data file
python -m epicast.monitor.retrain_trigger --force         # retrain regardless of drift
```

This is what turns the drift report from "something a person has to remember to check" into an
actual closed loop. It ingests fresh data itself (so it can genuinely be run standalone, e.g. on
a schedule — cron, Task Scheduler, a scheduled CI job — without any other setup), runs the same
season-aware drift check as above, and if drift is flagged, retrains LightGBM and promotes the
new version to `champion` via `epicast.train.lightgbm_model --register` (run as a subprocess, so
it reuses that already-tested CLI rather than re-implementing its training/registration logic).

Verified for real, not just unit-tested: run locally against live data (found drift, retrained,
registered v3 then v4 as champion across two runs) and again against the live `docker compose`
stack via `docker compose run --rm trainer python -m epicast.monitor.retrain_trigger --skip-ingest`
(found drift, registered v2 in the containerized MLflow registry).

**Known limitation:** a running `api` container loads its model once at startup and won't pick up
a newly-registered champion until it's restarted (`docker compose restart api`) — there's no
hot-reload. Wiring that up, or actually scheduling this to run periodically against a persistent
hosted MLflow server, would be the next real step past what's in this repo; not built here since
it needs infrastructure this project doesn't have (a persistently reachable MLflow instance), and
Non-Goals rules out standing up something like Kubernetes just to get one.

## CI/CD

`.github/workflows/ci.yml` runs on every push and PR to main:

- **test** — `ruff check .` then `pytest -q`
- **build** — builds the shared Dockerfile (build-only, doesn't push) to catch breakage early
- **deploy** — only on push to `main`: builds and pushes the image to GHCR as
  `ghcr.io/shah-meer/epicast:latest` and `:<commit-sha>`, using the built-in `GITHUB_TOKEN` (no
  external accounts or secrets needed)

Verified against a real run on GitHub Actions (not just YAML-checked): all three jobs passed, and
the pushed image was pulled back down from GHCR to confirm it's actually there and public.

## Status

All phases complete, including the Phase 7 stretch goal. See `CLAUDE.md` for the full phased
build order, definition of done, and a running log of what was built, verified, and fixed along
the way.
