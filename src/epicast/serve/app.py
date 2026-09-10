"""FastAPI service wrapping the registered LightGBM forecaster.

Loads the "champion" alias of the epicast-ili-forecaster model registry entry once at
startup and serves it behind /predict and /health. The model itself already knows how to
turn recent history into a forecast + confidence interval (see
epicast.train.lightgbm_model.RecursiveLightGBMForecaster) -- this layer just wires HTTP
around it and supplies the recent history from the ingested weekly series.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

import mlflow
from fastapi import FastAPI, HTTPException, Query, Request

from epicast.data import DEFAULT_DATA, DEFAULT_TARGET, load_series
from epicast.train.lightgbm_model import CHAMPION_ALIAS, REGISTERED_MODEL_NAME

MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"
DEFAULT_HORIZON = 4
MAX_HORIZON = 12
# Longest lag feature is 52 weeks; keep a bit of margin for rolling windows on top of that.
HISTORY_WEEKS = 60

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model from %s", MODEL_URI)
    app.state.model = mlflow.pyfunc.load_model(MODEL_URI)
    app.state.series = load_series(DEFAULT_DATA, DEFAULT_TARGET)
    logger.info("Loaded %d weekly points, most recent: %s", len(app.state.series), app.state.series["ds"].max().date())
    yield


app = FastAPI(
    title="Epicast",
    description="Real-time ILI (influenza-like-illness) forecasting API.",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict[str, Any]:
    model_loaded = getattr(request.app.state, "model", None) is not None
    return {"status": "ok" if model_loaded else "unavailable", "model_uri": MODEL_URI}


@app.get("/predict")
def predict(
    request: Request,
    horizon: int = Query(DEFAULT_HORIZON, ge=1, le=MAX_HORIZON, description="Weeks ahead to forecast"),
) -> dict[str, Any]:
    model = request.app.state.model
    series = request.app.state.series

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    history = series.tail(HISTORY_WEEKS)
    forecast = model.predict(history, params={"horizon": horizon})

    return {
        "target": DEFAULT_TARGET,
        "as_of": series["ds"].iloc[-1].date().isoformat(),
        "horizon_weeks": horizon,
        "forecast": [
            {
                "week_start": row.ds.date().isoformat(),
                "yhat": round(float(row.yhat), 4),
                "yhat_lower": round(float(row.yhat_lower), 4),
                "yhat_upper": round(float(row.yhat_upper), 4),
            }
            for row in forecast.itertuples()
        ],
    }
