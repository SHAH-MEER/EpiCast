"""LightGBM upgrade over the Prophet baseline: lag/rolling/seasonal features, tracked
in the same MLflow experiment for a direct baseline-vs-upgrade comparison.

One regressor is trained to predict next week's ILI rate from lag/rolling/seasonal
features. Multi-week forecasts are produced by forecasting one step, feeding that
prediction back in as if it were observed, and repeating -- the same recursive
strategy you'd reach for with any lag-feature model, since a single fit can't see
past the horizon its lags were built for.

Evaluation mirrors the Prophet baseline: hold out the last `horizon` weeks, fit on
everything before that, recursively forecast the holdout, and score it. The model
actually logged as an artifact is then refit on the full series.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd

from epicast.data import DEFAULT_DATA, DEFAULT_TARGET, load_series
from epicast.features import FEATURE_COLUMNS, build_features
from epicast.metrics import forecast_metrics

DEFAULT_EXPERIMENT = "epicast-ili-forecast"
DEFAULT_HORIZON = 4

DEFAULT_PARAMS = {
    "objective": "regression",
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 4,
    "num_leaves": 15,
    "min_child_samples": 10,
    "random_state": 42,
    "verbosity": -1,
}

logger = logging.getLogger(__name__)


def fit_one_step_model(features_df: pd.DataFrame, params: dict | None = None) -> lgb.LGBMRegressor:
    train_rows = features_df.dropna(subset=[*FEATURE_COLUMNS, "y"])
    model = lgb.LGBMRegressor(**(params or DEFAULT_PARAMS))
    model.fit(train_rows[list(FEATURE_COLUMNS)], train_rows["y"])
    return model


def forecast_recursive(model: lgb.LGBMRegressor, history: pd.DataFrame, horizon: int) -> np.ndarray:
    """Forecast `horizon` weeks past the end of `history` (a ds/y frame), one step at a time."""
    extended = history[["ds", "y"]].reset_index(drop=True).copy()
    predictions = []

    for _ in range(horizon):
        next_ds = extended["ds"].iloc[-1] + pd.Timedelta(weeks=1)
        candidate = pd.concat(
            [extended, pd.DataFrame({"ds": [next_ds], "y": [np.nan]})], ignore_index=True
        )
        next_features = build_features(candidate).iloc[[-1]][list(FEATURE_COLUMNS)]
        prediction = float(model.predict(next_features)[0])

        predictions.append(prediction)
        extended = pd.concat(
            [extended, pd.DataFrame({"ds": [next_ds], "y": [prediction]})], ignore_index=True
        )

    return np.array(predictions)


def evaluate_holdout(series: pd.DataFrame, horizon: int, params: dict | None = None) -> dict[str, float]:
    train, test = series.iloc[:-horizon], series.iloc[-horizon:]

    model = fit_one_step_model(build_features(train), params)
    predictions = forecast_recursive(model, train, horizon)

    return forecast_metrics(test["y"].to_numpy(), predictions)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="Weeks ahead to forecast/evaluate")
    parser.add_argument("--target", default=DEFAULT_TARGET, choices=["wili", "ili"])
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    args = parser.parse_args()

    series = load_series(args.data, args.target)
    logger.info("Loaded %d weekly points (%s -> %s)", len(series), series["ds"].min().date(), series["ds"].max().date())

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=f"lightgbm-h{args.horizon}"):
        mlflow.log_params(
            {
                "model": "lightgbm",
                "target": args.target,
                "horizon_weeks": args.horizon,
                "forecast_strategy": "recursive_one_step",
                "feature_columns": FEATURE_COLUMNS,
                **DEFAULT_PARAMS,
            }
        )

        metrics = evaluate_holdout(series, args.horizon)
        mlflow.log_metrics(metrics)
        logger.info("Holdout metrics (last %d weeks): %s", args.horizon, metrics)

        final_features = build_features(series)
        final_model = fit_one_step_model(final_features)
        input_example = final_features.dropna(subset=list(FEATURE_COLUMNS)).tail(5)[list(FEATURE_COLUMNS)]
        mlflow.lightgbm.log_model(final_model, name="model", input_example=input_example)
        logger.info("Logged final model (fit on full series) to MLflow run %s", mlflow.active_run().info.run_id)


if __name__ == "__main__":
    main()
