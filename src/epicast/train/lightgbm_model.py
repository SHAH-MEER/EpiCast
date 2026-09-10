"""LightGBM upgrade over the Prophet baseline: lag/rolling/seasonal features, tracked
in the same MLflow experiment for a direct baseline-vs-upgrade comparison.

Three regressors share one feature set: a point (mean) model plus two quantile models
(10th/90th percentile) that together give a forecast + confidence interval, since the
point model's L2 objective has no notion of uncertainty on its own. Multi-week forecasts
come from recursively predicting one step, feeding that prediction back in as if it were
observed, and repeating -- the interval models reuse the point model's trajectory rather
than branching recursively themselves, since forecasting "the 10th percentile assuming
the 10th percentile also happened last week" compounds pessimism rather than reflecting
real uncertainty growth.

Evaluation mirrors the Prophet baseline: hold out the last `horizon` weeks, fit on
everything before that, recursively forecast the holdout, and score it. The models
actually logged as artifacts are then refit on the full series, wrapped in a single
pyfunc model, and registered to the MLflow Model Registry so Phase 3 can load
"models:/epicast-ili-forecaster@champion" without knowing anything about feature
engineering or recursive forecasting.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import mlflow.pyfunc
import numpy as np
import pandas as pd

from epicast.data import DEFAULT_DATA, DEFAULT_TARGET, load_series
from epicast.features import FEATURE_COLUMNS, build_features
from epicast.metrics import forecast_metrics

DEFAULT_EXPERIMENT = "epicast-ili-forecast"
DEFAULT_HORIZON = 4
REGISTERED_MODEL_NAME = "epicast-ili-forecaster"
CHAMPION_ALIAS = "champion"

LOWER_QUANTILE = 0.1
UPPER_QUANTILE = 0.9

BASE_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 4,
    "num_leaves": 15,
    "min_child_samples": 10,
    "random_state": 42,
    "verbosity": -1,
}

logger = logging.getLogger(__name__)


def _fit_lgbm(features_df: pd.DataFrame, params: dict) -> lgb.LGBMRegressor:
    train_rows = features_df.dropna(subset=[*FEATURE_COLUMNS, "y"])
    model = lgb.LGBMRegressor(**params)
    model.fit(train_rows[list(FEATURE_COLUMNS)], train_rows["y"])
    return model


def fit_one_step_model(features_df: pd.DataFrame, params: dict | None = None) -> lgb.LGBMRegressor:
    """Fit the point (mean) one-step-ahead model."""
    return _fit_lgbm(features_df, {**BASE_PARAMS, "objective": "regression", **(params or {})})


def fit_quantile_model(features_df: pd.DataFrame, alpha: float, params: dict | None = None) -> lgb.LGBMRegressor:
    """Fit a one-step-ahead quantile model (e.g. alpha=0.1 for the 10th percentile)."""
    return _fit_lgbm(features_df, {**BASE_PARAMS, "objective": "quantile", "alpha": alpha, **(params or {})})


def _recursive_feature_trajectory(
    point_model: lgb.LGBMRegressor, history: pd.DataFrame, horizon: int
) -> tuple[list[pd.DataFrame], np.ndarray]:
    """Step `horizon` weeks past `history`, using point_model's own predictions as the
    "observed" values future lag/rolling features are computed from. Returns the feature
    row used at each step (so quantile models can be applied to the same trajectory) and
    the point predictions.
    """
    extended = history[["ds", "y"]].reset_index(drop=True).copy()
    feature_rows: list[pd.DataFrame] = []
    predictions = []

    for _ in range(horizon):
        next_ds = extended["ds"].iloc[-1] + pd.Timedelta(weeks=1)
        candidate = pd.concat(
            [extended, pd.DataFrame({"ds": [next_ds], "y": [np.nan]})], ignore_index=True
        )
        next_features = build_features(candidate).iloc[[-1]][list(FEATURE_COLUMNS)]
        prediction = float(point_model.predict(next_features)[0])

        feature_rows.append(next_features)
        predictions.append(prediction)
        extended = pd.concat(
            [extended, pd.DataFrame({"ds": [next_ds], "y": [prediction]})], ignore_index=True
        )

    return feature_rows, np.array(predictions)


def forecast_recursive(model: lgb.LGBMRegressor, history: pd.DataFrame, horizon: int) -> np.ndarray:
    """Forecast `horizon` weeks past the end of `history` (a ds/y frame), one step at a time."""
    _, predictions = _recursive_feature_trajectory(model, history, horizon)
    return predictions


def forecast_recursive_with_interval(
    point_model: lgb.LGBMRegressor,
    lower_model: lgb.LGBMRegressor,
    upper_model: lgb.LGBMRegressor,
    history: pd.DataFrame,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Like `forecast_recursive`, but also returns a per-step [lower, upper] interval."""
    feature_rows, predictions = _recursive_feature_trajectory(point_model, history, horizon)

    lower = np.array([float(lower_model.predict(row)[0]) for row in feature_rows])
    upper = np.array([float(upper_model.predict(row)[0]) for row in feature_rows])
    # Quantile crossing is rare with well-separated alphas but not impossible on small data.
    lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)

    return predictions, lower, upper


def evaluate_holdout(series: pd.DataFrame, horizon: int, params: dict | None = None) -> dict[str, float]:
    train, test = series.iloc[:-horizon], series.iloc[-horizon:]

    model = fit_one_step_model(build_features(train), params)
    predictions = forecast_recursive(model, train, horizon)

    return forecast_metrics(test["y"].to_numpy(), predictions)


def evaluate_holdout_with_interval(series: pd.DataFrame, horizon: int) -> dict[str, float]:
    train, test = series.iloc[:-horizon], series.iloc[-horizon:]
    train_features = build_features(train)

    point_model = fit_one_step_model(train_features)
    lower_model = fit_quantile_model(train_features, LOWER_QUANTILE)
    upper_model = fit_quantile_model(train_features, UPPER_QUANTILE)

    predictions, lower, upper = forecast_recursive_with_interval(point_model, lower_model, upper_model, train, horizon)
    actual = test["y"].to_numpy()

    metrics = forecast_metrics(actual, predictions)
    metrics["interval_coverage"] = float(np.mean((actual >= lower) & (actual <= upper)))
    metrics["mean_interval_width"] = float(np.mean(upper - lower))
    return metrics


class RecursiveLightGBMForecaster(mlflow.pyfunc.PythonModel):
    """Wraps the point + quantile LightGBM models and the feature/recursion logic behind
    a single `.predict(history_df, params={"horizon": N})` call, so serving code doesn't
    need to know about lag features or recursive forecasting at all.
    """

    def __init__(self, point_model: lgb.LGBMRegressor, lower_model: lgb.LGBMRegressor, upper_model: lgb.LGBMRegressor):
        self.point_model = point_model
        self.lower_model = lower_model
        self.upper_model = upper_model

    def predict(self, context, model_input: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
        horizon = (params or {}).get("horizon", DEFAULT_HORIZON)

        history = model_input[["ds", "y"]].copy()
        history["ds"] = pd.to_datetime(history["ds"])

        predictions, lower, upper = forecast_recursive_with_interval(
            self.point_model, self.lower_model, self.upper_model, history, horizon
        )
        future_ds = [history["ds"].iloc[-1] + pd.Timedelta(weeks=w) for w in range(1, horizon + 1)]

        return pd.DataFrame({"ds": future_ds, "yhat": predictions, "yhat_lower": lower, "yhat_upper": upper})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="Weeks ahead to forecast/evaluate")
    parser.add_argument("--target", default=DEFAULT_TARGET, choices=["wili", "ili"])
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--register", action="store_true", help="Register the final model as the registry champion")
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
                "interval_quantiles": [LOWER_QUANTILE, UPPER_QUANTILE],
                "feature_columns": FEATURE_COLUMNS,
                **BASE_PARAMS,
            }
        )

        metrics = evaluate_holdout_with_interval(series, args.horizon)
        mlflow.log_metrics(metrics)
        logger.info("Holdout metrics (last %d weeks): %s", args.horizon, metrics)

        final_features = build_features(series)
        point_model = fit_one_step_model(final_features)
        lower_model = fit_quantile_model(final_features, LOWER_QUANTILE)
        upper_model = fit_quantile_model(final_features, UPPER_QUANTILE)

        input_example = final_features.dropna(subset=list(FEATURE_COLUMNS)).tail(5)[list(FEATURE_COLUMNS)]
        mlflow.lightgbm.log_model(point_model, name="point_model", input_example=input_example)
        mlflow.lightgbm.log_model(lower_model, name="lower_model", input_example=input_example)
        mlflow.lightgbm.log_model(upper_model, name="upper_model", input_example=input_example)

        forecaster = RecursiveLightGBMForecaster(point_model, lower_model, upper_model)
        pyfunc_input_example = series[["ds", "y"]].tail(10)
        # A plain input_example only infers the *input* schema -- without an explicit
        # params schema here, MLflow silently drops any `params={"horizon": ...}` passed
        # at inference time and the model always falls back to DEFAULT_HORIZON.
        pyfunc_output_example = forecaster.predict(None, pyfunc_input_example, {"horizon": args.horizon})
        signature = mlflow.models.infer_signature(
            model_input=pyfunc_input_example,
            model_output=pyfunc_output_example,
            params={"horizon": args.horizon},
        )
        model_info = mlflow.pyfunc.log_model(
            python_model=forecaster,
            name="model",
            input_example=pyfunc_input_example,
            signature=signature,
            registered_model_name=REGISTERED_MODEL_NAME if args.register else None,
        )

        if args.register:
            client = mlflow.MlflowClient()
            client.set_registered_model_alias(
                REGISTERED_MODEL_NAME, CHAMPION_ALIAS, model_info.registered_model_version
            )
            logger.info(
                "Registered %s v%s and set alias '%s'",
                REGISTERED_MODEL_NAME,
                model_info.registered_model_version,
                CHAMPION_ALIAS,
            )

        logger.info("Logged forecaster pyfunc model to MLflow run %s", mlflow.active_run().info.run_id)


if __name__ == "__main__":
    main()
