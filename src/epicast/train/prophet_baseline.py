"""Prophet seasonal baseline for N-week-ahead ILI forecasting, tracked in MLflow.

Evaluation: hold out the last `horizon` weeks, fit on everything before that,
forecast the holdout, and score it. The model actually logged as an artifact
is then refit on the full series (so it's ready to serve forecasts past the
end of the training data), tagged with the honest holdout metrics.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mlflow
import mlflow.prophet
import pandas as pd
from prophet import Prophet

from epicast.metrics import forecast_metrics

DEFAULT_DATA = Path("data/processed/ilinet_national_weekly.csv")
DEFAULT_EXPERIMENT = "epicast-ili-forecast"
DEFAULT_HORIZON = 4
DEFAULT_TARGET = "wili"

logger = logging.getLogger(__name__)


def load_series(data_path: Path, target: str) -> pd.DataFrame:
    """Load the cleaned weekly ILI series as Prophet's expected ds/y frame."""
    df = pd.read_csv(data_path, parse_dates=["week_start"])
    return df[["week_start", target]].rename(columns={"week_start": "ds", target: "y"}).sort_values("ds")


def make_model() -> Prophet:
    # The series has exactly one point per week, so a "weekly seasonality"
    # component would just be aliasing noise; only annual flu seasonality is real.
    return Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)


def evaluate_holdout(series: pd.DataFrame, horizon: int) -> dict[str, float]:
    train, test = series.iloc[:-horizon], series.iloc[-horizon:]

    model = make_model().fit(train)
    forecast = model.predict(pd.concat([train[["ds"]], test[["ds"]]]))
    predicted = forecast.tail(horizon)["yhat"].to_numpy()

    return forecast_metrics(test["y"].to_numpy(), predicted)


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
    with mlflow.start_run(run_name=f"prophet-h{args.horizon}"):
        mlflow.log_params(
            {
                "model": "prophet",
                "target": args.target,
                "horizon_weeks": args.horizon,
                "yearly_seasonality": True,
                "n_train_holdout": len(series) - args.horizon,
            }
        )

        metrics = evaluate_holdout(series, args.horizon)
        mlflow.log_metrics(metrics)
        logger.info("Holdout metrics (last %d weeks): %s", args.horizon, metrics)

        final_model = make_model().fit(series)
        mlflow.prophet.log_model(final_model, name="model", input_example=series[["ds"]].tail(5))
        logger.info("Logged final model (fit on full series) to MLflow run %s", mlflow.active_run().info.run_id)


if __name__ == "__main__":
    main()
