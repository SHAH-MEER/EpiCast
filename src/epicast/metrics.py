"""Forecast accuracy metrics shared by every model trained in this project.

Kept dependency-free (plain numpy) so both the Prophet and LightGBM training
scripts log directly comparable numbers to the same MLflow experiment.
"""

from __future__ import annotations

import numpy as np


def forecast_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mape = float(np.mean(np.abs((actual - predicted) / actual)) * 100)

    return {"mae": mae, "rmse": rmse, "mape": mape}
