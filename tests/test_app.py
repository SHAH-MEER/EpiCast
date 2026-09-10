import mlflow
import pytest
from fastapi.testclient import TestClient

from epicast.serve.app import app
from epicast.train.lightgbm_model import CHAMPION_ALIAS, REGISTERED_MODEL_NAME


def _champion_registered() -> bool:
    try:
        mlflow.MlflowClient().get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _champion_registered(),
    reason="epicast-ili-forecaster@champion not registered locally; "
    "run `python -m epicast.train.lightgbm_model --register` first",
)


def test_health_reports_ok_once_model_is_loaded():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_uri": f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"}


def test_predict_returns_a_row_per_horizon_week():
    with TestClient(app) as client:
        response = client.get("/predict", params={"horizon": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["horizon_weeks"] == 3
    assert len(body["forecast"]) == 3
    assert set(body["forecast"][0]) == {"week_start", "yhat", "yhat_lower", "yhat_upper"}


def test_predict_defaults_to_four_weeks():
    with TestClient(app) as client:
        response = client.get("/predict")

    assert response.status_code == 200
    assert len(response.json()["forecast"]) == 4


def test_predict_rejects_horizon_outside_bounds():
    with TestClient(app) as client:
        response = client.get("/predict", params={"horizon": 100})

    assert response.status_code == 422
