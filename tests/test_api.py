import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import patch
from fastapi.testclient import TestClient


def _mock_load_models():
    """Return a minimal mock models dict so the API starts without real model files."""
    import numpy as np

    class _FakeModel:
        def predict(self, X):
            return np.array([0.05])  # log-ratio prediction ≈ 5% above rolling mean

    return {
        "linear":   _FakeModel(),
        "xgboost":  _FakeModel(),
        "catboost": _FakeModel(),
        "metrics": {
            "linear":   {"mae": 300000, "rmse": 400000, "smape": 7.58, "target": "raw"},
            "xgboost":  {"mae": 180000, "rmse": 240000, "smape": 4.77, "target": "log_ratio_to_rolling_mean_30"},
            "catboost": {"mae": 195000, "rmse": 265000, "smape": 5.08, "target": "log_ratio_to_rolling_mean_30"},
        },
        "history_df": _make_history_df(),
    }


def _make_history_df():
    import pandas as pd
    import numpy as np
    dates = pd.date_range(end="2024-12-31", periods=90, freq="D")
    np.random.seed(0)
    return pd.DataFrame({
        "date": dates,
        "transaction_volume": np.random.randint(2_000_000, 4_000_000, 90),
        "avg_transaction_value": np.random.randint(10_000, 15_000, 90),
        "usd_uzs_rate": np.linspace(12_500, 12_700, 90),
    })


@patch("api.main.load_models", side_effect=_mock_load_models)
def test_health_returns_200(mock_load):
    from api.main import app
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "linear" in data["models_loaded"]


@patch("api.main.load_models", side_effect=_mock_load_models)
def test_models_endpoint_returns_all_three(mock_load):
    from api.main import app
    with TestClient(app) as client:
        response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {"linear", "xgboost", "catboost"}
    assert "smape" in data["xgboost"]


@patch("api.main.load_models", side_effect=_mock_load_models)
def test_predict_returns_valid_response(mock_load):
    from api.main import app
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"date": "2024-12-15", "model": "xgboost"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["date"] == "2024-12-15"
    assert data["model_used"] == "xgboost"
    assert data["predicted_volume"] > 0
    assert "confidence_note" in data


@patch("api.main.load_models", side_effect=_mock_load_models)
def test_predict_invalid_date_returns_422(mock_load):
    from api.main import app
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"date": "not-a-date", "model": "catboost"},
        )
    assert response.status_code == 422
