import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # makes 'src' importable

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from src.predict import load_models, make_prediction


# ── Lifespan: load models once at startup, clean up at shutdown ───────────────
# Using lifespan (FastAPI 0.93+) — @app.on_event("startup") is deprecated.
_models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models into memory when the server starts."""
    print("🚀 Loading models into memory...")
    _models.update(load_models())
    loaded = [k for k in _models if k != "metrics" and k != "history_df"]
    print(f"✅ Models loaded: {loaded}")
    yield
    # Cleanup (nothing to do here, but good practice to have the hook)
    _models.clear()
    print("👋 Server shutting down — models released.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Uzbekistan Payment Volume Forecasting API",
    description=(
        "Predicts daily digital payment transaction volumes across Uzbekistan "
        "using Linear Regression, XGBoost, and CatBoost models trained on "
        "2019–2022 data and validated on 2023–2024 walk-forward test set."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response schemas ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    date: str
    model: Literal["linear", "xgboost", "catboost"] = "catboost"

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        try:
            import pandas as pd
            pd.Timestamp(v)
        except Exception:
            raise ValueError(f"Invalid date format: '{v}'. Use YYYY-MM-DD.")
        return v


class PredictResponse(BaseModel):
    date: str
    predicted_volume: int
    model_used: str
    smape_on_test: float
    is_navruz: bool
    is_ramadan: bool
    is_payday: bool
    is_weekend: bool
    confidence_note: str


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Status"])
def health():
    """Returns server health and list of loaded models."""
    loaded = [k for k in _models if k not in ("metrics", "history_df")]
    return {
        "status": "healthy" if loaded else "degraded",
        "models_loaded": loaded,
    }


@app.get("/models", tags=["Status"])
def list_models():
    """Returns available models and their test-set performance metrics."""
    if "metrics" not in _models:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")
    metrics = _models["metrics"]
    return {
        model: {
            "mae":   metrics[model]["mae"],
            "rmse":  metrics[model]["rmse"],
            "smape": metrics[model]["smape"],
            "target_type": metrics[model].get("target", "raw"),
        }
        for model in ("linear", "xgboost", "catboost")
        if model in metrics
    }


@app.post("/predict", response_model=PredictResponse, tags=["Forecasting"])
def predict(request: PredictRequest):
    """
    Predict daily transaction volume for a given date.

    - **date**: Target date in YYYY-MM-DD format
    - **model**: One of `linear`, `xgboost`, `catboost` (default)

    Returns the predicted transaction volume plus Uzbekistan-specific
    calendar context (Navruz, Ramadan, payday, weekend flags).
    """
    if not _models:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    try:
        result = make_prediction(request.date, request.model, _models)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return PredictResponse(**result)
