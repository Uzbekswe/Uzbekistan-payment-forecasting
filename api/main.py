import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Literal, Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))  # makes 'src' importable

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, field_validator

from src.predict import load_models, make_prediction


# ── Lifespan: load models once at startup, clean up at shutdown ───────────────
_models: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models into memory when the server starts."""
    logger.info("🚀 Loading models into memory...")
    try:
        _models.update(load_models())
        loaded = [k for k in _models if k not in ("metrics", "history_df")]
        logger.info(f"✅ Models loaded successfully: {loaded}")
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        # In production, you might want to exit here if models are critical
    
    yield
    
    _models.clear()
    logger.info("👋 Server shutting down — models released.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="🇺🇿 Uzbekistan Payment Volume Forecasting API",
    description="""
    ### 🚀 Production-grade Forecasting System
    This API predicts daily digital payment transaction volumes across Uzbekistan's 
    fast-growing payment ecosystem.
    
    **Key Features:**
    * **Multi-model inference:** Compare Linear Regression, XGBoost, and CatBoost.
    * **Domain-aware:** Incorporates Ramadan, Navruz, and local payday cycles.
    * **Macro-integrated:** Uses real-time USD/UZS exchange rates from the Central Bank.
    * **Explainable:** Models are evaluated using SHAP and rigorous backtesting.
    """,
    version="1.0.0",
    lifespan=lifespan,
    contact={
        "name": "Mukhammadali",
        "url": "https://github.com/Uzbekswe",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# ── Middleware (Hardening) ───────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["localhost", "127.0.0.1", "0.0.0.0"] # Add your domain here
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
