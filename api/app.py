"""
FastAPI REST API
Exposes the trained forecasting system as a production-grade HTTP service.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from data.preprocessor import DataPreprocessor
from models.registry import ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/api.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state (loaded once at startup)
# ---------------------------------------------------------------------------
DATA_PATH = os.getenv("DATA_PATH", "../data/sales_data.xlsx")
REGISTRY_PATH = os.getenv("REGISTRY_PATH", "../models/saved/registry.pkl")

preprocessor: DataPreprocessor = None
registry: ModelRegistry = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global preprocessor, registry
    logger.info("Starting up - loading data and models ...")
    try:
        preprocessor = DataPreprocessor(DATA_PATH)
        preprocessor.load().process_all()

        registry = ModelRegistry()
        if os.path.exists(REGISTRY_PATH):
            logger.info("Loading pre-trained registry ...")
            registry.load(REGISTRY_PATH)
        else:
            logger.info("No saved registry found - training now (this may take several minutes) ...")
            registry.train_all(preprocessor.processed)
            registry.save(REGISTRY_PATH)
        logger.info("System ready.")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Sales Forecasting API",
    description=(
        "Production-grade time series forecasting system. "
        "Predicts weekly beverage sales per US state using SARIMA, Prophet, XGBoost, and LSTM. "
        "Automatically selects the best-performing model per state."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ForecastPoint(BaseModel):
    week: str
    sales: float


class ForecastResponse(BaseModel):
    state: str
    model_used: str
    forecast: List[ForecastPoint]
    model_comparison: Optional[dict] = None


class HealthResponse(BaseModel):
    status: str
    states_loaded: int
    states_trained: int


class SummaryRow(BaseModel):
    state: str
    best_model: str
    rmse: float
    mae: float
    mape: float


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Health check — confirms data and models are loaded."""
    return {
        "status": "ok",
        "states_loaded": len(preprocessor.processed) if preprocessor else 0,
        "states_trained": len(registry.registry) if registry else 0,
    }


@app.get("/states", tags=["Data"])
def list_states():
    """Return all states available in the dataset."""
    if not preprocessor:
        raise HTTPException(503, "System not ready")
    return {"states": preprocessor.get_all_states()}


@app.get("/forecast", response_model=ForecastResponse, tags=["Forecasting"])
def forecast(
    state: str = Query(..., description="US state name (e.g. 'California')"),
    horizon: int = Query(8, ge=1, le=52, description="Number of weeks to forecast (1–52)"),
    show_comparison: bool = Query(False, description="Include per-model validation metrics"),
):
    """
    Forecast weekly sales for a given state.

    - **state**: Target US state
    - **horizon**: Number of future weeks (default = 8)
    - **show_comparison**: If true, include all model validation metrics
    """
    if not registry:
        raise HTTPException(503, "System not ready — models not trained yet")

    state_clean = state.strip().title()
    available = preprocessor.get_all_states() if preprocessor else []

    # Fuzzy match helper
    if state_clean not in registry.registry:
        close = [s for s in available if state_clean.lower() in s.lower()]
        if close:
            state_clean = close[0]
        else:
            raise HTTPException(
                404,
                f"State '{state}' not found. Available states: {available}",
            )

    try:
        result = registry.forecast(state_clean, horizon)
        if not show_comparison:
            result.pop("model_comparison", None)
        return result
    except KeyError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.exception(f"Forecast error for {state}: {e}")
        raise HTTPException(500, f"Forecasting failed: {str(e)}")


@app.get("/train/{state}", tags=["Training"])
def train_state(state: str):
    """
    (Re-)train models for a specific state and update the registry.
    Useful for incremental updates without retraining everything.
    """
    if not preprocessor or not registry:
        raise HTTPException(503, "System not ready")

    state_clean = state.strip().title()
    try:
        df = preprocessor.get_state(state_clean)
    except KeyError:
        raise HTTPException(404, f"State '{state}' not in dataset")

    try:
        entry = registry.train_state(state_clean, df)
        registry.save(REGISTRY_PATH)
        return {
            "state": state_clean,
            "best_model": entry["model_name"],
            "metrics": entry["metrics"],
        }
    except Exception as e:
        logger.exception(f"Training error for {state}: {e}")
        raise HTTPException(500, str(e))


@app.get("/summary", response_model=List[SummaryRow], tags=["Evaluation"])
def model_summary():
    """
    Return a leaderboard of best models per state, sorted by RMSE ascending.
    """
    if not registry:
        raise HTTPException(503, "System not ready")
    return registry.summary()


@app.get("/retrain", tags=["Training"])
def retrain_all():
    """Re-train all states from scratch and save updated registry."""
    if not preprocessor or not registry:
        raise HTTPException(503, "System not ready")
    try:
        registry.train_all(preprocessor.processed)
        registry.save(REGISTRY_PATH)
        return {"status": "success", "trained_states": len(registry.registry)}
    except Exception as e:
        logger.exception(f"Full retrain failed: {e}")
        raise HTTPException(500, str(e))
