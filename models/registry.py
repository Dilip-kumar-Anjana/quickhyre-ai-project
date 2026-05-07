"""
Model Registry & Trainer
Trains all models per state, evaluates on the validation fold, and picks the winner.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .evaluation import evaluate
from .lstm_model import LSTMForecaster
from .prophet_model import ProphetForecaster
from .sarima_model import SARIMAForecaster
from .xgboost_model import XGBoostForecaster

logger = logging.getLogger(__name__)

VAL_WEEKS = 8
PRIMARY_METRIC = "rmse"  # used for best-model selection


def _instantiate_models(preferred_model: Optional[str] = None):
    models = {
        "sarima": SARIMAForecaster(),
        "prophet": ProphetForecaster(),
        "xgboost": XGBoostForecaster(),
        "lstm": LSTMForecaster(epochs=60),
    }

    if preferred_model:
        key = preferred_model.strip().lower()
        if key not in models:
            raise ValueError(f"Unknown model '{preferred_model}'.")
        if key == "lstm":
            models[key] = LSTMForecaster(epochs=3)
        return [models[key]]

    return [
        models["sarima"],
        models["prophet"],
        models["xgboost"],
        models["lstm"],
    ]


class ModelRegistry:
    """Trains, evaluates, selects, and serves forecasts for every state."""

    def __init__(self, model_dir: str = "models/saved", preferred_model: Optional[str] = None):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.preferred_model = preferred_model
        # state -> {"model": fitted_model, "metrics": {...}, "last_date": ...}
        self.registry: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_state(
        self,
        state: str,
        df: pd.DataFrame,
        val_weeks: int = VAL_WEEKS,
    ) -> dict:
        logger.info(f"=== Training state: {state} ===")
        split = len(df) - val_weeks
        train = df.iloc[:split].copy()
        val = df.iloc[split:].copy()
        actual = val["Sales"].values

        results = {}
        best_model = None
        best_metric = float("inf")
        best_name = None

        for model in _instantiate_models(self.preferred_model):
            try:
                model.fit(train)
                preds = model.predict(val_weeks)
                metrics = evaluate(actual, preds)
                results[model.name] = metrics
                logger.info(
                    f"  {model.name:10s}  RMSE={metrics['rmse']:>12,.0f}  "
                    f"MAE={metrics['mae']:>12,.0f}  MAPE={metrics['mape']:.2f}%"
                )
                if metrics[PRIMARY_METRIC] < best_metric:
                    best_metric = metrics[PRIMARY_METRIC]
                    best_model = model
                    best_name = model.name
            except Exception as exc:
                logger.warning(f"  {model.name} failed: {exc}")
                results[model.name] = {"error": str(exc)}

        if best_model is None:
            raise RuntimeError(f"All models failed for state: {state}")

        # Refit best model on FULL data (train + val)
        logger.info(f"  -> Best: {best_name}. Refitting on full data.")
        best_model.fit(df)

        entry = {
            "model": best_model,
            "model_name": best_name,
            "metrics": results,
            "last_date": df["Date"].max(),
            "all_metrics": results,
        }
        self.registry[state] = entry
        return entry

    def train_all(self, processed: Dict[str, pd.DataFrame]):
        for state, df in processed.items():
            try:
                self.train_state(state, df)
            except Exception as e:
                logger.error(f"Skipping {state}: {e}")

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    def forecast(self, state: str, horizon: int = 8) -> dict:
        if state not in self.registry:
            raise KeyError(f"State '{state}' not trained yet.")

        entry = self.registry[state]
        model = entry["model"]
        last_date = entry["last_date"]

        preds = model.predict(horizon)
        dates = model.forecast_dates(last_date, horizon)

        return {
            "state": state,
            "model_used": entry["model_name"],
            "forecast": [
                {"week": d, "sales": round(float(v), 2)}
                for d, v in zip(dates, preds)
            ],
            "model_comparison": {
                name: {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}
                for name, metrics in entry["all_metrics"].items()
            },
        }

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------

    def summary(self) -> List[dict]:
        rows = []
        for state, entry in self.registry.items():
            best_metrics = entry["metrics"].get(entry["model_name"], {})
            rows.append(
                {
                    "state": state,
                    "best_model": entry["model_name"],
                    "rmse": round(best_metrics.get("rmse", float("nan")), 2),
                    "mae": round(best_metrics.get("mae", float("nan")), 2),
                    "mape": round(best_metrics.get("mape", float("nan")), 2),
                }
            )
        return sorted(rows, key=lambda r: r["rmse"])

    def save(self, path: Optional[str] = None):
        out = Path(path or self.model_dir / "registry.pkl")
        with open(out, "wb") as f:
            pickle.dump(self.registry, f)
        logger.info(f"Registry saved -> {out}")

    def load(self, path: Optional[str] = None):
        src = Path(path or self.model_dir / "registry.pkl")
        with open(src, "rb") as f:
            self.registry = pickle.load(f)
        logger.info(f"Registry loaded <- {src}")
