"""
XGBoost Forecaster
Recursive multi-step forecasting using lag + rolling + calendar features.
"""

import logging
import warnings

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .base import BaseForecaster

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "lag_1w", "lag_4w", "lag_8w", "lag_13w",
    "roll_mean_4w", "roll_std_4w", "roll_mean_8w", "roll_std_8w",
    "roll_mean_13w", "roll_min_13w", "roll_max_13w",
    "day_of_week", "month", "quarter", "week_of_year", "is_holiday",
]


class XGBoostForecaster(BaseForecaster):
    name = "XGBoost"

    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
        self._history: pd.DataFrame = None

    def _get_feature_cols(self, df: pd.DataFrame):
        return [c for c in FEATURE_COLS if c in df.columns]

    def fit(self, train: pd.DataFrame) -> "XGBoostForecaster":
        logger.info(f"Fitting XGBoost on {len(train)} points ...")
        self._history = train.copy()
        feat_cols = self._get_feature_cols(train)
        valid = train.dropna(subset=feat_cols + ["Sales"])
        X = valid[feat_cols]
        y = valid["Sales"]
        self.model.fit(X, y)
        self._feat_cols = feat_cols
        self._last_date = train["Date"].max()
        return self

    def _build_next_row(self, history_sales: list, next_date: pd.Timestamp) -> dict:
        s = pd.Series(history_sales)
        row = {
            "lag_1w": s.iloc[-1] if len(s) >= 1 else np.nan,
            "lag_4w": s.iloc[-4] if len(s) >= 4 else np.nan,
            "lag_8w": s.iloc[-8] if len(s) >= 8 else np.nan,
            "lag_13w": s.iloc[-13] if len(s) >= 13 else np.nan,
            "roll_mean_4w": s.iloc[-4:].mean() if len(s) >= 4 else s.mean(),
            "roll_std_4w": s.iloc[-4:].std() if len(s) >= 4 else 0,
            "roll_mean_8w": s.iloc[-8:].mean() if len(s) >= 8 else s.mean(),
            "roll_std_8w": s.iloc[-8:].std() if len(s) >= 8 else 0,
            "roll_mean_13w": s.iloc[-13:].mean() if len(s) >= 13 else s.mean(),
            "roll_min_13w": s.iloc[-13:].min() if len(s) >= 13 else s.min(),
            "roll_max_13w": s.iloc[-13:].max() if len(s) >= 13 else s.max(),
            "day_of_week": next_date.dayofweek,
            "month": next_date.month,
            "quarter": next_date.quarter,
            "week_of_year": next_date.isocalendar()[1],
            "is_holiday": 0,
        }
        return row

    def predict(self, horizon: int) -> np.ndarray:
        history_sales = list(self._history["Sales"].values)
        last_date = self._last_date
        predictions = []
        for i in range(horizon):
            next_date = last_date + pd.Timedelta(weeks=i + 1)
            row = self._build_next_row(history_sales, next_date)
            feat_vals = [row.get(c, np.nan) for c in self._feat_cols]
            X = pd.DataFrame([feat_vals], columns=self._feat_cols)
            pred = float(self.model.predict(X)[0])
            pred = max(pred, 0)
            predictions.append(pred)
            history_sales.append(pred)
        return np.array(predictions)

    def forecast_dates(self, last_date: pd.Timestamp, horizon: int):
        dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=horizon, freq="W")
        return [d.strftime("%Y-%m-%d") for d in dates]
