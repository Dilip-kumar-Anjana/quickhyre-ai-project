"""
SARIMA Model
Uses auto-selection of (p,d,q)(P,D,Q,s) parameters via AIC.
"""

import logging
import warnings
from itertools import product

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .base import BaseForecaster

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class SARIMAForecaster(BaseForecaster):
    name = "SARIMA"

    def __init__(self):
        self.model_fit = None
        self.best_order = None
        self.best_seasonal = None

    def _select_order(self, series: pd.Series):
        """
        Select SARIMA order via a focused grid search.
        Tries a small set of strong candidates to balance accuracy and speed.
        """
        best_aic = np.inf
        best_order = (1, 1, 1)
        best_seasonal = (1, 1, 1, 52)

        S = 52  # weekly seasonality
        candidates = [
            ((1, 1, 1), (1, 1, 1, S)),
            ((1, 1, 0), (1, 1, 0, S)),
            ((0, 1, 1), (0, 1, 1, S)),
            ((2, 1, 1), (1, 1, 0, S)),
            ((1, 1, 2), (0, 1, 1, S)),
            ((1, 0, 0), (1, 0, 0, S)),
        ]

        for order, seasonal in candidates:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = SARIMAX(
                        series,
                        order=order,
                        seasonal_order=seasonal,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False, maxiter=80)
                if m.aic < best_aic:
                    best_aic = m.aic
                    best_order = order
                    best_seasonal = seasonal
            except Exception:
                continue

        logger.debug(f"SARIMA best order: {best_order} x {best_seasonal} (AIC={best_aic:.1f})")
        return best_order, best_seasonal

    def fit(self, train: pd.DataFrame) -> "SARIMAForecaster":
        series = train.set_index("Date")["Sales"].asfreq("W").interpolate()
        logger.info(f"Fitting SARIMA on {len(series)} points ...")
        self.best_order, self.best_seasonal = self._select_order(series)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_fit = SARIMAX(
                series,
                order=self.best_order,
                seasonal_order=self.best_seasonal,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=150)
        self._last_date = series.index[-1]
        return self

    def predict(self, horizon: int) -> np.ndarray:
        fc = self.model_fit.forecast(steps=horizon)
        return np.maximum(fc.values, 0)

    def forecast_dates(self, last_date: pd.Timestamp, horizon: int):
        import pandas as pd
        dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=horizon, freq="W")
        return [d.strftime("%Y-%m-%d") for d in dates]
