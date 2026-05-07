"""
Facebook Prophet Model
Captures trend + weekly + yearly seasonality + holidays.
"""

import logging
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class ProphetForecaster:
    name = "Prophet"

    def __init__(self):
        self.model = None
        self._last_date = None

    def fit(self, train: pd.DataFrame) -> "ProphetForecaster":
        from prophet import Prophet

        logger.info(f"Fitting Prophet on {len(train)} points ...")
        df_prophet = train[["Date", "Sales"]].rename(columns={"Date": "ds", "Sales": "y"})
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
            changepoint_prior_scale=0.05,
        )
        self.model.add_country_holidays(country_name="US")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(df_prophet)
        self._last_date = train["Date"].max()
        return self

    def predict(self, horizon: int) -> np.ndarray:
        future = self.model.make_future_dataframe(periods=horizon, freq="W")
        forecast = self.model.predict(future)
        preds = forecast["yhat"].tail(horizon).values
        return np.maximum(preds, 0)

    def forecast_dates(self, last_date: pd.Timestamp, horizon: int):
        import pandas as pd
        dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=horizon, freq="W")
        return [d.strftime("%Y-%m-%d") for d in dates]
