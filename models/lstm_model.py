"""
LSTM Forecaster
Sequence-to-one LSTM for weekly sales forecasting with recursive multi-step prediction.
"""

import logging
import os
import warnings

import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

LOOKBACK = 26  # weeks of history used as input sequence


class LSTMForecaster:
    name = "LSTM"

    def __init__(self, lookback: int = LOOKBACK, epochs: int = 60, batch_size: int = 16):
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self._scaler_min = None
        self._scaler_max = None
        self._history_scaled = None
        self._last_date = None

    def _scale(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self._scaler_min) / (self._scaler_max - self._scaler_min + 1e-8)

    def _unscale(self, arr: np.ndarray) -> np.ndarray:
        return arr * (self._scaler_max - self._scaler_min + 1e-8) + self._scaler_min

    def _build_sequences(self, scaled: np.ndarray):
        X, y = [], []
        for i in range(self.lookback, len(scaled)):
            X.append(scaled[i - self.lookback : i])
            y.append(scaled[i])
        return np.array(X)[..., np.newaxis], np.array(y)

    def fit(self, train: pd.DataFrame) -> "LSTMForecaster":
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import Dense, Dropout, LSTM
        from tensorflow.keras.models import Sequential

        tf.get_logger().setLevel("ERROR")
        logger.info(f"Fitting LSTM on {len(train)} points ...")

        sales = train["Sales"].values.astype(float)
        self._scaler_min = sales.min()
        self._scaler_max = sales.max()
        scaled = self._scale(sales)
        self._history_scaled = scaled.copy()
        self._last_date = train["Date"].max()

        if len(scaled) <= self.lookback:
            logger.warning("Not enough data for LSTM; reducing lookback.")
            self.lookback = max(4, len(scaled) // 2)

        X, y = self._build_sequences(scaled)
        if len(X) == 0:
            raise ValueError("No sequences generated for LSTM training.")

        self.model = Sequential([
            LSTM(64, input_shape=(self.lookback, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        self.model.compile(optimizer="adam", loss="mse")
        cb = EarlyStopping(patience=10, restore_best_weights=True, verbose=0)
        self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=[cb],
            verbose=0,
        )
        return self

    def predict(self, horizon: int) -> np.ndarray:
        buffer = list(self._history_scaled[-self.lookback :])
        predictions_scaled = []
        for _ in range(horizon):
            seq = np.array(buffer[-self.lookback :])[np.newaxis, :, np.newaxis]
            pred_s = float(self.model.predict(seq, verbose=0)[0][0])
            predictions_scaled.append(pred_s)
            buffer.append(pred_s)
        return np.maximum(self._unscale(np.array(predictions_scaled)), 0)

    def forecast_dates(self, last_date: pd.Timestamp, horizon: int):
        dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=horizon, freq="W")
        return [d.strftime("%Y-%m-%d") for d in dates]
