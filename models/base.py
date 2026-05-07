"""
Base Forecaster Interface
Defines the common interface for all forecasting models.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseForecaster(ABC):
    """Abstract base class for all forecasting models."""
    
    name: str = "BaseForecaster"
    
    @abstractmethod
    def fit(self, train: pd.DataFrame) -> "BaseForecaster":
        """
        Fit the model on training data.
        
        Args:
            train: DataFrame with 'Date' and 'Sales' columns
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        """
        Generate forecasts for the specified horizon.
        
        Args:
            horizon: Number of periods to forecast
            
        Returns:
            Array of predictions
        """
        pass
    
    def forecast_dates(self, last_date: pd.Timestamp, horizon: int):
        """
        Generate forecast dates for the predictions.
        
        Args:
            last_date: Last date in the training data
            horizon: Number of periods to forecast
            
        Returns:
            List of date strings in YYYY-MM-DD format
        """
        dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=horizon, freq="W")
        return [d.strftime("%Y-%m-%d") for d in dates]
