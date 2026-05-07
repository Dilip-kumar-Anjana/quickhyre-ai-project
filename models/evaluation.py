"""
Evaluation Metrics
Computes RMSE, MAE, MAPE, and SMAPE metrics for time series forecasts.
"""

import numpy as np


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    """Root Mean Squared Error"""
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    """Mean Absolute Error"""
    return float(np.mean(np.abs(actual - pred)))


def mape(actual: np.ndarray, pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%)"""
    # Avoid division by zero
    mask = actual != 0
    if not np.any(mask):
        return 0.0
    return float(100 * np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])))


def smape(actual: np.ndarray, pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (%)"""
    denominator = (np.abs(actual) + np.abs(pred)) / 2.0
    mask = denominator != 0
    if not np.any(mask):
        return 0.0
    return float(100 * np.mean(np.abs(actual[mask] - pred[mask]) / denominator[mask]))


def evaluate(actual: np.ndarray, pred: np.ndarray) -> dict:
    """
    Compute all metrics for a forecast.
    
    Args:
        actual: Array of actual values
        pred: Array of predicted values
        
    Returns:
        Dictionary with RMSE, MAE, MAPE, and SMAPE metrics
    """
    actual = np.array(actual, dtype=float)
    pred = np.array(pred, dtype=float)
    
    return {
        "rmse": rmse(actual, pred),
        "mae": mae(actual, pred),
        "mape": mape(actual, pred),
        "smape": smape(actual, pred),
    }
