"""
Data Preprocessing Module
Handles loading, cleaning, resampling, and feature engineering for time series data.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import holidays
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles all data preprocessing and feature engineering for the forecasting system."""

    def __init__(self, data_path: str, country: str = "US"):
        self.data_path = Path(data_path)
        self.country = country
        self.raw_df: Optional[pd.DataFrame] = None
        self.processed: Dict[str, pd.DataFrame] = {}
        self._holiday_cache: Dict[int, set] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> "DataPreprocessor":
        logger.info(f"Loading data from {self.data_path}")
        ext = self.data_path.suffix.lower()
        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(self.data_path)
        elif ext == ".csv":
            df = pd.read_csv(self.data_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date"])
        df["Total"] = pd.to_numeric(df["Total"], errors="coerce")
        df = df.dropna(subset=["Total"])
        df["State"] = df["State"].astype(str).str.strip()
        self.raw_df = df.sort_values(["State", "Date"]).reset_index(drop=True)
        logger.info(
            f"Loaded {len(self.raw_df)} rows | {self.raw_df['State'].nunique()} states | "
            f"{self.raw_df['Date'].min().date()} -> {self.raw_df['Date'].max().date()}"
        )
        return self

    # ------------------------------------------------------------------
    # Per-state resampling to weekly frequency
    # ------------------------------------------------------------------

    def _resample_state(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.set_index("Date").sort_index()
        # Resample to weekly (Sunday-anchored) and sum
        weekly = df["Total"].resample("W").sum()
        # Forward-fill short gaps (≤2 weeks), interpolate longer ones
        weekly = weekly.replace(0, np.nan)
        weekly = weekly.interpolate(method="time", limit=2).ffill().bfill()
        result = weekly.reset_index()
        result.columns = ["Date", "Sales"]
        return result

    # ------------------------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------------------------

    def _get_holidays(self, years) -> set:
        us_holidays = set()
        for yr in years:
            if yr not in self._holiday_cache:
                self._holiday_cache[yr] = set(holidays.US(years=yr).keys())
            us_holidays |= self._holiday_cache[yr]
        return us_holidays

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_values("Date").reset_index(drop=True)

        # Date-based features
        df["day_of_week"] = df["Date"].dt.dayofweek
        df["month"] = df["Date"].dt.month
        df["quarter"] = df["Date"].dt.quarter
        df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
        df["year"] = df["Date"].dt.year

        # Holiday indicator (any US holiday within the week)
        years = df["Date"].dt.year.unique()
        holiday_dates = self._get_holidays(years)
        df["is_holiday"] = df["Date"].apply(
            lambda d: int(
                any(
                    (d + pd.Timedelta(days=i)).date() in holiday_dates
                    for i in range(7)
                )
            )
        )

        # Lag features (in weeks)
        for lag in [1, 4, 8, 13, 26, 52]:
            df[f"lag_{lag}w"] = df["Sales"].shift(lag)

        # Rolling statistics (on past data only)
        for window in [4, 8, 13]:
            roll = df["Sales"].shift(1).rolling(window, min_periods=max(1, window // 2))
            df[f"roll_mean_{window}w"] = roll.mean()
            df[f"roll_std_{window}w"] = roll.std()
            df[f"roll_min_{window}w"] = roll.min()
            df[f"roll_max_{window}w"] = roll.max()

        # Year-over-year change
        df["yoy_lag"] = df["Sales"].shift(52)
        df["yoy_pct"] = (df["Sales"] - df["yoy_lag"]) / (df["yoy_lag"].replace(0, np.nan))

        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_all(self) -> Dict[str, pd.DataFrame]:
        if self.raw_df is None:
            self.load()
        for state, grp in self.raw_df.groupby("State"):
            weekly = self._resample_state(grp)
            featured = self._add_features(weekly)
            self.processed[state] = featured
            logger.debug(f"  {state}: {len(featured)} weekly rows")
        logger.info(f"Processed {len(self.processed)} states")
        return self.processed

    def get_state(self, state: str) -> pd.DataFrame:
        if state not in self.processed:
            raise KeyError(f"State '{state}' not found. Available: {sorted(self.processed.keys())}")
        return self.processed[state]

    def get_all_states(self):
        return sorted(self.processed.keys())

    @staticmethod
    def train_val_split(
        df: pd.DataFrame, val_weeks: int = 8
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Strict chronological split — no leakage."""
        split_idx = len(df) - val_weeks
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
