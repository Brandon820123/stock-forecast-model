"""Feature engineering for price direction prediction."""

import numpy as np
import pandas as pd

from config import FUTURE_DIRECTION_DAYS


FEATURE_COLUMNS = [
    "daily_return",
    "MA5",
    "MA10",
    "MA20",
    "volatility_5",
    "volatility_20",
    "momentum_5",
    "momentum_10",
    "RSI",
    "MACD",
    "volume_change",
    "return_lag_1",
    "return_lag_2",
    "return_lag_5",
    "rolling_mean_return_5",
    "rolling_mean_return_20",
    "rolling_volatility_10",
    "rolling_volatility_20",
    "price_vs_ma20",
    "volume_zscore_20",
]


def calculate_rsi(close, window=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calculate_macd(close, fast=12, slow=26):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow


def build_features(df, future_direction_days=FUTURE_DIRECTION_DAYS):
    """Create leakage-safe features and direction labels.

    Rolling features only use the current row and earlier rows. Future prices are
    used only to build labels, never as model features.
    """
    data = df.copy().sort_index()

    data["daily_return"] = data["Close"].pct_change()
    data["MA5"] = data["Close"].rolling(window=5, min_periods=5).mean()
    data["MA10"] = data["Close"].rolling(window=10, min_periods=10).mean()
    data["MA20"] = data["Close"].rolling(window=20, min_periods=20).mean()
    data["volatility_5"] = data["daily_return"].rolling(window=5, min_periods=5).std()
    data["volatility_20"] = data["daily_return"].rolling(window=20, min_periods=20).std()
    data["momentum_5"] = data["Close"] / data["Close"].shift(5) - 1
    data["momentum_10"] = data["Close"] / data["Close"].shift(10) - 1
    data["RSI"] = calculate_rsi(data["Close"])
    data["MACD"] = calculate_macd(data["Close"])
    data["volume_change"] = data["Volume"].pct_change()
    data["return_lag_1"] = data["daily_return"].shift(1)
    data["return_lag_2"] = data["daily_return"].shift(2)
    data["return_lag_5"] = data["daily_return"].shift(5)
    data["rolling_mean_return_5"] = data["daily_return"].rolling(window=5, min_periods=5).mean()
    data["rolling_mean_return_20"] = data["daily_return"].rolling(window=20, min_periods=20).mean()
    data["rolling_volatility_10"] = data["daily_return"].rolling(window=10, min_periods=10).std()
    data["rolling_volatility_20"] = data["daily_return"].rolling(window=20, min_periods=20).std()
    data["price_vs_ma20"] = data["Close"] / data["MA20"] - 1

    volume_mean_20 = data["Volume"].rolling(window=20, min_periods=20).mean()
    volume_std_20 = data["Volume"].rolling(window=20, min_periods=20).std()
    data["volume_zscore_20"] = (data["Volume"] - volume_mean_20) / volume_std_20.replace(0, np.nan)

    data["next_close"] = data["Close"].shift(-1)
    data["future_close_5d"] = data["Close"].shift(-future_direction_days)
    data["label_next_day"] = (data["next_close"] > data["Close"]).astype(int)
    data["label_5d"] = (data["future_close_5d"] > data["Close"]).astype(int)
    data["label"] = data["label_5d"]

    label_columns = ["next_close", "future_close_5d", "label_next_day", "label_5d"]
    model_data = data.dropna(subset=FEATURE_COLUMNS + label_columns).copy()
    model_data[FEATURE_COLUMNS] = model_data[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    model_data = model_data.dropna(subset=FEATURE_COLUMNS + ["label_next_day", "label_5d"])

    return model_data
