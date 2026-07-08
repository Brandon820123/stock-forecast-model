"""Feature engineering for price direction prediction."""

import numpy as np
import pandas as pd


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


def build_features(df):
    """Create features and next-day direction label without feature leakage."""
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

    data["next_close"] = data["Close"].shift(-1)
    data["label"] = (data["next_close"] > data["Close"]).astype(int)

    model_data = data.dropna(subset=FEATURE_COLUMNS + ["next_close", "label"]).copy()
    model_data[FEATURE_COLUMNS] = model_data[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    model_data = model_data.dropna(subset=FEATURE_COLUMNS + ["label"])

    return model_data

