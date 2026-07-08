"""Download, cache, and fallback data loading utilities."""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from config import DATA_CACHE_DIR


def _cache_path(ticker, start_date, end_date):
    safe_ticker = ticker.replace("/", "-").replace("^", "")
    return DATA_CACHE_DIR / f"{safe_ticker}_{start_date}_{end_date}.csv"


def _normalize_yfinance_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _generate_sample_data(start_date, end_date, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, end=end_date)
    if len(dates) < 80:
        dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=260)

    returns = rng.normal(loc=0.0004, scale=0.018, size=len(dates))
    close = 150 * np.exp(np.cumsum(returns))
    open_price = close * (1 + rng.normal(0, 0.004, len(dates)))
    high = np.maximum(open_price, close) * (1 + np.abs(rng.normal(0.004, 0.006, len(dates))))
    low = np.minimum(open_price, close) * (1 - np.abs(rng.normal(0.004, 0.006, len(dates))))
    volume = rng.integers(40_000_000, 130_000_000, size=len(dates))

    df = pd.DataFrame(
        {
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Adj Close": close,
            "Volume": volume,
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


def load_stock_data(ticker, start_date, end_date):
    """Load stock OHLCV data, preferring cache over yfinance.

    Returns:
        tuple[pd.DataFrame, str]: DataFrame and one of
        "cache data", "yfinance real data", or "sample data".
    """
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(ticker, start_date, end_date)

    if cache_file.exists():
        df = pd.read_csv(cache_file, parse_dates=["Date"], index_col="Date")
        print(f"[DATA] Current data source: cache data ({cache_file})")
        return df.sort_index(), "cache data"

    try:
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        df = _normalize_yfinance_columns(df)
        if df.empty:
            raise ValueError("yfinance returned an empty DataFrame")

        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"yfinance data missing columns: {missing}")

        df = df.dropna(subset=["Close"]).sort_index()
        df.to_csv(cache_file)
        print(f"[DATA] Current data source: yfinance real data (cached to {cache_file})")
        return df, "yfinance real data"
    except Exception as exc:
        print(f"[DATA] yfinance download failed: {exc}")
        print("[DATA] Current data source: sample data")
        print("[DATA] WARNING: sample data is only for testing code flow, not real investment analysis.")
        df = _generate_sample_data(start_date, end_date)
        return df, "sample data"

