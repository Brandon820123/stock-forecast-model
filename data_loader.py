"""Download, cache, and fallback data loading utilities."""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    DATA_CACHE_DIR,
    LOCAL_CSV_PATH,
    PROJECT_ROOT,
    USE_CACHE,
    USE_LOCAL_CSV,
    USE_SAMPLE_DATA_IF_DOWNLOAD_FAILS,
)


def _cache_path(ticker, start_date, end_date):
    safe_ticker = ticker.replace("/", "-").replace("^", "")
    return DATA_CACHE_DIR / f"{safe_ticker}_{start_date}_{end_date}.csv"


def _normalize_yfinance_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _resolve_local_csv_path(csv_path):
    path = Path(csv_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _load_market_csv(csv_path):
    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df = pd.read_csv(csv_path)

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"CSV data missing required columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=required)
    df = df.set_index("Date").sort_index()

    if df.empty:
        raise ValueError("CSV data is empty after cleaning missing values.")

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
    """Load stock OHLCV data.

    Returns:
        tuple[pd.DataFrame, str]: DataFrame and one of
        "local CSV real market data", "cached market data",
        "yfinance real market data", or "sample random-walk data".
    """
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(ticker, start_date, end_date)

    local_csv_file = _resolve_local_csv_path(LOCAL_CSV_PATH)
    if USE_LOCAL_CSV and local_csv_file.exists():
        df = _load_market_csv(local_csv_file)
        print("[DATA] Using local CSV real market data")
        print(f"[DATA] Local CSV path: {local_csv_file}")
        return df, "local CSV real market data"

    if USE_CACHE and cache_file.exists():
        df = _load_market_csv(cache_file)
        print(f"[DATA] Using cached market data: {cache_file}")
        return df, "cached market data"

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
        if USE_CACHE:
            df.to_csv(cache_file)
            print(f"[DATA] Using yfinance real market data; cached to: {cache_file}")
        else:
            print("[DATA] Using yfinance real market data")
        return df, "yfinance real market data"
    except Exception as exc:
        print(f"[DATA] yfinance download failed: {exc}")

        if USE_CACHE and cache_file.exists():
            df = _load_market_csv(cache_file)
            print(f"[DATA] Using cached market data after yfinance failure: {cache_file}")
            return df, "cached market data"

        if not USE_SAMPLE_DATA_IF_DOWNLOAD_FAILS:
            raise RuntimeError(
                "No real market data available. Please provide a CSV file or retry yfinance later."
            ) from exc

        print("[DATA] Using sample random-walk data")
        print(
            "[DATA] WARNING: sample data was used. Results are only for testing workflow, "
            "not real investment analysis."
        )
        df = _generate_sample_data(start_date, end_date)
        return df, "sample random-walk data"
