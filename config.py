"""Project configuration for stock-forecast-model."""

from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_CACHE_DIR = PROJECT_ROOT / "data_cache"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TICKER = "AAPL"
START_DATE = "2018-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

TEST_SIZE = 0.2
RANDOM_STATE = 42

PRICE_COLUMN = "Close"

SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (0, 0, 0, 0)

