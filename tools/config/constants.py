"""Global constants and configuration values for the stock analyzer package."""

import os
from pathlib import Path

# Directory configuration
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# File naming patterns
MODEL_FILE_PATTERN = "train_{model_type}_price_predictor_{symbol}_model_{timestamp}.pkl"
RESULTS_FILE_PATTERN = "train_{model_type}_price_predictor_{symbol}_results_{timestamp}.json"
PREDICTIONS_FILE_PATTERN = "{model_type}_predictions_{symbol}_{timestamp}.csv"
BACKTEST_FILE_PATTERN = "backtest_{model_type}_{symbol}_{strategy_type}_{timestamp}.json"
VISUALIZATION_FILE_PATTERN = "{chart_type}_{symbol}_{timestamp}.html"

# Data fetching defaults
DEFAULT_PERIOD = "1y"
DEFAULT_INTERVAL = "1d"
VALID_PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
VALID_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]

# Model training defaults
DEFAULT_TEST_SIZE = 0.2
DEFAULT_TARGET_DAYS = 1
DEFAULT_RANDOM_STATE = 42

# Backtesting defaults
DEFAULT_INITIAL_CAPITAL = 10000.0
DEFAULT_TRANSACTION_COST = 0.001
DEFAULT_STRATEGY_TYPE = "directional"

# Technical indicator defaults
SMA_PERIODS = [20, 50, 200]
EMA_PERIODS = [12, 26]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

# Model type identifiers
MODEL_TYPES = {
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "svr": "Support Vector Regression",
    "gradient_boosting": "Gradient Boosting",
    "ridge_regression": "Ridge Regression",
    "extra_trees": "Extra Trees"
}

# Chart type identifiers
CHART_TYPES = {
    "line": "Line Chart",
    "candlestick": "Candlestick Chart",
    "volume": "Volume Chart",
    "backtesting": "Backtesting Chart",
    "comparison": "Model Comparison Chart"
}

# Logging configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Performance thresholds
MIN_R2_SCORE = 0.0
MAX_RMSE_RATIO = 2.0  # RMSE should not be more than 2x the price range
MIN_DIRECTIONAL_ACCURACY = 50.0  # Minimum 50% directional accuracy

# File size limits
MAX_CSV_SIZE_MB = 100
MAX_MODEL_SIZE_MB = 500

# Cache configuration
CACHE_EXPIRY_MINUTES = 15
ENABLE_CACHING = True