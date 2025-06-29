"""Backtesting module for testing trading strategies with trained models."""

from .engine import (
    backtest_model_strategy_impl,
    BacktestPortfolio,
    calculate_max_drawdown,
    calculate_win_rate
)
from .analyzers import (
    backtest_multiple_models_impl,
    discover_models,
    load_model_metadata,
    extract_metadata_from_filename,
    enhance_with_model_metadata,
    generate_model_comparison_matrix,
    calculate_model_rankings
)
from .strategies import (
    BaseTradingStrategy,
    DirectionalStrategy,
    ThresholdStrategy,
    PercentileStrategy,
    MomentumStrategy,
    VolatilityAdjustedStrategy,
    MeanReversionStrategy,
    get_strategy,
    get_available_strategies
)

__all__ = [
    "backtest_model_strategy_impl",
    "backtest_multiple_models_impl",
    "BacktestPortfolio",
    "calculate_max_drawdown",
    "calculate_win_rate",
    "discover_models",
    "load_model_metadata",
    "extract_metadata_from_filename",
    "enhance_with_model_metadata",
    "generate_model_comparison_matrix",
    "calculate_model_rankings",
    "BaseTradingStrategy",
    "DirectionalStrategy",
    "ThresholdStrategy",
    "PercentileStrategy",
    "MomentumStrategy",
    "VolatilityAdjustedStrategy",
    "MeanReversionStrategy",
    "get_strategy",
    "get_available_strategies"
]