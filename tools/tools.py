"""Tools facade - Re-exports all @tool decorated functions for backward compatibility."""

import os
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.offline import plot
from datetime import datetime, timedelta
from typing import Optional, Literal, Dict, Any, List
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
import pickle
import json
import numpy as np

# Import implementations from refactored modules
from .config import OUTPUT_DIR, logger, PARAMETER_SCHEMAS, MODELING_CONTEXTS
from .data import (
    fetch_yahoo_finance_data_impl,
    get_available_stock_periods_and_intervals_impl,
    read_csv_data_impl,
    apply_technical_indicators_and_transformations_impl,
    prepare_model_data
)
from .models.base import train_model_pipeline
from .backtesting import (
    backtest_model_strategy_impl,
    backtest_multiple_models_impl
)
from .visualization import (
    visualize_stock_data_impl,
    visualize_model_comparison_backtesting_impl,
    visualize_backtesting_results_impl,
    generate_comprehensive_html_report_impl
)
from .utils import (
    list_saved_stock_files_impl,
    save_text_to_file_impl,
    debug_file_system_impl,
    validate_model_parameters_impl,
    get_model_selection_guide_impl
)

# Tavily tool (external dependency)
tavily_tool = TavilySearchResults(max_results=5)

# =============================================================================
# DATA FETCHING AND PROCESSING TOOLS
# =============================================================================

@tool
def fetch_yahoo_finance_data(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    save_data: bool = True
) -> str:
    """
    Fetch stock data from Yahoo Finance with comprehensive error handling and validation.
    
    This function provides robust data fetching with automatic retries, data validation,
    and comprehensive reporting. It handles various edge cases and provides detailed
    feedback about the data quality and characteristics.
    
    Args:
        symbol (str): Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'TSLA', 'MSFT')
                     Automatically converts to uppercase for consistency
        
        period (str): Data period to retrieve
                     Valid options: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
                     - '1d': 1 day (intraday data only)
                     - '5d': 5 days 
                     - '1mo': 1 month (RECOMMENDED for short-term analysis)
                     - '3mo': 3 months
                     - '6mo': 6 months (RECOMMENDED for medium-term analysis)
                     - '1y': 1 year (DEFAULT - good balance)
                     - '2y': 2 years (RECOMMENDED for long-term analysis)
                     - '5y': 5 years (comprehensive historical data)
                     - '10y': 10 years (extensive historical analysis)
                     - 'ytd': Year to date
                     - 'max': Maximum available data
        
        interval (str): Data interval granularity
                       Valid options: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
                       - Minute intervals ('1m'-'90m'): Only available for last 60 days
                       - '1h': Hourly data (last 730 days)
                       - '1d': Daily data (DEFAULT - most reliable)
                       - '1wk': Weekly data (long-term trends)
                       - '1mo': Monthly data (very long-term analysis)
        
        save_data (bool): Whether to save data to CSV file with timestamp
                         TRUE RECOMMENDED for reproducible analysis
    
    Returns:
        str: Comprehensive data summary including:
             - Data quality metrics (completeness, validity)
             - Price statistics (range, volatility, trends)
             - Volume analysis (average, patterns)
             - Technical indicators preview
             - File save confirmation and location
             - Recommendations for further analysis
    """
    return fetch_yahoo_finance_data_impl(symbol, period, interval, save_data)


@tool
def get_available_stock_periods_and_intervals() -> str:
    """
    Get comprehensive information about available periods and intervals for Yahoo Finance data fetching.
    
    This function provides detailed guidance on data availability, limitations, and best practices
    for different types of analysis. Essential for AI agents to make informed decisions about
    data fetching parameters.
    
    Returns:
        str: Comprehensive guide covering:
             - Available periods with use case recommendations
             - Available intervals with data availability windows
             - Best practice combinations for different analysis types
             - Limitations and restrictions
             - Performance considerations
    """
    return get_available_stock_periods_and_intervals_impl()


@tool
def read_csv_data(
    filename: str,
    max_rows: int = 100,
    filepath: Optional[str] = None
) -> str:
    """
    Read and analyze CSV data from any location or the output directory.
    This allows the AI agent to examine stock data and extract insights.
    
    Args:
        filename: Name of the CSV file to read (include .csv extension)
        max_rows: Maximum number of rows to display (default 100, set to -1 for all)
        filepath: Full path to the file (if None, uses output directory)
        
    Returns:
        String with data summary, statistics, and sample data
    """
    return read_csv_data_impl(filename, max_rows, filepath)


@tool
def apply_technical_indicators_and_transformations(
    symbol: str,
    indicators: str = "sma_20,ema_12,rsi,macd,bollinger,volume_sma",
    source_file: Optional[str] = None,
    period: str = "3mo",
    save_results: bool = True
) -> str:
    """
    Apply various technical indicators and transformations to stock data.
    Can work with existing saved data files or fetch fresh data.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')
        indicators: Comma-separated list of indicators/transformations to apply.
                   Available options:
                   - sma_X: Simple Moving Average (X days, e.g., sma_20, sma_50, sma_200)
                   - ema_X: Exponential Moving Average (X days, e.g., ema_12, ema_26)
                   - rsi: Relative Strength Index (14-day default)
                   - rsi_X: RSI with custom period (e.g., rsi_30)
                   - macd: MACD indicator (12,26,9 default)
                   - bollinger: Bollinger Bands (20-day, 2 std dev)
                   - bollinger_X_Y: Custom Bollinger (X days, Y std dev)
                   - returns: Daily returns (percentage)
                   - log_returns: Logarithmic returns
                   - volatility: Rolling volatility (20-day default)
                   - volatility_X: Rolling volatility (X days)
                   - volume_sma_X: Volume moving average
                   - price_momentum_X: Price momentum (X days)
                   - support_resistance: Basic support/resistance levels
        source_file: Specific CSV file to use (if None, uses most recent or fetches new)
        period: Period for new data fetch if no source file specified
        save_results: Whether to save the enhanced data to a new CSV file
        
    Returns:
        String description of applied indicators and file location
    """
    return apply_technical_indicators_and_transformations_impl(
        symbol, indicators, source_file, period, save_results
    )


# =============================================================================
# MODEL TRAINING TOOLS (Using refactored pipeline)
# =============================================================================

@tool
def train_xgboost_price_predictor(
    symbol: str,
    source_file: Optional[str] = None,
    target_days: int = 1,
    test_size: float = 0.2,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    save_model: bool = True,
    save_predictions: bool = True
) -> str:
    """
    Train an XGBoost model to predict stock prices using technical indicators.
    
    XGBoost (Extreme Gradient Boosting) is a highly optimized gradient boosting framework
    designed for speed and performance. It uses gradient boosted decision trees and has
    built-in regularization to prevent overfitting.
    """
    def create_xgboost_model(n_estimators, max_depth, learning_rate, **kwargs):
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            n_jobs=-1
        )
    
    return train_model_pipeline(
        symbol=symbol,
        model_type='xgboost',
        model_factory_func=create_xgboost_model,
        source_file=source_file,
        target_days=target_days,
        test_size=test_size,
        save_model=save_model,
        save_predictions=save_predictions,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate
    )


@tool
def train_random_forest_price_predictor(
    symbol: str,
    source_file: Optional[str] = None,
    target_days: int = 1,
    test_size: float = 0.2,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    save_model: bool = True,
    save_predictions: bool = True
) -> str:
    """
    Train a Random Forest model to predict stock prices using technical indicators.
    
    Random Forest is a robust ensemble learning method that combines multiple decision
    trees using bootstrap aggregating (bagging). It provides excellent stability,
    interpretability, and resistance to overfitting.
    """
    def create_random_forest_model(n_estimators, max_depth, min_samples_split, **kwargs):
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1
        )
    
    return train_model_pipeline(
        symbol=symbol,
        model_type='random_forest',
        model_factory_func=create_random_forest_model,
        source_file=source_file,
        target_days=target_days,
        test_size=test_size,
        save_model=save_model,
        save_predictions=save_predictions,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split
    )


# =============================================================================
# BACKTESTING TOOLS
# =============================================================================

@tool
def backtest_model_strategy(
    symbol: str,
    model_file: str,
    data_file: Optional[str] = None,
    initial_capital: float = 10000.0,
    strategy_type: Literal["threshold", "directional", "percentile"] = "directional",
    threshold: float = 0.02,
    transaction_cost: float = 0.001,
    save_results: bool = True
) -> str:
    """
    Backtest a trained model's predictions using various trading strategies.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')
        model_file: Trained model file (.pkl) to use for predictions
        data_file: Enhanced CSV file with technical indicators (if None, uses most recent)
        initial_capital: Starting capital for backtesting ($10,000 default)
        strategy_type: Trading strategy type:
                      - "threshold": Buy if predicted return > threshold, sell if < -threshold
                      - "directional": Buy if predicted price > current, sell if < current
                      - "percentile": Buy/sell based on prediction percentiles
        threshold: Threshold for buy/sell signals (used in threshold strategy)
        transaction_cost: Transaction cost as percentage (0.001 = 0.1%)
        save_results: Whether to save detailed backtest results
        
    Returns:
        String with comprehensive backtesting results and performance metrics
    """
    return backtest_model_strategy_impl(
        symbol, model_file, data_file, initial_capital, strategy_type, threshold, transaction_cost, save_results
    )


@tool
def backtest_multiple_models(
    symbol: str,
    strategy_type: Literal["threshold", "directional", "percentile"] = "directional",
    initial_capital: float = 10000.0,
    transaction_cost: float = 0.001,
    save_results: bool = True
) -> str:
    """
    Backtest multiple trained models for a symbol and compare their performance.
    
    This function discovers all available trained models for a given symbol,
    runs backtesting on each model using the specified strategy, and provides
    comprehensive comparison analysis and rankings.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')
        strategy_type: Trading strategy to use for all models
        initial_capital: Starting capital for backtesting
        transaction_cost: Transaction cost as percentage
        save_results: Whether to save detailed comparison results
        
    Returns:
        String with comprehensive multi-model comparison results
    """
    return backtest_multiple_models_impl(symbol, strategy_type, initial_capital, transaction_cost, save_results)


# =============================================================================
# VISUALIZATION TOOLS
# =============================================================================

@tool
def visualize_stock_data(
    symbol: str,
    chart_type: Literal["line", "candlestick", "volume", "combined"] = "combined",
    source_file: Optional[str] = None,
    save_chart: bool = True,
    show_indicators: bool = True
) -> str:
    """
    Create interactive visualizations of stock data using Plotly.
    
    This function creates professional, interactive charts for stock data analysis
    with support for multiple chart types and technical indicators overlay.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')
        chart_type: Type of chart to create
                   - "line": Simple line chart of closing prices
                   - "candlestick": OHLC candlestick chart
                   - "volume": Volume bars chart
                   - "combined": All charts in subplots (RECOMMENDED)
        source_file: Specific CSV file to use (if None, uses most recent)
        save_chart: Whether to save chart as HTML file
        show_indicators: Whether to overlay technical indicators (if available)
        
    Returns:
        String with chart creation summary and file location
    """
    return visualize_stock_data_impl(symbol, chart_type, source_file, save_chart, show_indicators)


@tool
def visualize_model_comparison_backtesting(
    symbol: str,
    chart_type: Literal["performance_comparison", "parameter_sensitivity", "risk_return_scatter", "model_type_analysis"] = "performance_comparison",
    results_file: Optional[str] = None,
    save_chart: bool = True
) -> str:
    """
    Create comprehensive visualizations comparing multiple model backtesting results.
    
    This function creates professional comparison charts to analyze and compare
    the performance of different models across various metrics and parameters.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')
        chart_type: Type of comparison chart to create
                   - "performance_comparison": Bar charts comparing key metrics
                   - "parameter_sensitivity": Scatter plot of parameters vs performance
                   - "risk_return_scatter": Risk vs return scatter plot
                   - "model_type_analysis": Analysis by model type
        results_file: Specific multi-model results file (if None, uses most recent)
        save_chart: Whether to save chart as HTML file
        
    Returns:
        String with chart creation summary and file location
    """
    return visualize_model_comparison_backtesting_impl(symbol, chart_type, results_file, save_chart)


@tool
def visualize_backtesting_results(
    symbol: str,
    chart_type: Literal["portfolio_performance", "trading_signals", "model_predictions", "combined"] = "combined",
    results_file: Optional[str] = None,
    save_chart: bool = True
) -> str:
    """
    Create comprehensive visualizations of backtesting results.
    
    This function creates detailed charts showing how well trading strategies
    performed, including portfolio value, trading signals, and prediction accuracy.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')
        chart_type: Type of backtesting chart to create
                   - "portfolio_performance": Portfolio value vs benchmark over time
                   - "trading_signals": Buy/sell signals overlaid on price chart
                   - "model_predictions": Model predictions vs actual prices
                   - "combined": All charts in subplots (RECOMMENDED)
        results_file: Specific backtesting results file (if None, uses most recent)
        save_chart: Whether to save chart as HTML file
        
    Returns:
        String with chart creation summary and file location
    """
    return visualize_backtesting_results_impl(symbol, chart_type, results_file, save_chart)


@tool
def generate_comprehensive_html_report(
    symbol: str,
    title: Optional[str] = None,
    sections: Optional[List[str]] = None,
    include_charts: bool = True,
    custom_content: Optional[str] = None,
    save_report: bool = True
) -> str:
    """
    Generate a comprehensive HTML report with all analysis, charts, and results.
    
    This function creates a professional, interactive HTML document that consolidates
    all analysis results, charts, and insights into a single comprehensive report
    suitable for presentations, sharing, or archiving.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')
        title: Custom title for the report (if None, uses default)
        sections: List of sections to include (if None, includes all available)
                 Options: ['summary', 'data_analysis', 'model_results', 'backtesting', 'charts']
        include_charts: Whether to embed interactive charts in the report
        custom_content: Additional custom HTML content to include
        save_report: Whether to save the report as HTML file
        
    Returns:
        String with report generation summary and file location
    """
    return generate_comprehensive_html_report_impl(
        symbol, title, sections, include_charts, custom_content, save_report
    )


# =============================================================================
# UTILITY TOOLS
# =============================================================================

@tool
def list_saved_stock_files() -> str:
    """
    List all saved stock data files and charts in the output directory.
    
    This function provides a comprehensive overview of all generated files,
    categorized by type, with file sizes and modification timestamps.
    
    Returns:
        String with detailed file listing and statistics
    """
    return list_saved_stock_files_impl()


@tool
def save_text_to_file(
    content: str,
    filename: str,
    file_format: str = "txt",
    custom_header: Optional[str] = None
) -> str:
    """
    Save text content to files in various formats.
    
    This function provides a flexible way to save analysis results, reports,
    or any text content to files with proper formatting and metadata.
    
    Args:
        content: Text content to save
        filename: Base filename (without extension)
        file_format: File format ("txt", "md", "csv", "json")
        custom_header: Optional header to prepend to the content
        
    Returns:
        String with save confirmation and file location
    """
    return save_text_to_file_impl(content, filename, file_format, custom_header)


@tool
def debug_file_system(
    symbol: Optional[str] = None,
    show_content: bool = False
) -> str:
    """
    Debug tool to check file system status and help troubleshoot file-related issues.
    
    Args:
        symbol: Stock symbol to check files for (optional)
        show_content: Whether to show sample content from files
        
    Returns:
        String with detailed file system information
    """
    return debug_file_system_impl(symbol, show_content)


@tool
def validate_model_parameters(
    model_type: str,
    parameters: Dict[str, Any],
    symbol: Optional[str] = None,
    target_days: int = 1
) -> str:
    """
    Validate model parameters against schema and provide optimization suggestions.
    
    This function checks if provided parameters are valid for the specified model type,
    suggests improvements, and provides warnings about potential issues.
    
    Args:
        model_type: Type of model ('xgboost', 'random_forest', etc.)
        parameters: Dictionary of parameters to validate
        symbol: Stock symbol (for context-specific validation)
        target_days: Prediction horizon (for parameter optimization)
        
    Returns:
        String with validation results and recommendations
    """
    return validate_model_parameters_impl(model_type, parameters, symbol, target_days)


@tool
def get_model_selection_guide(
    context: Literal["short_term_trading", "medium_term_investing", "long_term_investing", "high_volatility"] = "medium_term_investing",
    target_days: int = 1,
    available_features: int = 10,
    computational_budget: Literal["low", "medium", "high"] = "medium"
) -> str:
    """
    AI decision support for model selection based on context and constraints.
    
    This function provides intelligent recommendations for model type and parameters
    based on trading context, prediction horizon, and computational constraints.
    
    Args:
        context: Trading/investment context
        target_days: Number of days ahead to predict
        available_features: Number of technical indicators available
        computational_budget: Available computational resources
        
    Returns:
        String with model selection recommendations and reasoning
    """
    return get_model_selection_guide_impl(context, target_days, available_features, computational_budget)


# =============================================================================
# IMPORT AND RE-EXPORT ALL REMAINING FUNCTIONS FROM ORIGINAL TOOLS.PY
# =============================================================================

# NOTE: For the proof of concept, I'm including essential functions here.
# In the full implementation, these would be moved to their respective modules:
# - Visualization functions -> visualization/
# - Backtesting functions -> backtesting/
# - Utility functions -> utils/
# - Model parameter functions -> models/

# Import the remaining functions from the original tools file
import sys
sys.path.append('.')

try:
    # Import remaining functions from original tools
    # from tools_original_backup import additional_functions
    # This ensures backward compatibility during migration
    pass  # Placeholder for now
except ImportError:
    logger.warning("Could not import from original tools backup - continuing with refactored functions only")

# Export all functions for the main package
__all__ = [
    # Data fetching and processing
    "fetch_yahoo_finance_data",
    "get_available_stock_periods_and_intervals", 
    "read_csv_data",
    "apply_technical_indicators_and_transformations",
    
    # Model training
    "train_xgboost_price_predictor",
    "train_random_forest_price_predictor",
    
    # Backtesting
    "backtest_model_strategy",
    "backtest_multiple_models",
    
    # Visualization
    "visualize_stock_data",
    "visualize_model_comparison_backtesting",
    "visualize_backtesting_results",
    "generate_comprehensive_html_report",
    
    # Utilities
    "list_saved_stock_files",
    "save_text_to_file",
    "debug_file_system",
    "validate_model_parameters",
    "get_model_selection_guide",
    
    # External tools
    "tavily_tool"
]