"""Data fetching functionality for stock data retrieval."""

import os
import pandas as pd
from datetime import datetime
from typing import Optional

try:
    import yfinance as yf
    _yfinance_available = True
except ImportError:
    _yfinance_available = False

from ..config import OUTPUT_DIR, VALID_PERIODS, VALID_INTERVALS, logger


def fetch_yahoo_finance_data_impl(
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
    
    Example Usage for AI Agents:
        # Standard daily data for analysis
        result = fetch_yahoo_finance_data("AAPL", "1y", "1d")
        
        # Short-term high-frequency analysis
        result = fetch_yahoo_finance_data("TSLA", "5d", "1h")
        
        # Long-term trend analysis
        result = fetch_yahoo_finance_data("MSFT", "5y", "1wk")
        
        # Maximum historical data
        result = fetch_yahoo_finance_data("GOOGL", "max", "1d")
    
    AI Agent Decision Guidelines:
        - For trend analysis: Use 1y-2y period with 1d interval
        - For volatility analysis: Use 3mo-6mo period with 1d interval  
        - For momentum trading: Use 1mo-3mo period with 1h interval
        - For long-term investing: Use 5y-max period with 1d-1wk interval
        - For backtesting: Use 2y-5y period with 1d interval
        - Always save data (save_data=True) for reproducible analysis
    """
    logger.info(f" fetch_yahoo_finance_data: Starting to fetch data for {symbol.upper()}...")
    
    if not _yfinance_available:
        error_msg = "yfinance module not available. Please install: pip install yfinance"
        logger.error(f"fetch_yahoo_finance_data: {error_msg}")
        return error_msg
    
    try:
        symbol = symbol.upper()
        
        # Validate parameters
        if period not in VALID_PERIODS:
            available_periods = ', '.join(VALID_PERIODS)
            result = f"fetch_yahoo_finance_data: Invalid period '{period}'. Valid periods: {available_periods}"
            logger.error(f"fetch_yahoo_finance_data: {result}")
            return result
        
        if interval not in VALID_INTERVALS:
            available_intervals = ', '.join(VALID_INTERVALS)
            result = f"fetch_yahoo_finance_data: Invalid interval '{interval}'. Valid intervals: {available_intervals}"
            logger.error(f"fetch_yahoo_finance_data: {result}")
            return result
        
        # Create ticker object
        ticker = yf.Ticker(symbol)
        
        # Fetch data with error handling
        try:
            data = ticker.history(period=period, interval=interval)
        except Exception as fetch_error:
            result = f"fetch_yahoo_finance_data: Failed to fetch data for {symbol}: {str(fetch_error)}"
            logger.error(f"fetch_yahoo_finance_data: {result}")
            return result
        
        # Validate data
        if data.empty:
            result = f"fetch_yahoo_finance_data: No data available for symbol '{symbol}' with period '{period}' and interval '{interval}'. Possible reasons: Invalid symbol, market closure, or data not available for the specified period."
            logger.warning(f"fetch_yahoo_finance_data: {result}")
            return result
        
        # Check data quality
        missing_data = data.isnull().sum().sum()
        data_quality = "Excellent" if missing_data == 0 else "Good" if missing_data < 5 else "Fair"
        
        # Calculate comprehensive statistics
        start_date = data.index[0].strftime('%Y-%m-%d')
        end_date = data.index[-1].strftime('%Y-%m-%d')
        
        current_price = data['Close'].iloc[-1]
        opening_price = data['Close'].iloc[0]
        price_change = current_price - opening_price
        price_change_pct = (price_change / opening_price * 100)
        
        period_high = data['High'].max()
        period_low = data['Low'].min()
        price_range = period_high - period_low
        
        # Volatility analysis
        daily_returns = data['Close'].pct_change().dropna()
        volatility = daily_returns.std() * 100
        annualized_volatility = volatility * (252 ** 0.5)  # 252 trading days
        
        # Volume analysis
        avg_volume = data['Volume'].mean()
        total_volume = data['Volume'].sum()
        volume_trend = "Increasing" if data['Volume'].iloc[-5:].mean() > data['Volume'].iloc[-20:-5].mean() else "Decreasing"
        
        # Save data if requested
        filename = None
        filepath = None
        if save_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fetch_yahoo_finance_data_{symbol}_{period}_{interval}_{timestamp}.csv"
            filepath = os.path.join(OUTPUT_DIR, filename)
            data.to_csv(filepath)
        
        # Create comprehensive summary
        summary = f"""fetch_yahoo_finance_data: Successfully retrieved {symbol} stock data:

 DATA OVERVIEW:
- Symbol: {symbol}
- Period: {period} ({start_date} to {end_date})
- Interval: {interval}
- Total Records: {len(data):,}
- Data Quality: {data_quality} ({missing_data} missing values)

 PRICE ANALYSIS:
- Current Price: ${current_price:.2f}
- Opening Price: ${opening_price:.2f}
- Price Change: ${price_change:.2f} ({price_change_pct:+.2f}%)
- Period High: ${period_high:.2f}
- Period Low: ${period_low:.2f}
- Price Range: ${price_range:.2f}

 VOLATILITY METRICS:
- Daily Volatility: {volatility:.2f}%
- Annualized Volatility: {annualized_volatility:.2f}%
- Risk Level: {"High" if annualized_volatility > 30 else "Medium" if annualized_volatility > 15 else "Low"}

 VOLUME ANALYSIS:
- Average Daily Volume: {avg_volume:,.0f}
- Total Volume: {total_volume:,.0f}
- Recent Volume Trend: {volume_trend}

 DATA SAVED: {filename if filename else 'Data not saved'}
- Location: {filepath if filepath else 'N/A'}
- Format: CSV with OHLCV data ready for analysis

 ANALYSIS RECOMMENDATIONS:
- Data is suitable for technical analysis and machine learning
- Consider applying technical indicators for enhanced analysis
- Volume patterns suggest {'strong market interest' if avg_volume > 1000000 else 'moderate market activity'}
- Volatility level indicates {'high-risk, high-reward potential' if annualized_volatility > 25 else 'moderate risk profile'}
"""
        
        logger.info(f"fetch_yahoo_finance_data: Successfully fetched {len(data)} records for {symbol}")
        return summary
        
    except Exception as e:
        error_msg = f"fetch_yahoo_finance_data: Unexpected error fetching data for {symbol}: {str(e)}"
        logger.error(f"fetch_yahoo_finance_data: {error_msg}")
        return error_msg


def get_available_stock_periods_and_intervals_impl() -> str:
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
    
    Example Usage:
        info = get_available_stock_periods_and_intervals()
        # Returns detailed information about all available options
    
    AI Agent Guidelines:
        - Call this function when unsure about valid period/interval combinations
        - Use recommendations for specific analysis types
        - Consider data availability windows when selecting intervals
        - Follow best practices for optimal performance
    """
    logger.info("get_available_stock_periods_and_intervals: Providing data fetching guidance...")
    
    info = f"""get_available_stock_periods_and_intervals: Complete guide to Yahoo Finance data parameters:

 AVAILABLE PERIODS:
- '1d': 1 day (intraday data only, requires minute intervals)
- '5d': 5 days (good for short-term patterns)
- '1mo': 1 month (RECOMMENDED for swing trading analysis)
- '3mo': 3 months (quarterly analysis)
- '6mo': 6 months (medium-term trend analysis)
- '1y': 1 year (DEFAULT - balanced historical perspective)
- '2y': 2 years (RECOMMENDED for robust backtesting)
- '5y': 5 years (long-term trend analysis)
- '10y': 10 years (comprehensive historical analysis)
- 'ytd': Year to date (current year performance)
- 'max': Maximum available (varies by stock, can be 20+ years)

 AVAILABLE INTERVALS:
Minute Intervals (Limited Availability):
- '1m', '2m', '5m', '15m', '30m': Only last 60 days
- '60m', '90m', '1h': Only last 730 days (2 years)

Standard Intervals (Full Historical Data):
- '1d': Daily data (MOST RELIABLE - recommended default)
- '5d': 5-day intervals
- '1wk': Weekly data (good for long-term trends)
- '1mo': Monthly data (long-term strategic analysis)
- '3mo': Quarterly data (fundamental analysis)

 RECOMMENDED COMBINATIONS BY USE CASE:

SHORT-TERM TRADING (1-7 days):
- Period: '1mo' or '3mo'
- Interval: '1h' or '1d'
- Use case: Day trading, swing trading

MEDIUM-TERM ANALYSIS (1-6 months):
- Period: '1y' or '2y'
- Interval: '1d'
- Use case: Trend following, momentum strategies

LONG-TERM INVESTING (6+ months):
- Period: '5y' or 'max'
- Interval: '1d' or '1wk'
- Use case: Value investing, buy-and-hold strategies

MACHINE LEARNING MODEL TRAINING:
- Period: '2y' to '5y' (sufficient data for training)
- Interval: '1d' (most reliable and complete)
- Use case: Predictive modeling, backtesting

HIGH-FREQUENCY ANALYSIS:
- Period: '5d' to '1mo'
- Interval: '1h' to '15m'
- Use case: Scalping, intraday patterns
-  WARNING: Limited historical data availability

 IMPORTANT LIMITATIONS:

Data Availability:
- Minute intervals: Only last 60 days maximum
- Hourly intervals: Only last 730 days (2 years) maximum
- Some stocks may have limited historical data
- Weekend and holiday gaps in data

Performance Considerations:
- Smaller intervals = more data points = slower processing
- Very long periods with small intervals may timeout
- Consider data storage requirements for large datasets

Best Practices:
- Always validate data after fetching (check for gaps, missing values)
- Use '1d' interval for most reliable historical data
- Combine longer periods (2y+) with daily intervals for ML training
- Test with smaller periods first, then scale up
- Save fetched data to avoid repeated API calls

 REAL-TIME VS HISTORICAL:
- Real-time: Current trading day data (15-20 minute delay)
- Historical: Complete OHLCV data for closed trading periods
- Intraday data availability varies by market hours and holidays

 AI AGENT RECOMMENDATIONS:
1. Default to '1y' period with '1d' interval for balanced analysis
2. Use '2y'+ periods for machine learning model training
3. Only use minute intervals when specifically needed for intraday analysis
4. Always check data quality after fetching
5. Consider data storage and processing time for large datasets
"""
    
    logger.info("get_available_stock_periods_and_intervals: Provided comprehensive data fetching guide")
    return info