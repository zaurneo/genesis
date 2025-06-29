"""Technical indicators calculation and application."""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List

try:
    import yfinance as yf
    _yfinance_available = True
except ImportError:
    _yfinance_available = False

from ..config import OUTPUT_DIR, logger


def apply_technical_indicators_and_transformations_impl(
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
    logger.info(f" apply_technical_indicators_and_transformations: Starting to apply indicators for {symbol.upper()}...")
    
    try:
        symbol = symbol.upper()
        
        # Load data
        if source_file:
            if not source_file.endswith('.csv'):
                source_file += '.csv'
            filepath = os.path.join(OUTPUT_DIR, source_file)
            if not os.path.exists(filepath):
                result = f"apply_technical_indicators_and_transformations: Source file '{source_file}' not found in output directory."
                logger.error(f"apply_technical_indicators_and_transformations: {result}")
                return result
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            data_source = f"file: {source_file}"
        else:
            # Try to find most recent data file
            data_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(f"fetch_yahoo_finance_data_{symbol}_") and f.endswith('.csv')]
            if data_files:
                latest_file = max(data_files, key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)))
                filepath = os.path.join(OUTPUT_DIR, latest_file)
                data = pd.read_csv(filepath, index_col=0, parse_dates=True)
                data_source = f"existing file: {latest_file}"
            else:
                # Fetch new data
                if not _yfinance_available:
                    result = f"apply_technical_indicators_and_transformations: yfinance not available and no existing data found for {symbol}."
                    logger.error(f"apply_technical_indicators_and_transformations: {result}")
                    return result
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                data_source = f"newly fetched data (period: {period})"
        
        if data.empty:
            result = f"apply_technical_indicators_and_transformations: No data available for {symbol}."
            logger.error(f"apply_technical_indicators_and_transformations: {result}")
            return result
        
        # Create a copy to avoid modifying original data
        enhanced_data = data.copy()
        applied_indicators = []
        
        # Parse indicators list
        indicator_list = [ind.strip().lower() for ind in indicators.split(',')]
        
        # Apply each indicator/transformation
        for indicator in indicator_list:
            try:
                if indicator.startswith('sma_'):
                    enhanced_data, applied = _apply_sma(enhanced_data, indicator, applied_indicators)
                elif indicator.startswith('ema_'):
                    enhanced_data, applied = _apply_ema(enhanced_data, indicator, applied_indicators)
                elif indicator.startswith('rsi'):
                    enhanced_data, applied = _apply_rsi(enhanced_data, indicator, applied_indicators)
                elif indicator == 'macd':
                    enhanced_data, applied = _apply_macd(enhanced_data, applied_indicators)
                elif indicator.startswith('bollinger'):
                    enhanced_data, applied = _apply_bollinger(enhanced_data, indicator, applied_indicators)
                elif indicator == 'returns':
                    enhanced_data, applied = _apply_returns(enhanced_data, applied_indicators)
                elif indicator == 'log_returns':
                    enhanced_data, applied = _apply_log_returns(enhanced_data, applied_indicators)
                elif indicator.startswith('volatility'):
                    enhanced_data, applied = _apply_volatility(enhanced_data, indicator, applied_indicators)
                elif indicator.startswith('volume_sma'):
                    enhanced_data, applied = _apply_volume_sma(enhanced_data, indicator, applied_indicators)
                elif indicator.startswith('price_momentum'):
                    enhanced_data, applied = _apply_price_momentum(enhanced_data, indicator, applied_indicators)
                elif indicator == 'support_resistance':
                    enhanced_data, applied = _apply_support_resistance(enhanced_data, applied_indicators)
                
                applied_indicators = applied
                
            except Exception as e:
                applied_indicators.append(f'{indicator}_ERROR: {str(e)}')
        
        # Add derived metrics
        if 'Close' in enhanced_data.columns:
            enhanced_data['Price_Change'] = enhanced_data['Close'].diff()
            enhanced_data['Price_Change_Pct'] = enhanced_data['Close'].pct_change() * 100
        
        # Save enhanced data if requested
        filename = None
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"apply_technical_indicators_and_transformations_{symbol}_enhanced_{timestamp}.csv"
            filepath = os.path.join(OUTPUT_DIR, filename)
            enhanced_data.to_csv(filepath)
        
        # Calculate summary statistics for new indicators
        new_columns = [col for col in enhanced_data.columns if col not in data.columns]
        stats_summary = ""
        
        if new_columns:
            stats_summary = "\\n NEW INDICATOR STATISTICS:\\n"
            for col in new_columns[:10]:  # Limit to first 10 for readability
                if enhanced_data[col].dtype in ['float64', 'int64']:
                    try:
                        mean_val = enhanced_data[col].mean()
                        std_val = enhanced_data[col].std()
                        min_val = enhanced_data[col].min()
                        max_val = enhanced_data[col].max()
                        stats_summary += f"- {col}: Mean={mean_val:.3f}, Std={std_val:.3f}, Range=[{min_val:.3f}, {max_val:.3f}]\\n"
                    except:
                        stats_summary += f"- {col}: Statistical calculation failed\\n"
        
        # Create comprehensive summary
        summary = f"""apply_technical_indicators_and_transformations: Successfully enhanced {symbol} stock data with technical indicators:

 DATA ENHANCEMENT SUMMARY:
- Symbol: {symbol}
- Data Source: {data_source}
- Original Data Points: {len(data)}
- Original Columns: {len(data.columns)}
- Enhanced Columns: {len(enhanced_data.columns)}
- New Indicators Added: {len(new_columns)}

🔧 APPLIED INDICATORS:
{chr(10).join([f"  ✓ {ind}" for ind in applied_indicators])}

 ENHANCED DATASET:
- Total Columns: {len(enhanced_data.columns)}
- Date Range: {enhanced_data.index[0].strftime('%Y-%m-%d')} to {enhanced_data.index[-1].strftime('%Y-%m-%d')}
- New Technical Columns: {', '.join(new_columns[:8])}{'...' if len(new_columns) > 8 else ''}

{stats_summary}

 ENHANCED DATA SAVED: {filename if filename else 'Data not saved'}
- Location: {os.path.join(OUTPUT_DIR, filename) if filename else 'N/A'}
- Format: CSV with all original data + technical indicators

 USAGE NOTES:
- Enhanced data includes all original OHLCV data plus technical indicators
- Indicators with rolling windows will have NaN values for initial periods
- Data is ready for advanced analysis and visualization
- Can be used directly by stock_analyzer for enhanced charting
"""
        
        logger.info(f"apply_technical_indicators_and_transformations: Successfully applied {len(applied_indicators)} indicators for {symbol}")
        return summary
        
    except Exception as e:
        error_msg = f"apply_technical_indicators_and_transformations: Error processing {symbol}: {str(e)}"
        logger.error(f"apply_technical_indicators_and_transformations: {error_msg}")
        return error_msg


def _apply_sma(data: pd.DataFrame, indicator: str, applied: List[str]) -> tuple:
    """Apply Simple Moving Average."""
    period_val = int(indicator.split('_')[1])
    data[f'SMA_{period_val}'] = data['Close'].rolling(window=period_val).mean()
    applied.append(f'SMA_{period_val}')
    return data, applied


def _apply_ema(data: pd.DataFrame, indicator: str, applied: List[str]) -> tuple:
    """Apply Exponential Moving Average."""
    period_val = int(indicator.split('_')[1])
    data[f'EMA_{period_val}'] = data['Close'].ewm(span=period_val).mean()
    applied.append(f'EMA_{period_val}')
    return data, applied


def _apply_rsi(data: pd.DataFrame, indicator: str, applied: List[str]) -> tuple:
    """Apply Relative Strength Index."""
    if '_' in indicator:
        period_val = int(indicator.split('_')[1])
    else:
        period_val = 14
    
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period_val).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period_val).mean()
    rs = gain / loss
    data[f'RSI_{period_val}'] = 100 - (100 / (1 + rs))
    applied.append(f'RSI_{period_val}')
    return data, applied


def _apply_macd(data: pd.DataFrame, applied: List[str]) -> tuple:
    """Apply MACD indicator."""
    ema_12 = data['Close'].ewm(span=12).mean()
    ema_26 = data['Close'].ewm(span=26).mean()
    data['MACD'] = ema_12 - ema_26
    data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
    data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
    applied.append('MACD')
    return data, applied


def _apply_bollinger(data: pd.DataFrame, indicator: str, applied: List[str]) -> tuple:
    """Apply Bollinger Bands."""
    if '_' in indicator:
        parts = indicator.split('_')
        period_val = int(parts[1]) if len(parts) > 1 else 20
        std_dev = float(parts[2]) if len(parts) > 2 else 2
    else:
        period_val = 20
        std_dev = 2
    
    rolling_mean = data['Close'].rolling(window=period_val).mean()
    rolling_std = data['Close'].rolling(window=period_val).std()
    data[f'BB_Upper_{period_val}'] = rolling_mean + (rolling_std * std_dev)
    data[f'BB_Lower_{period_val}'] = rolling_mean - (rolling_std * std_dev)
    data[f'BB_Middle_{period_val}'] = rolling_mean
    data[f'BB_Width_{period_val}'] = data[f'BB_Upper_{period_val}'] - data[f'BB_Lower_{period_val}']
    applied.append(f'Bollinger_Bands_{period_val}')
    return data, applied


def _apply_returns(data: pd.DataFrame, applied: List[str]) -> tuple:
    """Apply daily returns calculation."""
    data['Daily_Returns'] = data['Close'].pct_change() * 100
    applied.append('Daily_Returns')
    return data, applied


def _apply_log_returns(data: pd.DataFrame, applied: List[str]) -> tuple:
    """Apply log returns calculation."""
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1)) * 100
    applied.append('Log_Returns')
    return data, applied


def _apply_volatility(data: pd.DataFrame, indicator: str, applied: List[str]) -> tuple:
    """Apply rolling volatility calculation."""
    if '_' in indicator:
        period_val = int(indicator.split('_')[1])
    else:
        period_val = 20
    
    returns = data['Close'].pct_change()
    data[f'Volatility_{period_val}'] = returns.rolling(window=period_val).std() * np.sqrt(252) * 100
    applied.append(f'Volatility_{period_val}')
    return data, applied


def _apply_volume_sma(data: pd.DataFrame, indicator: str, applied: List[str]) -> tuple:
    """Apply volume moving average."""
    if '_' in indicator and len(indicator.split('_')) > 2:
        period_val = int(indicator.split('_')[2])
    else:
        period_val = 20
    
    data[f'Volume_SMA_{period_val}'] = data['Volume'].rolling(window=period_val).mean()
    data['Volume_Ratio'] = data['Volume'] / data[f'Volume_SMA_{period_val}']
    applied.append(f'Volume_SMA_{period_val}')
    return data, applied


def _apply_price_momentum(data: pd.DataFrame, indicator: str, applied: List[str]) -> tuple:
    """Apply price momentum calculation."""
    if '_' in indicator and len(indicator.split('_')) > 2:
        period_val = int(indicator.split('_')[2])
    else:
        period_val = 10
    
    data[f'Price_Momentum_{period_val}'] = data['Close'] / data['Close'].shift(period_val) - 1
    applied.append(f'Price_Momentum_{period_val}')
    return data, applied


def _apply_support_resistance(data: pd.DataFrame, applied: List[str]) -> tuple:
    """Apply basic support and resistance levels."""
    window = 20
    data['Local_Max'] = data['High'].rolling(window=window, center=True).max()
    data['Local_Min'] = data['Low'].rolling(window=window, center=True).min()
    
    # Resistance: price touches local max
    data['At_Resistance'] = (data['High'] >= data['Local_Max'] * 0.99).astype(int)
    # Support: price touches local min  
    data['At_Support'] = (data['Low'] <= data['Local_Min'] * 1.01).astype(int)
    applied.append('Support_Resistance_Levels')
    return data, applied