import os
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.offline import plot
from datetime import datetime, timedelta
from typing import Optional, Literal
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

# Tavily tool
tavily_tool = TavilySearchResults(max_results=5)

# Create output directory if it doesn't exist
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@tool
def read_csv_data(
    filename: str,
    max_rows: int = 100
) -> str:
    """
    Read and analyze CSV data from the output directory.
    This allows the AI agent to examine stock data and extract insights.
    
    Args:
        filename: Name of the CSV file to read (include .csv extension)
        max_rows: Maximum number of rows to display (default 100, set to -1 for all)
        
    Returns:
        String with data summary, statistics, and sample data
    """
    try:
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        if not os.path.exists(filepath):
            available_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.csv')]
            return f"File '{filename}' not found. Available CSV files: {', '.join(available_files) if available_files else 'None'}"
        
        # Read the CSV file
        data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        if data.empty:
            return f"The file '{filename}' is empty."
        
        # Calculate comprehensive statistics
        stats = {}
        if 'Close' in data.columns:
            stats['current_price'] = data['Close'].iloc[-1]
            stats['opening_price'] = data['Close'].iloc[0]
            stats['price_change'] = data['Close'].iloc[-1] - data['Close'].iloc[0]
            stats['price_change_pct'] = (stats['price_change'] / data['Close'].iloc[0] * 100)
            stats['period_high'] = data['High'].max() if 'High' in data.columns else data['Close'].max()
            stats['period_low'] = data['Low'].min() if 'Low' in data.columns else data['Close'].min()
            stats['volatility'] = data['Close'].pct_change().std() * 100
        
        if 'Volume' in data.columns:
            stats['avg_volume'] = data['Volume'].mean()
            stats['total_volume'] = data['Volume'].sum()
            stats['max_volume'] = data['Volume'].max()
        
        # Format data sample
        sample_rows = data.head(max_rows) if max_rows > 0 else data
        
        # Create comprehensive summary
        summary = f"""
📊 CSV DATA ANALYSIS for {filename}:

📈 DATASET OVERVIEW:
- Total Records: {len(data)}
- Date Range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}
- Columns: {', '.join(data.columns)}
- File Size: {os.path.getsize(filepath):,} bytes

"""
        
        if stats:
            summary += f"""💰 PRICE STATISTICS:
- Current Price: ${stats.get('current_price', 0):.2f}
- Opening Price: ${stats.get('opening_price', 0):.2f}
- Period High: ${stats.get('period_high', 0):.2f}
- Period Low: ${stats.get('period_low', 0):.2f}
- Price Change: ${stats.get('price_change', 0):.2f} ({stats.get('price_change_pct', 0):.2f}%)
- Volatility: {stats.get('volatility', 0):.2f}%

"""
        
        if 'avg_volume' in stats:
            summary += f"""📊 VOLUME STATISTICS:
- Average Volume: {stats['avg_volume']:,.0f}
- Total Volume: {stats['total_volume']:,.0f}
- Maximum Volume: {stats['max_volume']:,.0f}

"""
        
        summary += f"""📋 SAMPLE DATA ({min(len(sample_rows), max_rows)} of {len(data)} rows):
{sample_rows.to_string()}

💡 QUICK INSIGHTS:
- Data Quality: {'Complete' if not data.isnull().any().any() else 'Contains missing values'}
- Trend: {'Upward' if stats.get('price_change_pct', 0) > 0 else 'Downward' if stats.get('price_change_pct', 0) < 0 else 'Flat'}
- Volatility Level: {'High' if stats.get('volatility', 0) > 3 else 'Moderate' if stats.get('volatility', 0) > 1 else 'Low'}
"""
        
        return summary
        
    except Exception as e:
        return f"Error reading CSV file '{filename}': {str(e)}"


@tool
def fetch_yahoo_finance_data(
    symbol: str,
    period: str = "1mo",
    interval: str = "1d",
    save_data: bool = True
) -> str:
    """
    Fetch stock data from Yahoo Finance and optionally save to CSV.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')
        period: Data period - valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        interval: Data interval - valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        save_data: Whether to save data to CSV file
        
    Returns:
        String description of the fetched data and file location
    """
    try:
        # Create ticker object
        ticker = yf.Ticker(symbol.upper())
        
        # Fetch historical data
        data = ticker.history(period=period, interval=interval)
        
        if data.empty:
            return f"No data found for symbol {symbol}. Please check if the symbol is correct."
        
        # Get basic info about the stock
        try:
            info = ticker.info
            company_name = info.get('longName', symbol.upper())
        except:
            company_name = symbol.upper()
        
        # Prepare summary statistics
        latest_price = data['Close'].iloc[-1]
        price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2] if len(data) > 1 else 0
        price_change_pct = (price_change / data['Close'].iloc[-2] * 100) if len(data) > 1 and data['Close'].iloc[-2] != 0 else 0
        
        high_52w = data['High'].max()
        low_52w = data['Low'].min()
        avg_volume = data['Volume'].mean()
        
        # Save data if requested
        filename = None
        if save_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fetch_yahoo_finance_data_{symbol.upper()}_{period}_{interval}_{timestamp}.csv"
            filepath = os.path.join(OUTPUT_DIR, filename)
            data.to_csv(filepath)
        
        # Create summary
        summary = f"""
Successfully fetched {company_name} ({symbol.upper()}) stock data:

📊 DATA SUMMARY:
- Period: {period}
- Interval: {interval}
- Data points: {len(data)} records
- Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}

💰 CURRENT METRICS:
- Latest Price: ${latest_price:.2f}
- Price Change: ${price_change:.2f} ({price_change_pct:.2f}%)
- 52W High: ${high_52w:.2f}
- 52W Low: ${low_52w:.2f}
- Avg Volume: {avg_volume:,.0f}

📁 FILE SAVED: {filename if filename else 'Data not saved'}
- Location: {os.path.join(OUTPUT_DIR, filename) if filename else 'N/A'}

The data includes: Open, High, Low, Close, Volume, Dividends, and Stock Splits.
"""
        
        return summary
        
    except Exception as e:
        return f"Error fetching data for {symbol}: {str(e)}"



@tool
def get_available_stock_periods_and_intervals() -> str:
    """
    Get information about available periods and intervals for Yahoo Finance data.
    
    Returns:
        String with available options for periods and intervals
    """
    return """
📅 AVAILABLE PERIODS:
- 1d, 5d: Recent days
- 1mo, 3mo, 6mo: Monthly periods  
- 1y, 2y, 5y, 10y: Yearly periods
- ytd: Year to date
- max: Maximum available data

⏰ AVAILABLE INTERVALS:
- Intraday: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h
- Daily+: 1d, 5d, 1wk, 1mo, 3mo

📝 USAGE EXAMPLES:
- Recent month daily data: period='1mo', interval='1d'
- Last year weekly data: period='1y', interval='1wk'  
- Today's hourly data: period='1d', interval='1h'
- Maximum historical data: period='max', interval='1d'

⚠️ NOTE: Shorter intervals (minutes) are only available for recent periods (last 60 days).
"""



@tool
def visualize_stock_data(
    symbol: str,
    chart_type: Literal["line", "candlestick", "volume", "combined"] = "combined",
    period: str = "1mo"
) -> str:
    """
    Create interactive visualizations of stock data using the most recent data file or fetch new data.
    Charts are automatically saved to the output directory as HTML files.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')
        chart_type: Type of chart - 'line', 'candlestick', 'volume', or 'combined'
        period: Period for data if new fetch is needed
        
    Returns:
        String description of the created visualization
    """
    try:
        symbol = symbol.upper()
        
        # Try to find the most recent data file for this symbol
        data_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(f"fetch_yahoo_finance_data_{symbol}_") and f.endswith('.csv')]
        
        if data_files:
            # Use the most recent file
            latest_file = max(data_files, key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)))
            filepath = os.path.join(OUTPUT_DIR, latest_file)
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            data_source = f"existing file: {latest_file}"
        else:
            # Fetch new data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            data_source = f"newly fetched data (period: {period})"
        
        if data.empty:
            return f"No data available for {symbol} to create visualization."
        
        # Create the visualization using Plotly
        if chart_type == "line":
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=2)
            ))
            fig.update_layout(
                title=f'{symbol} Stock Price - Line Chart',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                showlegend=True,
                hovermode='x unified'
            )
            
        elif chart_type == "candlestick":
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name=symbol
            ))
            fig.update_layout(
                title=f'{symbol} Stock Price - Candlestick Chart',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                xaxis_rangeslider_visible=False
            )
            
        elif chart_type == "volume":
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color='orange',
                opacity=0.7
            ))
            fig.update_layout(
                title=f'{symbol} Trading Volume',
                xaxis_title='Date',
                yaxis_title='Volume',
                showlegend=True
            )
            
        elif chart_type == "combined":
            # Create subplots
            fig = sp.make_subplots(
                rows=2, cols=1,
                subplot_titles=(f'{symbol} Stock Price', f'{symbol} Trading Volume'),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )
            
            # Price chart (top)
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Close',
                line=dict(color='blue', width=2)
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['High'],
                mode='lines',
                name='High',
                line=dict(color='green', width=1),
                opacity=0.7
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Low'],
                mode='lines',
                name='Low',
                line=dict(color='red', width=1),
                opacity=0.7
            ), row=1, col=1)
            
            # Volume chart (bottom)
            fig.add_trace(go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color='orange',
                opacity=0.7
            ), row=2, col=1)
            
            # Update layout
            fig.update_layout(
                title=f'{symbol} Stock Analysis - Combined Chart',
                showlegend=True,
                hovermode='x unified'
            )
            fig.update_xaxes(title_text='Date', row=2, col=1)
            fig.update_yaxes(title_text='Price ($)', row=1, col=1)
            fig.update_yaxes(title_text='Volume', row=2, col=1)
        
        # Common layout updates
        fig.update_layout(
            template='plotly_white',
            width=1200,
            height=600 if chart_type != "combined" else 800,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Save chart as HTML file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_filename = f"visualize_stock_data_{symbol}_{chart_type}_chart_{timestamp}.html"
        chart_filepath = os.path.join(OUTPUT_DIR, chart_filename)
        
        # Save the interactive plot as HTML
        plot(fig, filename=chart_filepath, auto_open=False, include_plotlyjs=True)
        
        # Calculate some basic statistics
        price_stats = {
            'current_price': data['Close'].iloc[-1],
            'period_high': data['High'].max(),
            'period_low': data['Low'].min(),
            'price_change': data['Close'].iloc[-1] - data['Close'].iloc[0],
            'price_change_pct': ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100),
            'avg_volume': data['Volume'].mean()
        }
        
        summary = f"""
📈 INTERACTIVE VISUALIZATION CREATED for {symbol}:

🎨 CHART DETAILS:
- Chart Type: {chart_type.title()} (Interactive Plotly Chart)
- Data Source: {data_source}
- Data Points: {len(data)} records
- Date Range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}

📊 KEY STATISTICS:
- Current Price: ${price_stats['current_price']:.2f}
- Period High: ${price_stats['period_high']:.2f}
- Period Low: ${price_stats['period_low']:.2f}
- Price Change: ${price_stats['price_change']:.2f} ({price_stats['price_change_pct']:.2f}%)
- Average Volume: {price_stats['avg_volume']:,.0f}

📁 INTERACTIVE CHART SAVED: {chart_filename}
- Location: {os.path.join(OUTPUT_DIR, chart_filename)}
- Format: Interactive HTML with zoom, pan, and hover features

The interactive visualization includes hover data, zoom capabilities, and can be opened in any web browser.
"""
        
        return summary
        
    except Exception as e:
        return f"Error creating visualization for {symbol}: {str(e)}"



@tool
def list_saved_stock_files() -> str:
    """
    List all saved stock data files and charts in the output directory.
    
    Returns:
        String listing all available files with details
    """
    try:
        if not os.path.exists(OUTPUT_DIR):
            return "Output directory does not exist. No files have been saved yet."
        
        files = os.listdir(OUTPUT_DIR)
        if not files:
            return "No files found in the output directory."
        
        data_files = [f for f in files if f.endswith('.csv')]
        chart_files = [f for f in files if f.endswith('.html')]
        other_files = [f for f in files if not f.endswith(('.csv', '.html'))]
        
        summary = f"📁 FILES IN OUTPUT DIRECTORY ({OUTPUT_DIR}):\n\n"
        
        if data_files:
            summary += "📊 DATA FILES (.csv):\n"
            for file in sorted(data_files):
                filepath = os.path.join(OUTPUT_DIR, file)
                size = os.path.getsize(filepath)
                modified = datetime.fromtimestamp(os.path.getmtime(filepath))
                summary += f"  - {file} ({size:,} bytes, modified: {modified.strftime('%Y-%m-%d %H:%M:%S')})\n"
            summary += "\n"
        
        if chart_files:
            summary += "📈 INTERACTIVE CHART FILES (.html):\n"
            for file in sorted(chart_files):
                filepath = os.path.join(OUTPUT_DIR, file)
                size = os.path.getsize(filepath)
                modified = datetime.fromtimestamp(os.path.getmtime(filepath))
                summary += f"  - {file} ({size:,} bytes, modified: {modified.strftime('%Y-%m-%d %H:%M:%S')})\n"
            summary += "\n"
        
        if other_files:
            summary += "📄 OTHER FILES:\n"
            for file in sorted(other_files):
                filepath = os.path.join(OUTPUT_DIR, file)
                size = os.path.getsize(filepath)
                modified = datetime.fromtimestamp(os.path.getmtime(filepath))
                summary += f"  - {file} ({size:,} bytes, modified: {modified.strftime('%Y-%m-%d %H:%M:%S')})\n"
            summary += "\n"
        
        summary += f"📈 TOTAL FILES: {len(files)} ({len(data_files)} data, {len(chart_files)} charts, {len(other_files)} other)"
        
        return summary
        
    except Exception as e:
        return f"Error listing files: {str(e)}"


@tool
def save_text_to_file(
    content: str,
    filename: str,
    file_format: str = "md"
) -> str:
    """
    Save text content to a file in the output directory.
    The AI agent can use this to create reports, summaries, or any text documents.
    
    Args:
        content: The text content to save
        filename: Name for the file (without extension)
        file_format: File extension/format (md, txt, csv, etc.)
        
    Returns:
        String description of the saved file and its location
    """
    try:
        # Clean filename and add timestamp if needed
        clean_filename = filename.replace(" ", "_").replace("/", "_").replace("\\", "_")
        
        # Add timestamp if filename doesn't already have one
        if not any(char.isdigit() for char in clean_filename[-15:]):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            clean_filename = f"save_text_to_file_{clean_filename}_{timestamp}"
        else:
            clean_filename = f"save_text_to_file_{clean_filename}"
        
        # Add file extension
        if not clean_filename.endswith(f".{file_format}"):
            clean_filename = f"{clean_filename}.{file_format}"
        
        # Save to output directory
        filepath = os.path.join(OUTPUT_DIR, clean_filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        file_size = os.path.getsize(filepath)
        
        summary = f"""
📄 FILE SUCCESSFULLY SAVED:

📁 FILE DETAILS:
- Filename: {clean_filename}
- Location: {filepath}
- Format: {file_format.upper()}
- Size: {file_size:,} bytes ({len(content):,} characters)

📝 CONTENT SUMMARY:
- Lines: {content.count(chr(10)) + 1}
- Words: {len(content.split())}
- Characters: {len(content)}

The file has been saved to the output directory and is ready for use.
"""        
        return summary
        
    except Exception as e:
        return f"Error saving file: {str(e)}"
    

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
    try:
        import numpy as np
        
        symbol = symbol.upper()
        
        # Load data
        if source_file:
            if not source_file.endswith('.csv'):
                source_file += '.csv'
            filepath = os.path.join(OUTPUT_DIR, source_file)
            if not os.path.exists(filepath):
                return f"apply_technical_indicators_and_transformations: Source file '{source_file}' not found in output directory."
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
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                data_source = f"newly fetched data (period: {period})"
        
        if data.empty:
            return f"apply_technical_indicators_and_transformations: No data available for {symbol}."
        
        # Create a copy to avoid modifying original data
        enhanced_data = data.copy()
        applied_indicators = []
        
        # Parse indicators list
        indicator_list = [ind.strip().lower() for ind in indicators.split(',')]
        
        # Apply each indicator/transformation
        for indicator in indicator_list:
            try:
                if indicator.startswith('sma_'):
                    # Simple Moving Average
                    period_val = int(indicator.split('_')[1])
                    enhanced_data[f'SMA_{period_val}'] = enhanced_data['Close'].rolling(window=period_val).mean()
                    applied_indicators.append(f'SMA_{period_val}')
                
                elif indicator.startswith('ema_'):
                    # Exponential Moving Average
                    period_val = int(indicator.split('_')[1])
                    enhanced_data[f'EMA_{period_val}'] = enhanced_data['Close'].ewm(span=period_val).mean()
                    applied_indicators.append(f'EMA_{period_val}')
                
                elif indicator.startswith('rsi'):
                    # Relative Strength Index
                    if '_' in indicator:
                        period_val = int(indicator.split('_')[1])
                    else:
                        period_val = 14
                    
                    delta = enhanced_data['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=period_val).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=period_val).mean()
                    rs = gain / loss
                    enhanced_data[f'RSI_{period_val}'] = 100 - (100 / (1 + rs))
                    applied_indicators.append(f'RSI_{period_val}')
                
                elif indicator == 'macd':
                    # MACD
                    ema_12 = enhanced_data['Close'].ewm(span=12).mean()
                    ema_26 = enhanced_data['Close'].ewm(span=26).mean()
                    enhanced_data['MACD'] = ema_12 - ema_26
                    enhanced_data['MACD_Signal'] = enhanced_data['MACD'].ewm(span=9).mean()
                    enhanced_data['MACD_Histogram'] = enhanced_data['MACD'] - enhanced_data['MACD_Signal']
                    applied_indicators.append('MACD')
                
                elif indicator.startswith('bollinger'):
                    # Bollinger Bands
                    if '_' in indicator:
                        parts = indicator.split('_')
                        period_val = int(parts[1]) if len(parts) > 1 else 20
                        std_dev = float(parts[2]) if len(parts) > 2 else 2
                    else:
                        period_val = 20
                        std_dev = 2
                    
                    rolling_mean = enhanced_data['Close'].rolling(window=period_val).mean()
                    rolling_std = enhanced_data['Close'].rolling(window=period_val).std()
                    enhanced_data[f'BB_Upper_{period_val}'] = rolling_mean + (rolling_std * std_dev)
                    enhanced_data[f'BB_Lower_{period_val}'] = rolling_mean - (rolling_std * std_dev)
                    enhanced_data[f'BB_Middle_{period_val}'] = rolling_mean
                    enhanced_data[f'BB_Width_{period_val}'] = enhanced_data[f'BB_Upper_{period_val}'] - enhanced_data[f'BB_Lower_{period_val}']
                    applied_indicators.append(f'Bollinger_Bands_{period_val}')
                
                elif indicator == 'returns':
                    # Daily Returns
                    enhanced_data['Daily_Returns'] = enhanced_data['Close'].pct_change() * 100
                    applied_indicators.append('Daily_Returns')
                
                elif indicator == 'log_returns':
                    # Log Returns
                    enhanced_data['Log_Returns'] = np.log(enhanced_data['Close'] / enhanced_data['Close'].shift(1)) * 100
                    applied_indicators.append('Log_Returns')
                
                elif indicator.startswith('volatility'):
                    # Rolling Volatility
                    if '_' in indicator:
                        period_val = int(indicator.split('_')[1])
                    else:
                        period_val = 20
                    
                    returns = enhanced_data['Close'].pct_change()
                    enhanced_data[f'Volatility_{period_val}'] = returns.rolling(window=period_val).std() * np.sqrt(252) * 100
                    applied_indicators.append(f'Volatility_{period_val}')
                
                elif indicator.startswith('volume_sma'):
                    # Volume Moving Average
                    if '_' in indicator and len(indicator.split('_')) > 2:
                        period_val = int(indicator.split('_')[2])
                    else:
                        period_val = 20
                    
                    enhanced_data[f'Volume_SMA_{period_val}'] = enhanced_data['Volume'].rolling(window=period_val).mean()
                    enhanced_data['Volume_Ratio'] = enhanced_data['Volume'] / enhanced_data[f'Volume_SMA_{period_val}']
                    applied_indicators.append(f'Volume_SMA_{period_val}')
                
                elif indicator.startswith('price_momentum'):
                    # Price Momentum
                    if '_' in indicator and len(indicator.split('_')) > 2:
                        period_val = int(indicator.split('_')[2])
                    else:
                        period_val = 10
                    
                    enhanced_data[f'Price_Momentum_{period_val}'] = enhanced_data['Close'] / enhanced_data['Close'].shift(period_val) - 1
                    applied_indicators.append(f'Price_Momentum_{period_val}')
                
                elif indicator == 'support_resistance':
                    # Basic Support and Resistance Levels
                    window = 20
                    enhanced_data['Local_Max'] = enhanced_data['High'].rolling(window=window, center=True).max()
                    enhanced_data['Local_Min'] = enhanced_data['Low'].rolling(window=window, center=True).min()
                    
                    # Resistance: price touches local max
                    enhanced_data['At_Resistance'] = (enhanced_data['High'] >= enhanced_data['Local_Max'] * 0.99).astype(int)
                    # Support: price touches local min  
                    enhanced_data['At_Support'] = (enhanced_data['Low'] <= enhanced_data['Local_Min'] * 1.01).astype(int)
                    applied_indicators.append('Support_Resistance_Levels')
                
            except Exception as e:
                applied_indicators.append(f'{indicator}_ERROR: {str(e)}')
        
        # Add some derived metrics
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
            stats_summary = "\n📊 NEW INDICATOR STATISTICS:\n"
            for col in new_columns[:10]:  # Limit to first 10 for readability
                if enhanced_data[col].dtype in ['float64', 'int64']:
                    try:
                        mean_val = enhanced_data[col].mean()
                        std_val = enhanced_data[col].std()
                        min_val = enhanced_data[col].min()
                        max_val = enhanced_data[col].max()
                        stats_summary += f"- {col}: Mean={mean_val:.3f}, Std={std_val:.3f}, Range=[{min_val:.3f}, {max_val:.3f}]\n"
                    except:
                        stats_summary += f"- {col}: Statistical calculation failed\n"
        
        # Create comprehensive summary
        summary = f"""apply_technical_indicators_and_transformations: Successfully enhanced {symbol} stock data with technical indicators:

📈 DATA ENHANCEMENT SUMMARY:
- Symbol: {symbol}
- Data Source: {data_source}
- Original Data Points: {len(data)}
- Original Columns: {len(data.columns)}
- Enhanced Columns: {len(enhanced_data.columns)}
- New Indicators Added: {len(new_columns)}

🔧 APPLIED INDICATORS:
{chr(10).join([f"  ✓ {ind}" for ind in applied_indicators])}

📊 ENHANCED DATASET:
- Total Columns: {len(enhanced_data.columns)}
- Date Range: {enhanced_data.index[0].strftime('%Y-%m-%d')} to {enhanced_data.index[-1].strftime('%Y-%m-%d')}
- New Technical Columns: {', '.join(new_columns[:8])}{'...' if len(new_columns) > 8 else ''}

{stats_summary}

📁 ENHANCED DATA SAVED: {filename if filename else 'Data not saved'}
- Location: {os.path.join(OUTPUT_DIR, filename) if filename else 'N/A'}
- Format: CSV with all original data + technical indicators

💡 USAGE NOTES:
- Enhanced data includes all original OHLCV data plus technical indicators
- Indicators with rolling windows will have NaN values for initial periods
- Data is ready for advanced analysis and visualization
- Can be used directly by stock_analyzer for enhanced charting
"""
        
        return summary
        
    except Exception as e:
        return f"apply_technical_indicators_and_transformations: Error processing {symbol}: {str(e)}"

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
    try:
        symbol = symbol.upper()
        
        # Load trained model
        if not model_file.endswith('.pkl'):
            model_file += '.pkl'
        model_filepath = os.path.join(OUTPUT_DIR, model_file)
        
        if not os.path.exists(model_filepath):
            available_models = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('_model.pkl')]
            return f"backtest_model_strategy: Model file '{model_file}' not found. Available models: {', '.join(available_models)}"
        
        with open(model_filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        scaler = model_data['scaler']
        feature_cols = model_data['feature_cols']
        target_days = model_data['target_days']
        
        # Load data
        if data_file:
            if not data_file.endswith('.csv'):
                data_file += '.csv'
            filepath = os.path.join(OUTPUT_DIR, data_file)
            if not os.path.exists(filepath):
                return f"backtest_model_strategy: Data file '{data_file}' not found."
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            data_source = f"file: {data_file}"
        else:
            # Find most recent enhanced data file
            enhanced_files = [f for f in os.listdir(OUTPUT_DIR) if 
                            f.startswith(f"apply_technical_indicators_and_transformations_{symbol}_") and f.endswith('.csv')]
            if enhanced_files:
                latest_file = max(enhanced_files, key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)))
                filepath = os.path.join(OUTPUT_DIR, latest_file)
                data = pd.read_csv(filepath, index_col=0, parse_dates=True)
                data_source = f"enhanced file: {latest_file}"
            else:
                return f"backtest_model_strategy: No enhanced data files found for {symbol}."
        
        # Ensure we have the required features
        missing_features = set(feature_cols) - set(data.columns)
        if missing_features:
            return f"backtest_model_strategy: Missing required features: {', '.join(missing_features)}"
        
        # Prepare data for backtesting
        backtest_data = data[feature_cols + ['Close']].dropna().copy()
        
        if len(backtest_data) < 50:
            return f"backtest_model_strategy: Insufficient data for backtesting. Only {len(backtest_data)} records available."
        
        # Make predictions
        X = backtest_data[feature_cols]
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        
        # Add predictions to data
        backtest_data['Predicted_Price'] = predictions
        backtest_data['Current_Price'] = backtest_data['Close']
        backtest_data['Predicted_Return'] = (backtest_data['Predicted_Price'] / backtest_data['Current_Price'] - 1) * 100
        
        # Generate trading signals based on strategy
        if strategy_type == "threshold":
            backtest_data['Signal'] = 0
            backtest_data.loc[backtest_data['Predicted_Return'] > threshold * 100, 'Signal'] = 1  # Buy
            backtest_data.loc[backtest_data['Predicted_Return'] < -threshold * 100, 'Signal'] = -1  # Sell
        
        elif strategy_type == "directional":
            backtest_data['Signal'] = 0
            backtest_data.loc[backtest_data['Predicted_Price'] > backtest_data['Current_Price'], 'Signal'] = 1  # Buy
            backtest_data.loc[backtest_data['Predicted_Price'] < backtest_data['Current_Price'], 'Signal'] = -1  # Sell
        
        elif strategy_type == "percentile":
            pred_return_75 = backtest_data['Predicted_Return'].quantile(0.75)
            pred_return_25 = backtest_data['Predicted_Return'].quantile(0.25)
            backtest_data['Signal'] = 0
            backtest_data.loc[backtest_data['Predicted_Return'] > pred_return_75, 'Signal'] = 1  # Buy top 25%
            backtest_data.loc[backtest_data['Predicted_Return'] < pred_return_25, 'Signal'] = -1  # Sell bottom 25%
        
        # Simulate trading
        portfolio_value = initial_capital
        cash = initial_capital
        shares = 0
        position = 0  # 0: no position, 1: long, -1: short
        
        portfolio_history = []
        trades = []
        
        for i, (date, row) in enumerate(backtest_data.iterrows()):
            current_price = row['Current_Price']
            signal = row['Signal']
            
            # Execute trades based on signals
            if signal == 1 and position <= 0:  # Buy signal
                if position == -1:  # Cover short position first
                    cash += shares * current_price * (1 - transaction_cost)
                    trades.append({
                        'date': date,
                        'action': 'cover_short',
                        'price': current_price,
                        'shares': shares,
                        'value': shares * current_price
                    })
                    shares = 0
                
                # Open long position
                shares_to_buy = cash // (current_price * (1 + transaction_cost))
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price * (1 + transaction_cost)
                    cash -= cost
                    shares = shares_to_buy
                    position = 1
                    trades.append({
                        'date': date,
                        'action': 'buy',
                        'price': current_price,
                        'shares': shares_to_buy,
                        'value': cost
                    })
            
            elif signal == -1 and position >= 0:  # Sell signal
                if position == 1:  # Sell long position first
                    cash += shares * current_price * (1 - transaction_cost)
                    trades.append({
                        'date': date,
                        'action': 'sell',
                        'price': current_price,
                        'shares': shares,
                        'value': shares * current_price
                    })
                    shares = 0
                
                # Open short position (simplified - assume we can short)
                shares_to_short = cash // (current_price * (1 + transaction_cost))
                if shares_to_short > 0:
                    cash += shares_to_short * current_price * (1 - transaction_cost)
                    shares = shares_to_short
                    position = -1
                    trades.append({
                        'date': date,
                        'action': 'short',
                        'price': current_price,
                        'shares': shares_to_short,
                        'value': shares_to_short * current_price
                    })
            
            # Calculate portfolio value
            if position == 1:  # Long position
                portfolio_value = cash + shares * current_price
            elif position == -1:  # Short position
                portfolio_value = cash - shares * current_price
            else:  # No position
                portfolio_value = cash
            
            portfolio_history.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': cash,
                'shares': shares,
                'position': position,
                'price': current_price,
                'signal': signal
            })
        
        # Convert to DataFrame for analysis
        portfolio_df = pd.DataFrame(portfolio_history)
        portfolio_df.set_index('date', inplace=True)
        
        # Calculate performance metrics
        portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
        portfolio_df['cumulative_returns'] = (1 + portfolio_df['returns']).cumprod() - 1
        
        # Buy and hold benchmark
        initial_shares_bh = initial_capital / backtest_data['Current_Price'].iloc[0]
        portfolio_df['buy_hold_value'] = initial_shares_bh * portfolio_df['price']
        portfolio_df['buy_hold_returns'] = portfolio_df['buy_hold_value'].pct_change()
        portfolio_df['buy_hold_cumulative'] = (1 + portfolio_df['buy_hold_returns']).cumprod() - 1
        
        # Performance metrics
        total_return = (portfolio_df['portfolio_value'].iloc[-1] / initial_capital - 1) * 100
        buy_hold_return = (portfolio_df['buy_hold_value'].iloc[-1] / initial_capital - 1) * 100
        
        # Calculate additional metrics
        annual_return = ((portfolio_df['portfolio_value'].iloc[-1] / initial_capital) ** (252 / len(portfolio_df)) - 1) * 100
        volatility = portfolio_df['returns'].std() * np.sqrt(252) * 100
        sharpe_ratio = (annual_return - 2) / volatility if volatility > 0 else 0  # Assuming 2% risk-free rate
        
        # Maximum drawdown
        rolling_max = portfolio_df['portfolio_value'].expanding().max()
        drawdown = (portfolio_df['portfolio_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        # Win rate
        profitable_trades = len([t for t in trades if 
                               (t['action'] == 'sell' and len([t2 for t2 in trades if t2['action'] == 'buy' and t2['date'] < t['date']]) > 0) or
                               (t['action'] == 'cover_short' and len([t2 for t2 in trades if t2['action'] == 'short' and t2['date'] < t['date']]) > 0)])
        total_trades = len([t for t in trades if t['action'] in ['sell', 'cover_short']])
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Save results if requested
        results_filename = None
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            results = {
                'symbol': symbol,
                'model_file': model_file,
                'data_source': data_source,
                'strategy_type': strategy_type,
                'backtest_period': {
                    'start': str(portfolio_df.index[0]),
                    'end': str(portfolio_df.index[-1]),
                    'days': len(portfolio_df)
                },
                'performance': {
                    'total_return_pct': float(total_return),
                    'annualized_return_pct': float(annual_return),
                    'volatility_pct': float(volatility),
                    'sharpe_ratio': float(sharpe_ratio),
                    'max_drawdown_pct': float(max_drawdown),
                    'win_rate_pct': float(win_rate),
                    'total_trades': int(total_trades),
                    'final_portfolio_value': float(portfolio_df['portfolio_value'].iloc[-1])
                },
                'benchmark': {
                    'buy_hold_return_pct': float(buy_hold_return),
                    'excess_return_pct': float(total_return - buy_hold_return)
                },
                'trades': trades
            }
            
            results_filename = f"backtest_model_strategy_{symbol}_results_{timestamp}.json"
            results_filepath = os.path.join(OUTPUT_DIR, results_filename)
            with open(results_filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save portfolio history
            portfolio_filename = f"backtest_model_strategy_{symbol}_portfolio_{timestamp}.csv"
            portfolio_filepath = os.path.join(OUTPUT_DIR, portfolio_filename)
            portfolio_df.to_csv(portfolio_filepath)
        
        # Create summary
        summary = f"""backtest_model_strategy: Completed backtesting for {symbol} using {strategy_type} strategy:

🎯 BACKTEST CONFIGURATION:
- Symbol: {symbol}
- Model: {model_file}
- Data Source: {data_source}
- Strategy: {strategy_type.title()}
- Initial Capital: ${initial_capital:,.0f}
- Transaction Cost: {transaction_cost:.1%}
- Period: {portfolio_df.index[0].strftime('%Y-%m-%d')} to {portfolio_df.index[-1].strftime('%Y-%m-%d')} ({len(portfolio_df)} days)

📊 STRATEGY PERFORMANCE:
- Final Portfolio Value: ${portfolio_df['portfolio_value'].iloc[-1]:,.2f}
- Total Return: {total_return:.2f}%
- Annualized Return: {annual_return:.2f}%
- Volatility: {volatility:.2f}%
- Sharpe Ratio: {sharpe_ratio:.2f}
- Maximum Drawdown: {max_drawdown:.2f}%
- Win Rate: {win_rate:.1f}%
- Total Trades: {total_trades}

📈 BENCHMARK COMPARISON:
- Buy & Hold Return: {buy_hold_return:.2f}%
- Strategy Excess Return: {total_return - buy_hold_return:.2f}%
- Alpha: {'Positive' if total_return > buy_hold_return else 'Negative'}

💡 PERFORMANCE ASSESSMENT:
- Risk-Adjusted Performance: {'Excellent' if sharpe_ratio > 1.5 else 'Good' if sharpe_ratio > 1.0 else 'Fair' if sharpe_ratio > 0.5 else 'Poor'}
- Strategy Effectiveness: {'Outperforming' if total_return > buy_hold_return else 'Underperforming'} vs Buy & Hold
- Maximum Risk: {max_drawdown:.1f}% portfolio decline from peak
- Trading Activity: {'High' if total_trades > len(portfolio_df) * 0.1 else 'Moderate' if total_trades > len(portfolio_df) * 0.05 else 'Low'} frequency

📁 FILES SAVED:
- Detailed Results: {results_filename if results_filename else 'Not saved'}
- Portfolio History: {portfolio_filename if save_results else 'Not saved'}

⚠️ IMPORTANT NOTES:
- Results are based on historical data and may not reflect future performance
- Transaction costs and slippage are simplified
- Short selling assumptions may not reflect real market conditions
- This is for analysis purposes only, not investment advice
"""
        
        return summary
        
    except Exception as e:
        return f"backtest_model_strategy: Error running backtest for {symbol}: {str(e)}"
    

# Add these imports to the top of tools.py
import pickle
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

@tool
def train_xgboost_price_predictor(
    symbol: str,
    source_file: Optional[str] = None,
    target_days: int = 1,
    test_size: float = 0.2,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    save_model: bool = True
) -> str:
    """
    Train an XGBoost model to predict stock prices using technical indicators.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')
        source_file: Enhanced CSV file with technical indicators (if None, uses most recent)
        target_days: Number of days ahead to predict (1 = next day, 5 = next week)
        test_size: Proportion of data for testing (0.2 = 20%)
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate for boosting
        save_model: Whether to save the trained model
        
    Returns:
        String with model performance metrics and file locations
    """
    try:
        symbol = symbol.upper()
        
        # Load enhanced data with technical indicators
        if source_file:
            if not source_file.endswith('.csv'):
                source_file += '.csv'
            filepath = os.path.join(OUTPUT_DIR, source_file)
            if not os.path.exists(filepath):
                return f"train_xgboost_price_predictor: Source file '{source_file}' not found."
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            data_source = f"file: {source_file}"
        else:
            # Find most recent enhanced data file
            enhanced_files = [f for f in os.listdir(OUTPUT_DIR) if 
                            f.startswith(f"apply_technical_indicators_and_transformations_{symbol}_") and f.endswith('.csv')]
            if enhanced_files:
                latest_file = max(enhanced_files, key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)))
                filepath = os.path.join(OUTPUT_DIR, latest_file)
                data = pd.read_csv(filepath, index_col=0, parse_dates=True)
                data_source = f"enhanced file: {latest_file}"
            else:
                return f"train_xgboost_price_predictor: No enhanced data files found for {symbol}. Please run technical indicators first."
        
        if data.empty or len(data) < 50:
            return f"train_xgboost_price_predictor: Insufficient data for {symbol}. Need at least 50 records."
        
        # Prepare features and target
        # Target: future price (shifted by target_days)
        data['Target'] = data['Close'].shift(-target_days)
        
        # Select feature columns (exclude basic OHLCV and target)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'Target']
        feature_cols = [col for col in data.columns if col not in exclude_cols and not data[col].isnull().all()]
        
        if len(feature_cols) < 3:
            return f"train_xgboost_price_predictor: Insufficient technical indicators. Found only {len(feature_cols)} features. Need at least 3."
        
        # Remove rows with NaN values
        model_data = data[feature_cols + ['Target']].dropna()
        
        if len(model_data) < 30:
            return f"train_xgboost_price_predictor: Insufficient clean data after removing NaN values. Only {len(model_data)} records available."
        
        X = model_data[feature_cols]
        y = model_data['Target']
        
        # Time series split for more realistic evaluation
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = []
        
        # Also do a simple train-test split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train XGBoost model
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        # Cross-validation scores
        for train_idx, val_idx in tscv.split(X):
            X_cv_train, X_cv_val = X.iloc[train_idx], X.iloc[val_idx]
            y_cv_train, y_cv_val = y.iloc[train_idx], y.iloc[val_idx]
            
            X_cv_train_scaled = scaler.fit_transform(X_cv_train)
            X_cv_val_scaled = scaler.transform(X_cv_val)
            
            cv_model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                                     learning_rate=learning_rate, random_state=42)
            cv_model.fit(X_cv_train_scaled, y_cv_train)
            cv_pred = cv_model.predict(X_cv_val_scaled)
            cv_scores.append(r2_score(y_cv_val, cv_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save model and results if requested
        model_filename = None
        results_filename = None
        if save_model:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save model
            model_filename = f"train_xgboost_price_predictor_{symbol}_model_{timestamp}.pkl"
            model_filepath = os.path.join(OUTPUT_DIR, model_filename)
            with open(model_filepath, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'scaler': scaler,
                    'feature_cols': feature_cols,
                    'target_days': target_days,
                    'symbol': symbol
                }, f)
            
            # Save results
            results = {
                'symbol': symbol,
                'target_days': target_days,
                'data_source': data_source,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'features_used': feature_cols,
                'model_params': {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'learning_rate': learning_rate
                },
                'performance': {
                    'train_rmse': float(train_rmse),
                    'test_rmse': float(test_rmse),
                    'train_mae': float(train_mae),
                    'test_mae': float(test_mae),
                    'train_r2': float(train_r2),
                    'test_r2': float(test_r2),
                    'cv_r2_mean': float(np.mean(cv_scores)),
                    'cv_r2_std': float(np.std(cv_scores))
                },
                'feature_importance': feature_importance.to_dict('records')
            }
            
            results_filename = f"train_xgboost_price_predictor_{symbol}_results_{timestamp}.json"
            results_filepath = os.path.join(OUTPUT_DIR, results_filename)
            with open(results_filepath, 'w') as f:
                json.dump(results, f, indent=2)
        
        # Create summary
        summary = f"""train_xgboost_price_predictor: Successfully trained XGBoost model for {symbol}:

🤖 MODEL CONFIGURATION:
- Algorithm: XGBoost Regressor
- Symbol: {symbol}
- Target: {target_days}-day ahead price prediction
- Data Source: {data_source}
- Features: {len(feature_cols)} technical indicators
- Training Samples: {len(X_train)}
- Test Samples: {len(X_test)}

⚙️ HYPERPARAMETERS:
- N Estimators: {n_estimators}
- Max Depth: {max_depth}
- Learning Rate: {learning_rate}

📊 MODEL PERFORMANCE:
- Training RMSE: ${train_rmse:.3f}
- Test RMSE: ${test_rmse:.3f}
- Training MAE: ${train_mae:.3f}
- Test MAE: ${test_mae:.3f}
- Training R²: {train_r2:.3f}
- Test R²: {test_r2:.3f}
- Cross-Val R²: {np.mean(cv_scores):.3f} (±{np.std(cv_scores):.3f})

🎯 TOP 5 IMPORTANT FEATURES:
{chr(10).join([f"  {i+1}. {row['feature']}: {row['importance']:.3f}" for i, row in feature_importance.head().iterrows()])}

📁 FILES SAVED:
- Model: {model_filename if model_filename else 'Not saved'}
- Results: {results_filename if results_filename else 'Not saved'}

💡 MODEL INSIGHTS:
- Overfitting Risk: {'High' if train_r2 - test_r2 > 0.1 else 'Low' if train_r2 - test_r2 < 0.05 else 'Moderate'}
- Model Quality: {'Excellent' if test_r2 > 0.8 else 'Good' if test_r2 > 0.6 else 'Fair' if test_r2 > 0.4 else 'Poor'}
- Prediction Accuracy: ±${test_mae:.2f} average error on test set
"""
        
        return summary
        
    except Exception as e:
        return f"train_xgboost_price_predictor: Error training model for {symbol}: {str(e)}"


@tool
def train_random_forest_price_predictor(
    symbol: str,
    source_file: Optional[str] = None,
    target_days: int = 1,
    test_size: float = 0.2,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    save_model: bool = True
) -> str:
    """
    Train a Random Forest model to predict stock prices using technical indicators.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')
        source_file: Enhanced CSV file with technical indicators (if None, uses most recent)
        target_days: Number of days ahead to predict (1 = next day, 5 = next week)
        test_size: Proportion of data for testing (0.2 = 20%)
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of trees (None = unlimited)
        min_samples_split: Minimum samples required to split a node
        save_model: Whether to save the trained model
        
    Returns:
        String with model performance metrics and file locations
    """
    try:
        symbol = symbol.upper()
        
        # Load enhanced data with technical indicators
        if source_file:
            if not source_file.endswith('.csv'):
                source_file += '.csv'
            filepath = os.path.join(OUTPUT_DIR, source_file)
            if not os.path.exists(filepath):
                return f"train_random_forest_price_predictor: Source file '{source_file}' not found."
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            data_source = f"file: {source_file}"
        else:
            # Find most recent enhanced data file
            enhanced_files = [f for f in os.listdir(OUTPUT_DIR) if 
                            f.startswith(f"apply_technical_indicators_and_transformations_{symbol}_") and f.endswith('.csv')]
            if enhanced_files:
                latest_file = max(enhanced_files, key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)))
                filepath = os.path.join(OUTPUT_DIR, latest_file)
                data = pd.read_csv(filepath, index_col=0, parse_dates=True)
                data_source = f"enhanced file: {latest_file}"
            else:
                return f"train_random_forest_price_predictor: No enhanced data files found for {symbol}. Please run technical indicators first."
        
        if data.empty or len(data) < 50:
            return f"train_random_forest_price_predictor: Insufficient data for {symbol}. Need at least 50 records."
        
        # Prepare features and target
        data['Target'] = data['Close'].shift(-target_days)
        
        # Select feature columns
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'Target']
        feature_cols = [col for col in data.columns if col not in exclude_cols and not data[col].isnull().all()]
        
        if len(feature_cols) < 3:
            return f"train_random_forest_price_predictor: Insufficient technical indicators. Found only {len(feature_cols)} features."
        
        # Remove rows with NaN values
        model_data = data[feature_cols + ['Target']].dropna()
        
        if len(model_data) < 30:
            return f"train_random_forest_price_predictor: Insufficient clean data. Only {len(model_data)} records available."
        
        X = model_data[feature_cols]
        y = model_data['Target']
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = []
        
        # Train-test split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        # Cross-validation
        for train_idx, val_idx in tscv.split(X):
            X_cv_train, X_cv_val = X.iloc[train_idx], X.iloc[val_idx]
            y_cv_train, y_cv_val = y.iloc[train_idx], y.iloc[val_idx]
            
            X_cv_train_scaled = scaler.fit_transform(X_cv_train)
            X_cv_val_scaled = scaler.transform(X_cv_val)
            
            cv_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                          min_samples_split=min_samples_split, random_state=42)
            cv_model.fit(X_cv_train_scaled, y_cv_train)
            cv_pred = cv_model.predict(X_cv_val_scaled)
            cv_scores.append(r2_score(y_cv_val, cv_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save model and results
        model_filename = None
        results_filename = None
        if save_model:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save model
            model_filename = f"train_random_forest_price_predictor_{symbol}_model_{timestamp}.pkl"
            model_filepath = os.path.join(OUTPUT_DIR, model_filename)
            with open(model_filepath, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'scaler': scaler,
                    'feature_cols': feature_cols,
                    'target_days': target_days,
                    'symbol': symbol
                }, f)
            
            # Save results
            results = {
                'symbol': symbol,
                'target_days': target_days,
                'data_source': data_source,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'features_used': feature_cols,
                'model_params': {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split
                },
                'performance': {
                    'train_rmse': float(train_rmse),
                    'test_rmse': float(test_rmse),
                    'train_mae': float(train_mae),
                    'test_mae': float(test_mae),
                    'train_r2': float(train_r2),
                    'test_r2': float(test_r2),
                    'cv_r2_mean': float(np.mean(cv_scores)),
                    'cv_r2_std': float(np.std(cv_scores))
                },
                'feature_importance': feature_importance.to_dict('records')
            }
            
            results_filename = f"train_random_forest_price_predictor_{symbol}_results_{timestamp}.json"
            results_filepath = os.path.join(OUTPUT_DIR, results_filename)
            with open(results_filepath, 'w') as f:
                json.dump(results, f, indent=2)
        
        # Create summary
        summary = f"""train_random_forest_price_predictor: Successfully trained Random Forest model for {symbol}:

🌲 MODEL CONFIGURATION:
- Algorithm: Random Forest Regressor
- Symbol: {symbol}
- Target: {target_days}-day ahead price prediction
- Data Source: {data_source}
- Features: {len(feature_cols)} technical indicators
- Training Samples: {len(X_train)}
- Test Samples: {len(X_test)}

⚙️ HYPERPARAMETERS:
- N Estimators: {n_estimators}
- Max Depth: {max_depth if max_depth else 'Unlimited'}
- Min Samples Split: {min_samples_split}

📊 MODEL PERFORMANCE:
- Training RMSE: ${train_rmse:.3f}
- Test RMSE: ${test_rmse:.3f}
- Training MAE: ${train_mae:.3f}
- Test MAE: ${test_mae:.3f}
- Training R²: {train_r2:.3f}
- Test R²: {test_r2:.3f}
- Cross-Val R²: {np.mean(cv_scores):.3f} (±{np.std(cv_scores):.3f})

🎯 TOP 5 IMPORTANT FEATURES:
{chr(10).join([f"  {i+1}. {row['feature']}: {row['importance']:.3f}" for i, row in feature_importance.head().iterrows()])}

📁 FILES SAVED:
- Model: {model_filename if model_filename else 'Not saved'}
- Results: {results_filename if results_filename else 'Not saved'}

💡 MODEL INSIGHTS:
- Overfitting Risk: {'High' if train_r2 - test_r2 > 0.1 else 'Low' if train_r2 - test_r2 < 0.05 else 'Moderate'}
- Model Quality: {'Excellent' if test_r2 > 0.8 else 'Good' if test_r2 > 0.6 else 'Fair' if test_r2 > 0.4 else 'Poor'}
- Prediction Accuracy: ±${test_mae:.2f} average error on test set
- Model Stability: {'High' if np.std(cv_scores) < 0.1 else 'Moderate' if np.std(cv_scores) < 0.2 else 'Low'}
"""
        
        return summary
        
    except Exception as e:
        return f"train_random_forest_price_predictor: Error training model for {symbol}: {str(e)}"
    

@tool
def debug_file_system(
    symbol: Optional[str] = None,
    show_content: bool = False
) -> str:
    """
    Debug tool to check file system status and help troubleshoot file-related issues.
    
    Args:
        symbol: Stock symbol to check files for (optional)
        show_content: Whether to show sample content from CSV files
        
    Returns:
        String with detailed file system information
    """
    try:
        # Check if output directory exists
        if not os.path.exists(OUTPUT_DIR):
            return f"debug_file_system: Output directory '{OUTPUT_DIR}' does not exist. Creating it now..."
        
        # Get all files in output directory
        try:
            all_files = os.listdir(OUTPUT_DIR)
        except Exception as e:
            return f"debug_file_system: Error reading output directory: {str(e)}"
        
        if not all_files:
            return f"debug_file_system: Output directory '{OUTPUT_DIR}' is empty. No files found."
        
        # Categorize files
        csv_files = [f for f in all_files if f.endswith('.csv')]
        pkl_files = [f for f in all_files if f.endswith('.pkl')]
        json_files = [f for f in all_files if f.endswith('.json')]
        html_files = [f for f in all_files if f.endswith('.html')]
        other_files = [f for f in all_files if not any(f.endswith(ext) for ext in ['.csv', '.pkl', '.json', '.html'])]
        
        # If symbol specified, filter for that symbol
        if symbol:
            symbol = symbol.upper()
            symbol_csv = [f for f in csv_files if symbol in f.upper()]
            symbol_pkl = [f for f in pkl_files if symbol in f.upper()]
            symbol_json = [f for f in json_files if symbol in f.upper()]
            symbol_html = [f for f in html_files if symbol in f.upper()]
        
        # Build detailed report
        report = f"debug_file_system: File system analysis for output directory '{OUTPUT_DIR}':\n\n"
        
        # Overall statistics
        report += f"📊 DIRECTORY OVERVIEW:\n"
        report += f"- Total Files: {len(all_files)}\n"
        report += f"- CSV Files: {len(csv_files)}\n"
        report += f"- Model Files (.pkl): {len(pkl_files)}\n"
        report += f"- Results Files (.json): {len(json_files)}\n"
        report += f"- Chart Files (.html): {len(html_files)}\n"
        report += f"- Other Files: {len(other_files)}\n\n"
        
        # Symbol-specific analysis
        if symbol:
            report += f"🎯 FILES FOR SYMBOL '{symbol}':\n"
            report += f"- CSV Files: {len(symbol_csv)}\n"
            report += f"- Model Files: {len(symbol_pkl)}\n"
            report += f"- Results Files: {len(symbol_json)}\n"
            report += f"- Chart Files: {len(symbol_html)}\n\n"
            
            if symbol_csv:
                report += f"📋 {symbol} CSV FILES:\n"
                for file in sorted(symbol_csv):
                    filepath = os.path.join(OUTPUT_DIR, file)
                    size = os.path.getsize(filepath)
                    modified = datetime.fromtimestamp(os.path.getmtime(filepath))
                    report += f"  - {file} ({size:,} bytes, {modified.strftime('%Y-%m-%d %H:%M:%S')})\n"
                report += "\n"
        
        # All CSV files with details
        if csv_files:
            report += f"📈 ALL CSV FILES:\n"
            for file in sorted(csv_files):
                filepath = os.path.join(OUTPUT_DIR, file)
                size = os.path.getsize(filepath)
                modified = datetime.fromtimestamp(os.path.getmtime(filepath))
                
                # Try to get basic info about the CSV
                try:
                    df = pd.read_csv(filepath, index_col=0, parse_dates=True, nrows=5)
                    rows_info = f"{len(df)} rows (sample)"
                    cols_info = f"{len(df.columns)} columns"
                    date_range = f"dates: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}" if len(df) > 0 else "no dates"
                except:
                    rows_info = "unknown rows"
                    cols_info = "unknown columns"
                    date_range = "date info unavailable"
                
                report += f"  - {file}\n"
                report += f"    Size: {size:,} bytes | Modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}\n"
                report += f"    Data: {rows_info}, {cols_info}, {date_range}\n"
                
                # Show content sample if requested
                if show_content:
                    try:
                        sample_df = pd.read_csv(filepath, index_col=0, parse_dates=True, nrows=3)
                        report += f"    Sample:\n{sample_df.to_string()}\n"
                    except Exception as e:
                        report += f"    Sample: Error reading content - {str(e)}\n"
                report += "\n"
        
        # Model files
        if pkl_files:
            report += f"🤖 MODEL FILES (.pkl):\n"
            for file in sorted(pkl_files):
                filepath = os.path.join(OUTPUT_DIR, file)
                size = os.path.getsize(filepath)
                modified = datetime.fromtimestamp(os.path.getmtime(filepath))
                report += f"  - {file} ({size:,} bytes, {modified.strftime('%Y-%m-%d %H:%M:%S')})\n"
            report += "\n"
        
        # Results files
        if json_files:
            report += f"📊 RESULTS FILES (.json):\n"
            for file in sorted(json_files):
                filepath = os.path.join(OUTPUT_DIR, file)
                size = os.path.getsize(filepath)
                modified = datetime.fromtimestamp(os.path.getmtime(filepath))
                report += f"  - {file} ({size:,} bytes, {modified.strftime('%Y-%m-%d %H:%M:%S')})\n"
            report += "\n"
        
        # File type patterns
        report += f"🔍 FILE PATTERN ANALYSIS:\n"
        
        # Count by pattern
        fetch_files = [f for f in csv_files if f.startswith('fetch_yahoo_finance_data_')]
        indicator_files = [f for f in csv_files if f.startswith('apply_technical_indicators_and_transformations_')]
        model_files_xgb = [f for f in pkl_files if 'xgboost' in f]
        model_files_rf = [f for f in pkl_files if 'random_forest' in f]
        backtest_files = [f for f in json_files if 'backtest' in f]
        
        report += f"- Raw Stock Data Files: {len(fetch_files)}\n"
        report += f"- Enhanced Indicator Files: {len(indicator_files)}\n"
        report += f"- XGBoost Models: {len(model_files_xgb)}\n"
        report += f"- Random Forest Models: {len(model_files_rf)}\n"
        report += f"- Backtest Results: {len(backtest_files)}\n\n"
        
        # Recommendations
        report += f"💡 RECOMMENDATIONS:\n"
        if not csv_files:
            report += "- No CSV files found. Run fetch_yahoo_finance_data first.\n"
        elif not indicator_files:
            report += "- No enhanced data files found. Run apply_technical_indicators_and_transformations.\n"
        elif not pkl_files:
            report += "- No trained models found. Train models using train_xgboost_price_predictor or train_random_forest_price_predictor.\n"
        elif not backtest_files:
            report += "- No backtest results found. Run backtest_model_strategy to evaluate models.\n"
        else:
            report += "- All file types present. System appears to be working correctly.\n"
        
        # Show file paths for debugging
        report += f"\n🛠️ DEBUGGING INFO:\n"
        report += f"- Output Directory: {os.path.abspath(OUTPUT_DIR)}\n"
        report += f"- Directory Exists: {os.path.exists(OUTPUT_DIR)}\n"
        report += f"- Directory Writable: {os.access(OUTPUT_DIR, os.W_OK) if os.path.exists(OUTPUT_DIR) else 'N/A'}\n"
        report += f"- Current Working Directory: {os.getcwd()}\n"
        
        return report
        
    except Exception as e:
        return f"debug_file_system: Error during analysis: {str(e)}"