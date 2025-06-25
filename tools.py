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