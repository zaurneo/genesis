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
import pickle
import json
import numpy as np

# Tavily tool
tavily_tool = TavilySearchResults(max_results=5)

# Create output directory if it doesn't exist
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    print(f"🔄 read_csv_data: Starting to read CSV file '{filename}'...")
    
    try:
        # Determine file path
        if filepath:
            file_path = filepath
        else:
            file_path = os.path.join(OUTPUT_DIR, filename)
        
        if not os.path.exists(file_path):
            if not filepath:  # Only show available files if using output directory
                available_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.csv')]
                result = f"File '{filename}' not found. Available CSV files: {', '.join(available_files) if available_files else 'None'}"
            else:
                result = f"File not found at path: {file_path}"
            print(f"❌ read_csv_data: {result}")
            return result
        
        # Read the CSV file
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        if data.empty:
            result = f"The file '{filename}' is empty."
            print(f"⚠️ read_csv_data: {result}")
            return result
        
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
- File Size: {os.path.getsize(file_path):,} bytes

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
        
        print(f"✅ read_csv_data: Successfully read and analyzed '{filename}' with {len(data)} records")
        return summary
        
    except Exception as e:
        error_msg = f"Error reading CSV file '{filename}': {str(e)}"
        print(f"❌ read_csv_data: {error_msg}")
        return error_msg


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
    print(f"🔄 fetch_yahoo_finance_data: Starting to fetch data for {symbol.upper()}...")
    
    try:
        # Create ticker object
        ticker = yf.Ticker(symbol.upper())
        
        # Fetch historical data
        data = ticker.history(period=period, interval=interval)
        
        if data.empty:
            result = f"No data found for symbol {symbol}. Please check if the symbol is correct."
            print(f"❌ fetch_yahoo_finance_data: {result}")
            return result
        
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
        
        print(f"✅ fetch_yahoo_finance_data: Successfully fetched and saved data for {symbol.upper()}")
        return summary
        
    except Exception as e:
        error_msg = f"Error fetching data for {symbol}: {str(e)}"
        print(f"❌ fetch_yahoo_finance_data: {error_msg}")
        return error_msg



@tool
def get_available_stock_periods_and_intervals() -> str:
    """
    Get information about available periods and intervals for Yahoo Finance data.
    
    Returns:
        String with available options for periods and intervals
    """
    print("🔄 get_available_stock_periods_and_intervals: Starting to provide period and interval information...")
    
    result = """
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
    
    print("✅ get_available_stock_periods_and_intervals: Successfully provided period and interval information")
    return result



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
    print(f"🔄 visualize_stock_data: Starting to create {chart_type} visualization for {symbol.upper()}...")
    
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
            result = f"No data available for {symbol} to create visualization."
            print(f"❌ visualize_stock_data: {result}")
            return result
        
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
        
        print(f"✅ visualize_stock_data: Successfully created {chart_type} visualization for {symbol}")
        return summary
        
    except Exception as e:
        error_msg = f"Error creating visualization for {symbol}: {str(e)}"
        print(f"❌ visualize_stock_data: {error_msg}")
        return error_msg



@tool
def list_saved_stock_files() -> str:
    """
    List all saved stock data files and charts in the output directory.
    
    Returns:
        String listing all available files with details
    """
    print("🔄 list_saved_stock_files: Starting to list all saved files...")
    
    try:
        if not os.path.exists(OUTPUT_DIR):
            result = "Output directory does not exist. No files have been saved yet."
            print(f"⚠️ list_saved_stock_files: {result}")
            return result
        
        files = os.listdir(OUTPUT_DIR)
        if not files:
            result = "No files found in the output directory."
            print(f"⚠️ list_saved_stock_files: {result}")
            return result
        
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
        
        print(f"✅ list_saved_stock_files: Successfully listed {len(files)} files")
        return summary
        
    except Exception as e:
        error_msg = f"Error listing files: {str(e)}"
        print(f"❌ list_saved_stock_files: {error_msg}")
        return error_msg


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
    print(f"🔄 save_text_to_file: Starting to save content to '{filename}.{file_format}'...")
    
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
        
        print(f"✅ save_text_to_file: Successfully saved '{clean_filename}' ({file_size:,} bytes)")
        return summary
        
    except Exception as e:
        error_msg = f"Error saving file: {str(e)}"
        print(f"❌ save_text_to_file: {error_msg}")
        return error_msg
    

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
    print(f"🔄 apply_technical_indicators_and_transformations: Starting to apply indicators for {symbol.upper()}...")
    
    try:
        import numpy as np
        
        symbol = symbol.upper()
        
        # Load data
        if source_file:
            if not source_file.endswith('.csv'):
                source_file += '.csv'
            filepath = os.path.join(OUTPUT_DIR, source_file)
            if not os.path.exists(filepath):
                result = f"apply_technical_indicators_and_transformations: Source file '{source_file}' not found in output directory."
                print(f"❌ apply_technical_indicators_and_transformations: {result}")
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
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                data_source = f"newly fetched data (period: {period})"
        
        if data.empty:
            result = f"apply_technical_indicators_and_transformations: No data available for {symbol}."
            print(f"❌ apply_technical_indicators_and_transformations: {result}")
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
        
        print(f"✅ apply_technical_indicators_and_transformations: Successfully applied {len(applied_indicators)} indicators for {symbol}")
        return summary
        
    except Exception as e:
        error_msg = f"apply_technical_indicators_and_transformations: Error processing {symbol}: {str(e)}"
        print(f"❌ apply_technical_indicators_and_transformations: {error_msg}")
        return error_msg

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
    print(f"🔄 backtest_model_strategy: Starting backtesting for {symbol.upper()} using {strategy_type} strategy...")
    
    try:
        symbol = symbol.upper()
        
        # Load trained model
        if not model_file.endswith('.pkl'):
            model_file += '.pkl'
        model_filepath = os.path.join(OUTPUT_DIR, model_file)
        
        if not os.path.exists(model_filepath):
            available_models = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('_model.pkl')]
            result = f"backtest_model_strategy: Model file '{model_file}' not found. Available models: {', '.join(available_models)}"
            print(f"❌ backtest_model_strategy: {result}")
            return result
        
        with open(model_filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        scaler = model_data['scaler']
        feature_cols = model_data['feature_cols']
        target_days = model_data['target_days']
        
        # Load data using read_csv_data functionality
        if data_file:
            if not data_file.endswith('.csv'):
                data_file += '.csv'
            filepath = os.path.join(OUTPUT_DIR, data_file)
            if not os.path.exists(filepath):
                result = f"backtest_model_strategy: Data file '{data_file}' not found."
                print(f"❌ backtest_model_strategy: {result}")
                return result
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
                result = f"backtest_model_strategy: No enhanced data files found for {symbol}."
                print(f"❌ backtest_model_strategy: {result}")
                return result
        
        # Ensure we have the required features
        missing_features = set(feature_cols) - set(data.columns)
        if missing_features:
            result = f"backtest_model_strategy: Missing required features: {', '.join(missing_features)}"
            print(f"❌ backtest_model_strategy: {result}")
            return result
        
        # Prepare data for backtesting
        backtest_data = data[feature_cols + ['Close']].dropna().copy()
        
        if len(backtest_data) < 50:
            result = f"backtest_model_strategy: Insufficient data for backtesting. Only {len(backtest_data)} records available."
            print(f"❌ backtest_model_strategy: {result}")
            return result
        
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
        
        print(f"✅ backtest_model_strategy: Successfully completed backtesting for {symbol} ({total_trades} trades, {total_return:.2f}% return)")
        return summary
        
    except Exception as e:
        error_msg = f"backtest_model_strategy: Error running backtest for {symbol}: {str(e)}"
        print(f"❌ backtest_model_strategy: {error_msg}")
        return error_msg
    

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
    print(f"🔄 train_xgboost_price_predictor: Starting XGBoost training for {symbol.upper()}...")
    
    try:
        import numpy as np
        
        symbol = symbol.upper()
        
        # Load enhanced data with technical indicators
        if source_file:
            if not source_file.endswith('.csv'):
                source_file += '.csv'
            filepath = os.path.join(OUTPUT_DIR, source_file)
            if not os.path.exists(filepath):
                result = f"train_xgboost_price_predictor: Source file '{source_file}' not found."
                print(f"❌ train_xgboost_price_predictor: {result}")
                return result
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
                result = f"train_xgboost_price_predictor: No enhanced data files found for {symbol}. Please run technical indicators first."
                print(f"❌ train_xgboost_price_predictor: {result}")
                return result
        
        if data.empty or len(data) < 50:
            result = f"train_xgboost_price_predictor: Insufficient data for {symbol}. Need at least 50 records."
            print(f"❌ train_xgboost_price_predictor: {result}")
            return result
        
        # Prepare features and target
        # Target: future price (shifted by target_days)
        data['Target'] = data['Close'].shift(-target_days)
        
        # Select feature columns (exclude basic OHLCV and target)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'Target']
        feature_cols = [col for col in data.columns if col not in exclude_cols and not data[col].isnull().all()]
        
        if len(feature_cols) < 3:
            result = f"train_xgboost_price_predictor: Insufficient technical indicators. Found only {len(feature_cols)} features. Need at least 3."
            print(f"❌ train_xgboost_price_predictor: {result}")
            return result
        
        # Remove rows with NaN values
        model_data = data[feature_cols + ['Target']].dropna()
        
        if len(model_data) < 30:
            result = f"train_xgboost_price_predictor: Insufficient clean data after removing NaN values. Only {len(model_data)} records available."
            print(f"❌ train_xgboost_price_predictor: {result}")
            return result
        
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
        
        print(f"✅ train_xgboost_price_predictor: Successfully trained XGBoost model for {symbol} (R²: {test_r2:.3f})")
        return summary
        
    except Exception as e:
        error_msg = f"train_xgboost_price_predictor: Error training model for {symbol}: {str(e)}"
        print(f"❌ train_xgboost_price_predictor: {error_msg}")
        return error_msg


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
    print(f"🔄 train_random_forest_price_predictor: Starting Random Forest training for {symbol.upper()}...")
    
    try:
        symbol = symbol.upper()
        
        # Load enhanced data with technical indicators using read_csv_data functionality
        if source_file:
            if not source_file.endswith('.csv'):
                source_file += '.csv'
            filepath = os.path.join(OUTPUT_DIR, source_file)
            if not os.path.exists(filepath):
                result = f"train_random_forest_price_predictor: Source file '{source_file}' not found."
                print(f"❌ train_random_forest_price_predictor: {result}")
                return result
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
                result = f"train_random_forest_price_predictor: No enhanced data files found for {symbol}. Please run technical indicators first."
                print(f"❌ train_random_forest_price_predictor: {result}")
                return result
        
        if data.empty or len(data) < 50:
            result = f"train_random_forest_price_predictor: Insufficient data for {symbol}. Need at least 50 records."
            print(f"❌ train_random_forest_price_predictor: {result}")
            return result
        
        # Prepare features and target
        data['Target'] = data['Close'].shift(-target_days)
        
        # Select feature columns
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'Target']
        feature_cols = [col for col in data.columns if col not in exclude_cols and not data[col].isnull().all()]
        
        if len(feature_cols) < 3:
            result = f"train_random_forest_price_predictor: Insufficient technical indicators. Found only {len(feature_cols)} features."
            print(f"❌ train_random_forest_price_predictor: {result}")
            return result
        
        # Remove rows with NaN values
        model_data = data[feature_cols + ['Target']].dropna()
        
        if len(model_data) < 30:
            result = f"train_random_forest_price_predictor: Insufficient clean data. Only {len(model_data)} records available."
            print(f"❌ train_random_forest_price_predictor: {result}")
            return result
        
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
        
        print(f"✅ train_random_forest_price_predictor: Successfully trained Random Forest model for {symbol} (R²: {test_r2:.3f})")
        return summary
        
    except Exception as e:
        error_msg = f"train_random_forest_price_predictor: Error training model for {symbol}: {str(e)}"
        print(f"❌ train_random_forest_price_predictor: {error_msg}")
        return error_msg
    

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
    print(f"🔄 debug_file_system: Starting file system analysis{' for ' + symbol.upper() if symbol else ''}...")
    
    try:
        # Check if output directory exists
        if not os.path.exists(OUTPUT_DIR):
            result = f"debug_file_system: Output directory '{OUTPUT_DIR}' does not exist. Creating it now..."
            print(f"⚠️ debug_file_system: {result}")
            return result
        
        # Get all files in output directory
        try:
            all_files = os.listdir(OUTPUT_DIR)
        except Exception as e:
            result = f"debug_file_system: Error reading output directory: {str(e)}"
            print(f"❌ debug_file_system: {result}")
            return result
        
        if not all_files:
            result = f"debug_file_system: Output directory '{OUTPUT_DIR}' is empty. No files found."
            print(f"⚠️ debug_file_system: {result}")
            return result
        
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
        
        print(f"✅ debug_file_system: Analysis completed - found {len(all_files)} files total")
        return report
        
    except Exception as e:
        error_msg = f"debug_file_system: Error during analysis: {str(e)}"
        print(f"❌ debug_file_system: {error_msg}")
        return error_msg
    



# Add this tool to tools.py

@tool
def generate_comprehensive_html_report(
    symbol: str,
    report_title: Optional[str] = None,
    include_charts: bool = True,
    include_model_results: bool = True,
    include_backtest_results: bool = True,
    custom_analysis: Optional[str] = None,
    save_report: bool = True
) -> str:
    """
    Generate a comprehensive HTML report with all analysis, charts, and results.
    Creates a professional, interactive HTML document with embedded charts and detailed analysis.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')
        report_title: Custom title for the report (if None, auto-generates)
        include_charts: Whether to embed interactive charts in the report
        include_model_results: Whether to include ML model performance results
        include_backtest_results: Whether to include backtesting analysis
        custom_analysis: Additional custom analysis text to include
        save_report: Whether to save the HTML report to file
        
    Returns:
        String description of the generated report and its location
    """
    print(f"🔄 generate_comprehensive_html_report: Starting comprehensive HTML report generation for {symbol.upper()}...")
    
    try:
        symbol = symbol.upper()
        
        # Default report title
        if not report_title:
            report_title = f"{symbol} Stock Analysis Report"
        
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Collect all available files
        available_files = os.listdir(OUTPUT_DIR) if os.path.exists(OUTPUT_DIR) else []
        
        # Filter files by symbol
        symbol_files = {
            'data_files': [f for f in available_files if f.endswith('.csv') and symbol in f.upper()],
            'chart_files': [f for f in available_files if f.endswith('.html') and symbol in f.upper()],
            'model_files': [f for f in available_files if f.endswith('.pkl') and symbol in f.upper()],
            'result_files': [f for f in available_files if f.endswith('.json') and symbol in f.upper()]
        }
        
        # Start building HTML report
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report_title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid #007acc;
        }}
        .header h1 {{
            color: #007acc;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}
        .header .subtitle {{
            color: #666;
            font-size: 1.2em;
        }}
        .section {{
            margin: 30px 0;
            padding: 20px;
            border-left: 4px solid #007acc;
            background-color: #f9f9f9;
        }}
        .section h2 {{
            color: #005580;
            margin-top: 0;
            font-size: 1.8em;
        }}
        .section h3 {{
            color: #007acc;
            font-size: 1.3em;
            margin-top: 25px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-left: 4px solid #28a745;
        }}
        .metric-card.negative {{
            border-left-color: #dc3545;
        }}
        .metric-card h4 {{
            margin: 0 0 10px 0;
            color: #333;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .metric-card .value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #007acc;
        }}
        .chart-container {{
            margin: 30px 0;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        .chart-title {{
            font-size: 1.4em;
            color: #005580;
            margin-bottom: 15px;
            text-align: center;
        }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .data-table th, .data-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .data-table th {{
            background-color: #007acc;
            color: white;
            font-weight: bold;
        }}
        .data-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .file-list {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #ddd;
            margin: 15px 0;
        }}
        .file-list ul {{
            margin: 0;
            padding-left: 20px;
        }}
        .file-list li {{
            margin: 5px 0;
            color: #666;
        }}
        .analysis-text {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            line-height: 1.8;
        }}
        .footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 2px solid #eee;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
        .warning {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
            padding: 15px;
            margin: 20px 0;
            color: #856404;
        }}
        .success {{
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 5px;
            padding: 15px;
            margin: 20px 0;
            color: #155724;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{report_title}</h1>
            <div class="subtitle">Comprehensive Stock Analysis Report</div>
            <div class="subtitle">Generated on {timestamp}</div>
        </div>
"""

        # Executive Summary Section
        html_content += f"""
        <div class="section">
            <h2>📈 Executive Summary</h2>
            <p>This comprehensive analysis report for <strong>{symbol}</strong> includes stock data analysis, 
            technical indicators, machine learning predictions, and backtesting results. The report combines 
            multiple data sources and analytical methods to provide actionable insights.</p>
        </div>
"""

        # Stock Data Analysis Section
        if symbol_files['data_files']:
            html_content += f"""
        <div class="section">
            <h2>📊 Stock Data Analysis</h2>
"""
            
            # Find the most recent data file and analyze it
            latest_data_file = max(symbol_files['data_files'], 
                                 key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)))
            
            try:
                # Read the latest data file
                filepath = os.path.join(OUTPUT_DIR, latest_data_file)
                data = pd.read_csv(filepath, index_col=0, parse_dates=True)
                
                if not data.empty:
                    # Calculate key metrics
                    current_price = data['Close'].iloc[-1]
                    opening_price = data['Close'].iloc[0]
                    price_change = current_price - opening_price
                    price_change_pct = (price_change / opening_price * 100)
                    period_high = data['High'].max()
                    period_low = data['Low'].min()
                    volatility = data['Close'].pct_change().std() * 100
                    avg_volume = data['Volume'].mean()
                    
                    html_content += f"""
            <div class="metrics-grid">
                <div class="metric-card {'negative' if price_change < 0 else ''}">
                    <h4>Current Price</h4>
                    <div class="value">${current_price:.2f}</div>
                </div>
                <div class="metric-card {'negative' if price_change_pct < 0 else ''}">
                    <h4>Price Change</h4>
                    <div class="value">{price_change_pct:+.2f}%</div>
                </div>
                <div class="metric-card">
                    <h4>Period High</h4>
                    <div class="value">${period_high:.2f}</div>
                </div>
                <div class="metric-card">
                    <h4>Period Low</h4>
                    <div class="value">${period_low:.2f}</div>
                </div>
                <div class="metric-card">
                    <h4>Volatility</h4>
                    <div class="value">{volatility:.2f}%</div>
                </div>
                <div class="metric-card">
                    <h4>Average Volume</h4>
                    <div class="value">{avg_volume:,.0f}</div>
                </div>
            </div>
            
            <h3>📋 Data Summary</h3>
            <ul>
                <li><strong>Data Period:</strong> {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}</li>
                <li><strong>Total Records:</strong> {len(data):,}</li>
                <li><strong>Data Source:</strong> {latest_data_file}</li>
                <li><strong>Available Indicators:</strong> {len([col for col in data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']])}</li>
            </ul>
"""
            except Exception as e:
                html_content += f"""
            <div class="warning">
                <strong>Warning:</strong> Could not analyze data file {latest_data_file}. Error: {str(e)}
            </div>
"""
            
            # List all data files
            html_content += f"""
            <h3>📁 Available Data Files</h3>
            <div class="file-list">
                <ul>
"""
            for file in sorted(symbol_files['data_files']):
                file_path = os.path.join(OUTPUT_DIR, file)
                file_size = os.path.getsize(file_path)
                modified = datetime.fromtimestamp(os.path.getmtime(file_path))
                html_content += f"                    <li>{file} ({file_size:,} bytes, modified {modified.strftime('%Y-%m-%d %H:%M')})</li>\n"
            
            html_content += """
                </ul>
            </div>
        </div>
"""

        # Interactive Charts Section
        if include_charts and symbol_files['chart_files']:
            html_content += f"""
        <div class="section">
            <h2>📈 Interactive Charts</h2>
            <p>The following interactive charts provide visual analysis of {symbol} stock performance:</p>
"""
            
            for chart_file in sorted(symbol_files['chart_files']):
                chart_path = os.path.join(OUTPUT_DIR, chart_file)
                chart_name = chart_file.replace('visualize_stock_data_', '').replace('.html', '').replace('_', ' ').title()
                
                # Try to read and embed the chart HTML
                try:
                    with open(chart_path, 'r', encoding='utf-8') as f:
                        chart_html = f.read()
                    
                    # Extract just the Plotly div and script parts
                    if 'plotly-div' in chart_html or 'Plotly.newPlot' in chart_html:
                        html_content += f"""
            <div class="chart-container">
                <div class="chart-title">{chart_name}</div>
                {chart_html}
            </div>
"""
                    else:
                        html_content += f"""
            <div class="chart-container">
                <div class="chart-title">{chart_name}</div>
                <p><strong>Chart File:</strong> <a href="{chart_file}" target="_blank">{chart_file}</a></p>
                <p><em>Interactive chart available in separate file.</em></p>
            </div>
"""
                except Exception as e:
                    html_content += f"""
            <div class="chart-container">
                <div class="chart-title">{chart_name}</div>
                <div class="warning">Could not embed chart. File: {chart_file}</div>
            </div>
"""
            
            html_content += """
        </div>
"""

        # Machine Learning Model Results Section
        if include_model_results and (symbol_files['model_files'] or symbol_files['result_files']):
            html_content += f"""
        <div class="section">
            <h2>🤖 Machine Learning Model Results</h2>
"""
            
            # Find model result files
            model_result_files = [f for f in symbol_files['result_files'] if 'model' in f and 'backtest' not in f]
            
            if model_result_files:
                for result_file in sorted(model_result_files):
                    try:
                        result_path = os.path.join(OUTPUT_DIR, result_file)
                        with open(result_path, 'r') as f:
                            results = json.load(f)
                        
                        model_type = 'XGBoost' if 'xgboost' in result_file else 'Random Forest' if 'random_forest' in result_file else 'ML Model'
                        
                        html_content += f"""
            <h3>📊 {model_type} Performance</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h4>Test R² Score</h4>
                    <div class="value">{results['performance']['test_r2']:.3f}</div>
                </div>
                <div class="metric-card">
                    <h4>Test RMSE</h4>
                    <div class="value">${results['performance']['test_rmse']:.3f}</div>
                </div>
                <div class="metric-card">
                    <h4>Test MAE</h4>
                    <div class="value">${results['performance']['test_mae']:.3f}</div>
                </div>
                <div class="metric-card">
                    <h4>Cross-Val R²</h4>
                    <div class="value">{results['performance']['cv_r2_mean']:.3f}</div>
                </div>
            </div>
            
            <h4>🎯 Top Features</h4>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Feature</th>
                        <th>Importance</th>
                    </tr>
                </thead>
                <tbody>
"""
                        
                        for i, feature in enumerate(results['feature_importance'][:10]):
                            html_content += f"""
                    <tr>
                        <td>{i+1}</td>
                        <td>{feature['feature']}</td>
                        <td>{feature['importance']:.4f}</td>
                    </tr>
"""
                        
                        html_content += """
                </tbody>
            </table>
"""
                    except Exception as e:
                        html_content += f"""
            <div class="warning">
                Could not load model results from {result_file}. Error: {str(e)}
            </div>
"""
            else:
                html_content += """
            <div class="warning">
                No machine learning model results found. Train models using XGBoost or Random Forest tools.
            </div>
"""
            
            # List model files
            if symbol_files['model_files']:
                html_content += f"""
            <h3>💾 Trained Models</h3>
            <div class="file-list">
                <ul>
"""
                for file in sorted(symbol_files['model_files']):
                    file_path = os.path.join(OUTPUT_DIR, file)
                    file_size = os.path.getsize(file_path)
                    modified = datetime.fromtimestamp(os.path.getmtime(file_path))
                    html_content += f"                    <li>{file} ({file_size:,} bytes, modified {modified.strftime('%Y-%m-%d %H:%M')})</li>\n"
                
                html_content += """
                </ul>
            </div>
"""
            
            html_content += """
        </div>
"""

        # Backtesting Results Section
        if include_backtest_results:
            backtest_files = [f for f in symbol_files['result_files'] if 'backtest' in f]
            
            if backtest_files:
                html_content += f"""
        <div class="section">
            <h2>📊 Backtesting Results</h2>
"""
                
                for backtest_file in sorted(backtest_files):
                    try:
                        backtest_path = os.path.join(OUTPUT_DIR, backtest_file)
                        with open(backtest_path, 'r') as f:
                            backtest_results = json.load(f)
                        
                        strategy_type = backtest_results.get('strategy_type', 'Unknown').title()
                        
                        html_content += f"""
            <h3>🎯 {strategy_type} Strategy Performance</h3>
            <div class="metrics-grid">
                <div class="metric-card {'negative' if backtest_results['performance']['total_return_pct'] < 0 else ''}">
                    <h4>Total Return</h4>
                    <div class="value">{backtest_results['performance']['total_return_pct']:.2f}%</div>
                </div>
                <div class="metric-card">
                    <h4>Sharpe Ratio</h4>
                    <div class="value">{backtest_results['performance']['sharpe_ratio']:.2f}</div>
                </div>
                <div class="metric-card {'negative' if backtest_results['performance']['max_drawdown_pct'] < -10 else ''}">
                    <h4>Max Drawdown</h4>
                    <div class="value">{backtest_results['performance']['max_drawdown_pct']:.2f}%</div>
                </div>
                <div class="metric-card">
                    <h4>Win Rate</h4>
                    <div class="value">{backtest_results['performance']['win_rate_pct']:.1f}%</div>
                </div>
                <div class="metric-card">
                    <h4>Total Trades</h4>
                    <div class="value">{backtest_results['performance']['total_trades']}</div>
                </div>
                <div class="metric-card {'negative' if backtest_results['benchmark']['excess_return_pct'] < 0 else ''}">
                    <h4>Excess Return</h4>
                    <div class="value">{backtest_results['benchmark']['excess_return_pct']:+.2f}%</div>
                </div>
            </div>
            
            <h4>📊 Strategy vs Benchmark</h4>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Strategy</th>
                        <th>Buy & Hold</th>
                        <th>Difference</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Total Return</td>
                        <td>{backtest_results['performance']['total_return_pct']:.2f}%</td>
                        <td>{backtest_results['benchmark']['buy_hold_return_pct']:.2f}%</td>
                        <td>{backtest_results['benchmark']['excess_return_pct']:+.2f}%</td>
                    </tr>
                    <tr>
                        <td>Final Value</td>
                        <td>${backtest_results['performance']['final_portfolio_value']:,.2f}</td>
                        <td>-</td>
                        <td>-</td>
                    </tr>
                </tbody>
            </table>
"""
                    except Exception as e:
                        html_content += f"""
            <div class="warning">
                Could not load backtesting results from {backtest_file}. Error: {str(e)}
            </div>
"""
                
                html_content += """
        </div>
"""

        # Custom Analysis Section
        if custom_analysis:
            html_content += f"""
        <div class="section">
            <h2>💡 Additional Analysis</h2>
            <div class="analysis-text">
                {custom_analysis.replace('\n', '<br>')}
            </div>
        </div>
"""

        # File Summary Section
        html_content += f"""
        <div class="section">
            <h2>📁 File Summary</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h4>Data Files</h4>
                    <div class="value">{len(symbol_files['data_files'])}</div>
                </div>
                <div class="metric-card">
                    <h4>Chart Files</h4>
                    <div class="value">{len(symbol_files['chart_files'])}</div>
                </div>
                <div class="metric-card">
                    <h4>Model Files</h4>
                    <div class="value">{len(symbol_files['model_files'])}</div>
                </div>
                <div class="metric-card">
                    <h4>Result Files</h4>
                    <div class="value">{len(symbol_files['result_files'])}</div>
                </div>
            </div>
        </div>
"""

        # Footer
        html_content += f"""
        <div class="footer">
            <p><strong>Disclaimer:</strong> This analysis is for informational purposes only and should not be considered as investment advice. 
            Past performance does not guarantee future results. Please consult with a qualified financial advisor before making investment decisions.</p>
            <p>Report generated on {timestamp} | Total files analyzed: {sum(len(files) for files in symbol_files.values())}</p>
        </div>
    </div>
</body>
</html>"""

        # Save the report if requested
        report_filename = None
        if save_report:
            report_filename = f"generate_comprehensive_html_report_{symbol}_report_{file_timestamp}.html"
            report_filepath = os.path.join(OUTPUT_DIR, report_filename)
            
            with open(report_filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            file_size = os.path.getsize(report_filepath)
        
        # Create summary
        summary = f"""generate_comprehensive_html_report: Successfully generated comprehensive HTML report for {symbol}:

📊 REPORT OVERVIEW:
- Symbol: {symbol}
- Title: {report_title}
- Generated: {timestamp}
- Content Sections: Executive Summary, Stock Data, {'Charts, ' if include_charts else ''}{'ML Models, ' if include_model_results else ''}{'Backtesting, ' if include_backtest_results else ''}File Summary

📈 INCLUDED ANALYSIS:
- Data Files Analyzed: {len(symbol_files['data_files'])}
- Interactive Charts: {len(symbol_files['chart_files'])} {'(embedded)' if include_charts else '(referenced)'}
- ML Model Results: {len([f for f in symbol_files['result_files'] if 'model' in f and 'backtest' not in f])}
- Backtesting Results: {len([f for f in symbol_files['result_files'] if 'backtest' in f])}

🎨 REPORT FEATURES:
- Professional HTML styling with responsive design
- Interactive elements and modern UI
- Comprehensive metrics and visualizations
- Performance comparisons and insights
- Embedded charts and analysis
- Mobile-friendly layout

📁 REPORT SAVED: {report_filename if report_filename else 'Report not saved'}
- Location: {os.path.join(OUTPUT_DIR, report_filename) if report_filename else 'N/A'}
- File Size: {file_size:,} bytes ({len(html_content):,} characters)
- Format: Interactive HTML with embedded CSS styling

💡 USAGE:
- Open the HTML file in any web browser
- All charts and data are self-contained
- Professional presentation ready for sharing
- Includes all analysis and interactive elements
"""
        
        print(f"✅ generate_comprehensive_html_report: Successfully generated comprehensive HTML report for {symbol}")
        return summary
        
    except Exception as e:
        error_msg = f"generate_comprehensive_html_report: Error generating report for {symbol}: {str(e)}"
        print(f"❌ generate_comprehensive_html_report: {error_msg}")
        return error_msg
    

# Add this new function to tools.py, after the existing visualization functions

@tool
def visualize_backtesting_results(
    symbol: str,
    chart_type: Literal["portfolio_performance", "trading_signals", "model_predictions", "combined"] = "combined",
    backtest_files: Optional[str] = None,
    strategies_to_show: Optional[str] = None,
    include_benchmark: bool = True,
    save_chart: bool = True
) -> str:
    """
    Create comprehensive visualizations of backtesting results showing model performance,
    trading signals, and portfolio value compared to actual stock data and benchmarks.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')
        chart_type: Type of visualization:
                   - "portfolio_performance": Portfolio value vs benchmark over time
                   - "trading_signals": Stock price with buy/sell signals
                   - "model_predictions": Model predictions vs actual prices
                   - "combined": All visualizations in subplots
        backtest_files: Specific backtest result files to visualize (comma-separated)
        strategies_to_show: Specific strategies to display (comma-separated, e.g., "xgboost,random_forest")
        include_benchmark: Whether to include buy-and-hold benchmark comparison
        save_chart: Whether to save the interactive chart as HTML file
        
    Returns:
        String description of the created visualization and insights
    """
    print(f"[DEBUG] [visualize_backtesting_results] Starting backtesting visualization for {symbol.upper()}...")
    
    try:
        symbol = symbol.upper()
        
        # Find all backtesting result files for this symbol
        available_files = os.listdir(OUTPUT_DIR) if os.path.exists(OUTPUT_DIR) else []
        
        if backtest_files:
            # Use specified files
            specified_files = [f.strip() for f in backtest_files.split(',')]
            backtest_result_files = []
            for file in specified_files:
                if not file.endswith('.json'):
                    file += '.json'
                if file in available_files and symbol in file.upper() and 'backtest' in file:
                    backtest_result_files.append(file)
        else:
            # Find all backtest files for this symbol
            backtest_result_files = [f for f in available_files if 
                                   f.endswith('.json') and symbol in f.upper() and 'backtest' in f]
        
        if not backtest_result_files:
            result = f"[DEBUG] [visualize_backtesting_results] No backtesting result files found for {symbol}."
            print(result)
            return result
        
        print(f"[DEBUG] [visualize_backtesting_results] Found {len(backtest_result_files)} backtest files for {symbol}")
        
        # Load backtesting results and portfolio data
        all_backtest_data = {}
        portfolio_data = {}
        
        for result_file in backtest_result_files:
            try:
                # Load backtest results
                result_path = os.path.join(OUTPUT_DIR, result_file)
                with open(result_path, 'r') as f:
                    backtest_results = json.load(f)
                
                # Find corresponding portfolio CSV file
                portfolio_file = result_file.replace('_results_', '_portfolio_').replace('.json', '.csv')
                portfolio_path = os.path.join(OUTPUT_DIR, portfolio_file)
                
                if os.path.exists(portfolio_path):
                    portfolio_df = pd.read_csv(portfolio_path, index_col=0, parse_dates=True)
                    
                    # Store data with cleaner strategy name
                    model_type = "XGBoost" if "xgboost" in backtest_results.get('model_file', '') else \
                                "RandomForest" if "random_forest" in backtest_results.get('model_file', '') else "Model"
                    strategy_type = backtest_results.get('strategy_type', 'unknown').replace('_', ' ').title()
                    strategy_key = f"{model_type}_{strategy_type}"
                    
                    all_backtest_data[strategy_key] = backtest_results
                    portfolio_data[strategy_key] = portfolio_df
                    
                    print(f"[DEBUG] [visualize_backtesting_results] Loaded data for strategy: {strategy_key}")
                else:
                    print(f"[DEBUG] [visualize_backtesting_results] Portfolio file not found: {portfolio_file}")
                    
            except Exception as e:
                print(f"[DEBUG] [visualize_backtesting_results] Error loading {result_file}: {str(e)}")
                continue
        
        if not all_backtest_data:
            result = f"[DEBUG] [visualize_backtesting_results] No valid backtesting data could be loaded for {symbol}."
            print(result)
            return result
        
        # Filter strategies if specified
        if strategies_to_show:
            filter_strategies = [s.strip().lower() for s in strategies_to_show.split(',')]
            filtered_data = {}
            filtered_portfolio = {}
            
            for strategy_key in all_backtest_data.keys():
                strategy_lower = strategy_key.lower()
                if any(filter_str in strategy_lower for filter_str in filter_strategies):
                    filtered_data[strategy_key] = all_backtest_data[strategy_key]
                    filtered_portfolio[strategy_key] = portfolio_data[strategy_key]
            
            if filtered_data:
                all_backtest_data = filtered_data
                portfolio_data = filtered_portfolio
                print(f"[DEBUG] [visualize_backtesting_results] Filtered to {len(all_backtest_data)} strategies")
            else:
                print(f"[DEBUG] [visualize_backtesting_results] No strategies matched filter, showing all")
        
        # Create visualizations based on chart_type
        if chart_type == "portfolio_performance":
            fig = create_portfolio_performance_chart(symbol, all_backtest_data, portfolio_data, include_benchmark)
        elif chart_type == "trading_signals":
            fig = create_trading_signals_chart(symbol, all_backtest_data, portfolio_data)
        elif chart_type == "model_predictions":
            fig = create_model_predictions_chart(symbol, all_backtest_data, portfolio_data)
        elif chart_type == "combined":
            fig = create_combined_backtesting_chart(symbol, all_backtest_data, portfolio_data, include_benchmark)
        else:
            result = f"[DEBUG] [visualize_backtesting_results] Unknown chart type: {chart_type}"
            print(result)
            return result
        
        # Save chart if requested
        chart_filename = None
        if save_chart:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_filename = f"visualize_backtesting_results_{symbol}_{chart_type}_chart_{timestamp}.html"
            chart_filepath = os.path.join(OUTPUT_DIR, chart_filename)
            
            plot(fig, filename=chart_filepath, auto_open=False, include_plotlyjs=True)
            file_size = os.path.getsize(chart_filepath)
            print(f"[DEBUG] [visualize_backtesting_results] Saved chart: {chart_filename} ({file_size:,} bytes)")
        
        # Calculate summary statistics
        performance_summary = {}
        for strategy_key, backtest_data in all_backtest_data.items():
            performance_summary[strategy_key] = {
                'total_return': backtest_data['performance']['total_return_pct'],
                'sharpe_ratio': backtest_data['performance']['sharpe_ratio'],
                'max_drawdown': backtest_data['performance']['max_drawdown_pct'],
                'total_trades': backtest_data['performance']['total_trades']
            }
        
        # Find best performing strategy
        best_strategy = max(performance_summary.keys(), 
                          key=lambda x: performance_summary[x]['total_return'])
        
        summary = f"""[DEBUG] [visualize_backtesting_results] Successfully created backtesting visualization for {symbol}:

🎯 BACKTESTING VISUALIZATION SUMMARY:
- Symbol: {symbol}
- Chart Type: {chart_type.replace('_', ' ').title()}
- Strategies Analyzed: {len(all_backtest_data)}
- Strategies Filter: {'Applied' if strategies_to_show else 'None'}
- Benchmark Included: {'Yes' if include_benchmark else 'No'}

📊 STRATEGIES COMPARED:
{chr(10).join([f"  • {strategy}: {data['total_return']:.2f}% return, {data['sharpe_ratio']:.2f} Sharpe, {data['total_trades']} trades" 
               for strategy, data in performance_summary.items()])}

🏆 BEST PERFORMING STRATEGY:
- Strategy: {best_strategy}
- Total Return: {performance_summary[best_strategy]['total_return']:.2f}%
- Sharpe Ratio: {performance_summary[best_strategy]['sharpe_ratio']:.2f}
- Max Drawdown: {performance_summary[best_strategy]['max_drawdown']:.2f}%

📈 VISUALIZATION FEATURES:
- Independent charts with side legends for better readability
- Proper date formatting and cumulative return calculations
- Clean strategy names (e.g., "XGBoost Directional" instead of filenames)
- Interactive Plotly charts with zoom/pan capabilities
- Multiple strategy performance comparison
- Trading signals overlaid on price data
- Portfolio value evolution over time
- Model predictions vs actual prices
- Benchmark comparison (if enabled)

📁 CHART SAVED: {chart_filename if chart_filename else 'Not saved'}
- Location: {os.path.join(OUTPUT_DIR, chart_filename) if chart_filename else 'N/A'}
- Format: Interactive HTML with embedded JavaScript

💡 KEY INSIGHTS:
- Strategy Count: {len(performance_summary)} backtesting strategies analyzed
- Performance Range: {min(data['total_return'] for data in performance_summary.values()):.2f}% to {max(data['total_return'] for data in performance_summary.values()):.2f}%
- Best Risk-Adjusted: {max(performance_summary.keys(), key=lambda x: performance_summary[x]['sharpe_ratio'])} (Sharpe: {max(data['sharpe_ratio'] for data in performance_summary.values()):.2f})
- Most Active: {max(performance_summary.keys(), key=lambda x: performance_summary[x]['total_trades'])} ({max(data['total_trades'] for data in performance_summary.values())} trades)

The interactive visualization allows detailed analysis of model performance, trading patterns, and risk-return characteristics across all backtested strategies with improved legends and date formatting.
"""
        
        print(f"[DEBUG] [visualize_backtesting_results] Completed successfully for {symbol}")
        return summary
        
    except Exception as e:
        error_msg = f"[DEBUG] [visualize_backtesting_results] Error creating visualization for {symbol}: {str(e)}"
        print(error_msg)
        return error_msg


def create_portfolio_performance_chart(symbol, all_backtest_data, portfolio_data, include_benchmark):
    """Create portfolio performance comparison chart."""
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    color_idx = 0
    
    for strategy_key, portfolio_df in portfolio_data.items():
        # Calculate cumulative returns properly
        initial_value = portfolio_df['portfolio_value'].iloc[0]
        cumulative_returns = (portfolio_df['portfolio_value'] / initial_value - 1) * 100
        
        # Ensure we have valid dates and returns
        valid_dates = portfolio_df.index
        valid_returns = cumulative_returns
        
        fig.add_trace(go.Scatter(
            x=valid_dates,
            y=valid_returns,
            mode='lines',
            name=strategy_key,
            line=dict(color=colors[color_idx % len(colors)], width=2),
            hovertemplate=f'<b>{strategy_key}</b><br>Date: %{{x}}<br>Return: %{{y:.2f}}%<extra></extra>'
        ))
        color_idx += 1
        
        # Add buy and hold benchmark if available and requested
        if include_benchmark and 'buy_hold_cumulative' in portfolio_df.columns:
            benchmark_returns = portfolio_df['buy_hold_cumulative'] * 100
            fig.add_trace(go.Scatter(
                x=valid_dates,
                y=benchmark_returns,
                mode='lines',
                name='Buy & Hold Benchmark',
                line=dict(color='gray', width=2, dash='dash'),
                hovertemplate='<b>Buy & Hold</b><br>Date: %{{x}}<br>Return: %{{y:.2f}}%<extra></extra>'
            ))
            include_benchmark = False  # Only add benchmark once
    
    fig.update_layout(
        title=f'{symbol} Portfolio Performance Comparison',
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)',
        template='plotly_white',
        hovermode='x unified',
        width=1200,
        height=600,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        )
    )
    
    # Ensure proper date formatting on x-axis
    fig.update_xaxes(
        tickformat='%Y-%m-%d',
        tickangle=45
    )
    
    return fig


def create_trading_signals_chart(symbol, all_backtest_data, portfolio_data):
    """Create trading signals overlaid on price chart."""
    # Create individual charts for each strategy instead of subplots
    if len(portfolio_data) == 1:
        # Single strategy - use simple layout
        strategy_key = list(portfolio_data.keys())[0]
        portfolio_df = list(portfolio_data.values())[0]
        
        fig = go.Figure()
        
        # Price line
        fig.add_trace(go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df['price'],
            mode='lines',
            name='Price',
            line=dict(color='blue', width=2)
        ))
        
        # Buy signals (signal = 1)
        buy_signals = portfolio_df[portfolio_df['signal'] == 1]
        if not buy_signals.empty:
            fig.add_trace(go.Scatter(
                x=buy_signals.index,
                y=buy_signals['price'],
                mode='markers',
                name='Buy Signal',
                marker=dict(color='green', size=10, symbol='triangle-up')
            ))
        
        # Sell signals (signal = -1)
        sell_signals = portfolio_df[portfolio_df['signal'] == -1]
        if not sell_signals.empty:
            fig.add_trace(go.Scatter(
                x=sell_signals.index,
                y=sell_signals['price'],
                mode='markers',
                name='Sell Signal',
                marker=dict(color='red', size=10, symbol='triangle-down')
            ))
        
        fig.update_layout(
            title=f'{symbol} Trading Signals - {strategy_key}',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            template='plotly_white',
            width=1200,
            height=600,
            hovermode='x unified',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1
            )
        )
        
    else:
        # Multiple strategies - use subplots but with better spacing
        fig = sp.make_subplots(
            rows=len(portfolio_data), cols=1,
            subplot_titles=[f'{symbol} - {strategy}' for strategy in portfolio_data.keys()],
            vertical_spacing=0.15,
            specs=[[{"secondary_y": False}] for _ in range(len(portfolio_data))]
        )
        
        row = 1
        for strategy_key, portfolio_df in portfolio_data.items():
            # Price line
            fig.add_trace(go.Scatter(
                x=portfolio_df.index,
                y=portfolio_df['price'],
                mode='lines',
                name=f'{strategy_key} Price',
                line=dict(color='blue', width=1),
                showlegend=(row == 1)
            ), row=row, col=1)
            
            # Buy signals (signal = 1)
            buy_signals = portfolio_df[portfolio_df['signal'] == 1]
            if not buy_signals.empty:
                fig.add_trace(go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['price'],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(color='green', size=8, symbol='triangle-up'),
                    showlegend=(row == 1)
                ), row=row, col=1)
            
            # Sell signals (signal = -1)
            sell_signals = portfolio_df[portfolio_df['signal'] == -1]
            if not sell_signals.empty:
                fig.add_trace(go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals['price'],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(color='red', size=8, symbol='triangle-down'),
                    showlegend=(row == 1)
                ), row=row, col=1)
            
            row += 1
        
        fig.update_layout(
            title=f'{symbol} Trading Signals Analysis',
            template='plotly_white',
            width=1200,
            height=400 * len(portfolio_data),
            hovermode='x unified',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1
            )
        )
        
        # Update y-axis titles
        for i in range(len(portfolio_data)):
            fig.update_yaxes(title_text='Price ($)', row=i+1, col=1)
    
    return fig


def create_model_predictions_chart(symbol, all_backtest_data, portfolio_data):
    """Create model predictions vs actual prices chart."""
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    color_idx = 0
    
    # Get actual price data (use first portfolio data as reference)
    first_portfolio = list(portfolio_data.values())[0]
    fig.add_trace(go.Scatter(
        x=first_portfolio.index,
        y=first_portfolio['price'],
        mode='lines',
        name='Actual Price',
        line=dict(color='black', width=2),
        hovertemplate='<b>Actual Price</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
    ))
    
    # Add model predictions (if available in backtest data)
    for strategy_key in all_backtest_data.keys():
        # Try to find corresponding enhanced data file with predictions
        enhanced_files = [f for f in os.listdir(OUTPUT_DIR) if 
                         f.startswith(f"apply_technical_indicators_and_transformations_{symbol}_") and f.endswith('.csv')]
        
        if enhanced_files:
            # Use most recent enhanced file
            latest_enhanced = max(enhanced_files, key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)))
            enhanced_path = os.path.join(OUTPUT_DIR, latest_enhanced)
            
            try:
                enhanced_data = pd.read_csv(enhanced_path, index_col=0, parse_dates=True)
                
                # If we have prediction data, add it
                if 'Predicted_Price' in enhanced_data.columns:
                    # Align with portfolio data timeframe
                    portfolio_df = portfolio_data[strategy_key]
                    common_dates = enhanced_data.index.intersection(portfolio_df.index)
                    
                    if not common_dates.empty:
                        aligned_predictions = enhanced_data.loc[common_dates, 'Predicted_Price']
                        
                        fig.add_trace(go.Scatter(
                            x=common_dates,
                            y=aligned_predictions,
                            mode='lines',
                            name=f'{strategy_key} Predictions',
                            line=dict(color=colors[color_idx % len(colors)], width=1, dash='dot'),
                            opacity=0.7,
                            hovertemplate=f'<b>{strategy_key} Prediction</b><br>Date: %{{x}}<br>Price: $%{{y:.2f}}<extra></extra>'
                        ))
                        color_idx += 1
            except Exception as e:
                print(f"[DEBUG] [create_model_predictions_chart] Could not load predictions for {strategy_key}: {e}")
                continue
    
    fig.update_layout(
        title=f'{symbol} Model Predictions vs Actual Prices',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_white',
        hovermode='x unified',
        width=1200,
        height=600,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        )
    )
    
    return fig


def create_combined_backtesting_chart(symbol, all_backtest_data, portfolio_data, include_benchmark):
    """Create combined chart with all backtesting visualizations."""
    # Calculate the number of subplots needed
    n_strategies = len(portfolio_data)
    n_rows = 3  # Portfolio performance, signals, predictions
    
    subplot_titles = [
        f'{symbol} Portfolio Performance Comparison',
        f'{symbol} Trading Signals (All Strategies)',
        f'{symbol} Model Predictions vs Actual Prices'
    ]
    
    fig = sp.make_subplots(
        rows=n_rows, cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
        row_heights=[0.4, 0.35, 0.25]
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # 1. Portfolio Performance (Row 1)
    color_idx = 0
    for strategy_key, portfolio_df in portfolio_data.items():
        initial_value = portfolio_df['portfolio_value'].iloc[0]
        cumulative_returns = (portfolio_df['portfolio_value'] / initial_value - 1) * 100
        
        # Ensure we have valid dates and returns
        valid_dates = portfolio_df.index
        valid_returns = cumulative_returns
        
        fig.add_trace(go.Scatter(
            x=valid_dates,
            y=valid_returns,
            mode='lines',
            name=strategy_key,
            line=dict(color=colors[color_idx % len(colors)], width=2),
            legendgroup='strategies',
            hovertemplate=f'<b>{strategy_key}</b><br>Date: %{{x}}<br>Return: %{{y:.2f}}%<extra></extra>'
        ), row=1, col=1)
        color_idx += 1
    
    # Add benchmark if requested
    if include_benchmark and portfolio_data:
        first_portfolio = list(portfolio_data.values())[0]
        if 'buy_hold_cumulative' in first_portfolio.columns:
            benchmark_returns = first_portfolio['buy_hold_cumulative'] * 100
            fig.add_trace(go.Scatter(
                x=first_portfolio.index,
                y=benchmark_returns,
                mode='lines',
                name='Buy & Hold',
                line=dict(color='gray', width=2, dash='dash'),
                legendgroup='benchmark',
                hovertemplate='<b>Buy & Hold</b><br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
            ), row=1, col=1)
    
    # 2. Trading Signals (Row 2) - Combined view
    first_portfolio = list(portfolio_data.values())[0]
    
    # Price line
    fig.add_trace(go.Scatter(
        x=first_portfolio.index,
        y=first_portfolio['price'],
        mode='lines',
        name='Price',
        line=dict(color='black', width=1),
        legendgroup='price',
        showlegend=True
    ), row=2, col=1)
    
    # Combine all signals
    all_buy_dates = []
    all_buy_prices = []
    all_sell_dates = []
    all_sell_prices = []
    
    for strategy_key, portfolio_df in portfolio_data.items():
        buy_signals = portfolio_df[portfolio_df['signal'] == 1]
        sell_signals = portfolio_df[portfolio_df['signal'] == -1]
        
        all_buy_dates.extend(buy_signals.index)
        all_buy_prices.extend(buy_signals['price'])
        all_sell_dates.extend(sell_signals.index)
        all_sell_prices.extend(sell_signals['price'])
    
    if all_buy_dates:
        fig.add_trace(go.Scatter(
            x=all_buy_dates,
            y=all_buy_prices,
            mode='markers',
            name='Buy Signals',
            marker=dict(color='green', size=8, symbol='triangle-up'),
            legendgroup='signals'
        ), row=2, col=1)
    
    if all_sell_dates:
        fig.add_trace(go.Scatter(
            x=all_sell_dates,
            y=all_sell_prices,
            mode='markers',
            name='Sell Signals',
            marker=dict(color='red', size=8, symbol='triangle-down'),
            legendgroup='signals'
        ), row=2, col=1)
    
    # 3. Model Predictions (Row 3)
    fig.add_trace(go.Scatter(
        x=first_portfolio.index,
        y=first_portfolio['price'],
        mode='lines',
        name='Actual Price',
        line=dict(color='black', width=2),
        legendgroup='actual',
        showlegend=False  # Already shown above
    ), row=3, col=1)
    
    # Add predictions if available
    enhanced_files = [f for f in os.listdir(OUTPUT_DIR) if 
                     f.startswith(f"apply_technical_indicators_and_transformations_{symbol}_") and f.endswith('.csv')]
    
    if enhanced_files:
        latest_enhanced = max(enhanced_files, key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)))
        enhanced_path = os.path.join(OUTPUT_DIR, latest_enhanced)
        
        try:
            enhanced_data = pd.read_csv(enhanced_path, index_col=0, parse_dates=True)
            
            if 'Predicted_Price' in enhanced_data.columns:
                common_dates = enhanced_data.index.intersection(first_portfolio.index)
                if not common_dates.empty:
                    aligned_predictions = enhanced_data.loc[common_dates, 'Predicted_Price']
                    
                    fig.add_trace(go.Scatter(
                        x=common_dates,
                        y=aligned_predictions,
                        mode='lines',
                        name='Model Predictions',
                        line=dict(color='orange', width=1, dash='dot'),
                        opacity=0.7,
                        legendgroup='predictions'
                    ), row=3, col=1)
        except Exception as e:
            print(f"[DEBUG] [create_combined_backtesting_chart] Could not load predictions: {e}")
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} Comprehensive Backtesting Analysis',
        template='plotly_white',
        hovermode='x unified',
        width=1200,
        height=1200,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        )
    )
    
    # Update axis labels
    fig.update_yaxes(title_text='Cumulative Return (%)', row=1, col=1)
    fig.update_yaxes(title_text='Price ($)', row=2, col=1)
    fig.update_yaxes(title_text='Price ($)', row=3, col=1)
    fig.update_xaxes(title_text='Date', row=3, col=1)
    
    return fig