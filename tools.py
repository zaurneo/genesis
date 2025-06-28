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




# =============================================================================
# PARAMETER SCHEMAS AND CONFIGURATION
# =============================================================================

PARAMETER_SCHEMAS = {
    "xgboost": {
        "required": ["n_estimators", "max_depth", "learning_rate"],
        "optional": ["subsample", "colsample_bytree", "reg_alpha", "reg_lambda"],
        "defaults": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
            "n_jobs": -1
        },
        "ranges": {
            "n_estimators": [50, 100, 200, 500],
            "max_depth": [3, 6, 10, 15],
            "learning_rate": [0.01, 0.1, 0.2, 0.3]
        },
        "description": "Gradient boosting framework optimized for speed and performance"
    },
    "random_forest": {
        "required": ["n_estimators"],
        "optional": ["max_depth", "min_samples_split", "min_samples_leaf", "max_features"],
        "defaults": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "random_state": 42,
            "n_jobs": -1
        },
        "ranges": {
            "n_estimators": [50, 100, 200, 500],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10]
        },
        "description": "Ensemble of decision trees using bootstrap aggregating"
    },
    "svr": {
        "required": ["C", "gamma", "kernel"],
        "optional": ["epsilon", "degree"],
        "defaults": {
            "C": 1.0,
            "gamma": 'scale',
            "kernel": 'rbf'
        },
        "ranges": {
            "C": [0.1, 1.0, 10.0],
            "gamma": ['scale', 'auto', 0.1, 1.0],
            "kernel": ['rbf', 'linear', 'poly']
        },
        "description": "Support Vector Regression for complex non-linear relationships"
    },
    "gradient_boosting": {
        "required": ["n_estimators", "learning_rate", "max_depth"],
        "optional": ["subsample", "max_features"],
        "defaults": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "random_state": 42
        },
        "ranges": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 6, 10]
        },
        "description": "Sequential ensemble method with error correction"
    },
    "ridge_regression": {
        "required": ["alpha"],
        "optional": ["fit_intercept", "max_iter"],
        "defaults": {
            "alpha": 1.0,
            "fit_intercept": True
        },
        "ranges": {
            "alpha": [0.1, 1.0, 10.0]
        },
        "description": "Linear regression with L2 regularization"
    },
    "extra_trees": {
        "required": ["n_estimators"],
        "optional": ["max_depth", "min_samples_split"],
        "defaults": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "random_state": 42,
            "n_jobs": -1
        },
        "ranges": {
            "n_estimators": [50, 100, 200, 500],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10]
        },
        "description": "Extremely randomized trees with random thresholds"
    },
    "common": {
        "target_days": {
            "default": 1,
            "options": [1, 3, 5, 7, 14, 30],
            "description": "Number of days ahead to predict (1=next day, 5=next week, 30=next month)"
        },
        "test_size": {
            "default": 0.2,
            "range": [0.1, 0.3],
            "description": "Proportion of data reserved for testing"
        },
        "data_requirements": {
            "min_records": 50,
            "min_features": 3,
            "required_indicators": ["SMA", "EMA", "RSI"],
            "description": "Minimum data requirements for reliable model training"
        }
    }
}

MODELING_CONTEXTS = {
    "short_term_trading": {
        "target_days": [1, 3],
        "preferred_models": ["xgboost", "random_forest"],
        "key_features": ["RSI", "MACD", "Bollinger_Bands", "Volume_SMA"],
        "description": "Optimized for day trading and short-term position holding"
    },
    "medium_term_investing": {
        "target_days": [7, 14],
        "preferred_models": ["random_forest", "xgboost"],
        "key_features": ["SMA_20", "EMA_50", "Price_Momentum", "Volatility"],
        "description": "Suitable for swing trading and medium-term investments"
    },
    "long_term_forecasting": {
        "target_days": [30, 60],
        "preferred_models": ["random_forest"],
        "key_features": ["SMA_200", "Long_term_trends", "Fundamental_indicators"],
        "description": "Designed for long-term investment decisions"
    }
}


# =============================================================================
# PARAMETER DECISION AND VALIDATION TOOLS
# =============================================================================

@tool
def decide_model_parameters(
    context: str = "general_prediction",
    model_type: Optional[str] = None,
    symbol: Optional[str] = None,
    prediction_horizon: Optional[str] = None,
    risk_tolerance: str = "medium"
) -> str:
    """
    AI-assisted parameter selection tool that recommends optimal model parameters
    based on context, market conditions, and prediction goals.
    
    This tool helps AI agents make informed decisions about model configuration
    by analyzing the trading context and providing reasoning for each parameter choice.
    
    Args:
        context: Trading/prediction context. Options:
                 - "short_term_trading": Day trading, scalping (1-3 days)
                 - "medium_term_investing": Swing trading (1-2 weeks)  
                 - "long_term_forecasting": Investment planning (1+ months)
                 - "general_prediction": Balanced approach for various timeframes
        
        model_type: Preferred model type. Options:
                   - "xgboost": Fast, accurate, handles non-linearity well
                   - "random_forest": Robust, interpretable, less prone to overfitting
                   - "svr": Support vector regression for complex patterns
                   - "gradient_boosting": Sequential error correction approach
                   - "ridge_regression": Linear with regularization
                   - "extra_trees": Extremely randomized trees
                   - None: Auto-select based on context
        
        symbol: Stock symbol (e.g., "AAPL", "TSLA") for symbol-specific optimization
        
        prediction_horizon: Specific prediction timeframe. Options:
                           - "next_day": 1-day ahead prediction
                           - "next_week": 5-7 day prediction
                           - "next_month": 30-day prediction
                           - None: Auto-select based on context
        
        risk_tolerance: Risk management preference. Options:
                       - "conservative": Prioritize stability over accuracy
                       - "medium": Balanced risk-reward approach
                       - "aggressive": Maximize accuracy, accept higher risk
    
    Returns:
        Formatted string with recommended parameters and detailed reasoning
        that AI agents can parse and use for model configuration decisions.
    
    Example Usage for AI Agents:
        - For day trading: decide_model_parameters("short_term_trading", "xgboost", "AAPL", "next_day", "aggressive")
        - For investment analysis: decide_model_parameters("long_term_forecasting", None, "GOOGL", "next_month", "conservative")
        - For general analysis: decide_model_parameters("general_prediction", "random_forest")
    """
    print(f"🔄 decide_model_parameters: Analyzing context '{context}' for parameter recommendations...")
    
    try:
        recommendations = {}
        reasoning = {}
        
        # Context-based recommendations
        if context in MODELING_CONTEXTS:
            context_config = MODELING_CONTEXTS[context]
            
            # Target days selection
            if prediction_horizon == "next_day":
                target_days = 1
            elif prediction_horizon == "next_week":
                target_days = 7
            elif prediction_horizon == "next_month":
                target_days = 30
            else:
                target_days = context_config["target_days"][0]  # Use first recommended option
            
            recommendations["target_days"] = target_days
            reasoning["target_days"] = f"Selected {target_days} days based on {context} context and {prediction_horizon if prediction_horizon else 'default'} horizon"
            
            # Model type selection
            if model_type is None:
                recommended_model = context_config["preferred_models"][0]
            else:
                recommended_model = model_type if model_type in context_config["preferred_models"] else context_config["preferred_models"][0]
            
            recommendations["model_type"] = recommended_model
            reasoning["model_type"] = f"Selected {recommended_model} - {PARAMETER_SCHEMAS[recommended_model]['description']}"
            
        else:
            # Default recommendations
            recommendations["target_days"] = 1
            recommendations["model_type"] = model_type if model_type else "xgboost"
            reasoning["target_days"] = "Default 1-day prediction for general analysis"
            reasoning["model_type"] = f"Selected {recommendations['model_type']} as specified or default choice"
        
        # Risk-based parameter adjustment
        selected_model = recommendations["model_type"]
        model_schema = PARAMETER_SCHEMAS[selected_model]
        
        if risk_tolerance == "conservative":
            if selected_model == "xgboost":
                recommendations.update({
                    "n_estimators": 50,  # Fewer trees = less overfitting
                    "max_depth": 3,      # Shallow trees = more conservative
                    "learning_rate": 0.01 # Slower learning = more stable
                })
                reasoning["parameters"] = "Conservative XGBoost: fewer estimators, shallow depth, slow learning rate for stability"
            elif selected_model == "random_forest":
                recommendations.update({
                    "n_estimators": 100,
                    "max_depth": 10,     # Limited depth
                    "min_samples_split": 10  # Require more samples to split
                })
                reasoning["parameters"] = "Conservative Random Forest: limited depth, higher split threshold for robustness"
            else:
                # Use conservative defaults for other models
                recommendations.update(model_schema["defaults"])
                reasoning["parameters"] = f"Conservative {selected_model} with default parameters for stability"
                
        elif risk_tolerance == "aggressive":
            if selected_model == "xgboost":
                recommendations.update({
                    "n_estimators": 200,  # More trees = better fitting
                    "max_depth": 10,      # Deeper trees = capture complexity
                    "learning_rate": 0.2  # Faster learning = quicker adaptation
                })
                reasoning["parameters"] = "Aggressive XGBoost: more estimators, deeper trees, faster learning for maximum accuracy"
            elif selected_model == "random_forest":
                recommendations.update({
                    "n_estimators": 200,
                    "max_depth": None,    # Unlimited depth
                    "min_samples_split": 2  # Split with minimum samples
                })
                reasoning["parameters"] = "Aggressive Random Forest: more trees, unlimited depth, minimal split threshold for complexity"
            else:
                # Use defaults but potentially more aggressive
                recommendations.update(model_schema["defaults"])
                reasoning["parameters"] = f"Aggressive {selected_model} parameters for maximum performance"
        
        else:  # medium risk
            # Use default parameters from schema
            recommendations.update(model_schema["defaults"])
            reasoning["parameters"] = f"Balanced {selected_model} parameters optimized for general performance"
        
        # Symbol-specific adjustments (if applicable)
        if symbol:
            # High volatility stocks (examples)
            volatile_stocks = ["TSLA", "GME", "AMC", "MEME"]
            if symbol.upper() in volatile_stocks:
                if selected_model == "random_forest":
                    recommendations["n_estimators"] = min(recommendations.get("n_estimators", 100) + 50, 300)
                    reasoning["symbol_adjustment"] = f"Increased ensemble size for high-volatility stock {symbol}"
                else:
                    if "learning_rate" in recommendations:
                        recommendations["learning_rate"] = max(recommendations.get("learning_rate", 0.1) - 0.05, 0.01)
                    reasoning["symbol_adjustment"] = f"Adjusted parameters for volatile stock {symbol}"
        
        # Additional recommendations
        recommendations.update({
            "test_size": 0.2,
            "save_model": True,
            "save_predictions": True
        })
        reasoning["data_split"] = "80/20 train-test split provides good balance of training data and validation"
        reasoning["saving"] = "Enable model and prediction saving for analysis and backtesting"
        
        # Format response for AI agent consumption
        summary = f"""decide_model_parameters: Parameter recommendations for {context} context:

🎯 CONTEXT ANALYSIS:
- Use Case: {context.replace('_', ' ').title()}
- Model Type: {recommendations['model_type'].upper()}
- Prediction Horizon: {recommendations['target_days']} day(s)
- Risk Profile: {risk_tolerance.title()}
- Symbol: {symbol if symbol else 'Generic'}

🔧 RECOMMENDED PARAMETERS:
"""
        
        # Add model-specific parameters
        for param, value in recommendations.items():
            if param in ["model_type", "target_days", "test_size", "save_model", "save_predictions"]:
                continue
            summary += f"- {param}: {value}\n"
        
        summary += f"""
📊 CONFIGURATION DETAILS:
- Target Days: {recommendations['target_days']}
- Test Size: {recommendations['test_size']}
- Save Model: {recommendations['save_model']}
- Save Predictions: {recommendations['save_predictions']}

💡 REASONING:
"""
        
        for key, reason in reasoning.items():
            summary += f"- {key.replace('_', ' ').title()}: {reason}\n"
        
        summary += f"""
🚀 NEXT STEPS FOR AI AGENT:
1. Use recommended model_type: '{recommendations['model_type']}'
2. Apply suggested parameters in training function
3. Set target_days to {recommendations['target_days']} for optimal horizon
4. Consider market conditions and adjust if needed

📋 COPY-PASTE READY PARAMETERS:
model_type='{recommendations['model_type']}', target_days={recommendations['target_days']}, test_size={recommendations['test_size']}"""
        
        # Add model-specific parameter string
        model_params = []
        for param, value in recommendations.items():
            if param not in ["model_type", "target_days", "test_size", "save_model", "save_predictions"]:
                if isinstance(value, str):
                    model_params.append(f"{param}='{value}'")
                else:
                    model_params.append(f"{param}={value}")
        
        if model_params:
            summary += f", {', '.join(model_params)}"
        
        print(f"✅ decide_model_parameters: Generated recommendations for {context} context")
        return summary
        
    except Exception as e:
        error_msg = f"decide_model_parameters: Error generating recommendations: {str(e)}"
        print(f"❌ decide_model_parameters: {error_msg}")
        return error_msg


@tool
def validate_model_parameters(
    model_type: str,
    parameters: dict
) -> str:
    """
    Validate model parameters against schema and provide warnings/suggestions.
    
    This tool helps AI agents ensure their parameter choices are valid and optimal
    before training models, preventing errors and suboptimal configurations.
    
    Args:
        model_type: Type of model to validate parameters for ("xgboost", "random_forest", etc.)
        parameters: Dictionary of parameters to validate
    
    Returns:
        Validation report with status, warnings, and suggestions for AI agents
    
    Example Usage:
        validate_model_parameters("xgboost", {"n_estimators": 1000, "max_depth": 20, "learning_rate": 0.5})
    """
    print(f"🔄 validate_model_parameters: Validating {model_type} parameters...")
    
    try:
        if model_type not in PARAMETER_SCHEMAS:
            return f"validate_model_parameters: Unknown model type '{model_type}'. Supported types: {list(PARAMETER_SCHEMAS.keys())}"
        
        schema = PARAMETER_SCHEMAS[model_type]
        validation_results = {
            "status": "valid",
            "warnings": [],
            "suggestions": [],
            "missing_required": [],
            "out_of_range": []
        }
        
        # Check required parameters
        for required_param in schema["required"]:
            if required_param not in parameters:
                validation_results["missing_required"].append(required_param)
                validation_results["status"] = "invalid"
        
        # Check parameter ranges and values
        for param, value in parameters.items():
            if param in schema.get("ranges", {}):
                expected_range = schema["ranges"][param]
                if isinstance(expected_range, list):
                    if value not in expected_range and not any(isinstance(opt, type(value)) and opt == value for opt in expected_range):
                        validation_results["out_of_range"].append(f"{param}={value} (expected: {expected_range})")
                        validation_results["warnings"].append(f"Parameter {param}={value} may not be optimal")
        
        # Performance warnings
        if model_type == "xgboost":
            if parameters.get("n_estimators", 0) > 500:
                validation_results["warnings"].append("High n_estimators (>500) may cause overfitting and slow training")
            if parameters.get("learning_rate", 0) > 0.3:
                validation_results["warnings"].append("High learning_rate (>0.3) may cause unstable training")
            if parameters.get("max_depth", 0) > 15:
                validation_results["warnings"].append("Very deep trees (>15) may overfit to training data")
        
        elif model_type == "random_forest":
            if parameters.get("n_estimators", 0) > 500:
                validation_results["warnings"].append("High n_estimators (>500) increases training time with diminishing returns")
            if parameters.get("min_samples_split", 10) < 2:
                validation_results["warnings"].append("Very low min_samples_split may cause overfitting")
        
        # Generate suggestions
        if not validation_results["warnings"]:
            validation_results["suggestions"].append("Parameters look good for general use")
        else:
            validation_results["suggestions"].append("Consider using decide_model_parameters() for optimized configuration")
        
        # Format response
        status_icon = "✅" if validation_results["status"] == "valid" else "❌"
        summary = f"""validate_model_parameters: {status_icon} Parameter validation for {model_type}:

📊 VALIDATION STATUS: {validation_results['status'].upper()}

"""
        
        if validation_results["missing_required"]:
            summary += f"❌ MISSING REQUIRED PARAMETERS:\n"
            for param in validation_results["missing_required"]:
                default_val = schema["defaults"].get(param, "N/A")
                summary += f"  - {param} (default: {default_val})\n"
            summary += "\n"
        
        if validation_results["warnings"]:
            summary += f"⚠️ WARNINGS:\n"
            for warning in validation_results["warnings"]:
                summary += f"  - {warning}\n"
            summary += "\n"
        
        if validation_results["suggestions"]:
            summary += f"💡 SUGGESTIONS:\n"
            for suggestion in validation_results["suggestions"]:
                summary += f"  - {suggestion}\n"
            summary += "\n"
        
        summary += f"🎯 RECOMMENDED NEXT STEPS:\n"
        if validation_results["status"] == "valid":
            summary += "  - Parameters validated successfully\n"
            summary += "  - Proceed with model training\n"
        else:
            summary += "  - Fix missing required parameters\n"
            summary += "  - Consider parameter optimization\n"
        
        print(f"✅ validate_model_parameters: Validation completed for {model_type}")
        return summary
        
    except Exception as e:
        error_msg = f"validate_model_parameters: Error validating parameters: {str(e)}"
        print(f"❌ validate_model_parameters: {error_msg}")
        return error_msg


@tool
def get_model_selection_guide(
    use_case: str = "general",
    symbol: Optional[str] = None,
    data_characteristics: str = "unknown"
) -> str:
    """
    AI Agent decision support tool for selecting optimal model type and parameters.
    
    This tool provides intelligent recommendations for model selection based on
    trading context, data characteristics, and performance requirements.
    
    Args:
        use_case: Trading/prediction context
                 - "day_trading": 1-day predictions, high frequency
                 - "swing_trading": 3-7 day predictions, medium frequency  
                 - "position_trading": 14-30 day predictions, low frequency
                 - "long_term_investing": 30+ day predictions
                 - "general": Balanced approach
        
        symbol: Stock symbol for symbol-specific recommendations
        
        data_characteristics: Data quality and patterns
                            - "high_volatility": TSLA, MEME stocks
                            - "stable": AAPL, MSFT, large caps
                            - "trending": Strong directional movement
                            - "sideways": Range-bound trading
                            - "noisy": Inconsistent patterns
                            - "unknown": No specific characteristics
    
    Returns:
        Comprehensive model selection guide with specific recommendations
    """
    
    recommendations = {
        'day_trading': {
            'primary_model': 'xgboost', 
            'secondary_model': 'svr',
            'target_days': 1,
            'rationale': 'XGBoost excels at capturing short-term non-linear patterns',
            'xgboost_params': {'n_estimators': 150, 'max_depth': 4, 'learning_rate': 0.1},
            'avoid': 'ridge_regression (too simple for intraday patterns)'
        },
        
        'swing_trading': {
            'primary_model': 'random_forest',
            'secondary_model': 'xgboost', 
            'target_days': 5,
            'rationale': 'Random Forest provides stable predictions for medium-term trends',
            'random_forest_params': {'n_estimators': 200, 'max_depth': None},
            'avoid': 'linear models (insufficient for multi-day complexity)'
        },
        
        'position_trading': {
            'primary_model': 'random_forest',
            'secondary_model': 'gradient_boosting',
            'target_days': 14,
            'rationale': 'Ensemble methods handle longer-term trend analysis well',
            'random_forest_params': {'n_estimators': 300, 'min_samples_split': 5},
            'avoid': 'svr (computationally expensive for longer horizons)'
        },
        
        'long_term_investing': {
            'primary_model': 'ridge_regression',
            'secondary_model': 'random_forest',
            'target_days': 30,
            'rationale': 'Linear models capture fundamental long-term relationships',
            'ridge_params': {'alpha': 2.0},
            'avoid': 'xgboost (may overfit to short-term noise)'
        }
    }
    
    volatility_adjustments = {
        'high_volatility': {
            'recommendation': 'Increase regularization, use ensemble methods',
            'xgboost_adjust': {'learning_rate': 0.05, 'max_depth': 3},
            'random_forest_adjust': {'min_samples_split': 10, 'n_estimators': 250}
        },
        
        'stable': {
            'recommendation': 'Standard parameters work well',
            'note': 'Can use more aggressive parameters for higher accuracy'
        },
        
        'noisy': {
            'recommendation': 'Prioritize Random Forest and regularization',
            'primary_model_override': 'random_forest',
            'avoid': 'svr (sensitive to noise)'
        }
    }
    
    # Generate recommendation
    context = recommendations.get(use_case, recommendations['general'] if 'general' in recommendations else recommendations['day_trading'])
    vol_context = volatility_adjustments.get(data_characteristics, {})
    
    summary = f"""get_model_selection_guide: Model Selection Recommendations for {use_case.replace('_', ' ').title()}:

🎯 PRIMARY RECOMMENDATION:
- Model Type: {context['primary_model'].upper()}
- Target Days: {context['target_days']}
- Rationale: {context['rationale']}

🔧 RECOMMENDED PARAMETERS:
"""
    
    # Add parameter recommendations
    param_key = f"{context['primary_model']}_params"
    if param_key in context:
        for param, value in context[param_key].items():
            summary += f"- {param}: {value}\n"
    
    # Add volatility adjustments
    if data_characteristics != 'unknown' and vol_context:
        summary += f"\n📊 {data_characteristics.replace('_', ' ').title().upper()} ADJUSTMENTS:\n"
        summary += f"- Strategy: {vol_context['recommendation']}\n"
        
        if f"{context['primary_model']}_adjust" in vol_context:
            summary += "- Parameter Adjustments:\n"
            for param, value in vol_context[f"{context['primary_model']}_adjust"].items():
                summary += f"  * {param}: {value}\n"
    
    summary += f"""
🥈 ALTERNATIVE MODEL:
- Secondary Choice: {context['secondary_model'].upper()}
- Use When: Primary model shows signs of overfitting or underperformance

❌ AVOID:
- {context['avoid']}

🚀 QUICK START COMMANDS:
# Primary recommendation
train_{context['primary_model']}_price_predictor(
    symbol="{symbol if symbol else 'YOUR_SYMBOL'}",
    target_days={context['target_days']},
"""
    
    if param_key in context:
        for param, value in context[param_key].items():
            if isinstance(value, str):
                summary += f"    {param}='{value}',\n"
            else:
                summary += f"    {param}={value},\n"
    
    summary += """)

# Get parameter recommendations first
decide_model_parameters(
    context="{use_case}",
    model_type="{context['primary_model']}",
    symbol="{symbol if symbol else 'YOUR_SYMBOL'}"
)

💡 AI AGENT DECISION TREE:
1. ✅ Use decide_model_parameters() for intelligent parameter selection
2. ✅ Start with primary model recommendation
3. ✅ Compare results with secondary model
4. ✅ Adjust parameters based on initial results
5. ✅ Use validate_model_parameters() before training
6. ✅ Always enable save_model=True for backtesting

📈 PERFORMANCE EXPECTATIONS:
- R² Score: {'>0.6' if use_case in ['day_trading', 'swing_trading'] else '>0.4'} (good performance)
- Directional Accuracy: {'>55%' if use_case == 'day_trading' else '>60%'} 
- Information Ratio: >-0.5 (acceptable risk-adjusted performance)"""

    return summary


# =============================================================================
# SCALABLE CORE FUNCTIONS
# =============================================================================

def prepare_model_data(
    symbol: str,
    source_file: Optional[str] = None,
    target_days: int = 1,
    test_size: float = 0.2
) -> tuple:
    """
    Universal data preparation pipeline for all machine learning models.
    
    This function provides a standardized approach to loading, cleaning, and preparing
    stock market data with technical indicators for predictive modeling. It handles
    feature selection, target variable creation, train-test splitting, and feature scaling.
    
    Key Features:
    - Automatic enhanced data file detection
    - Robust feature selection and validation
    - Time-series aware train-test splitting
    - Standardized feature scaling
    - Comprehensive error handling and validation
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL', 'GOOGL', 'TSLA', 'MSFT').
                     Must be uppercase format. Used to locate relevant data files.
        
        source_file (Optional[str]): Specific enhanced CSV file with technical indicators.
                                   If None, automatically finds most recent enhanced data file.
                                   File should contain OHLCV data plus technical indicators.
                                   Example: "apply_technical_indicators_AAPL_enhanced_20241127_143022.csv"
        
        target_days (int): Number of days ahead to predict (1-90 recommended).
                          - 1: Next day prediction (day trading)
                          - 3-7: Short-term swing trading
                          - 14-30: Medium-term investing
                          - 30+: Long-term forecasting
        
        test_size (float): Proportion of data reserved for testing (0.1-0.3 recommended).
                          Uses time-series split (latest data for testing).
                          0.2 = 20% test set, 80% training set.
    
    Returns:
        tuple: (model_data_dict, error_message)
            - model_data_dict (dict): Contains all prepared data if successful:
                * 'X_train', 'X_test': Raw feature DataFrames
                * 'X_train_scaled', 'X_test_scaled': Scaled feature arrays
                * 'y_train', 'y_test': Target variable Series
                * 'scaler': Fitted StandardScaler object
                * 'feature_cols': List of feature column names
                * 'data_source': Description of data source used
                * 'full_X', 'full_y': Complete dataset for cross-validation
            - error_message (str): Error description if preparation failed, None if successful
    
    Data Requirements:
        - Minimum 50 records for reliable model training
        - At least 3 technical indicators/features
        - Clean data without excessive missing values
        - Enhanced data file with technical indicators pre-computed
    
    Example Usage for AI Agents:
        # Auto-detect latest data file
        data, error = prepare_model_data("AAPL", target_days=5, test_size=0.2)
        
        # Use specific data file
        data, error = prepare_model_data("TSLA", "enhanced_data.csv", target_days=1, test_size=0.15)
        
        # Check for errors
        if error:
            print(f"Data preparation failed: {error}")
        else:
            print(f"Prepared {len(data['X_train'])} training samples")
    """
    symbol = symbol.upper()
    
    # Load enhanced data with technical indicators
    if source_file:
        if not source_file.endswith('.csv'):
            source_file += '.csv'
        filepath = os.path.join(OUTPUT_DIR, source_file)
        if not os.path.exists(filepath):
            return None, f"Source file '{source_file}' not found."
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
            return None, f"No enhanced data files found for {symbol}. Please run technical indicators first."
    
    if data.empty or len(data) < 50:
        return None, f"Insufficient data for {symbol}. Need at least 50 records."
    
    # Prepare features and target
    data['Target'] = data['Close'].shift(-target_days)
    
    # Select feature columns
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'Target']
    feature_cols = [col for col in data.columns if col not in exclude_cols and not data[col].isnull().all()]
    
    if len(feature_cols) < 3:
        return None, f"Insufficient technical indicators. Found only {len(feature_cols)} features."
    
    # Remove rows with NaN values
    model_data = data[feature_cols + ['Target']].dropna()
    
    if len(model_data) < 30:
        return None, f"Insufficient clean data. Only {len(model_data)} records available."
    
    X = model_data[feature_cols]
    y = model_data['Target']
    
    # Train-test split (time series split)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'data_source': data_source,
        'full_X': X,
        'full_y': y
    }, None


def get_train_test_predictions(model, model_data: dict) -> dict:
    """
    Universal prediction generator for all trained models.
    
    Generates predictions for both training and test sets using any trained
    scikit-learn compatible model. Organizes results with timestamps for
    comprehensive model evaluation and analysis.
    
    Args:
        model: Trained machine learning model with predict() method.
               Compatible with scikit-learn, XGBoost, LightGBM, etc.
               Must be already fitted/trained on training data.
        
        model_data (dict): Dictionary containing prepared model data from prepare_model_data().
                          Must include scaled features and target variables for both sets.
    
    Returns:
        dict: Comprehensive predictions dictionary containing:
            - 'train_predictions': Array of training set predictions
            - 'test_predictions': Array of test set predictions  
            - 'train_actuals': Array of actual training values
            - 'test_actuals': Array of actual test values
            - 'train_dates': DatetimeIndex of training dates
            - 'test_dates': DatetimeIndex of test dates
            - 'train_features': DataFrame of training features
            - 'test_features': DataFrame of test features
    
    Example Usage:
        model = XGBRegressor().fit(model_data['X_train_scaled'], model_data['y_train'])
        predictions = get_train_test_predictions(model, model_data)
        
        # Access predictions
        train_rmse = np.sqrt(mean_squared_error(predictions['train_actuals'], predictions['train_predictions']))
    """
    train_pred = model.predict(model_data['X_train_scaled'])
    test_pred = model.predict(model_data['X_test_scaled'])
    
    return {
        'train_predictions': train_pred,
        'test_predictions': test_pred,
        'train_actuals': model_data['y_train'].values,
        'test_actuals': model_data['y_test'].values,
        'train_dates': model_data['y_train'].index,
        'test_dates': model_data['y_test'].index,
        'train_features': model_data['X_train'],
        'test_features': model_data['X_test']
    }


def assess_model_metrics(predictions_data: dict, model, model_data: dict) -> dict:
    """
    Universal model performance assessment for any machine learning model.
    
    Calculates comprehensive performance metrics including traditional regression metrics,
    financial-specific indicators, and cross-validation scores. Fully scalable to work
    with any scikit-learn compatible model without hard-coded model types.
    
    Key Metrics Calculated:
    - Regression: RMSE, MAE, R², MAPE
    - Financial: Information Ratio (Sharpe-like), Directional Accuracy
    - Validation: Time-series cross-validation scores
    - Error Analysis: Error distributions and volatility
    
    Args:
        predictions_data (dict): Dictionary from get_train_test_predictions() containing
                               predictions and actuals for both training and test sets.
        
        model: Trained model instance (any scikit-learn compatible model).
               Used for cross-validation. Must have get_params() method.
               Examples: XGBRegressor, RandomForestRegressor, SVR, etc.
        
        model_data (dict): Dictionary from prepare_model_data() containing
                          full dataset and scaler for cross-validation.
    
    Returns:
        dict: Comprehensive metrics dictionary:
            - 'train_metrics': Training set performance metrics
            - 'test_metrics': Test set performance metrics  
            - 'cross_validation': Cross-validation scores and statistics
            
            Each metrics subdictionary contains:
            - 'rmse': Root Mean Square Error
            - 'mae': Mean Absolute Error  
            - 'r2': R-squared coefficient
            - 'information_ratio': Risk-adjusted accuracy measure
            - 'mape': Mean Absolute Percentage Error
            - 'directional_accuracy': Percentage of correct trend predictions
            - 'error_std': Standard deviation of prediction errors
            - 'mean_abs_error': Mean absolute prediction error
    
    Scalability Features:
    - Works with any model that has get_params() method
    - Generic cross-validation using model's parameter structure
    - No hard-coded model type assumptions
    - Automatic feature importance detection (if available)
    - Robust error handling for different model types
    
    Example Usage:
        predictions = get_train_test_predictions(trained_model, model_data)
        metrics = assess_model_metrics(predictions, trained_model, model_data)
        
        print(f"Test R²: {metrics['test_metrics']['r2']:.3f}")
        print(f"Cross-val mean: {metrics['cross_validation']['cv_r2_mean']:.3f}")
    """
    import numpy as np
    
    # Extract data
    train_pred = predictions_data['train_predictions']
    test_pred = predictions_data['test_predictions']
    train_actual = predictions_data['train_actuals']
    test_actual = predictions_data['test_actuals']
    
    # Calculate basic metrics
    train_rmse = np.sqrt(mean_squared_error(train_actual, train_pred))
    test_rmse = np.sqrt(mean_squared_error(test_actual, test_pred))
    train_mae = mean_absolute_error(train_actual, train_pred)
    test_mae = mean_absolute_error(test_actual, test_pred)
    train_r2 = r2_score(train_actual, train_pred)
    test_r2 = r2_score(test_actual, test_pred)
    
    # Calculate prediction errors
    train_errors = train_pred - train_actual
    test_errors = test_pred - test_actual
    
    # Information Ratio (Sharpe-like ratio for predictions)
    train_mean_abs_error = np.mean(np.abs(train_errors))
    test_mean_abs_error = np.mean(np.abs(test_errors))
    train_error_std = np.std(train_errors)
    test_error_std = np.std(test_errors)
    
    # Information ratio: negative because we want lower error/volatility to be better
    train_info_ratio = -train_mean_abs_error / train_error_std if train_error_std > 0 else 0
    test_info_ratio = -test_mean_abs_error / test_error_std if test_error_std > 0 else 0
    
    # SCALABLE Cross-validation using generic model recreation
    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = []
    
    X_full = model_data['full_X']
    y_full = model_data['full_y']
    
    for train_idx, val_idx in tscv.split(X_full):
        try:
            X_cv_train, X_cv_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
            y_cv_train, y_cv_val = y_full.iloc[train_idx], y_full.iloc[val_idx]
            
            # Scale features for CV
            cv_scaler = StandardScaler()
            X_cv_train_scaled = cv_scaler.fit_transform(X_cv_train)
            X_cv_val_scaled = cv_scaler.transform(X_cv_val)
            
            # SCALABLE: Create new model instance using original model's parameters
            # This works for ANY scikit-learn compatible model
            model_params = model.get_params()
            cv_model = type(model)(**model_params)
            
            # Train and predict
            cv_model.fit(X_cv_train_scaled, y_cv_train)
            cv_pred = cv_model.predict(X_cv_val_scaled)
            cv_scores.append(r2_score(y_cv_val, cv_pred))
            
        except Exception as e:
            # If model recreation fails, skip this fold
            print(f"Warning: Cross-validation fold failed: {str(e)}")
            continue
    
    # Additional metrics
    train_mape = np.mean(np.abs((train_actual - train_pred) / np.maximum(np.abs(train_actual), 1e-8))) * 100
    test_mape = np.mean(np.abs((test_actual - test_pred) / np.maximum(np.abs(test_actual), 1e-8))) * 100
    
    # Directional accuracy (percentage of correct direction predictions)
    if len(train_actual) > 1:
        train_actual_direction = np.diff(train_actual) > 0
        train_pred_direction = np.diff(train_pred) > 0
        train_directional_accuracy = np.mean(train_actual_direction == train_pred_direction) * 100
    else:
        train_directional_accuracy = 0
    
    if len(test_actual) > 1:
        test_actual_direction = np.diff(test_actual) > 0
        test_pred_direction = np.diff(test_pred) > 0
        test_directional_accuracy = np.mean(test_actual_direction == test_pred_direction) * 100
    else:
        test_directional_accuracy = 0
    
    return {
        'train_metrics': {
            'rmse': float(train_rmse),
            'mae': float(train_mae),
            'r2': float(train_r2),
            'information_ratio': float(train_info_ratio),
            'mape': float(train_mape),
            'directional_accuracy': float(train_directional_accuracy),
            'error_std': float(train_error_std),
            'mean_abs_error': float(train_mean_abs_error)
        },
        'test_metrics': {
            'rmse': float(test_rmse),
            'mae': float(test_mae),
            'r2': float(test_r2),
            'information_ratio': float(test_info_ratio),
            'mape': float(test_mape),
            'directional_accuracy': float(test_directional_accuracy),
            'error_std': float(test_error_std),
            'mean_abs_error': float(test_mean_abs_error)
        },
        'cross_validation': {
            'cv_r2_mean': float(np.mean(cv_scores)) if cv_scores else 0.0,
            'cv_r2_std': float(np.std(cv_scores)) if cv_scores else 0.0,
            'cv_scores': [float(score) for score in cv_scores],
            'cv_folds_completed': len(cv_scores)
        }
    }


def save_model_artifacts(
    model,
    model_data: dict,
    predictions_data: dict,
    metrics: dict,
    feature_importance: pd.DataFrame,
    symbol: str,
    model_type: str,
    model_params: dict,
    target_days: int,
    save_model: bool = True,
    save_predictions: bool = True
) -> dict:
    """
    Universal model artifact saving with standardized naming conventions.
    
    Saves trained models, comprehensive results, and prediction data with consistent
    file naming and structure. Works with any model type and automatically handles
    serialization, JSON formatting, and CSV exports.
    
    Args:
        model: Trained model instance (any serializable model)
        model_data (dict): Prepared model data dictionary
        predictions_data (dict): Predictions and actuals dictionary
        metrics (dict): Comprehensive performance metrics
        feature_importance (pd.DataFrame): Feature importance rankings
        symbol (str): Stock symbol for file naming
        model_type (str): Model type identifier ('xgboost', 'random_forest', etc.)
        model_params (dict): Model parameters for metadata
        target_days (int): Prediction horizon for metadata
        save_model (bool): Whether to save model pickle file
        save_predictions (bool): Whether to save predictions CSV
    
    Returns:
        dict: Dictionary with saved filenames:
            - 'model': Model pickle filename (or None)
            - 'results': Results JSON filename (or None)  
            - 'predictions': Predictions CSV filename (or None)
    
    File Naming Convention:
        - Models: "train_{model_type}_price_predictor_{symbol}_model_{timestamp}.pkl"
        - Results: "train_{model_type}_price_predictor_{symbol}_results_{timestamp}.json"
        - Predictions: "{model_type}_predictions_{symbol}_{timestamp}.csv"
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filenames = {'model': None, 'results': None, 'predictions': None}
    
    if save_model:
        # Save model
        model_filename = f"train_{model_type}_price_predictor_{symbol}_model_{timestamp}.pkl"
        model_filepath = os.path.join(OUTPUT_DIR, model_filename)
        with open(model_filepath, 'wb') as f:
            pickle.dump({
                'model': model,
                'scaler': model_data['scaler'],
                'feature_cols': model_data['feature_cols'],
                'target_days': target_days,
                'symbol': symbol,
                'model_type': model_type
            }, f)
        filenames['model'] = model_filename
        
        # Save results
        results = {
            'symbol': symbol,
            'model_type': model_type.replace('_', ' ').title(),
            'target_days': target_days,
            'data_source': model_data['data_source'],
            'training_samples': len(model_data['X_train']),
            'test_samples': len(model_data['X_test']),
            'features_used': model_data['feature_cols'],
            'model_params': model_params,
            'performance': {
                'train_metrics': metrics['train_metrics'],
                'test_metrics': metrics['test_metrics'],
                'cross_validation': metrics['cross_validation']
            },
            'feature_importance': feature_importance.to_dict('records'),
            'timestamp': timestamp
        }
        
        results_filename = f"train_{model_type}_price_predictor_{symbol}_results_{timestamp}.json"
        results_filepath = os.path.join(OUTPUT_DIR, results_filename)
        with open(results_filepath, 'w') as f:
            json.dump(results, f, indent=2)
        filenames['results'] = results_filename
    
    if save_predictions:
        # Save predictions
        train_df = pd.DataFrame({
            'Date': predictions_data['train_dates'],
            'Actual': predictions_data['train_actuals'],
            'Predicted': predictions_data['train_predictions'],
            'Set': 'Train'
        }).set_index('Date')
        
        test_df = pd.DataFrame({
            'Date': predictions_data['test_dates'],
            'Actual': predictions_data['test_actuals'],
            'Predicted': predictions_data['test_predictions'],
            'Set': 'Test'
        }).set_index('Date')
        
        # Combine both sets
        combined_df = pd.concat([train_df, test_df])
        combined_df['Error'] = combined_df['Predicted'] - combined_df['Actual']
        combined_df['Absolute_Error'] = abs(combined_df['Error'])
        combined_df['Percentage_Error'] = (combined_df['Error'] / np.maximum(abs(combined_df['Actual']), 1e-8)) * 100
        
        predictions_filename = f"{model_type}_predictions_{symbol}_{timestamp}.csv"
        filepath = os.path.join(OUTPUT_DIR, predictions_filename)
        combined_df.to_csv(filepath)
        filenames['predictions'] = predictions_filename
    
    return filenames


def generate_model_summary(
    symbol: str,
    model_type: str,
    model_data: dict,
    metrics: dict,
    feature_importance: pd.DataFrame,
    model_params: dict,
    target_days: int,
    filenames: dict
) -> str:
    """
    Universal model training summary generator for all model types.
    
    Creates comprehensive, standardized training reports with performance metrics,
    parameter details, feature importance, and actionable insights. Adapts display
    based on model type while maintaining consistent structure.
    
    Args:
        symbol (str): Stock symbol
        model_type (str): Model type identifier  
        model_data (dict): Prepared model data
        metrics (dict): Performance metrics
        feature_importance (pd.DataFrame): Feature rankings
        model_params (dict): Model parameters
        target_days (int): Prediction horizon
        filenames (dict): Saved file references
    
    Returns:
        str: Formatted comprehensive training summary report
    """
    train_metrics = metrics['train_metrics']
    test_metrics = metrics['test_metrics']
    cv_metrics = metrics['cross_validation']
    
    # Model-specific configuration details
    model_display_name = model_type.replace('_', ' ').title()
    model_icons = {
        'xgboost': '🤖',
        'random_forest': '🌲', 
        'svr': '🎯',
        'gradient_boosting': '📈',
        'ridge_regression': '📊',
        'extra_trees': '🌳'
    }
    model_icon = model_icons.get(model_type, '🔬')
    
    # Format model parameters
    param_lines = []
    for key, value in model_params.items():
        if key in ['random_state', 'n_jobs']:  # Skip internal parameters
            continue
        param_lines.append(f"- {key.replace('_', ' ').title()}: {value if value is not None else 'Unlimited'}")
    
    summary = f"""train_{model_type}_price_predictor: Successfully trained {model_display_name} model for {symbol}:

{model_icon} MODEL CONFIGURATION:
- Algorithm: {model_display_name} Regressor
- Symbol: {symbol}
- Target: {target_days}-day ahead price prediction
- Data Source: {model_data['data_source']}
- Features: {len(model_data['feature_cols'])} technical indicators
- Training Samples: {len(model_data['X_train'])}
- Test Samples: {len(model_data['X_test'])}

⚙️ HYPERPARAMETERS:
{chr(10).join(param_lines) if param_lines else '- Using default parameters'}

📊 COMPREHENSIVE PERFORMANCE METRICS:

🎯 TRAIN SET PERFORMANCE:
- RMSE: ${train_metrics['rmse']:.3f}
- MAE: ${train_metrics['mae']:.3f}
- R²: {train_metrics['r2']:.3f}
- Information Ratio: {train_metrics['information_ratio']:.3f}
- MAPE: {train_metrics['mape']:.2f}%
- Directional Accuracy: {train_metrics['directional_accuracy']:.1f}%

🎯 TEST SET PERFORMANCE:
- RMSE: ${test_metrics['rmse']:.3f}
- MAE: ${test_metrics['mae']:.3f}
- R²: {test_metrics['r2']:.3f}
- Information Ratio: {test_metrics['information_ratio']:.3f}
- MAPE: {test_metrics['mape']:.2f}%
- Directional Accuracy: {test_metrics['directional_accuracy']:.1f}%

🔄 CROSS-VALIDATION SCORES:
- Mean R²: {cv_metrics['cv_r2_mean']:.3f}
- Std R²: {cv_metrics['cv_r2_std']:.3f}
- Completed Folds: {cv_metrics['cv_folds_completed']}/3
- Individual Scores: {[f"{score:.3f}" for score in cv_metrics['cv_scores']]}

🎯 TOP 5 IMPORTANT FEATURES:
{chr(10).join([f"  {i+1}. {row['feature']}: {row['importance']:.3f}" for i, row in feature_importance.head().iterrows()])}

📁 FILES SAVED:
- Model: {filenames['model'] if filenames['model'] else 'Not saved'}
- Results: {filenames['results'] if filenames['results'] else 'Not saved'}
- Predictions: {filenames['predictions'] if filenames['predictions'] else 'Not saved'}

💡 MODEL INSIGHTS:
- Overfitting Risk: {'High' if train_metrics['r2'] - test_metrics['r2'] > 0.1 else 'Low' if train_metrics['r2'] - test_metrics['r2'] < 0.05 else 'Moderate'}
- Model Quality: {'Excellent' if test_metrics['r2'] > 0.8 else 'Good' if test_metrics['r2'] > 0.6 else 'Fair' if test_metrics['r2'] > 0.4 else 'Poor'}
- Prediction Accuracy: ±${test_metrics['mae']:.2f} average error on test set
- Direction Prediction: {test_metrics['directional_accuracy']:.1f}% correct trend predictions
- Model Stability: {'High' if cv_metrics['cv_r2_std'] < 0.1 else 'Moderate' if cv_metrics['cv_r2_std'] < 0.2 else 'Low'}
"""
    
    return summary


def train_model_pipeline(
    symbol: str,
    model_type: str,
    model_factory_func,
    source_file: Optional[str] = None,
    target_days: int = 1,
    test_size: float = 0.2,
    save_model: bool = True,
    save_predictions: bool = True,
    **model_params
) -> str:
    """
    Universal machine learning model training pipeline.
    
    This is the core orchestration function that handles the complete model training
    workflow for any scikit-learn compatible model. It provides a standardized,
    scalable approach to training, evaluation, and artifact management.
    
    Pipeline Stages:
    1. Data preparation and validation
    2. Model creation and training  
    3. Prediction generation
    4. Comprehensive metrics assessment
    5. Feature importance analysis
    6. Artifact saving (model, results, predictions)
    7. Summary report generation
    
    Scalability Features:
    - Model-agnostic design works with any scikit-learn compatible model
    - Generic parameter handling via factory function pattern
    - Consistent evaluation metrics across all model types
    - Standardized file naming and artifact structure
    - Comprehensive error handling and validation
    
    Args:
        symbol (str): Stock symbol for training (e.g., 'AAPL', 'TSLA')
        
        model_type (str): Model identifier for naming and classification
                         Examples: 'xgboost', 'random_forest', 'svr', 'linear_regression'
        
        model_factory_func (callable): Function that creates model instance
                                     Must accept **model_params and return fitted model
                                     Example: lambda **p: XGBRegressor(**p)
        
        source_file (Optional[str]): Enhanced data file with technical indicators
                                   If None, auto-detects latest enhanced file
        
        target_days (int): Prediction horizon in days (1-90 recommended)
        
        test_size (float): Test set proportion (0.1-0.3 recommended)
        
        save_model (bool): Whether to save trained model and results
        
        save_predictions (bool): Whether to save prediction data
        
        **model_params: Model-specific parameters passed to factory function
                       Examples: n_estimators=100, max_depth=6, learning_rate=0.1
    
    Returns:
        str: Comprehensive training summary with performance metrics,
             parameter details, feature importance, and file locations
    
    Example Usage for AI Agents:
        # XGBoost training
        result = train_model_pipeline(
            symbol="AAPL",
            model_type="xgboost", 
            model_factory_func=lambda **p: XGBRegressor(**p),
            target_days=5,
            n_estimators=200,
            max_depth=8
        )
        
        # Random Forest training  
        result = train_model_pipeline(
            symbol="TSLA",
            model_type="random_forest",
            model_factory_func=lambda **p: RandomForestRegressor(**p),
            n_estimators=100,
            max_depth=None
        )
    
    Error Handling:
        - Validates data availability and quality
        - Handles model training failures gracefully
        - Provides detailed error messages for debugging
        - Continues with partial results if some steps fail
    """
    print(f"🔄 train_{model_type}_price_predictor: Starting {model_type.replace('_', ' ').title()} training for {symbol.upper()}...")
    
    try:
        import numpy as np
        
        symbol = symbol.upper()
        
        # 1. Prepare data
        model_data, error_msg = prepare_model_data(symbol, source_file, target_days, test_size)
        if error_msg:
            print(f"❌ train_{model_type}_price_predictor: {error_msg}")
            return f"train_{model_type}_price_predictor: {error_msg}"
        
        # 2. Create and train model
        model = model_factory_func(**model_params)
        model.fit(model_data['X_train_scaled'], model_data['y_train'])
        
        # 3. Get predictions
        predictions_data = get_train_test_predictions(model, model_data)
        
        # 4. Assess metrics
        metrics = assess_model_metrics(predictions_data, model, model_data)
        
        # 5. Calculate feature importance (handle models without feature_importances_)
        try:
            if hasattr(model, 'feature_importances_'):
                importance_values = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # For linear models, use absolute coefficients
                importance_values = np.abs(model.coef_)
            else:
                # Create dummy importance for models without feature importance
                importance_values = np.ones(len(model_data['feature_cols'])) / len(model_data['feature_cols'])
            
            feature_importance = pd.DataFrame({
                'feature': model_data['feature_cols'],
                'importance': importance_values
            }).sort_values('importance', ascending=False)
        except Exception as e:
            print(f"Warning: Could not calculate feature importance: {e}")
            feature_importance = pd.DataFrame({
                'feature': model_data['feature_cols'],
                'importance': np.ones(len(model_data['feature_cols'])) / len(model_data['feature_cols'])
            })
        
        # 6. Save artifacts
        filenames = save_model_artifacts(
            model, model_data, predictions_data, metrics, feature_importance,
            symbol, model_type, model_params, target_days, save_model, save_predictions
        )
        
        # 7. Generate summary
        summary = generate_model_summary(
            symbol, model_type, model_data, metrics, feature_importance,
            model_params, target_days, filenames
        )
        
        print(f"✅ train_{model_type}_price_predictor: Successfully trained {model_type.replace('_', ' ').title()} model for {symbol} (R²: {metrics['test_metrics']['r2']:.3f})")
        return summary
        
    except Exception as e:
        error_msg = f"train_{model_type}_price_predictor: Error training model for {symbol}: {str(e)}"
        print(f"❌ train_{model_type}_price_predictor: {error_msg}")
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
        
        # Simulate trading with improved position tracking
        portfolio_value = initial_capital
        cash = initial_capital
        shares = 0
        position = 0  # 0: no position, 1: long, -1: short
        entry_price = 0  # Track entry price for proper PnL calculation
        
        portfolio_history = []
        trades = []
        
        for i, (date, row) in enumerate(backtest_data.iterrows()):
            current_price = row['Current_Price']
            signal = row['Signal']
            
            # Execute trades based on signals
            if signal == 1 and position <= 0:  # Buy signal
                if position == -1:  # Cover short position first
                    # Calculate PnL for short position: profit when price goes down
                    pnl = (entry_price - current_price) * shares
                    cash += pnl - (shares * current_price * transaction_cost)  # Add PnL minus transaction cost
                    trades.append({
                        'date': date,
                        'action': 'cover_short',
                        'price': current_price,
                        'shares': shares,
                        'value': shares * current_price,
                        'pnl': pnl,
                        'entry_price': entry_price
                    })
                    shares = 0
                    position = 0  # Reset position
                
                # Open long position
                shares_to_buy = cash // (current_price * (1 + transaction_cost))
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price * (1 + transaction_cost)
                    cash -= cost
                    shares = shares_to_buy
                    position = 1
                    entry_price = current_price  # Track entry price
                    trades.append({
                        'date': date,
                        'action': 'buy',
                        'price': current_price,
                        'shares': shares_to_buy,
                        'value': cost,
                        'pnl': 0,
                        'entry_price': entry_price
                    })
            
            elif signal == -1 and position >= 0:  # Sell signal
                if position == 1:  # Sell long position first
                    # Calculate PnL for long position: profit when price goes up
                    pnl = (current_price - entry_price) * shares
                    cash += (shares * current_price * (1 - transaction_cost))  # Add proceeds minus transaction cost
                    trades.append({
                        'date': date,
                        'action': 'sell',
                        'price': current_price,
                        'shares': shares,
                        'value': shares * current_price,
                        'pnl': pnl,
                        'entry_price': entry_price
                    })
                    shares = 0
                    position = 0  # Reset position
                
                # Open short position (simplified - assume margin account)
                # Use available cash as collateral for short position
                shares_to_short = (cash * 0.5) // (current_price * (1 + transaction_cost))  # 50% margin requirement
                if shares_to_short > 0:
                    # For short selling, we receive cash proceeds but owe shares
                    # Simplified: treat as borrowing shares and receiving cash
                    proceeds = shares_to_short * current_price * (1 - transaction_cost)
                    cash += proceeds
                    shares = shares_to_short
                    position = -1
                    entry_price = current_price  # Track entry price for short position
                    trades.append({
                        'date': date,
                        'action': 'short',
                        'price': current_price,
                        'shares': shares_to_short,
                        'value': proceeds,
                        'pnl': 0,
                        'entry_price': entry_price
                    })
            
            # Calculate portfolio value with proper mark-to-market
            if position == 1:  # Long position
                portfolio_value = cash + shares * current_price
            elif position == -1:  # Short position - mark to market properly
                # Short position value = cash + (entry_price - current_price) * shares
                unrealized_pnl = (entry_price - current_price) * shares
                portfolio_value = cash + unrealized_pnl
            else:  # No position
                portfolio_value = cash
            
            portfolio_history.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': cash,
                'shares': shares,
                'position': position,
                'price': current_price,
                'signal': signal,
                'entry_price': entry_price
            })
        
        # Close any remaining positions at the end
        if position != 0:
            final_price = backtest_data['Current_Price'].iloc[-1]
            final_date = backtest_data.index[-1]
            
            if position == 1:  # Close long position
                pnl = (final_price - entry_price) * shares
                cash += shares * final_price * (1 - transaction_cost)
                trades.append({
                    'date': final_date,
                    'action': 'sell',
                    'price': final_price,
                    'shares': shares,
                    'value': shares * final_price,
                    'pnl': pnl,
                    'entry_price': entry_price
                })
            elif position == -1:  # Cover short position
                pnl = (entry_price - final_price) * shares
                cash += pnl - (shares * final_price * transaction_cost)
                trades.append({
                    'date': final_date,
                    'action': 'cover_short',
                    'price': final_price,
                    'shares': shares,
                    'value': shares * final_price,
                    'pnl': pnl,
                    'entry_price': entry_price
                })
            
            # Update final portfolio value
            portfolio_history[-1]['portfolio_value'] = cash
            portfolio_history[-1]['cash'] = cash
            portfolio_history[-1]['shares'] = 0
            portfolio_history[-1]['position'] = 0
        
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
        
        # Calculate additional metrics with sufficient data check
        trading_days = len(portfolio_df)
        if trading_days >= 30:  # Only calculate annualized metrics if sufficient data
            annual_return = ((portfolio_df['portfolio_value'].iloc[-1] / initial_capital) ** (252 / trading_days) - 1) * 100
            volatility = portfolio_df['returns'].std() * np.sqrt(252) * 100
        else:
            # Use simple metrics for insufficient data
            annual_return = total_return  # Don't annualize
            volatility = portfolio_df['returns'].std() * 100  # Don't annualize
        
        sharpe_ratio = (annual_return - 2) / volatility if volatility > 0 else 0  # Assuming 2% risk-free rate
        
        # Maximum drawdown
        rolling_max = portfolio_df['portfolio_value'].expanding().max()
        drawdown = (portfolio_df['portfolio_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        # Fixed win rate calculation - check actual profitability
        profitable_trades = 0
        total_closed_trades = 0
        
        for trade in trades:
            if 'pnl' in trade and trade['action'] in ['sell', 'cover_short']:
                total_closed_trades += 1
                if trade['pnl'] > 0:
                    profitable_trades += 1
        
        win_rate = (profitable_trades / total_closed_trades * 100) if total_closed_trades > 0 else 0
        
        # Save results if requested
        results_filename = None
        portfolio_filename = None
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
                    'total_trades': int(total_closed_trades),
                    'profitable_trades': int(profitable_trades),
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
- {'Annualized' if trading_days >= 30 else 'Total'} Return: {annual_return:.2f}%
- {'Annualized' if trading_days >= 30 else 'Total'} Volatility: {volatility:.2f}%
- Sharpe Ratio: {sharpe_ratio:.2f}
- Maximum Drawdown: {max_drawdown:.2f}%
- Win Rate: {win_rate:.1f}% ({profitable_trades}/{total_closed_trades} trades)
- Total Closed Trades: {total_closed_trades}

📈 BENCHMARK COMPARISON:
- Buy & Hold Return: {buy_hold_return:.2f}%
- Strategy Excess Return: {total_return - buy_hold_return:.2f}%
- Alpha: {'Positive' if total_return > buy_hold_return else 'Negative'}

💡 PERFORMANCE ASSESSMENT:
- Risk-Adjusted Performance: {'Excellent' if sharpe_ratio > 1.5 else 'Good' if sharpe_ratio > 1.0 else 'Fair' if sharpe_ratio > 0.5 else 'Poor'}
- Strategy Effectiveness: {'Outperforming' if total_return > buy_hold_return else 'Underperforming'} vs Buy & Hold
- Maximum Risk: {max_drawdown:.1f}% portfolio decline from peak
- Trading Activity: {'High' if total_closed_trades > len(portfolio_df) * 0.1 else 'Moderate' if total_closed_trades > len(portfolio_df) * 0.05 else 'Low'} frequency
- Data Sufficiency: {'Sufficient for annualized metrics' if trading_days >= 30 else 'Limited - using total period metrics'}

📁 FILES SAVED:
- Detailed Results: {results_filename if results_filename else 'Not saved'}
- Portfolio History: {portfolio_filename if save_results else 'Not saved'}

⚠️ IMPORTANT NOTES:
- Results are based on historical data and may not reflect future performance
- Short selling uses simplified margin accounting (50% margin requirement)
- Transaction costs and slippage are simplified but consistently applied
- Position tracking includes proper mark-to-market valuation
- Win rate based on actual trade profitability, not signal accuracy
- This is for analysis purposes only, not investment advice
"""
        
        print(f"✅ backtest_model_strategy: Successfully completed backtesting for {symbol} ({total_closed_trades} trades, {total_return:.2f}% return)")
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

# Replace the existing train_xgboost_price_predictor and train_random_forest_price_predictor functions with these ultra-minimal versions:





# =============================================================================
# MULTI-MODEL BACKTESTING EXTENSIONS
# Add these functions to tools.py after the existing backtest_model_strategy function
# =============================================================================

import hashlib

def generate_model_signature(model_type: str, params: dict) -> str:
    """Create human-readable model signature for identification."""
    key_params = get_key_parameters(model_type, params)
    signature_parts = [model_type]
    for k, v in key_params.items():
        if isinstance(v, float):
            signature_parts.append(f"{k}{v:.3f}")
        else:
            signature_parts.append(f"{k}{v}")
    return "_".join(signature_parts)


def get_key_parameters(model_type: str, params: dict) -> dict:
    """Extract most important parameters for model identification."""
    key_param_map = {
        'xgboost': ['n_estimators', 'max_depth', 'learning_rate'],
        'random_forest': ['n_estimators', 'max_depth', 'min_samples_split'],
        'svr': ['C', 'gamma', 'kernel'],
        'gradient_boosting': ['n_estimators', 'learning_rate', 'max_depth'],
        'ridge_regression': ['alpha'],
        'extra_trees': ['n_estimators', 'max_depth', 'min_samples_split']
    }
    key_params = key_param_map.get(model_type, [])
    return {k: params.get(k, 'default') for k in key_params if k in params}


def create_parameter_summary(model_type: str, params: dict) -> str:
    """Create concise parameter summary for display."""
    key_params = get_key_parameters(model_type, params)
    if not key_params:
        return "default_params"
    
    summary_parts = []
    for k, v in key_params.items():
        if isinstance(v, float):
            summary_parts.append(f"{k}={v:.3f}")
        elif v is None:
            summary_parts.append(f"{k}=None")
        else:
            summary_parts.append(f"{k}={v}")
    
    return ", ".join(summary_parts)


def calculate_feature_entropy(feature_importance: pd.DataFrame) -> float:
    """Calculate entropy of feature importance distribution."""
    if feature_importance.empty:
        return 0.0
    
    importances = feature_importance['importance'].values
    importances = importances / importances.sum()  # Normalize
    importances = importances[importances > 0]  # Remove zeros
    
    if len(importances) <= 1:
        return 0.0
    
    entropy = -np.sum(importances * np.log2(importances))
    return float(entropy)


def discover_models(symbol: str, model_filter: Optional[str] = None) -> list:
    """Discover available model files for a symbol."""
    print(f"🔍 discover_models: Looking for models for symbol '{symbol}' in directory '{OUTPUT_DIR}'")
    
    if not os.path.exists(OUTPUT_DIR):
        print(f"❌ discover_models: Output directory '{OUTPUT_DIR}' does not exist")
        return []
    
    try:
        all_files = os.listdir(OUTPUT_DIR)
        print(f"📁 discover_models: Found {len(all_files)} total files in directory")
        
        # First, find all .pkl files
        pkl_files = [f for f in all_files if f.endswith('.pkl')]
        print(f"📦 discover_models: Found {len(pkl_files)} .pkl files")
        
        # Then, find model files (containing _model_ and ending with .pkl)
        model_pkl_files = [f for f in all_files if '_model_' in f and f.endswith('.pkl')]
        print(f"🤖 discover_models: Found {len(model_pkl_files)} files containing '_model_' and ending with '.pkl'")
        
        # Filter for the specific symbol
        symbol_upper = symbol.upper()
        model_files = [f for f in model_pkl_files if symbol_upper in f.upper()]
        print(f"🎯 discover_models: Found {len(model_files)} model files containing '{symbol_upper}':")
        for f in model_files:
            print(f"   - {f}")
        
        # Apply additional filter if specified
        if model_filter:
            print(f"🔍 discover_models: Applying filter '{model_filter}'")
            filtered_files = [f for f in model_files if model_filter.lower() in f.lower()]
            print(f"📊 discover_models: After filtering: {len(filtered_files)} files")
            for f in filtered_files:
                print(f"   - {f}")
            model_files = filtered_files
        
        result = sorted(model_files)
        print(f"✅ discover_models: Returning {len(result)} sorted model files")
        return result
        
    except Exception as e:
        print(f"❌ discover_models: Error accessing directory '{OUTPUT_DIR}': {str(e)}")
        return []

def load_model_metadata(model_file: str) -> dict:
    """Load model metadata from corresponding JSON result file."""
    try:
        # Convert model file name to result file name
        result_file = model_file.replace('_model.pkl', '_results.json')
        result_path = os.path.join(OUTPUT_DIR, result_file)
        
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                return json.load(f)
        else:
            # If no result file, try to extract from model file name
            return extract_metadata_from_filename(model_file)
    except Exception as e:
        print(f"Warning: Could not load metadata for {model_file}: {e}")
        return extract_metadata_from_filename(model_file)


def extract_metadata_from_filename(model_file: str) -> dict:
    """Extract basic metadata from model filename."""
    parts = model_file.split('_')
    model_type = 'unknown'
    
    for part in parts:
        if part in ['xgboost', 'random', 'forest', 'svr', 'gradient', 'boosting', 'ridge', 'regression', 'extra', 'trees']:
            if 'random' in parts and 'forest' in parts:
                model_type = 'random_forest'
            elif 'gradient' in parts and 'boosting' in parts:
                model_type = 'gradient_boosting'
            elif 'ridge' in parts and 'regression' in parts:
                model_type = 'ridge_regression'
            elif 'extra' in parts and 'trees' in parts:
                model_type = 'extra_trees'
            else:
                model_type = part
            break
    
    return {
        'model_type': model_type,
        'model_params': {},
        'symbol': extract_symbol_from_filename(model_file),
        'timestamp': extract_timestamp_from_filename(model_file)
    }


def extract_symbol_from_filename(filename: str) -> str:
    """Extract symbol from filename."""
    parts = filename.split('_')
    for part in parts:
        if len(part) >= 2 and part.isupper():
            return part
    return 'UNKNOWN'


def extract_timestamp_from_filename(filename: str) -> str:
    """Extract timestamp from filename."""
    parts = filename.split('_')
    for part in parts:
        if len(part) == 15 and part.isdigit():
            return part
    return '000000000000000'


def parse_model_list(model_files: str) -> list:
    """Parse comma-separated model file list."""
    models = [f.strip() for f in model_files.split(',')]
    return [f if f.endswith('.pkl') else f + '.pkl' for f in models]


def enhance_with_model_metadata(backtest_result_raw: str, model_metadata: dict) -> dict:
    """Parse raw backtest result string and enhance with model metadata."""
    try:
        # Extract key metrics from the backtest result string
        lines = backtest_result_raw.split('\n')
        
        # Initialize result structure
        result = {
            'model_metadata': model_metadata,
            'performance': {},
            'summary': backtest_result_raw
        }
        
        # Parse performance metrics from the backtest result string
        for line in lines:
            if 'Total Return:' in line:
                try:
                    value = float(line.split(':')[1].strip().replace('%', ''))
                    result['performance']['total_return_pct'] = value
                except:
                    pass
            elif 'Sharpe Ratio:' in line:
                try:
                    value = float(line.split(':')[1].strip())
                    result['performance']['sharpe_ratio'] = value
                except:
                    pass
            elif 'Maximum Drawdown:' in line:
                try:
                    value = float(line.split(':')[1].strip().replace('%', ''))
                    result['performance']['max_drawdown_pct'] = value
                except:
                    pass
            elif 'Win Rate:' in line:
                try:
                    value = float(line.split(':')[1].strip().split('%')[0])
                    result['performance']['win_rate_pct'] = value
                except:
                    pass
            elif 'Total Closed Trades:' in line:
                try:
                    value = int(line.split(':')[1].strip())
                    result['performance']['total_trades'] = value
                except:
                    pass
            elif 'Final Portfolio Value:' in line:
                try:
                    value_str = line.split(':')[1].strip().replace('$', '').replace(',', '')
                    value = float(value_str)
                    result['performance']['final_portfolio_value'] = value
                except:
                    pass
        
        return result
        
    except Exception as e:
        print(f"Warning: Could not parse backtest result: {e}")
        return {
            'model_metadata': model_metadata,
            'performance': {},
            'summary': backtest_result_raw
        }


def generate_model_comparison_matrix(all_results: list) -> pd.DataFrame:
    """Generate comparison matrix from multiple backtest results."""
    comparison_data = []
    
    for result in all_results:
        try:
            metadata = result.get('model_metadata', {})
            performance = result.get('performance', {})
            
            model_type = metadata.get('model_type', 'unknown')
            model_params = metadata.get('model_params', {})
            
            comparison_data.append({
                'model_id': generate_model_signature(model_type, model_params),
                'model_type': model_type.replace('_', ' ').title(),
                'parameters': create_parameter_summary(model_type, model_params),
                'total_return': performance.get('total_return_pct', 0.0),
                'sharpe_ratio': performance.get('sharpe_ratio', 0.0),
                'max_drawdown': performance.get('max_drawdown_pct', 0.0),
                'win_rate': performance.get('win_rate_pct', 0.0),
                'total_trades': performance.get('total_trades', 0),
                'final_value': performance.get('final_portfolio_value', 0.0),
                'timestamp': metadata.get('timestamp', ''),
                'target_days': metadata.get('target_days', 1)
            })
        except Exception as e:
            print(f"Warning: Could not process result for comparison: {e}")
            continue
    
    if not comparison_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(comparison_data)
    return df.sort_values('total_return', ascending=False)


def calculate_model_rankings(all_results: list) -> dict:
    """Calculate model rankings across different metrics."""
    if not all_results:
        return {}
    
    comparison_df = generate_model_comparison_matrix(all_results)
    if comparison_df.empty:
        return {}
    
    rankings = {}
    
    # Best total return
    best_return_idx = comparison_df['total_return'].idxmax()
    rankings['best_return'] = {
        'model_id': comparison_df.loc[best_return_idx, 'model_id'],
        'model_type': comparison_df.loc[best_return_idx, 'model_type'],
        'value': comparison_df.loc[best_return_idx, 'total_return']
    }
    
    # Best Sharpe ratio
    best_sharpe_idx = comparison_df['sharpe_ratio'].idxmax()
    rankings['best_sharpe'] = {
        'model_id': comparison_df.loc[best_sharpe_idx, 'model_id'],
        'model_type': comparison_df.loc[best_sharpe_idx, 'model_type'],
        'value': comparison_df.loc[best_sharpe_idx, 'sharpe_ratio']
    }
    
    # Smallest drawdown (least negative)
    best_drawdown_idx = comparison_df['max_drawdown'].idxmax()  # Max because less negative is better
    rankings['best_drawdown'] = {
        'model_id': comparison_df.loc[best_drawdown_idx, 'model_id'],
        'model_type': comparison_df.loc[best_drawdown_idx, 'model_type'],
        'value': comparison_df.loc[best_drawdown_idx, 'max_drawdown']
    }
    
    # Best win rate
    best_winrate_idx = comparison_df['win_rate'].idxmax()
    rankings['best_winrate'] = {
        'model_id': comparison_df.loc[best_winrate_idx, 'model_id'],
        'model_type': comparison_df.loc[best_winrate_idx, 'model_type'],
        'value': comparison_df.loc[best_winrate_idx, 'win_rate']
    }
    
    # Most active (most trades)
    most_active_idx = comparison_df['total_trades'].idxmax()
    rankings['most_active'] = {
        'model_id': comparison_df.loc[most_active_idx, 'model_id'],
        'model_type': comparison_df.loc[most_active_idx, 'model_type'],
        'value': comparison_df.loc[most_active_idx, 'total_trades']
    }
    
    return rankings


def save_multi_model_results(symbol: str, all_results: list, comparison_matrix: pd.DataFrame, rankings: dict) -> dict:
    """Save multi-model comparison results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    detailed_results = {
        'symbol': symbol,
        'comparison_timestamp': timestamp,
        'models_analyzed': len(all_results),
        'comparison_matrix': comparison_matrix.to_dict('records') if not comparison_matrix.empty else [],
        'rankings': rankings,
        'detailed_results': all_results
    }
    
    results_filename = f"backtest_multiple_models_{symbol}_detailed_{timestamp}.json"
    results_filepath = os.path.join(OUTPUT_DIR, results_filename)
    
    with open(results_filepath, 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    
    # Save comparison matrix as CSV
    matrix_filename = None
    if not comparison_matrix.empty:
        matrix_filename = f"backtest_multiple_models_{symbol}_comparison_{timestamp}.csv"
        matrix_filepath = os.path.join(OUTPUT_DIR, matrix_filename)
        comparison_matrix.to_csv(matrix_filepath, index=False)
    
    return {
        'detailed_results': results_filename,
        'comparison_matrix': matrix_filename
    }


def format_multi_model_summary(comparison_matrix: pd.DataFrame, rankings: dict, symbol: str, models_analyzed: int) -> str:
    """Format comprehensive multi-model comparison summary."""
    
    summary = f"""backtest_multiple_models: Successfully analyzed {models_analyzed} models for {symbol}:

🎯 MULTI-MODEL BACKTESTING SUMMARY:
- Symbol: {symbol}
- Models Analyzed: {models_analyzed}
- Successful Backtests: {len(comparison_matrix) if not comparison_matrix.empty else 0}
- Analysis Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
    
    if comparison_matrix.empty:
        summary += """❌ NO SUCCESSFUL BACKTESTS:
- No models could be successfully backtested
- Check model files and data availability
- Ensure models have corresponding result files
"""
        return summary
    
    # Top 5 performers by return
    top_5 = comparison_matrix.head(5)
    summary += "🏆 TOP 5 PERFORMERS (by Total Return):\n"
    for i, (_, row) in enumerate(top_5.iterrows(), 1):
        summary += f"  {i}. {row['model_type']}: {row['total_return']:.2f}% return, {row['sharpe_ratio']:.2f} Sharpe\n"
        summary += f"     Parameters: {row['parameters']}\n"
    
    summary += "\n📊 PERFORMANCE RANKINGS:\n"
    
    for metric, data in rankings.items():
        metric_display = metric.replace('_', ' ').title()
        summary += f"- {metric_display}: {data['model_type']} ({data['value']:.2f})\n"
    
    # Performance distribution
    if len(comparison_matrix) > 1:
        summary += f"""
📈 PERFORMANCE DISTRIBUTION:
- Return Range: {comparison_matrix['total_return'].min():.2f}% to {comparison_matrix['total_return'].max():.2f}%
- Average Return: {comparison_matrix['total_return'].mean():.2f}%
- Sharpe Range: {comparison_matrix['sharpe_ratio'].min():.2f} to {comparison_matrix['sharpe_ratio'].max():.2f}
- Average Sharpe: {comparison_matrix['sharpe_ratio'].mean():.2f}
- Drawdown Range: {comparison_matrix['max_drawdown'].min():.2f}% to {comparison_matrix['max_drawdown'].max():.2f}%

"""
    
    # Model type analysis
    model_type_counts = comparison_matrix['model_type'].value_counts()
    summary += "🤖 MODEL TYPE BREAKDOWN:\n"
    for model_type, count in model_type_counts.items():
        avg_return = comparison_matrix[comparison_matrix['model_type'] == model_type]['total_return'].mean()
        summary += f"- {model_type}: {count} models, {avg_return:.2f}% avg return\n"
    
    summary += f"""
📁 RESULTS SAVED:
- Detailed Analysis: backtest_multiple_models_{symbol}_detailed_[timestamp].json
- Comparison Matrix: backtest_multiple_models_{symbol}_comparison_[timestamp].csv

💡 KEY INSIGHTS:
- Best Overall: {rankings.get('best_return', {}).get('model_type', 'N/A')} model
- Most Consistent: {rankings.get('best_sharpe', {}).get('model_type', 'N/A')} model  
- Safest: {rankings.get('best_drawdown', {}).get('model_type', 'N/A')} model
- Performance Spread: {comparison_matrix['total_return'].max() - comparison_matrix['total_return'].min():.2f}% difference between best and worst
"""
    
    return summary


@tool
def backtest_multiple_models(
    symbol: str,
    model_files: str = "auto",
    model_filter: Optional[str] = None,
    strategy_type: str = "directional",
    initial_capital: float = 10000.0,
    transaction_cost: float = 0.001,
    save_comparison: bool = True
) -> str:
    """
    Backtest multiple trained models simultaneously and compare their performance.
    
    This tool orchestrates backtesting across multiple models, providing comprehensive
    comparison analysis, rankings, and insights to identify the best performing models
    and understand parameter impacts on trading performance.
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')
                     Must have trained models available for backtesting
        
        model_files (str): Model selection strategy:
                          - "auto": Automatically discover all available models for symbol
                          - "model1.pkl,model2.pkl": Comma-separated specific model files
                          - "latest_5": Use 5 most recently trained models
        
        model_filter (Optional[str]): Filter models by type:
                                     - "xgboost": Only XGBoost models
                                     - "random_forest": Only Random Forest models
                                     - "svr": Only Support Vector Regression models
                                     - None: Include all model types
        
        strategy_type (str): Trading strategy for all models:
                           - "directional": Buy if predicted > current, sell if < current
                           - "threshold": Buy/sell based on return thresholds
                           - "percentile": Buy/sell based on prediction percentiles
        
        initial_capital (float): Starting capital for each backtest ($10,000 default)
                               Same capital used for fair comparison across models
        
        transaction_cost (float): Transaction cost percentage (0.001 = 0.1%)
                                Applied consistently across all model backtests
        
        save_comparison (bool): Whether to save detailed comparison results and matrix
                              TRUE RECOMMENDED for analysis and reporting
    
    Returns:
        str: Comprehensive multi-model comparison report including:
             - Performance rankings and top performers
             - Model type analysis and parameter impacts  
             - Statistical performance distribution
             - Best models by different metrics (return, Sharpe, drawdown)
             - Detailed comparison matrix with all models
             - File locations for saved comparison data
    
    Key Features:
        - **Auto-Discovery**: Automatically finds all trained models for symbol
        - **Fair Comparison**: Uses identical backtesting parameters for all models
        - **Multiple Rankings**: Ranks by return, Sharpe ratio, drawdown, win rate
        - **Parameter Analysis**: Shows how parameter differences impact performance
        - **Model Type Insights**: Compares XGBoost vs Random Forest vs other models
        - **Statistical Analysis**: Performance distributions and spreads
        - **Scalable**: Efficiently handles 20+ models without performance issues
    
    Example Usage for AI Agents:
        # Compare all available models
        result = backtest_multiple_models("AAPL")
        
        # Compare only XGBoost models
        result = backtest_multiple_models("TSLA", model_filter="xgboost")
        
        # Compare specific models
        result = backtest_multiple_models("GOOGL", model_files="model1.pkl,model2.pkl,model3.pkl")
        
        # Use threshold strategy for comparison
        result = backtest_multiple_models("MSFT", strategy_type="threshold")
    
    Expected Outputs:
        - Models ranked by total return, Sharpe ratio, maximum drawdown
        - Parameter impact analysis (e.g., how learning_rate affects performance)
        - Model type comparison (which algorithm works best for this symbol)
        - Performance consistency analysis across different models
        - Detailed comparison matrix saved as CSV for further analysis
    
    AI Agent Decision Guidelines:
        - Use "auto" for comprehensive model comparison
        - Filter by model_type when comparing algorithm effectiveness
        - Analyze parameter patterns in top performers
        - Consider both return and risk metrics for final model selection
        - Use results to guide future model training parameter choices
    """
    print(f"🔄 backtest_multiple_models: Starting multi-model backtesting for {symbol.upper()}...")
    
    try:
        symbol = symbol.upper()
        
        # 1. Discovery phase - find available models
        if model_files == "auto":
            available_models = discover_models(symbol, model_filter)
        elif model_files == "latest_5":
            all_models = discover_models(symbol, model_filter)
            # Sort by modification time and take latest 5
            model_times = [(f, os.path.getmtime(os.path.join(OUTPUT_DIR, f))) for f in all_models]
            sorted_models = sorted(model_times, key=lambda x: x[1], reverse=True)
            available_models = [f[0] for f in sorted_models[:5]]
        else:
            available_models = parse_model_list(model_files)
            # Filter to only existing files
            available_models = [f for f in available_models if os.path.exists(os.path.join(OUTPUT_DIR, f))]
        
        if not available_models:
            result = f"backtest_multiple_models: No models found for {symbol}. Train models first or check model_filter."
            print(f"❌ backtest_multiple_models: {result}")
            return result
        
        print(f"🔍 backtest_multiple_models: Found {len(available_models)} models for {symbol}")
        
        # 2. Backtesting phase - run each model through existing backtest function
        all_results = []
        successful_backtests = 0
        
        for i, model_file in enumerate(available_models, 1):
            try:
                print(f"🔄 backtest_multiple_models: Backtesting model {i}/{len(available_models)}: {model_file}")
                
                # Load model metadata
                model_metadata = load_model_metadata(model_file)
                
                # Run existing backtesting function (don't save individual results)
                # Call the underlying function directly, not through the tool wrapper
                backtest_result_raw = backtest_model_strategy.func(
                    symbol=symbol,
                    model_file=model_file,
                    strategy_type=strategy_type,
                    initial_capital=initial_capital,
                    transaction_cost=transaction_cost,
                    save_results=False  # We'll aggregate results
                )
                
                # Check if backtest was successful
                if "Error" not in backtest_result_raw and "Total Return:" in backtest_result_raw:
                    # Enhance with metadata
                    enhanced_result = enhance_with_model_metadata(backtest_result_raw, model_metadata)
                    enhanced_result['model_file'] = model_file
                    all_results.append(enhanced_result)
                    successful_backtests += 1
                    print(f"✅ backtest_multiple_models: Successfully backtested {model_file}")
                else:
                    print(f"❌ backtest_multiple_models: Failed to backtest {model_file}")
                
            except Exception as e:
                print(f"❌ backtest_multiple_models: Error backtesting {model_file}: {str(e)}")
                continue
        
        if successful_backtests == 0:
            result = f"backtest_multiple_models: No models could be successfully backtested for {symbol}."
            print(f"❌ backtest_multiple_models: {result}")
            return result
        
        print(f"✅ backtest_multiple_models: Successfully backtested {successful_backtests}/{len(available_models)} models")
        
        # 3. Comparison phase - generate comparison matrix and rankings
        comparison_matrix = generate_model_comparison_matrix(all_results)
        rankings = calculate_model_rankings(all_results)
        
        # 4. Save results if requested
        saved_files = {}
        if save_comparison:
            saved_files = save_multi_model_results(symbol, all_results, comparison_matrix, rankings)
        
        # 5. Generate comprehensive summary
        summary = format_multi_model_summary(comparison_matrix, rankings, symbol, len(available_models))
        
        # Add file information
        if saved_files:
            summary += f"\n📁 COMPARISON FILES SAVED:\n"
            summary += f"- Detailed Results: {saved_files['detailed_results']}\n"
            if saved_files['comparison_matrix']:
                summary += f"- Comparison Matrix: {saved_files['comparison_matrix']}\n"
        
        print(f"✅ backtest_multiple_models: Completed multi-model analysis for {symbol} ({successful_backtests} models)")
        return summary
        
    except Exception as e:
        error_msg = f"backtest_multiple_models: Error analyzing models for {symbol}: {str(e)}"
        print(f"❌ backtest_multiple_models: {error_msg}")
        return error_msg


@tool
def visualize_model_comparison_backtesting(
    symbol: str,
    comparison_results_file: Optional[str] = None,
    chart_type: Literal["performance_comparison", "parameter_sensitivity", "risk_return_scatter", "model_type_analysis"] = "performance_comparison",
    top_n_models: int = 10,
    save_chart: bool = True
) -> str:
    """
    Create comprehensive visualizations comparing multiple model backtesting results.
    
    This tool generates interactive charts that help analyze model performance patterns,
    parameter impacts, and identify the best performing models across different metrics.
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')
                     Must have multi-model backtest results available
        
        comparison_results_file (Optional[str]): Specific comparison results file to visualize
                                               If None, uses most recent multi-model results
                                               Format: "backtest_multiple_models_SYMBOL_detailed_TIMESTAMP.json"
        
        chart_type (str): Type of visualization to create:
                         - "performance_comparison": Bar charts of returns, Sharpe, drawdown
                         - "parameter_sensitivity": How parameters affect performance
                         - "risk_return_scatter": Risk vs return scatter plot
                         - "model_type_analysis": Performance by model algorithm type
        
        top_n_models (int): Number of top models to include in visualizations (default 10)
                           Helps focus on best performers for readability
        
        save_chart (bool): Whether to save interactive chart as HTML file
                          TRUE RECOMMENDED for detailed analysis and sharing
    
    Returns:
        str: Description of created visualizations with insights and file locations
    
    Visualization Features:
        - **Interactive Charts**: Hover details, zoom, pan capabilities
        - **Model Identification**: Clear model names with key parameters
        - **Performance Metrics**: Return, Sharpe ratio, drawdown, win rate
        - **Parameter Analysis**: How learning rate, depth, etc. impact results
        - **Risk-Return Analysis**: Optimal risk-adjusted performance identification
        - **Model Type Comparison**: Algorithm effectiveness comparison
    
    Example Usage for AI Agents:
        # Visualize performance comparison
        result = visualize_model_comparison_backtesting("AAPL")
        
        # Parameter sensitivity analysis
        result = visualize_model_comparison_backtesting("TSLA", chart_type="parameter_sensitivity")
        
        # Risk-return analysis
        result = visualize_model_comparison_backtesting("GOOGL", chart_type="risk_return_scatter")
        
        # Model type effectiveness
        result = visualize_model_comparison_backtesting("MSFT", chart_type="model_type_analysis")
    """
    print(f"🔄 visualize_model_comparison_backtesting: Creating visualization for {symbol.upper()}...")
    
    try:
        symbol = symbol.upper()
        
        # Find comparison results file
        if comparison_results_file:
            if not comparison_results_file.endswith('.json'):
                comparison_results_file += '.json'
            results_path = os.path.join(OUTPUT_DIR, comparison_results_file)
        else:
            # Find most recent multi-model results file
            available_files = os.listdir(OUTPUT_DIR) if os.path.exists(OUTPUT_DIR) else []
            comparison_files = [f for f in available_files if 
                              f.startswith(f"backtest_multiple_models_{symbol}_detailed_") and f.endswith('.json')]
            
            if not comparison_files:
                result = f"visualize_model_comparison_backtesting: No multi-model results found for {symbol}. Run backtest_multiple_models first."
                print(f"❌ visualize_model_comparison_backtesting: {result}")
                return result
            
            latest_file = max(comparison_files, key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)))
            results_path = os.path.join(OUTPUT_DIR, latest_file)
            comparison_results_file = latest_file
        
        # Load comparison results
        with open(results_path, 'r') as f:
            comparison_data = json.load(f)
        
        comparison_matrix = pd.DataFrame(comparison_data['comparison_matrix'])
        if comparison_matrix.empty:
            result = f"visualize_model_comparison_backtesting: No model data found in {comparison_results_file}."
            print(f"❌ visualize_model_comparison_backtesting: {result}")
            return result
        
        # Limit to top N models for readability
        comparison_matrix = comparison_matrix.head(top_n_models)
        
        print(f"📊 visualize_model_comparison_backtesting: Loaded {len(comparison_matrix)} models from {comparison_results_file}")
        
        # Create visualization based on chart type
        if chart_type == "performance_comparison":
            fig = create_performance_comparison_chart(comparison_matrix, symbol)
        elif chart_type == "parameter_sensitivity":
            fig = create_parameter_sensitivity_chart(comparison_matrix, symbol)
        elif chart_type == "risk_return_scatter":
            fig = create_risk_return_scatter_chart(comparison_matrix, symbol)
        elif chart_type == "model_type_analysis":
            fig = create_model_type_analysis_chart(comparison_matrix, symbol)
        else:
            result = f"visualize_model_comparison_backtesting: Unknown chart type: {chart_type}"
            print(f"❌ visualize_model_comparison_backtesting: {result}")
            return result
        
        # Save chart if requested
        chart_filename = None
        if save_chart:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_filename = f"visualize_model_comparison_backtesting_{symbol}_{chart_type}_{timestamp}.html"
            chart_filepath = os.path.join(OUTPUT_DIR, chart_filename)
            
            plot(fig, filename=chart_filepath, auto_open=False, include_plotlyjs=True)
            file_size = os.path.getsize(chart_filepath)
        
        # Calculate insights
        best_return = comparison_matrix.loc[comparison_matrix['total_return'].idxmax()]
        best_sharpe = comparison_matrix.loc[comparison_matrix['sharpe_ratio'].idxmax()]
        
        summary = f"""visualize_model_comparison_backtesting: Successfully created {chart_type.replace('_', ' ')} visualization for {symbol}:

        📊 VISUALIZATION SUMMARY:
        - Symbol: {symbol}
        - Chart Type: {chart_type.replace('_', ' ').title()}
        - Models Displayed: {len(comparison_matrix)}
        - Data Source: {comparison_results_file}
        - Performance Range: {comparison_matrix['total_return'].min():.2f}% to {comparison_matrix['total_return'].max():.2f}%

        🏆 TOP PERFORMERS:
        - Best Return: {best_return['model_type']} ({best_return['total_return']:.2f}%)
        Parameters: {best_return['parameters']}
        - Best Sharpe: {best_sharpe['model_type']} ({best_sharpe['sharpe_ratio']:.2f})
        Parameters: {best_sharpe['parameters']}

        📈 VISUALIZATION FEATURES:
        - Interactive Plotly charts with hover details
        - Model identification with key parameters
        - Performance metrics comparison
        - Clear visual ranking and patterns
        - Professional formatting for presentations

        📁 CHART SAVED: {chart_filename if chart_filename else 'Not saved'}
        - Location: {os.path.join(OUTPUT_DIR, chart_filename) if chart_filename else 'N/A'}
        - File Size: {file_size:,} bytes if chart_filename else 'N/A'
        - Format: Interactive HTML with embedded JavaScript

        💡 KEY INSIGHTS:
        - Performance Spread: {comparison_matrix['total_return'].max() - comparison_matrix['total_return'].min():.2f}% difference
        - Sharpe Range: {comparison_matrix['sharpe_ratio'].min():.2f} to {comparison_matrix['sharpe_ratio'].max():.2f}
        - Model Types: {comparison_matrix['model_type'].nunique()} different algorithms tested
        - Average Return: {comparison_matrix['total_return'].mean():.2f}%

        The visualization provides clear insights into model performance patterns and helps identify optimal parameter configurations for future model training.
        """
        
        print(f"✅ visualize_model_comparison_backtesting: Successfully created {chart_type} chart for {symbol}")
        return summary
        
    except Exception as e:
        error_msg = f"visualize_model_comparison_backtesting: Error creating visualization for {symbol}: {str(e)}"
        print(f"❌ visualize_model_comparison_backtesting: {error_msg}")
        return error_msg


def create_performance_comparison_chart(comparison_matrix: pd.DataFrame, symbol: str):
    """Create performance comparison bar charts."""
    fig = sp.make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f'{symbol} Total Return Comparison',
            f'{symbol} Sharpe Ratio Comparison', 
            f'{symbol} Maximum Drawdown Comparison',
            f'{symbol} Win Rate Comparison'
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.12
    )
    
    # Prepare data - truncate long model names for display
    display_names = []
    for _, row in comparison_matrix.iterrows():
        name = f"{row['model_type'][:8]}..{row['model_id'][-8:]}" if len(row['model_id']) > 20 else row['model_id']
        display_names.append(name)
    
    # Total Return (Row 1, Col 1)
    fig.add_trace(go.Bar(
        x=display_names,
        y=comparison_matrix['total_return'],
        name='Total Return (%)',
        marker_color='lightblue',
        hovertemplate='<b>%{x}</b><br>Return: %{y:.2f}%<extra></extra>'
    ), row=1, col=1)
    
    # Sharpe Ratio (Row 1, Col 2)
    fig.add_trace(go.Bar(
        x=display_names,
        y=comparison_matrix['sharpe_ratio'],
        name='Sharpe Ratio',
        marker_color='lightgreen',
        hovertemplate='<b>%{x}</b><br>Sharpe: %{y:.2f}<extra></extra>',
        showlegend=False
    ), row=1, col=2)
    
    # Max Drawdown (Row 2, Col 1) - less negative is better
    fig.add_trace(go.Bar(
        x=display_names,
        y=comparison_matrix['max_drawdown'],
        name='Max Drawdown (%)',
        marker_color='lightcoral',
        hovertemplate='<b>%{x}</b><br>Drawdown: %{y:.2f}%<extra></extra>',
        showlegend=False
    ), row=2, col=1)
    
    # Win Rate (Row 2, Col 2)
    fig.add_trace(go.Bar(
        x=display_names,
        y=comparison_matrix['win_rate'],
        name='Win Rate (%)',
        marker_color='lightyellow',
        hovertemplate='<b>%{x}</b><br>Win Rate: %{y:.1f}%<extra></extra>',
        showlegend=False
    ), row=2, col=2)
    
    fig.update_layout(
        title=f'{symbol} Model Performance Comparison',
        template='plotly_white',
        height=800,
        width=1200,
        showlegend=False
    )
    
    # Update y-axis titles
    fig.update_yaxes(title_text='Return (%)', row=1, col=1)
    fig.update_yaxes(title_text='Sharpe Ratio', row=1, col=2)
    fig.update_yaxes(title_text='Drawdown (%)', row=2, col=1)
    fig.update_yaxes(title_text='Win Rate (%)', row=2, col=2)
    
    # Rotate x-axis labels for better readability
    fig.update_xaxes(tickangle=45)
    
    return fig


def create_parameter_sensitivity_chart(comparison_matrix: pd.DataFrame, symbol: str):
    """Create parameter sensitivity analysis chart."""
    fig = go.Figure()
    
    # Extract parameter information and plot
    model_types = comparison_matrix['model_type'].unique()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, model_type in enumerate(model_types):
        model_data = comparison_matrix[comparison_matrix['model_type'] == model_type]
        
        fig.add_trace(go.Scatter(
            x=list(range(len(model_data))),
            y=model_data['total_return'],
            mode='markers+lines',
            name=model_type,
            marker=dict(
                size=model_data['sharpe_ratio'] * 5 + 5,  # Size by Sharpe ratio
                color=colors[i % len(colors)],
                opacity=0.7
            ),
            line=dict(color=colors[i % len(colors)]),
            hovertemplate=f'<b>{model_type}</b><br>' +
                         'Return: %{y:.2f}%<br>' +
                         'Parameters: %{text}<extra></extra>',
            text=model_data['parameters']
        ))
    
    fig.update_layout(
        title=f'{symbol} Parameter Sensitivity Analysis',
        xaxis_title='Model Variants',
        yaxis_title='Total Return (%)',
        template='plotly_white',
        height=600,
        width=1200,
        hovermode='closest'
    )
    
    return fig


def create_risk_return_scatter_chart(comparison_matrix: pd.DataFrame, symbol: str):
    """Create risk-return scatter plot."""
    fig = go.Figure()
    
    # Use max drawdown as risk measure (absolute value for better visualization)
    risk_measure = comparison_matrix['max_drawdown'].abs()
    
    model_types = comparison_matrix['model_type'].unique()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, model_type in enumerate(model_types):
        model_data = comparison_matrix[comparison_matrix['model_type'] == model_type]
        model_risk = risk_measure[comparison_matrix['model_type'] == model_type]
        
        fig.add_trace(go.Scatter(
            x=model_risk,
            y=model_data['total_return'],
            mode='markers',
            name=model_type,
            marker=dict(
                size=model_data['total_trades'] / 5 + 8,  # Size by number of trades
                color=colors[i % len(colors)],
                opacity=0.7
            ),
            hovertemplate=f'<b>{model_type}</b><br>' +
                         'Risk (Drawdown): %{x:.2f}%<br>' +
                         'Return: %{y:.2f}%<br>' +
                         'Trades: %{text}<extra></extra>',
            text=model_data['total_trades']
        ))
    
    fig.update_layout(
        title=f'{symbol} Risk-Return Analysis',
        xaxis_title='Risk (Max Drawdown %)',
        yaxis_title='Total Return (%)',
        template='plotly_white',
        height=600,
        width=1200,
        hovermode='closest'
    )
    
    # Add diagonal lines for Sharpe ratio reference
    max_risk = risk_measure.max()
    max_return = comparison_matrix['total_return'].max()
    
    # Add reference lines for different Sharpe ratios
    for sharpe in [0.5, 1.0, 1.5]:
        fig.add_shape(
            type="line",
            x0=0, y0=0,
            x1=max_risk, y1=max_risk * sharpe,
            line=dict(color="gray", width=1, dash="dot"),
        )
        fig.add_annotation(
            x=max_risk * 0.8,
            y=max_risk * 0.8 * sharpe,
            text=f"Sharpe {sharpe}",
            showarrow=False,
            font=dict(size=10, color="gray")
        )
    
    return fig


def create_model_type_analysis_chart(comparison_matrix: pd.DataFrame, symbol: str):
    """Create model type analysis chart."""
    # Calculate average performance by model type
    model_type_stats = comparison_matrix.groupby('model_type').agg({
        'total_return': ['mean', 'std', 'count'],
        'sharpe_ratio': 'mean',
        'max_drawdown': 'mean',
        'win_rate': 'mean'
    }).round(2)
    
    model_type_stats.columns = ['avg_return', 'return_std', 'count', 'avg_sharpe', 'avg_drawdown', 'avg_winrate']
    model_type_stats = model_type_stats.reset_index()
    
    fig = sp.make_subplots(
        rows=1, cols=2,
        subplot_titles=[f'{symbol} Average Return by Model Type', f'{symbol} Risk-Adjusted Performance'],
        specs=[[{"secondary_y": False}, {"secondary_y": True}]]
    )
    
    # Average return with error bars (Row 1, Col 1)
    fig.add_trace(go.Bar(
        x=model_type_stats['model_type'],
        y=model_type_stats['avg_return'],
        error_y=dict(type='data', array=model_type_stats['return_std']),
        name='Average Return',
        marker_color='lightblue',
        hovertemplate='<b>%{x}</b><br>Avg Return: %{y:.2f}%<br>Std Dev: %{error_y.array:.2f}%<br>Models: %{text}<extra></extra>',
        text=model_type_stats['count']
    ), row=1, col=1)
    
    # Sharpe ratio (Row 1, Col 2)
    fig.add_trace(go.Bar(
        x=model_type_stats['model_type'],
        y=model_type_stats['avg_sharpe'],
        name='Average Sharpe',
        marker_color='lightgreen',
        hovertemplate='<b>%{x}</b><br>Avg Sharpe: %{y:.2f}<extra></extra>',
        showlegend=False
    ), row=1, col=2)
    
    # Win rate as secondary y-axis
    fig.add_trace(go.Scatter(
        x=model_type_stats['model_type'],
        y=model_type_stats['avg_winrate'],
        mode='markers+lines',
        name='Average Win Rate',
        marker=dict(color='red', size=10),
        line=dict(color='red', dash='dot'),
        hovertemplate='<b>%{x}</b><br>Avg Win Rate: %{y:.1f}%<extra></extra>',
        yaxis='y2'
    ), row=1, col=2)
    
    fig.update_layout(
        title=f'{symbol} Model Type Performance Analysis',
        template='plotly_white',
        height=500,
        width=1200
    )
    
    # Update y-axis titles
    fig.update_yaxes(title_text='Average Return (%)', row=1, col=1)
    fig.update_yaxes(title_text='Average Sharpe Ratio', row=1, col=2)
    fig.update_yaxes(title_text='Average Win Rate (%)', secondary_y=True, row=1, col=2)
    
    return fig


# =============================================================================
# ENHANCED MODEL METADATA STORAGE
# This replaces the existing save_model_artifacts function in tools.py
# =============================================================================

def save_model_artifacts(
    model,
    model_data: dict,
    predictions_data: dict,
    metrics: dict,
    feature_importance: pd.DataFrame,
    symbol: str,
    model_type: str,
    model_params: dict,
    target_days: int,
    save_model: bool = True,
    save_predictions: bool = True
) -> dict:
    """
    Enhanced version of save_model_artifacts with richer metadata for multi-model comparison.
    This replaces the existing save_model_artifacts function.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filenames = {'model': None, 'results': None, 'predictions': None}
    
    if save_model:
        # Save model with enhanced metadata
        model_filename = f"train_{model_type}_price_predictor_{symbol}_model_{timestamp}.pkl"
        model_filepath = os.path.join(OUTPUT_DIR, model_filename)
        
        # Enhanced model metadata
        model_metadata = {
            'model': model,
            'scaler': model_data['scaler'],
            'feature_cols': model_data['feature_cols'],
            'target_days': target_days,
            'symbol': symbol,
            'model_type': model_type,
            'model_id': generate_model_signature(model_type, model_params),
            'parameter_summary': create_parameter_summary(model_type, model_params),
            'training_context': {
                'data_source': model_data['data_source'],
                'training_samples': len(model_data['X_train']),
                'test_samples': len(model_data['X_test']),
                'features_used': len(model_data['feature_cols']),
                'target_days': target_days
            },
            'performance_fingerprint': {
                'test_r2': metrics['test_metrics']['r2'],
                'cv_stability': metrics['cross_validation']['cv_r2_std'],
                'feature_importance_entropy': calculate_feature_entropy(feature_importance)
            }
        }
        
        with open(model_filepath, 'wb') as f:
            pickle.dump(model_metadata, f)
        filenames['model'] = model_filename
        
        # Enhanced results with metadata
        results = {
            'symbol': symbol,
            'model_type': model_type.replace('_', ' ').title(),
            'model_id': generate_model_signature(model_type, model_params),
            'parameter_summary': create_parameter_summary(model_type, model_params),
            'target_days': target_days,
            'data_source': model_data['data_source'],
            'training_samples': len(model_data['X_train']),
            'test_samples': len(model_data['X_test']),
            'features_used': model_data['feature_cols'],
            'model_params': model_params,
            'performance': {
                'train_r2': metrics['train_metrics']['r2'],
                'test_r2': metrics['test_metrics']['r2'],
                'train_rmse': metrics['train_metrics']['rmse'],
                'test_rmse': metrics['test_metrics']['rmse'],
                'train_mae': metrics['train_metrics']['mae'],
                'test_mae': metrics['test_metrics']['mae'],
                'cv_r2_mean': metrics['cross_validation']['cv_r2_mean'],
                'cv_r2_std': metrics['cross_validation']['cv_r2_std']
            },
            'feature_importance': feature_importance.to_dict('records'),
            'timestamp': timestamp,
            'model_signature': generate_model_signature(model_type, model_params)
        }
        
        results_filename = f"train_{model_type}_price_predictor_{symbol}_results_{timestamp}.json"
        results_filepath = os.path.join(OUTPUT_DIR, results_filename)
        with open(results_filepath, 'w') as f:
            json.dump(results, f, indent=2)
        filenames['results'] = results_filename
    
    if save_predictions:
        # Save predictions (unchanged from original)
        train_df = pd.DataFrame({
            'Date': predictions_data['train_dates'],
            'Actual': predictions_data['train_actuals'],
            'Predicted': predictions_data['train_predictions'],
            'Set': 'Train'
        }).set_index('Date')
        
        test_df = pd.DataFrame({
            'Date': predictions_data['test_dates'],
            'Actual': predictions_data['test_actuals'],
            'Predicted': predictions_data['test_predictions'],
            'Set': 'Test'
        }).set_index('Date')
        
        combined_df = pd.concat([train_df, test_df])
        combined_df['Error'] = combined_df['Predicted'] - combined_df['Actual']
        combined_df['Absolute_Error'] = abs(combined_df['Error'])
        combined_df['Percentage_Error'] = (combined_df['Error'] / np.maximum(abs(combined_df['Actual']), 1e-8)) * 100
        
        predictions_filename = f"{model_type}_predictions_{symbol}_{timestamp}.csv"
        filepath = os.path.join(OUTPUT_DIR, predictions_filename)
        combined_df.to_csv(filepath)
        filenames['predictions'] = predictions_filename
    
    return filenames










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
    
    XGBoost (eXtreme Gradient Boosting) is a powerful ensemble learning method that uses
    gradient boosting framework optimized for speed and performance. It's particularly
    effective for structured/tabular data and handles non-linear relationships well.
    
    Key Advantages:
    - Excellent performance on structured data
    - Built-in regularization prevents overfitting
    - Handles missing values automatically
    - Fast training with parallel processing
    - Feature importance ranking included
    
    Best Use Cases:
    - Short to medium-term trading (1-14 days)
    - High-frequency data with complex patterns
    - When maximum accuracy is priority
    - Stocks with non-linear price relationships
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL', 'GOOGL', 'TSLA', 'MSFT')
                     Must have enhanced technical indicators data available
        
        source_file (Optional[str]): Specific enhanced CSV file with technical indicators
                                   If None, automatically uses most recent enhanced file
                                   Should contain OHLCV + technical indicators
        
        target_days (int): Number of days ahead to predict
                          - 1: Next day (day trading) - RECOMMENDED
                          - 3-5: Short-term swing trading
                          - 7-14: Medium-term trading
                          - 30+: Long-term (may be less accurate)
        
        test_size (float): Proportion of data for testing (0.1-0.3)
                          0.2 = 20% test, 80% train (RECOMMENDED)
                          Larger test sets = more reliable validation
        
        n_estimators (int): Number of boosting rounds/trees
                           - 50-100: Fast training, may underfit
                           - 100-200: Good balance (RECOMMENDED)
                           - 200-500: High accuracy, slower training
                           - 500+: Risk of overfitting
        
        max_depth (int): Maximum depth of each tree
                        - 3-6: Conservative, less overfitting (RECOMMENDED)
                        - 6-10: More complex patterns
                        - 10+: High risk of overfitting
        
        learning_rate (float): Step size shrinkage to prevent overfitting
                              - 0.01-0.05: Very conservative, needs more estimators
                              - 0.1: Good default balance (RECOMMENDED)
                              - 0.2-0.3: Aggressive learning
                              - 0.3+: High risk of instability
        
        save_model (bool): Whether to save trained model (.pkl) and results (.json)
                          TRUE RECOMMENDED for analysis and backtesting
        
        save_predictions (bool): Whether to save train/test predictions to CSV
                               TRUE RECOMMENDED for detailed analysis
    
    Returns:
        str: Comprehensive training report including:
             - Model configuration and parameters
             - Performance metrics (RMSE, MAE, R², Information Ratio)
             - Cross-validation scores
             - Feature importance rankings
             - Model quality assessment and insights
             - File locations for saved artifacts
    
    Example Usage for AI Agents:
        # Conservative day trading model
        result = train_xgboost_price_predictor("AAPL", target_days=1, n_estimators=100, max_depth=3, learning_rate=0.05)
        
        # Aggressive short-term model
        result = train_xgboost_price_predictor("TSLA", target_days=3, n_estimators=200, max_depth=8, learning_rate=0.2)
        
        # Quick prototyping
        result = train_xgboost_price_predictor("GOOGL")  # Uses all defaults
    
    AI Agent Decision Guidelines:
        - For volatile stocks (TSLA, MEME): Use lower learning_rate (0.05-0.1), more estimators
        - For stable stocks (AAPL, MSFT): Standard parameters work well
        - For day trading: target_days=1, max_depth=3-6
        - For swing trading: target_days=5-7, max_depth=6-10
        - Always enable save_model=True for backtesting capability
    """
    def create_xgboost_model(n_estimators, max_depth, learning_rate, **kwargs):
        return xgb.XGBRegressor(
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
    
    Key Advantages:
    - Highly robust and stable predictions
    - Less prone to overfitting than single trees
    - Handles noisy data well
    - Provides reliable feature importance
    - Works well with default parameters
    - Good for various prediction horizons
    
    Best Use Cases:
    - Medium to long-term predictions (7-30+ days)
    - When model stability is more important than peak accuracy
    - Noisy or inconsistent market conditions
    - When interpretability is important
    - Conservative trading strategies
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL', 'GOOGL', 'TSLA', 'MSFT')
                     Must have enhanced technical indicators data available
        
        source_file (Optional[str]): Specific enhanced CSV file with technical indicators
                                   If None, automatically uses most recent enhanced file
                                   Should contain OHLCV + technical indicators
        
        target_days (int): Number of days ahead to predict
                          - 1: Next day prediction
                          - 7: Next week (RECOMMENDED for Random Forest)
                          - 14: Two weeks ahead
                          - 30: Next month (good performance)
        
        test_size (float): Proportion of data for testing (0.1-0.3)
                          0.2 = 20% test, 80% train (RECOMMENDED)
        
        n_estimators (int): Number of trees in the forest
                           - 50-100: Fast training, good for prototyping
                           - 100-200: Excellent balance (RECOMMENDED)
                           - 200-500: Marginal improvements, slower training
                           - 500+: Diminishing returns
        
        max_depth (Optional[int]): Maximum depth of trees
                                  - None: Unlimited depth (RECOMMENDED - Random Forest handles this well)
                                  - 10-20: Conservative depth limiting
                                  - 5-10: Very conservative, may underfit
        
        min_samples_split (int): Minimum samples required to split internal node
                                - 2: Maximum granularity (RECOMMENDED)
                                - 5-10: More conservative, prevents overfitting
                                - 10+: Very conservative, may underfit
        
        save_model (bool): Whether to save trained model (.pkl) and results (.json)
                          TRUE RECOMMENDED for analysis and backtesting
        
        save_predictions (bool): Whether to save train/test predictions to CSV
                               TRUE RECOMMENDED for detailed analysis
    
    Returns:
        str: Comprehensive training report including:
             - Model configuration and parameters
             - Performance metrics (RMSE, MAE, R², Information Ratio)
             - Cross-validation scores
             - Feature importance rankings
             - Model quality assessment and insights
             - File locations for saved artifacts
    
    Example Usage for AI Agents:
        # Conservative long-term model
        result = train_random_forest_price_predictor("AAPL", target_days=14, n_estimators=200)
        
        # Quick swing trading model
        result = train_random_forest_price_predictor("MSFT", target_days=7, n_estimators=100)
        
        # High-stability model for volatile stock
        result = train_random_forest_price_predictor("TSLA", target_days=30, n_estimators=300, min_samples_split=10)
        
        # Default parameters (good starting point)
        result = train_random_forest_price_predictor("GOOGL")
    
    AI Agent Decision Guidelines:
        - For stable predictions: Use Random Forest over XGBoost
        - For longer horizons (>7 days): Random Forest often outperforms
        - For volatile markets: Increase n_estimators, increase min_samples_split
        - For conservative trading: Use max_depth=15, min_samples_split=5
        - For diversified portfolios: Random Forest provides more consistent results
        - Default parameters (max_depth=None) work well in most cases
    """
    def create_random_forest_model(n_estimators, max_depth, min_samples_split, **kwargs):
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
# EXAMPLE: ADDING NEW MODELS IS NOW TRIVIAL (ZERO DUPLICATION)
# =============================================================================

# Example 1: Support Vector Regression - Only 15 lines of code!
@tool
def train_svr_price_predictor(
    symbol: str,
    source_file: Optional[str] = None,
    target_days: int = 1,
    test_size: float = 0.2,
    C: float = 1.0,
    gamma: str = 'scale',
    kernel: str = 'rbf',
    save_model: bool = True,
    save_predictions: bool = True
) -> str:
    """
    Train Support Vector Regression model for stock price prediction.
    
    SVR uses support vector machines for regression, finding optimal hyperplane
    that best fits the data with maximum margin. Excellent for complex non-linear
    relationships and robust to outliers.
    
    Best Use Cases:
    - Complex non-linear patterns
    - Robust predictions with outliers
    - When you have sufficient computational resources
    - Medium-term predictions (3-14 days)
    
    Parameter Guidelines:
        C: Regularization parameter (0.1-10)
           - Lower values: More regularization, simpler model
           - Higher values: Less regularization, more complex model
        
        gamma: Kernel coefficient ('scale', 'auto', or float)
               - 'scale': 1/(n_features * X.var()) - RECOMMENDED
               - 'auto': 1/n_features
               - Custom float: Manual control
        
        kernel: Kernel type ('rbf', 'linear', 'poly')
               - 'rbf': Radial basis function - RECOMMENDED for most cases
               - 'linear': Linear relationships only
               - 'poly': Polynomial relationships
    """
    def create_svr_model(C, gamma, kernel, **kwargs):
        from sklearn.svm import SVR
        return SVR(C=C, gamma=gamma, kernel=kernel)
    
    return train_model_pipeline(
        symbol=symbol,
        model_type='svr',
        model_factory_func=create_svr_model,
        source_file=source_file,
        target_days=target_days,
        test_size=test_size,
        save_model=save_model,
        save_predictions=save_predictions,
        C=C,
        gamma=gamma,
        kernel=kernel
    )


# Example 2: Gradient Boosting Regressor - Another 15 lines!
@tool
def train_gradient_boosting_price_predictor(
    symbol: str,
    source_file: Optional[str] = None,
    target_days: int = 1,
    test_size: float = 0.2,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    save_model: bool = True,
    save_predictions: bool = True
) -> str:
    """
    Train Gradient Boosting model for stock price prediction.
    
    Gradient Boosting builds models sequentially, with each new model correcting
    errors from previous models. Similar to XGBoost but uses scikit-learn implementation.
    
    Best Use Cases:
    - Alternative to XGBoost with scikit-learn ecosystem
    - Sequential error correction approach
    - Good balance of accuracy and interpretability
    - Short to medium-term predictions
    
    Parameter Guidelines:
        n_estimators: Number of boosting stages (50-200 recommended)
        learning_rate: Learning rate shrinks contribution (0.01-0.2)
        max_depth: Maximum depth of trees (3-8 recommended)
    """
    def create_gb_model(n_estimators, learning_rate, max_depth, **kwargs):
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
    
    return train_model_pipeline(
        symbol=symbol,
        model_type='gradient_boosting',
        model_factory_func=create_gb_model,
        source_file=source_file,
        target_days=target_days,
        test_size=test_size,
        save_model=save_model,
        save_predictions=save_predictions,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth
    )


# Example 3: Linear Regression with Ridge Regularization
@tool
def train_ridge_regression_price_predictor(
    symbol: str,
    source_file: Optional[str] = None,
    target_days: int = 1,
    test_size: float = 0.2,
    alpha: float = 1.0,
    fit_intercept: bool = True,
    save_model: bool = True,
    save_predictions: bool = True
) -> str:
    """
    Train Ridge Regression model for stock price prediction.
    
    Ridge Regression is linear regression with L2 regularization, preventing
    overfitting by penalizing large coefficients. Excellent baseline model
    and provides interpretable linear relationships.
    
    Best Use Cases:
    - Baseline model for comparison
    - When linear relationships are expected
    - High interpretability requirements
    - Quick prototyping and testing
    - When you have limited training data
    
    Parameter Guidelines:
        alpha: Regularization strength (0.1-10)
               - Higher values: More regularization
               - Lower values: Less regularization
        fit_intercept: Whether to fit intercept term (usually True)
    """
    def create_ridge_model(alpha, fit_intercept, **kwargs):
        from sklearn.linear_model import Ridge
        
        # Add dummy feature_importances_ for consistency
        class RidgeWithImportance(Ridge):
            @property
            def feature_importances_(self):
                import numpy as np
                return np.abs(self.coef_) / np.sum(np.abs(self.coef_))
        
        return RidgeWithImportance(alpha=alpha, fit_intercept=fit_intercept)
    
    return train_model_pipeline(
        symbol=symbol,
        model_type='ridge_regression',
        model_factory_func=create_ridge_model,
        source_file=source_file,
        target_days=target_days,
        test_size=test_size,
        save_model=save_model,
        save_predictions=save_predictions,
        alpha=alpha,
        fit_intercept=fit_intercept
    )


# Example 4: Extra Trees Regressor (Extremely Randomized Trees)
@tool
def train_extra_trees_price_predictor(
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
    Train Extra Trees (Extremely Randomized Trees) model for stock price prediction.
    
    Extra Trees uses random thresholds for each feature rather than searching for
    the best thresholds like Random Forest. This reduces variance and can provide
    better generalization.
    
    Best Use Cases:
    - Similar to Random Forest but with more randomization
    - Good for reducing overfitting
    - Faster training than Random Forest
    - When you want more model diversity
    """
    def create_extra_trees_model(n_estimators, max_depth, min_samples_split, **kwargs):
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1
        )
    
    return train_model_pipeline(
        symbol=symbol,
        model_type='extra_trees',
        model_factory_func=create_extra_trees_model,
        source_file=source_file,
        target_days=target_days,
        test_size=test_size,
        save_model=save_model,
        save_predictions=save_predictions,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split
    )


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