import os
import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid threading issues
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
            filename = f"{symbol.upper()}_{period}_{interval}_{timestamp}.csv"
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
    Create visualizations of stock data using the most recent data file or fetch new data.
    Charts are automatically saved to the output directory.
    
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
        data_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(f"{symbol}_") and f.endswith('.csv')]
        
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
        
        # Create the visualization
        if chart_type == "line":
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data.index, data['Close'], linewidth=2, color='blue')
            ax.set_title(f'{symbol} Stock Price - Line Chart')
            ax.set_ylabel('Price ($)')
            ax.grid(True, alpha=0.3)
            
        elif chart_type == "candlestick":
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Create candlestick-like visualization using matplotlib
            for i, (date, row) in enumerate(data.iterrows()):
                color = 'green' if row['Close'] >= row['Open'] else 'red'
                # Body
                ax.plot([date, date], [row['Open'], row['Close']], color=color, linewidth=3)
                # Wicks
                ax.plot([date, date], [row['Low'], row['High']], color=color, linewidth=1)
            
            ax.set_title(f'{symbol} Stock Price - Candlestick Chart')
            ax.set_ylabel('Price ($)')
            ax.grid(True, alpha=0.3)
            
        elif chart_type == "volume":
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(data.index, data['Volume'], alpha=0.7, color='orange')
            ax.set_title(f'{symbol} Trading Volume')
            ax.set_ylabel('Volume')
            ax.grid(True, alpha=0.3)
            
        elif chart_type == "combined":
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            
            # Price chart
            ax1.plot(data.index, data['Close'], linewidth=2, color='blue', label='Close')
            ax1.plot(data.index, data['High'], linewidth=1, color='green', alpha=0.7, label='High')
            ax1.plot(data.index, data['Low'], linewidth=1, color='red', alpha=0.7, label='Low')
            ax1.set_title(f'{symbol} Stock Analysis - Combined Chart')
            ax1.set_ylabel('Price ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Volume chart
            ax2.bar(data.index, data['Volume'], alpha=0.7, color='orange')
            ax2.set_ylabel('Volume')
            ax2.set_xlabel('Date')
            ax2.grid(True, alpha=0.3)
        
        # Format x-axis dates based on chart type
        if chart_type == "combined":
            # For combined charts, format the bottom subplot
            if len(data) > 30:
                ax2.xaxis.set_major_locator(mdates.MonthLocator())
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            else:
                ax2.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(data)//10)))
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        else:
            # For single charts
            if len(data) > 30:
                ax.xaxis.set_major_locator(mdates.MonthLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            else:
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(data)//10)))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save chart (always save in multi-agent environment)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_filename = f"{symbol}_{chart_type}_chart_{timestamp}.png"
        chart_filepath = os.path.join(OUTPUT_DIR, chart_filename)
        plt.savefig(chart_filepath, dpi=300, bbox_inches='tight')
        
        # Always close the figure to prevent memory leaks and threading issues
        plt.close(fig)
        
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
📈 VISUALIZATION CREATED for {symbol}:

🎨 CHART DETAILS:
- Chart Type: {chart_type.title()}
- Data Source: {data_source}
- Data Points: {len(data)} records
- Date Range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}

📊 KEY STATISTICS:
- Current Price: ${price_stats['current_price']:.2f}
- Period High: ${price_stats['period_high']:.2f}
- Period Low: ${price_stats['period_low']:.2f}
- Price Change: ${price_stats['price_change']:.2f} ({price_stats['price_change_pct']:.2f}%)
- Average Volume: {price_stats['avg_volume']:,.0f}

📁 CHART SAVED: {chart_filename}
- Location: {os.path.join(OUTPUT_DIR, chart_filename)}

The visualization shows price movements and trading patterns for the selected period.
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
        chart_files = [f for f in files if f.endswith('.png')]
        
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
            summary += "📈 CHART FILES (.png):\n"
            for file in sorted(chart_files):
                filepath = os.path.join(OUTPUT_DIR, file)
                size = os.path.getsize(filepath)
                modified = datetime.fromtimestamp(os.path.getmtime(filepath))
                summary += f"  - {file} ({size:,} bytes, modified: {modified.strftime('%Y-%m-%d %H:%M:%S')})\n"
            summary += "\n"
        
        summary += f"📈 TOTAL FILES: {len(files)} ({len(data_files)} data, {len(chart_files)} charts)"
        
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
            clean_filename = f"{clean_filename}_{timestamp}"
        
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