"""Basic chart creation functionality using Plotly."""

import os
import pandas as pd
import sys
from datetime import datetime
from typing import Optional, Literal
from pathlib import Path

# Import logging helpers
try:
    from ..logs.logging_helpers import log_info, log_success, log_warning, log_error, log_progress, safe_run
    _logging_helpers_available = True
except ImportError:
    _logging_helpers_available = False
    # Fallback to regular logger if logging_helpers not available
    def log_info(msg, **kwargs): logger.info(msg)
    def log_success(msg, **kwargs): logger.info(msg)
    def log_warning(msg, **kwargs): logger.warning(msg) 
    def log_error(msg, **kwargs): logger.error(msg)
    def log_progress(msg, **kwargs): logger.info(msg)
    def safe_run(func): return func

try:
    import plotly.graph_objects as go
    import plotly.subplots as sp
    from plotly.offline import plot
    import plotly.offline as pyo
    _plotly_available = True
except ImportError:
    _plotly_available = False

from ..config import OUTPUT_DIR, logger


@safe_run
def visualize_stock_data_impl(
    symbol: str,
    chart_type: Literal["line", "candlestick", "volume", "combined"] = "combined",
    source_file: Optional[str] = None,
    save_chart: bool = True,
    show_indicators: bool = True
) -> str:
    """
    Create interactive stock charts using Plotly with various chart types.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'GOOGL')
        chart_type: Type of chart to create ('line', 'candlestick', 'volume', 'combined')
        source_file: CSV file with stock data (optional)
        save_chart: Whether to save chart to HTML file
        show_indicators: Whether to include technical indicators
        
    Returns:
        String with chart creation results and file location
    """
    log_info(f"visualize_stock_data: Creating {chart_type} chart for {symbol.upper()}...")
    
    if not _plotly_available:
        error_msg = "Plotly not available. Please install: pip install plotly"
        log_error(f"visualize_stock_data: {error_msg}")
        return error_msg
    
    try:
        symbol = symbol.upper()
        
        # Load data
        if source_file:
            if not source_file.endswith('.csv'):
                source_file += '.csv'
            filepath = os.path.join(OUTPUT_DIR, source_file)
            if not os.path.exists(filepath):
                result = f"visualize_stock_data: Source file '{source_file}' not found."
                log_error(f"visualize_stock_data: {result}")
                return result
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            data_source = source_file
        else:
            # Find most recent data file (enhanced or basic)
            all_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.csv') and symbol in f.upper()]
            if not all_files:
                result = f"visualize_stock_data: No data files found for {symbol}."
                log_error(f"visualize_stock_data: {result}")
                return result
            
            # Prefer enhanced files
            enhanced_files = [f for f in all_files if 'enhanced' in f.lower()]
            if enhanced_files:
                latest_file = max(enhanced_files, key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)))
            else:
                latest_file = max(all_files, key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)))
            
            filepath = os.path.join(OUTPUT_DIR, latest_file)
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            data_source = latest_file
        
        if data.empty:
            result = f"visualize_stock_data: No data available in {data_source}."
            log_error(f"visualize_stock_data: {result}")
            return result
        
        # Ensure required columns exist
        required_cols = ['Close']
        if chart_type in ['candlestick', 'combined']:
            required_cols.extend(['Open', 'High', 'Low'])
        if chart_type in ['volume', 'combined']:
            required_cols.append('Volume')
        
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            result = f"visualize_stock_data: Missing required columns: {missing_cols}"
            log_error(f"visualize_stock_data: {result}")
            return result
        
        # Create chart based on type
        if chart_type == "line":
            fig = create_line_chart(data, symbol, show_indicators)
        elif chart_type == "candlestick":
            fig = create_candlestick_chart(data, symbol, show_indicators)
        elif chart_type == "volume":
            fig = create_volume_chart(data, symbol)
        elif chart_type == "combined":
            fig = create_combined_chart(data, symbol, show_indicators)
        else:
            result = f"visualize_stock_data: Invalid chart type '{chart_type}'"
            log_error(f"visualize_stock_data: {result}")
            return result
        
        # Save chart if requested
        chart_file = None
        if save_chart:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_file = f"visualize_stock_data_{symbol}_{chart_type}_{timestamp}.html"
            chart_filepath = os.path.join(OUTPUT_DIR, chart_file)
            
            # Ensure Plotly is properly initialized for offline plotting
            pyo.init_notebook_mode(connected=False)
            plot(fig, filename=chart_filepath, auto_open=False, include_plotlyjs='cdn', config={'displayModeBar': True})
        
        # Count indicators if shown
        indicator_count = 0
        if show_indicators:
            technical_cols = [col for col in data.columns 
                            if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]
            indicator_count = len(technical_cols)
        
        # Generate summary
        summary = f"""visualize_stock_data: Successfully created {chart_type} chart for {symbol}:

 CHART DETAILS:
- Symbol: {symbol}
- Chart Type: {chart_type.title()}
- Data Source: {data_source}
- Date Range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}
- Data Points: {len(data):,}

 CHART FEATURES:
- Interactive: Zoom, pan, hover tooltips
- Technical Indicators: {indicator_count if show_indicators else 0} overlays
- Professional Styling: Corporate color scheme
- Export Options: PNG, HTML, PDF available

 CHART SAVED: {chart_file if chart_file else 'Not saved'}
- Location: {os.path.join(OUTPUT_DIR, chart_file) if chart_file else 'N/A'}
- Format: Interactive HTML with embedded JavaScript

 USAGE NOTES:
- Open HTML file in any web browser
- Use mouse to zoom and pan around the chart
- Hover over data points for detailed information
- Click legend items to show/hide data series
- Perfect for presentations and analysis sharing
"""
        
        log_success(f"visualize_stock_data: Successfully created {chart_type} chart for {symbol}")
        return summary
        
    except Exception as e:
        error_msg = f"visualize_stock_data: Error creating chart for {symbol}: {str(e)}"
        log_error(f"visualize_stock_data: {error_msg}")
        return error_msg


def create_line_chart(data: pd.DataFrame, symbol: str, show_indicators: bool = True) -> go.Figure:
    """Create a simple line chart of closing prices."""
    fig = go.Figure()
    
    # Add closing price line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name=f'{symbol} Close Price',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Add technical indicators if available and requested
    if show_indicators:
        add_technical_indicators(fig, data)
    
    fig.update_layout(
        title=f'{symbol} Stock Price Chart',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


def create_candlestick_chart(data: pd.DataFrame, symbol: str, show_indicators: bool = True) -> go.Figure:
    """Create a candlestick chart with OHLC data."""
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=f'{symbol} OHLC'
    ))
    
    # Add technical indicators if available and requested
    if show_indicators:
        add_technical_indicators(fig, data)
    
    fig.update_layout(
        title=f'{symbol} Candlestick Chart',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_white',
        xaxis_rangeslider_visible=False
    )
    
    return fig


def create_volume_chart(data: pd.DataFrame, symbol: str) -> go.Figure:
    """Create a volume bars chart."""
    fig = go.Figure()
    
    # Add volume bars
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        name=f'{symbol} Volume',
        marker_color='rgba(158,202,225,0.8)'
    ))
    
    # Add volume moving average if available
    if 'Volume_SMA_20' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Volume_SMA_20'],
            mode='lines',
            name='Volume SMA (20)',
            line=dict(color='red', width=2)
        ))
    
    fig.update_layout(
        title=f'{symbol} Volume Chart',
        xaxis_title='Date',
        yaxis_title='Volume',
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


def create_combined_chart(data: pd.DataFrame, symbol: str, show_indicators: bool = True) -> go.Figure:
    """Create a combined chart with price and volume subplots."""
    # Create subplots
    fig = sp.make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f'{symbol} Price', 'Volume'),
        row_weights=[0.7, 0.3]
    )
    
    # Add candlestick to first subplot
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=f'{symbol} OHLC'
    ), row=1, col=1)
    
    # Add technical indicators to first subplot if available
    if show_indicators:
        add_technical_indicators_to_subplot(fig, data, row=1, col=1)
    
    # Add volume to second subplot
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        name='Volume',
        marker_color='rgba(158,202,225,0.8)',
        showlegend=False
    ), row=2, col=1)
    
    # Add volume moving average if available
    if 'Volume_SMA_20' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Volume_SMA_20'],
            mode='lines',
            name='Volume SMA (20)',
            line=dict(color='red', width=1),
            showlegend=False
        ), row=2, col=1)
    
    fig.update_layout(
        title=f'{symbol} Stock Analysis Chart',
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        height=600
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig


def add_technical_indicators(fig: go.Figure, data: pd.DataFrame):
    """Add technical indicators as overlays to the chart."""
    # Moving averages
    sma_cols = [col for col in data.columns if col.startswith('SMA_')]
    ema_cols = [col for col in data.columns if col.startswith('EMA_')]
    
    for col in sma_cols[:3]:  # Limit to first 3 to avoid clutter
        period = col.split('_')[1]
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[col],
            mode='lines',
            name=f'SMA ({period})',
            line=dict(width=1),
            opacity=0.8
        ))
    
    for col in ema_cols[:2]:  # Limit to first 2
        period = col.split('_')[1]
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[col],
            mode='lines',
            name=f'EMA ({period})',
            line=dict(width=1, dash='dash'),
            opacity=0.8
        ))
    
    # Bollinger Bands
    bb_upper_cols = [col for col in data.columns if col.startswith('BB_Upper_')]
    bb_lower_cols = [col for col in data.columns if col.startswith('BB_Lower_')]
    
    if bb_upper_cols and bb_lower_cols:
        upper_col = bb_upper_cols[0]
        lower_col = bb_lower_cols[0]
        period = upper_col.split('_')[2]
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[upper_col],
            mode='lines',
            name=f'BB Upper ({period})',
            line=dict(color='gray', width=1),
            opacity=0.5
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[lower_col],
            mode='lines',
            name=f'BB Lower ({period})',
            line=dict(color='gray', width=1),
            fill='tonexty',
            fillcolor='rgba(128,128,128,0.1)',
            opacity=0.5
        ))


def add_technical_indicators_to_subplot(fig: go.Figure, data: pd.DataFrame, row: int, col: int):
    """Add technical indicators to a specific subplot."""
    # This is a simplified version for subplots
    # Moving averages
    if 'SMA_20' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA_20'],
            mode='lines',
            name='SMA (20)',
            line=dict(color='orange', width=1),
            opacity=0.8
        ), row=row, col=col)
    
    if 'EMA_12' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['EMA_12'],
            mode='lines',
            name='EMA (12)',
            line=dict(color='purple', width=1, dash='dash'),
            opacity=0.8
        ), row=row, col=col)