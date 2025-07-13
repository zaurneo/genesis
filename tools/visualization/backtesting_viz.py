"""Backtesting results visualization functionality."""

import os
import json
import pandas as pd
import numpy as np
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
def visualize_backtesting_results_impl(
    symbol: str,
    chart_type: Literal["portfolio_performance", "trading_signals", "model_predictions", "combined"] = "combined",
    results_file: Optional[str] = None,
    save_chart: bool = True
) -> str:
    """
    Create visualizations for backtesting results and trading performance.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'GOOGL')
        chart_type: Type of chart to create
        results_file: JSON file with backtesting results (optional)
        save_chart: Whether to save chart to HTML file
        
    Returns:
        String with chart creation results and file location
    """
    log_info(f"visualize_backtesting_results: Creating {chart_type} chart for {symbol.upper()}...")
    
    if not _plotly_available:
        error_msg = "Plotly not available. Please install: pip install plotly"
        log_error(f"visualize_backtesting_results: {error_msg}")
        return error_msg
    
    try:
        symbol = symbol.upper()
        
        # Load backtesting results
        if results_file:
            if not results_file.endswith('.json'):
                results_file += '.json'
            filepath = os.path.join(OUTPUT_DIR, results_file)
        else:
            # Find most recent backtesting results file
            results_files = [f for f in os.listdir(OUTPUT_DIR) 
                           if f.startswith(f"backtest_{symbol}_") and f.endswith('.json')]
            if not results_files:
                result = f"visualize_backtesting_results: No backtesting results found for {symbol}"
                log_error(f"visualize_backtesting_results: {result}")
                return result
            
            latest_file = max(results_files, key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)))
            filepath = os.path.join(OUTPUT_DIR, latest_file)
        
        if not os.path.exists(filepath):
            result = f"visualize_backtesting_results: Results file not found: {filepath}"
            log_error(f"visualize_backtesting_results: {result}")
            return result
        
        # Load results data
        with open(filepath, 'r') as f:
            results_data = json.load(f)
        
        signals_df = pd.DataFrame(results_data['signals'])
        signals_df['date'] = pd.to_datetime(signals_df['date'])
        signals_df.set_index('date', inplace=True)
        
        if signals_df.empty:
            result = f"visualize_backtesting_results: No signal data available"
            log_error(f"visualize_backtesting_results: {result}")
            return result
        
        # Create chart based on type
        if chart_type == "portfolio_performance":
            fig = create_portfolio_performance_chart(signals_df, results_data, symbol)
        elif chart_type == "trading_signals":
            fig = create_trading_signals_chart(signals_df, results_data, symbol)
        elif chart_type == "model_predictions":
            fig = create_model_predictions_chart(signals_df, results_data, symbol)
        elif chart_type == "combined":
            fig = create_combined_backtesting_chart(signals_df, results_data, symbol)
        else:
            result = f"visualize_backtesting_results: Invalid chart type '{chart_type}'"
            log_error(f"visualize_backtesting_results: {result}")
            return result
        
        # Save chart if requested
        chart_file = None
        if save_chart:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_file = f"backtesting_{symbol}_{chart_type}_{timestamp}.html"
            chart_filepath = os.path.join(OUTPUT_DIR, chart_file)
            
            # Ensure Plotly is properly initialized for offline plotting
            pyo.init_notebook_mode(connected=False)
            plot(fig, filename=chart_filepath, auto_open=False, include_plotlyjs='cdn', config={'displayModeBar': True})
        
        # Generate summary
        summary = f"""visualize_backtesting_results: Successfully created {chart_type} chart for {symbol}:

 BACKTESTING CHART DETAILS:
- Symbol: {symbol}
- Chart Type: {chart_type.replace('_', ' ').title()}
- Strategy: {results_data.get('strategy_type', 'Unknown')}
- Data Source: {os.path.basename(filepath)}
- Trading Period: {signals_df.index[0].strftime('%Y-%m-%d')} to {signals_df.index[-1].strftime('%Y-%m-%d')}

 PERFORMANCE SUMMARY:
- Total Return: {results_data.get('total_return', 0):+.2f}%
- Buy & Hold Return: {results_data.get('buy_hold_return', 0):+.2f}%
- Sharpe Ratio: {results_data.get('sharpe_ratio', 0):.3f}
- Max Drawdown: {results_data.get('max_drawdown', 0):.2f}%
- Total Trades: {results_data.get('total_trades', 0)}

 CHART SAVED: {chart_file if chart_file else 'Not saved'}
- Location: {os.path.join(OUTPUT_DIR, chart_file) if chart_file else 'N/A'}
- Format: Interactive HTML with embedded JavaScript

 CHART FEATURES:
- Interactive: Zoom, pan, hover tooltips
- Trading Signals: Buy/sell markers on price chart
- Performance Tracking: Portfolio value vs benchmark
- Prediction Accuracy: Model vs actual price comparison
"""
        
        log_success(f"visualize_backtesting_results: Successfully created {chart_type} chart for {symbol}")
        return summary
        
    except Exception as e:
        error_msg = f"visualize_backtesting_results: Error creating chart: {str(e)}"
        log_error(f"visualize_backtesting_results: {error_msg}")
        return error_msg


def create_portfolio_performance_chart(signals_df: pd.DataFrame, results_data: dict, symbol: str) -> go.Figure:
    """Create portfolio performance comparison chart."""
    
    fig = go.Figure()
    
    # Calculate buy and hold performance
    initial_price = signals_df['price'].iloc[0]
    buy_hold_values = signals_df['price'] / initial_price * results_data['initial_capital']
    
    # Add portfolio performance
    fig.add_trace(go.Scatter(
        x=signals_df.index,
        y=signals_df['portfolio_value'],
        mode='lines',
        name=f'Strategy ({results_data.get("strategy_type", "Unknown")})',
        line=dict(color='blue', width=2)
    ))
    
    # Add buy and hold benchmark
    fig.add_trace(go.Scatter(
        x=signals_df.index,
        y=buy_hold_values,
        mode='lines',
        name='Buy & Hold',
        line=dict(color='gray', width=2, dash='dash')
    ))
    
    # Add initial capital reference line
    fig.add_hline(
        y=results_data['initial_capital'],
        line_dash="dot",
        line_color="black",
        annotation_text="Initial Capital"
    )
    
    fig.update_layout(
        title=f'{symbol} Portfolio Performance Comparison',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


def create_trading_signals_chart(signals_df: pd.DataFrame, results_data: dict, symbol: str) -> go.Figure:
    """Create trading signals overlaid on price chart."""
    
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=signals_df.index,
        y=signals_df['price'],
        mode='lines',
        name=f'{symbol} Price',
        line=dict(color='black', width=1)
    ))
    
    # Add buy signals
    buy_signals = signals_df[signals_df['signal'] == 1]
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals['price'],
            mode='markers',
            name='Buy Signal',
            marker=dict(color='green', size=8, symbol='triangle-up')
        ))
    
    # Add sell signals
    sell_signals = signals_df[signals_df['signal'] == -1]
    if not sell_signals.empty:
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals['price'],
            mode='markers',
            name='Sell Signal',
            marker=dict(color='red', size=8, symbol='triangle-down')
        ))
    
    fig.update_layout(
        title=f'{symbol} Trading Signals ({results_data.get("strategy_type", "Unknown")} Strategy)',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


def create_model_predictions_chart(signals_df: pd.DataFrame, results_data: dict, symbol: str) -> go.Figure:
    """Create model predictions vs actual prices chart."""
    
    fig = go.Figure()
    
    # Add actual prices
    fig.add_trace(go.Scatter(
        x=signals_df.index,
        y=signals_df['price'],
        mode='lines',
        name='Actual Price',
        line=dict(color='blue', width=2)
    ))
    
    # Add predicted prices
    fig.add_trace(go.Scatter(
        x=signals_df.index,
        y=signals_df['predicted_price'],
        mode='lines',
        name='Predicted Price',
        line=dict(color='red', width=2, dash='dash'),
        opacity=0.8
    ))
    
    # Calculate prediction accuracy metrics
    actual_prices = signals_df['price'].values
    predicted_prices = signals_df['predicted_price'].values
    mse = np.mean((actual_prices - predicted_prices) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual_prices - predicted_prices))
    
    # Add annotation with accuracy metrics
    fig.add_annotation(
        x=0.02, y=0.98,
        xref="paper", yref="paper",
        text=f"RMSE: ${rmse:.2f}<br>MAE: ${mae:.2f}",
        showarrow=False,
        font=dict(size=12),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1
    )
    
    fig.update_layout(
        title=f'{symbol} Model Predictions vs Actual Prices',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


def create_combined_backtesting_chart(signals_df: pd.DataFrame, results_data: dict, symbol: str) -> go.Figure:
    """Create combined chart with all backtesting visualizations."""
    
    # Create subplots
    fig = sp.make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            f'{symbol} Portfolio Performance',
            'Trading Signals & Price',
            'Model Predictions vs Actual'
        ),
        row_heights=[0.4, 0.3, 0.3]
    )
    
    # Subplot 1: Portfolio Performance
    initial_price = signals_df['price'].iloc[0]
    buy_hold_values = signals_df['price'] / initial_price * results_data['initial_capital']
    
    fig.add_trace(go.Scatter(
        x=signals_df.index,
        y=signals_df['portfolio_value'],
        mode='lines',
        name=f'Strategy ({results_data.get("strategy_type", "Unknown")})',
        line=dict(color='blue', width=2)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=signals_df.index,
        y=buy_hold_values,
        mode='lines',
        name='Buy & Hold',
        line=dict(color='gray', width=2, dash='dash')
    ), row=1, col=1)
    
    # Subplot 2: Trading Signals
    fig.add_trace(go.Scatter(
        x=signals_df.index,
        y=signals_df['price'],
        mode='lines',
        name=f'{symbol} Price',
        line=dict(color='black', width=1),
        showlegend=False
    ), row=2, col=1)
    
    # Buy signals
    buy_signals = signals_df[signals_df['signal'] == 1]
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals['price'],
            mode='markers',
            name='Buy Signal',
            marker=dict(color='green', size=6, symbol='triangle-up'),
            showlegend=False
        ), row=2, col=1)
    
    # Sell signals
    sell_signals = signals_df[signals_df['signal'] == -1]
    if not sell_signals.empty:
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals['price'],
            mode='markers',
            name='Sell Signal',
            marker=dict(color='red', size=6, symbol='triangle-down'),
            showlegend=False
        ), row=2, col=1)
    
    # Subplot 3: Model Predictions
    fig.add_trace(go.Scatter(
        x=signals_df.index,
        y=signals_df['price'],
        mode='lines',
        name='Actual Price',
        line=dict(color='blue', width=1),
        showlegend=False
    ), row=3, col=1)
    
    fig.add_trace(go.Scatter(
        x=signals_df.index,
        y=signals_df['predicted_price'],
        mode='lines',
        name='Predicted Price',
        line=dict(color='red', width=1, dash='dash'),
        opacity=0.8,
        showlegend=False
    ), row=3, col=1)
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} Comprehensive Backtesting Analysis',
        template='plotly_white',
        height=800,
        hovermode='x unified'
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Price ($)", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=3, col=1)
    
    return fig