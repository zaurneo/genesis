"""Model comparison and analysis visualizations."""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal

try:
    import plotly.graph_objects as go
    import plotly.subplots as sp
    from plotly.offline import plot
    _plotly_available = True
except ImportError:
    _plotly_available = False

from ..config import OUTPUT_DIR, logger


def visualize_model_comparison_backtesting_impl(
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
    logger.info(f" visualize_model_comparison_backtesting: Creating {chart_type} chart for {symbol.upper()}...")
    
    if not _plotly_available:
        error_msg = "Plotly not available. Please install: pip install plotly"
        logger.error(f"visualize_model_comparison_backtesting: {error_msg}")
        return error_msg
    
    try:
        symbol = symbol.upper()
        
        # Load multi-model results
        if results_file:
            if not results_file.endswith('.json'):
                results_file += '.json'
            filepath = os.path.join(OUTPUT_DIR, results_file)
        else:
            # Find most recent multi-model results file
            results_files = [f for f in os.listdir(OUTPUT_DIR) 
                           if f.startswith(f"multi_model_backtest_{symbol}_") and f.endswith('.json')]
            if not results_files:
                result = f"visualize_model_comparison_backtesting: No multi-model results found for {symbol}"
                logger.error(f"visualize_model_comparison_backtesting: {result}")
                return result
            
            latest_file = max(results_files, key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)))
            filepath = os.path.join(OUTPUT_DIR, latest_file)
        
        if not os.path.exists(filepath):
            result = f"visualize_model_comparison_backtesting: Results file not found: {filepath}"
            logger.error(f"visualize_model_comparison_backtesting: {result}")
            return result
        
        # Load results data
        with open(filepath, 'r') as f:
            results_data = json.load(f)
        
        comparison_matrix = pd.DataFrame(results_data['comparison_matrix'])
        if comparison_matrix.empty:
            result = f"visualize_model_comparison_backtesting: No comparison data available"
            logger.error(f"visualize_model_comparison_backtesting: {result}")
            return result
        
        # Create chart based on type
        if chart_type == "performance_comparison":
            fig = create_performance_comparison_chart(comparison_matrix, symbol)
        elif chart_type == "parameter_sensitivity":
            fig = create_parameter_sensitivity_chart(comparison_matrix, symbol)
        elif chart_type == "risk_return_scatter":
            fig = create_risk_return_scatter_chart(comparison_matrix, symbol)
        elif chart_type == "model_type_analysis":
            fig = create_model_type_analysis_chart(comparison_matrix, symbol)
        else:
            result = f"visualize_model_comparison_backtesting: Invalid chart type '{chart_type}'"
            logger.error(f"visualize_model_comparison_backtesting: {result}")
            return result
        
        # Save chart if requested
        chart_file = None
        if save_chart:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_file = f"model_comparison_{symbol}_{chart_type}_{timestamp}.html"
            chart_filepath = os.path.join(OUTPUT_DIR, chart_file)
            
            plot(fig, filename=chart_filepath, auto_open=False)
        
        # Generate summary
        summary = f"""visualize_model_comparison_backtesting: Successfully created {chart_type} chart for {symbol}:

 COMPARISON CHART DETAILS:
- Symbol: {symbol}
- Chart Type: {chart_type.replace('_', ' ').title()}
- Models Compared: {len(comparison_matrix)}
- Data Source: {os.path.basename(filepath)}

 CHART FEATURES:
- Interactive: Hover tooltips with detailed metrics
- Professional Styling: Corporate color scheme and layout
- Comparative Analysis: Side-by-side model performance
- Export Options: PNG, HTML, PDF available

 CHART SAVED: {chart_file if chart_file else 'Not saved'}
- Location: {os.path.join(OUTPUT_DIR, chart_file) if chart_file else 'N/A'}
- Format: Interactive HTML with embedded JavaScript

 INSIGHTS AVAILABLE:
- Best performing models by different metrics
- Parameter sensitivity analysis
- Risk-return trade-offs
- Model type effectiveness comparison
"""
        
        logger.info(f"visualize_model_comparison_backtesting: Successfully created {chart_type} chart for {symbol}")
        return summary
        
    except Exception as e:
        error_msg = f"visualize_model_comparison_backtesting: Error creating chart: {str(e)}"
        logger.error(f"visualize_model_comparison_backtesting: {error_msg}")
        return error_msg


def create_performance_comparison_chart(comparison_df: pd.DataFrame, symbol: str) -> go.Figure:
    """Create performance comparison bar charts."""
    
    # Create subplots for different metrics
    fig = sp.make_subplots(
        rows=2, cols=2,
        subplot_titles=('Total Return (%)', 'Sharpe Ratio', 'Maximum Drawdown (%)', 'Win Rate (%)'),
        vertical_spacing=0.12,
        horizontal_spacing=0.12
    )
    
    # Prepare model names (shortened for display)
    model_names = [name.replace('train_', '').replace('_price_predictor', '').replace('_model', '')[:15] 
                   for name in comparison_df['model_file']]
    
    # Total Return
    fig.add_trace(go.Bar(
        x=model_names,
        y=comparison_df['total_return'],
        name='Total Return',
        marker_color='lightblue',
        showlegend=False
    ), row=1, col=1)
    
    # Sharpe Ratio
    fig.add_trace(go.Bar(
        x=model_names,
        y=comparison_df['sharpe_ratio'],
        name='Sharpe Ratio',
        marker_color='lightgreen',
        showlegend=False
    ), row=1, col=2)
    
    # Maximum Drawdown
    fig.add_trace(go.Bar(
        x=model_names,
        y=comparison_df['max_drawdown'],
        name='Max Drawdown',
        marker_color='lightcoral',
        showlegend=False
    ), row=2, col=1)
    
    # Win Rate
    fig.add_trace(go.Bar(
        x=model_names,
        y=comparison_df['win_rate'],
        name='Win Rate',
        marker_color='lightyellow',
        showlegend=False
    ), row=2, col=2)
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} Model Performance Comparison',
        template='plotly_white',
        height=600,
        showlegend=False
    )
    
    # Rotate x-axis labels for better readability
    fig.update_xaxes(tickangle=45)
    
    return fig


def create_parameter_sensitivity_chart(comparison_df: pd.DataFrame, symbol: str) -> go.Figure:
    """Create parameter sensitivity analysis chart."""
    
    fig = go.Figure()
    
    # Create scatter plot with features_count vs total_return
    # Size based on Sharpe ratio, color based on model type
    fig.add_trace(go.Scatter(
        x=comparison_df['features_count'],
        y=comparison_df['total_return'],
        mode='markers',
        marker=dict(
            size=comparison_df['sharpe_ratio'] * 10 + 10,  # Scale for visibility
            color=comparison_df['max_drawdown'],
            colorscale='RdYlGn_r',  # Red for high drawdown, green for low
            showscale=True,
            colorbar=dict(title="Max Drawdown (%)")
        ),
        text=[f"Model: {name}<br>Return: {ret:.2f}%<br>Sharpe: {sharpe:.3f}<br>Drawdown: {dd:.2f}%" 
              for name, ret, sharpe, dd in zip(
                  comparison_df['model_file'], 
                  comparison_df['total_return'],
                  comparison_df['sharpe_ratio'],
                  comparison_df['max_drawdown']
              )],
        hovertemplate='%{text}<extra></extra>',
        name='Models'
    ))
    
    fig.update_layout(
        title=f'{symbol} Parameter Sensitivity Analysis',
        xaxis_title='Number of Features',
        yaxis_title='Total Return (%)',
        template='plotly_white',
        hovermode='closest'
    )
    
    return fig


def create_risk_return_scatter_chart(comparison_df: pd.DataFrame, symbol: str) -> go.Figure:
    """Create risk-return scatter plot."""
    
    fig = go.Figure()
    
    # Use max_drawdown as risk measure (x-axis) and total_return as return (y-axis)
    fig.add_trace(go.Scatter(
        x=comparison_df['max_drawdown'],
        y=comparison_df['total_return'],
        mode='markers+text',
        marker=dict(
            size=12,
            color=comparison_df['sharpe_ratio'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Sharpe Ratio")
        ),
        text=[name.split('_')[1] for name in comparison_df['model_file']],  # Model type
        textposition="top center",
        hovertemplate='Risk (Max DD): %{x:.2f}%<br>Return: %{y:.2f}%<br>Sharpe: %{marker.color:.3f}<extra></extra>',
        name='Models'
    ))
    
    # Add diagonal lines for reference
    max_risk = comparison_df['max_drawdown'].max()
    max_return = comparison_df['total_return'].max()
    min_return = comparison_df['total_return'].min()
    
    # Add efficient frontier reference line (simplified)
    efficient_x = np.linspace(0, max_risk, 100)
    efficient_y = np.sqrt(efficient_x) * (max_return / np.sqrt(max_risk)) if max_risk > 0 else [0] * 100
    
    fig.add_trace(go.Scatter(
        x=efficient_x,
        y=efficient_y,
        mode='lines',
        line=dict(dash='dash', color='gray', width=1),
        name='Reference Line',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=f'{symbol} Risk-Return Analysis',
        xaxis_title='Risk (Maximum Drawdown %)',
        yaxis_title='Return (%)',
        template='plotly_white',
        hovermode='closest'
    )
    
    return fig


def create_model_type_analysis_chart(comparison_df: pd.DataFrame, symbol: str) -> go.Figure:
    """Create model type analysis chart."""
    
    # Group by model type
    model_type_stats = comparison_df.groupby('model_type').agg({
        'total_return': ['mean', 'std', 'count'],
        'sharpe_ratio': ['mean', 'std'],
        'max_drawdown': ['mean', 'std'],
        'win_rate': ['mean', 'std']
    }).round(3)
    
    # Flatten column names
    model_type_stats.columns = ['_'.join(col).strip() for col in model_type_stats.columns]
    model_type_stats = model_type_stats.reset_index()
    
    # Create subplots
    fig = sp.make_subplots(
        rows=2, cols=2,
        subplot_titles=('Average Total Return', 'Average Sharpe Ratio', 'Average Max Drawdown', 'Average Win Rate'),
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    # Total Return
    fig.add_trace(go.Bar(
        x=model_type_stats['model_type'],
        y=model_type_stats['total_return_mean'],
        error_y=dict(type='data', array=model_type_stats['total_return_std']),
        name='Return',
        marker_color='lightblue',
        showlegend=False
    ), row=1, col=1)
    
    # Sharpe Ratio
    fig.add_trace(go.Bar(
        x=model_type_stats['model_type'],
        y=model_type_stats['sharpe_ratio_mean'],
        error_y=dict(type='data', array=model_type_stats['sharpe_ratio_std']),
        name='Sharpe',
        marker_color='lightgreen',
        showlegend=False
    ), row=1, col=2)
    
    # Max Drawdown
    fig.add_trace(go.Bar(
        x=model_type_stats['model_type'],
        y=model_type_stats['max_drawdown_mean'],
        error_y=dict(type='data', array=model_type_stats['max_drawdown_std']),
        name='Drawdown',
        marker_color='lightcoral',
        showlegend=False
    ), row=2, col=1)
    
    # Win Rate
    fig.add_trace(go.Bar(
        x=model_type_stats['model_type'],
        y=model_type_stats['win_rate_mean'],
        error_y=dict(type='data', array=model_type_stats['win_rate_std']),
        name='Win Rate',
        marker_color='lightyellow',
        showlegend=False
    ), row=2, col=2)
    
    fig.update_layout(
        title=f'{symbol} Model Type Performance Analysis',
        template='plotly_white',
        height=600,
        showlegend=False
    )
    
    return fig