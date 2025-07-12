"""
Enhanced visualization wrapper that sends Plotly data to the UI via WebSocket.
This module wraps the existing visualization tools to capture and send chart data.
"""

import json
import os
import re
from typing import Optional, Dict, Any, Literal
from pathlib import Path
import plotly.graph_objects as go
import plotly.io as pio

from tools.visualization.charts import visualize_stock_data_impl as original_visualize_stock_data
from tools.visualization.backtesting_viz import visualize_backtesting_results_impl as original_visualize_backtesting
from tools.visualization.comparisons import visualize_model_comparison_backtesting_impl as original_visualize_comparison

# Global variable to store the current message handler
_current_message_handler = None

def set_message_handler(handler):
    """Set the message handler for sending data to UI."""
    global _current_message_handler
    _current_message_handler = handler

def extract_plotly_data_from_html(html_file_path: str) -> Optional[Dict[str, Any]]:
    """Extract Plotly data and layout from saved HTML file."""
    try:
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Extract Plotly.newPlot call data
        data_match = re.search(
            r'Plotly\.newPlot\([^,]+,\s*(\[.*?\]),\s*(\{.*?\})',
            html_content,
            re.DOTALL
        )
        
        if data_match:
            data_str = data_match.group(1)
            layout_str = data_match.group(2)
            
            # Parse JSON, handling potential JavaScript objects
            data = json.loads(data_str)
            layout = json.loads(layout_str)
            
            return {
                "data": data,
                "layout": layout
            }
    except Exception as e:
        print(f"Error extracting Plotly data: {e}")
    
    return None

def visualize_stock_data_with_ui(
    symbol: str,
    chart_type: Literal["line", "candlestick", "volume", "combined"] = "combined",
    source_file: Optional[str] = None,
    save_chart: bool = True,
    show_indicators: bool = True
) -> str:
    """Enhanced version that sends Plotly data to UI."""
    
    # Call original function
    result = original_visualize_stock_data(
        symbol=symbol,
        chart_type=chart_type,
        source_file=source_file,
        save_chart=save_chart,
        show_indicators=show_indicators
    )
    
    # Extract chart file path from result
    if "Location:" in result and ".html" in result:
        lines = result.split('\n')
        for line in lines:
            if "Location:" in line and ".html" in line:
                file_path = line.split("Location:")[1].strip()
                
                # Extract Plotly data
                plotly_data = extract_plotly_data_from_html(file_path)
                
                if plotly_data and _current_message_handler:
                    # Send to UI
                    enhanced_message = {
                        "content": result,
                        "additional_kwargs": {
                            "plotly_data": plotly_data
                        }
                    }
                    _current_message_handler(enhanced_message)
                    return result
    
    return result

def visualize_backtesting_results_with_ui(
    symbol: str,
    chart_type: Literal["portfolio_performance", "trading_signals", "model_predictions", "combined"] = "combined",
    results_file: Optional[str] = None,
    save_chart: bool = True
) -> str:
    """Enhanced version that sends Plotly data to UI."""
    
    # Call original function
    result = original_visualize_backtesting(
        symbol=symbol,
        chart_type=chart_type,
        results_file=results_file,
        save_chart=save_chart
    )
    
    # Extract chart file path from result
    if "Location:" in result and ".html" in result:
        lines = result.split('\n')
        for line in lines:
            if "Location:" in line and ".html" in line:
                file_path = line.split("Location:")[1].strip()
                
                # Extract Plotly data
                plotly_data = extract_plotly_data_from_html(file_path)
                
                if plotly_data and _current_message_handler:
                    # Send to UI
                    enhanced_message = {
                        "content": result,
                        "additional_kwargs": {
                            "plotly_data": plotly_data
                        }
                    }
                    _current_message_handler(enhanced_message)
                    return result
    
    return result

def visualize_model_comparison_with_ui(
    symbol: str,
    chart_type: Literal["performance_comparison", "parameter_sensitivity", "risk_return_scatter", "model_type_analysis"] = "performance_comparison",
    results_file: Optional[str] = None,
    save_chart: bool = True
) -> str:
    """Enhanced version that sends Plotly data to UI."""
    
    # Call original function
    result = original_visualize_comparison(
        symbol=symbol,
        chart_type=chart_type,
        results_file=results_file,
        save_chart=save_chart
    )
    
    # Extract chart file path from result
    if "Location:" in result and ".html" in result:
        lines = result.split('\n')
        for line in lines:
            if "Location:" in line and ".html" in line:
                file_path = line.split("Location:")[1].strip()
                
                # Extract Plotly data
                plotly_data = extract_plotly_data_from_html(file_path)
                
                if plotly_data and _current_message_handler:
                    # Send to UI
                    enhanced_message = {
                        "content": result,
                        "additional_kwargs": {
                            "plotly_data": plotly_data
                        }
                    }
                    _current_message_handler(enhanced_message)
                    return result
    
    return result

# Create wrapper functions that can be used by the agents
def create_ui_enhanced_tools():
    """Create UI-enhanced versions of visualization tools."""
    return {
        'visualize_stock_data': visualize_stock_data_with_ui,
        'visualize_backtesting_results': visualize_backtesting_results_with_ui,
        'visualize_model_comparison_backtesting': visualize_model_comparison_with_ui
    }