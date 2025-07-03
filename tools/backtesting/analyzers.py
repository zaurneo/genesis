"""Multi-model backtesting analysis and comparison utilities."""

import os
import json
import pickle
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
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

from ..config import OUTPUT_DIR, logger
from .engine import backtest_model_strategy_impl


@safe_run
def backtest_multiple_models_impl(
    symbol: str,
    strategy_type: Literal["threshold", "directional", "percentile"] = "directional",
    initial_capital: float = 10000.0,
    transaction_cost: float = 0.001,
    save_results: bool = True
) -> str:
    """
    Backtest multiple trained models for a symbol and compare their performance.
    
    This function discovers all available trained models for a given symbol,
    runs backtesting on each model using the specified strategy, and provides
    comprehensive comparison analysis and rankings.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')
        strategy_type: Trading strategy to use for all models
        initial_capital: Starting capital for backtesting
        transaction_cost: Transaction cost as percentage
        save_results: Whether to save detailed comparison results
        
    Returns:
        String with comprehensive multi-model comparison results
    """
    log_info(f" backtest_multiple_models: Starting multi-model backtesting for {symbol.upper()}...")
    
    try:
        symbol = symbol.upper()
        
        # Discover available models
        model_files = discover_models(symbol)
        if not model_files:
            result = f"backtest_multiple_models: No trained models found for {symbol}"
            log_warning(f"backtest_multiple_models: {result}")
            return result
        
        log_info(f" backtest_multiple_models: Found {len(model_files)} models for {symbol}")
        
        all_results = []
        successful_backtests = 0
        
        for i, model_file in enumerate(model_files, 1):
            log_info(f"backtest_multiple_models: Backtesting model {i}/{len(model_files)}: {model_file}")
            
            try:
                # Load model metadata
                model_metadata = load_model_metadata(model_file)
                
                # Run backtesting
                backtest_result_raw = backtest_model_strategy_impl(
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
                    log_success(f"backtest_multiple_models: Model {i} completed successfully")
                else:
                    log_error(f"backtest_multiple_models: Error backtesting {model_file}: {backtest_result_raw}")
                    
            except Exception as e:
                log_error(f"backtest_multiple_models: Error backtesting {model_file}: {str(e)}")
                continue
        
        if successful_backtests == 0:
            result = f"backtest_multiple_models: No successful backtests completed for {symbol}"
            log_error(f"backtest_multiple_models: {result}")
            return result
        
        # Generate comparison analysis
        comparison_matrix = generate_model_comparison_matrix(all_results)
        rankings = calculate_model_rankings(comparison_matrix)
        
        # Save results if requested
        results_file = None
        if save_results:
            results_file = save_multi_model_results(symbol, strategy_type, all_results, comparison_matrix, rankings)
        
        # Format summary
        summary = format_multi_model_summary(symbol, strategy_type, successful_backtests, len(model_files), rankings, results_file)
        
        log_success(f"backtest_multiple_models: Completed {successful_backtests}/{len(model_files)} backtests for {symbol}")
        return summary
        
    except Exception as e:
        error_msg = f"backtest_multiple_models: Unexpected error: {str(e)}"
        log_error(f"backtest_multiple_models: {error_msg}")
        return error_msg


def discover_models(symbol: str) -> List[str]:
    """
    Discover all available trained model files for a given symbol.
    
    Args:
        symbol: Stock symbol to search for
        
    Returns:
        List of model filenames sorted by modification time (newest first)
    """
    try:
        if not os.path.exists(OUTPUT_DIR):
            return []
        
        all_files = os.listdir(OUTPUT_DIR)
        model_files = [
            f for f in all_files 
            if f.endswith('.pkl') and symbol.upper() in f.upper()
        ]
        
        # Sort by modification time (newest first)
        model_files.sort(
            key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)), 
            reverse=True
        )
        
        log_info(f"discover_models: Returning {len(model_files)} sorted model files")
        return model_files
        
    except Exception as e:
        log_error(f"discover_models: Error discovering models: {str(e)}")
        return []


def load_model_metadata(model_file: str) -> Dict[str, Any]:
    """
    Load metadata from a saved model file.
    
    Args:
        model_file: Name of the model file
        
    Returns:
        Dictionary with model metadata
    """
    try:
        model_filepath = os.path.join(OUTPUT_DIR, model_file)
        
        # Try to load metadata from JSON file first
        json_file = model_file.replace('_model.pkl', '_results.json')
        json_filepath = os.path.join(OUTPUT_DIR, json_file)
        
        if os.path.exists(json_filepath):
            with open(json_filepath, 'r') as f:
                metadata = json.load(f)
            log_debug(f" load_model_metadata: Loaded metadata from JSON for {model_file}")
            return metadata
        
        # Fallback: extract from pickle file
        with open(model_filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        metadata = {
            'model_type': model_data.get('model_type', 'unknown'),
            'symbol': model_data.get('symbol', 'unknown'),
            'target_days': model_data.get('target_days', 1),
            'feature_cols': model_data.get('feature_cols', []),
            'features_count': len(model_data.get('feature_cols', [])),
        }
        
        # Try to extract additional metadata from filename
        filename_metadata = extract_metadata_from_filename(model_file)
        metadata.update(filename_metadata)
        
        log_debug(f" load_model_metadata: Extracted metadata from pickle for {model_file}")
        return metadata
        
    except Exception as e:
        log_warning(f"Warning: Could not load metadata for {model_file}: {str(e)}")
        return extract_metadata_from_filename(model_file)


def extract_metadata_from_filename(filename: str) -> Dict[str, Any]:
    """
    Extract metadata from model filename patterns.
    
    Expected pattern: train_{model_type}_price_predictor_{symbol}_model_{timestamp}.pkl
    
    Args:
        filename: Model filename
        
    Returns:
        Dictionary with extracted metadata
    """
    try:
        # Remove .pkl extension
        base_name = filename.replace('.pkl', '')
        parts = base_name.split('_')
        
        metadata = {
            'model_type': 'unknown',
            'symbol': 'unknown',
            'target_days': 1,
            'timestamp': 'unknown',
            'features_count': 0
        }
        
        # Extract model type (usually parts[1])
        if len(parts) > 1:
            metadata['model_type'] = parts[1]
        
        # Extract symbol (usually before 'model')
        if 'model' in parts:
            model_index = parts.index('model')
            if model_index > 0:
                metadata['symbol'] = parts[model_index - 1]
        
        # Extract timestamp (usually last part)
        if len(parts) > 0:
            metadata['timestamp'] = parts[-1]
        
        return metadata
        
    except Exception as e:
        log_warning(f"Warning: Could not extract metadata from filename {filename}: {str(e)}")
        return {
            'model_type': 'unknown',
            'symbol': 'unknown', 
            'target_days': 1,
            'timestamp': 'unknown',
            'features_count': 0
        }


def enhance_with_model_metadata(backtest_result: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhance backtest results with model metadata.
    
    Args:
        backtest_result: Raw backtest result string
        metadata: Model metadata dictionary
        
    Returns:
        Enhanced result dictionary
    """
    try:
        # Parse key metrics from backtest result string
        lines = backtest_result.split('\n')
        
        enhanced = {
            'model_type': metadata.get('model_type', 'unknown'),
            'symbol': metadata.get('symbol', 'unknown'),
            'target_days': metadata.get('target_days', 1),
            'features_count': metadata.get('features_count', 0),
            'timestamp': metadata.get('timestamp', 'unknown'),
            'total_return': 0.0,
            'buy_hold_return': 0.0,
            'excess_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'total_trades': 0,
            'raw_result': backtest_result
        }
        
        # Extract metrics from result string
        for line in lines:
            if 'Total Return:' in line:
                try:
                    enhanced['total_return'] = float(line.split(':')[1].strip().replace('%', ''))
                except:
                    pass
            elif 'Buy & Hold Return:' in line:
                try:
                    enhanced['buy_hold_return'] = float(line.split(':')[1].strip().replace('%', ''))
                except:
                    pass
            elif 'Excess Return:' in line:
                try:
                    enhanced['excess_return'] = float(line.split(':')[1].strip().replace('%', ''))
                except:
                    pass
            elif 'Sharpe Ratio:' in line:
                try:
                    enhanced['sharpe_ratio'] = float(line.split(':')[1].strip())
                except:
                    pass
            elif 'Maximum Drawdown:' in line:
                try:
                    enhanced['max_drawdown'] = float(line.split(':')[1].strip().replace('%', ''))
                except:
                    pass
            elif 'Win Rate:' in line:
                try:
                    enhanced['win_rate'] = float(line.split(':')[1].strip().replace('%', ''))
                except:
                    pass
            elif 'Total Trades:' in line:
                try:
                    enhanced['total_trades'] = int(line.split(':')[1].strip())
                except:
                    pass
        
        return enhanced
        
    except Exception as e:
        log_warning(f"Warning: Could not enhance results with metadata: {str(e)}")
        return {
            'model_type': metadata.get('model_type', 'unknown'),
            'total_return': 0.0,
            'raw_result': backtest_result
        }


def generate_model_comparison_matrix(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Generate comparison matrix from multiple model results.
    
    Args:
        results: List of enhanced backtest results
        
    Returns:
        DataFrame with comparison metrics
    """
    try:
        comparison_data = []
        
        for result in results:
            comparison_data.append({
                'model_file': result.get('model_file', 'unknown'),
                'model_type': result.get('model_type', 'unknown'),
                'total_return': result.get('total_return', 0.0),
                'excess_return': result.get('excess_return', 0.0),
                'sharpe_ratio': result.get('sharpe_ratio', 0.0),
                'max_drawdown': result.get('max_drawdown', 0.0),
                'win_rate': result.get('win_rate', 0.0),
                'total_trades': result.get('total_trades', 0),
                'features_count': result.get('features_count', 0),
                'target_days': result.get('target_days', 1)
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by total return (descending)
        df = df.sort_values('total_return', ascending=False)
        
        log_info(f"generate_model_comparison_matrix: Created comparison matrix with {len(df)} models")
        return df
        
    except Exception as e:
        log_error(f"generate_model_comparison_matrix: Error creating matrix: {str(e)}")
        return pd.DataFrame()


def calculate_model_rankings(comparison_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate rankings and best performers across different metrics.
    
    Args:
        comparison_df: Comparison matrix DataFrame
        
    Returns:
        Dictionary with rankings and insights
    """
    try:
        if comparison_df.empty:
            return {}
        
        rankings = {
            'best_total_return': {
                'model': comparison_df.iloc[0]['model_file'],
                'value': comparison_df.iloc[0]['total_return']
            },
            'best_excess_return': {
                'model': comparison_df.loc[comparison_df['excess_return'].idxmax()]['model_file'],
                'value': comparison_df['excess_return'].max()
            },
            'best_sharpe_ratio': {
                'model': comparison_df.loc[comparison_df['sharpe_ratio'].idxmax()]['model_file'],
                'value': comparison_df['sharpe_ratio'].max()
            },
            'lowest_drawdown': {
                'model': comparison_df.loc[comparison_df['max_drawdown'].idxmin()]['model_file'],
                'value': comparison_df['max_drawdown'].min()
            },
            'highest_win_rate': {
                'model': comparison_df.loc[comparison_df['win_rate'].idxmax()]['model_file'],
                'value': comparison_df['win_rate'].max()
            },
            'summary_stats': {
                'avg_return': comparison_df['total_return'].mean(),
                'avg_sharpe': comparison_df['sharpe_ratio'].mean(),
                'avg_drawdown': comparison_df['max_drawdown'].mean(),
                'avg_win_rate': comparison_df['win_rate'].mean()
            }
        }
        
        log_info(f"calculate_model_rankings: Calculated rankings for {len(comparison_df)} models")
        return rankings
        
    except Exception as e:
        log_error(f"calculate_model_rankings: Error calculating rankings: {str(e)}")
        return {}


def save_multi_model_results(symbol: str, strategy_type: str, results: List[Dict], comparison_df: pd.DataFrame, rankings: Dict) -> str:
    """Save comprehensive multi-model results."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"multi_model_backtest_{symbol}_{strategy_type}_{timestamp}.json"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        data = {
            'symbol': symbol,
            'strategy_type': strategy_type,
            'timestamp': timestamp,
            'models_tested': len(results),
            'individual_results': results,
            'comparison_matrix': comparison_df.to_dict('records'),
            'rankings': rankings
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        log_info(f"save_multi_model_results: Saved results to {filename}")
        return filename
        
    except Exception as e:
        log_error(f"save_multi_model_results: Error saving results: {str(e)}")
        return None


def format_multi_model_summary(symbol: str, strategy_type: str, successful: int, total: int, rankings: Dict, results_file: Optional[str]) -> str:
    """Format comprehensive multi-model summary."""
    
    if not rankings:
        return f"backtest_multiple_models: No successful backtests to summarize for {symbol}"
    
    summary = f"""backtest_multiple_models: Multi-model backtesting completed for {symbol}:

 BACKTESTING OVERVIEW:
- Symbol: {symbol}
- Strategy: {strategy_type.title()}
- Models Tested: {successful}/{total}
- Success Rate: {successful/total*100:.1f}%

 TOP PERFORMERS:

 Best Total Return:
- Model: {rankings['best_total_return']['model']}
- Return: {rankings['best_total_return']['value']:+.2f}%

 Best Risk-Adjusted (Sharpe):
- Model: {rankings['best_sharpe_ratio']['model']}
- Sharpe Ratio: {rankings['best_sharpe_ratio']['value']:.3f}

 Lowest Drawdown:
- Model: {rankings['lowest_drawdown']['model']}
- Max Drawdown: {rankings['lowest_drawdown']['value']:.2f}%

 Highest Win Rate:
- Model: {rankings['highest_win_rate']['model']}
- Win Rate: {rankings['highest_win_rate']['value']:.1f}%

 PORTFOLIO STATISTICS:
- Average Return: {rankings['summary_stats']['avg_return']:+.2f}%
- Average Sharpe: {rankings['summary_stats']['avg_sharpe']:.3f}
- Average Drawdown: {rankings['summary_stats']['avg_drawdown']:.2f}%
- Average Win Rate: {rankings['summary_stats']['avg_win_rate']:.1f}%

 DETAILED RESULTS: {results_file if results_file else 'Not saved'}

 INSIGHTS:
- Model Diversity: {'High' if successful > 3 else 'Medium' if successful > 1 else 'Low'} ({successful} different models)
- Performance Spread: {'Wide' if rankings['best_total_return']['value'] - rankings['summary_stats']['avg_return'] > 10 else 'Narrow'} variation in returns
- Risk Management: {'Strong' if rankings['summary_stats']['avg_drawdown'] < 15 else 'Moderate'} overall drawdown control
- Recommendation: {'Deploy best performer' if rankings['best_total_return']['value'] > 5 else 'Consider parameter optimization'}
"""
    
    return summary