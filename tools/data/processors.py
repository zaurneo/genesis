"""Data processing and preparation utilities for machine learning models."""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Optional, Tuple, Dict, Any
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


@safe_run
def read_csv_data_impl(
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
    log_info(f"read_csv_data: Starting to read CSV file '{filename}'...")
    
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
            log_error(f"read_csv_data: {result}")
            return result
        
        # Read the CSV file
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        if data.empty:
            result = f"The file '{filename}' is empty."
            log_warning(f"read_csv_data: {result}")
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
 CSV DATA ANALYSIS for {filename}:

 DATASET OVERVIEW:
- Total Records: {len(data)}
- Date Range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}
- Columns: {', '.join(data.columns)}
- File Size: {os.path.getsize(file_path):,} bytes

"""
        
        if stats:
            summary += f""" PRICE STATISTICS:
- Current Price: ${stats.get('current_price', 0):.2f}
- Opening Price: ${stats.get('opening_price', 0):.2f}
- Price Change: ${stats.get('price_change', 0):.2f} ({stats.get('price_change_pct', 0):+.2f}%)
- Period High: ${stats.get('period_high', 0):.2f}
- Period Low: ${stats.get('period_low', 0):.2f}
- Daily Volatility: {stats.get('volatility', 0):.2f}%

"""
        
        if 'avg_volume' in stats:
            summary += f""" VOLUME STATISTICS:
- Average Volume: {stats['avg_volume']:,.0f}
- Total Volume: {stats['total_volume']:,.0f}
- Max Volume: {stats['max_volume']:,.0f}

"""
        
        summary += f""" DATA SAMPLE (First {min(max_rows, len(data))} rows):
{sample_rows.to_string()}

 DATA INSIGHTS:
- Data quality: {'Excellent' if data.isnull().sum().sum() == 0 else 'Good with some missing values'}
- Suitable for: {'Technical analysis, ML modeling, backtesting' if len(data) > 100 else 'Basic analysis only (limited data)'}
- Recommendation: {'Ready for advanced analysis' if 'Close' in data.columns and len(data) > 252 else 'Consider fetching more data for better analysis'}
"""
        
        log_success(f"read_csv_data: Successfully analyzed {len(data)} records from {filename}")
        return summary
        
    except Exception as e:
        error_msg = f"read_csv_data: Error reading file '{filename}': {str(e)}"
        log_error(f"read_csv_data: {error_msg}")
        return error_msg


def prepare_model_data(
    symbol: str,
    source_file: Optional[str] = None,
    target_days: int = 1,
    test_size: float = 0.2
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
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
    """
    log_info(f"prepare_model_data: Preparing data for {symbol} with {target_days}-day target...")
    
    symbol = symbol.upper()
    
    # Load enhanced data with technical indicators
    if source_file:
        if not source_file.endswith('.csv'):
            source_file += '.csv'
        filepath = os.path.join(OUTPUT_DIR, source_file)
        if not os.path.exists(filepath):
            error_msg = f"Source file '{source_file}' not found."
            log_error(f"prepare_model_data: {error_msg}")
            return None, error_msg
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
            error_msg = f"No enhanced data files found for {symbol}. Please run technical indicators first."
            log_error(f"prepare_model_data: {error_msg}")
            return None, error_msg
    
    if data.empty or len(data) < 50:
        error_msg = f"Insufficient data for {symbol}. Need at least 50 records, found {len(data)}."
        log_error(f"prepare_model_data: {error_msg}")
        return None, error_msg
    
    # Prepare features and target
    data['Target'] = data['Close'].shift(-target_days)
    
    # Select feature columns
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'Target']
    feature_cols = [col for col in data.columns if col not in exclude_cols and not data[col].isnull().all()]
    
    if len(feature_cols) < 3:
        error_msg = f"Insufficient technical indicators. Found only {len(feature_cols)} features."
        log_error(f"prepare_model_data: {error_msg}")
        return None, error_msg
    
    # Remove rows with NaN values
    model_data = data[feature_cols + ['Target']].dropna()
    
    if len(model_data) < 30:
        error_msg = f"Insufficient clean data. Only {len(model_data)} records available."
        log_error(f"prepare_model_data: {error_msg}")
        return None, error_msg
    
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
    
    result_data = {
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
    }
    
    log_success(f"prepare_model_data: Successfully prepared {len(X_train)} training and {len(X_test)} test samples")
    return result_data, None


def get_train_test_predictions(model, model_data: Dict[str, Any]) -> Dict[str, Any]:
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
    """
    log_info("get_train_test_predictions: Generating predictions for train and test sets...")
    
    train_pred = model.predict(model_data['X_train_scaled'])
    test_pred = model.predict(model_data['X_test_scaled'])
    
    predictions = {
        'train_predictions': train_pred,
        'test_predictions': test_pred,
        'train_actuals': model_data['y_train'].values,
        'test_actuals': model_data['y_test'].values,
        'train_dates': model_data['y_train'].index,
        'test_dates': model_data['y_test'].index,
        'train_features': model_data['X_train'],
        'test_features': model_data['X_test']
    }
    
    log_success(f"get_train_test_predictions: Generated {len(train_pred)} training and {len(test_pred)} test predictions")
    return predictions


def assess_model_metrics(predictions_data: Dict[str, Any], model, model_data: Dict[str, Any]) -> Dict[str, Any]:
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
    """
    log_info("assess_model_metrics: Calculating comprehensive performance metrics...")
    
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
            log_warning(f"Cross-validation fold failed: {str(e)}")
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
    
    metrics = {
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
    
    log_success(f"assess_model_metrics: Calculated metrics with R² {test_r2:.3f} and {len(cv_scores)} CV folds")
    return metrics