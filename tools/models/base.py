"""Base classes and interfaces for machine learning models."""

import os
import pickle
import json
import sys
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
from datetime import datetime

# Import logging helpers
import sys
from pathlib import Path

# Add parent directory to path to import logging_helpers
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

try:
    from tools.logs.logging_helpers import log_info, log_success, log_warning, log_error, log_progress, safe_run
    _logging_helpers_available = True
except ImportError:
    _logging_helpers_available = False
    # Fallback to regular logger if logging_helpers not available
    def log_info(msg, **kwargs): logger.info(msg)
    def log_success(msg, **kwargs): logger.info(msg)
    def log_warning(msg, **kwargs): logger.warning(msg) 
    def log_error(msg, **kwargs): logger.error(msg)
    def log_progress(msg, **kwargs): logger.info(msg)
    def safe_run(func): return func  # No-op decorator if not available

from ..config import OUTPUT_DIR, logger
from ..data import prepare_model_data, get_train_test_predictions, assess_model_metrics


class BaseModelTrainer(ABC):
    """Abstract base class for model trainers."""
    
    def __init__(self, symbol: str, model_type: str, **config):
        self.symbol = symbol.upper()
        self.model_type = model_type
        self.config = config
        self.model = None
        self.model_data = None
        self.predictions = None
        self.metrics = None
        
    @abstractmethod
    def create_model(self) -> Any:
        """Create and return a model instance."""
        pass
    
    def prepare_data(
        self, 
        source_file: Optional[str] = None,
        target_days: int = 1,
        test_size: float = 0.2
    ) -> bool:
        """Prepare data for training."""
        self.model_data, error = prepare_model_data(
            self.symbol, source_file, target_days, test_size
        )
        if error:
            log_error(f"Data preparation failed: {error}")
            return False
        return True
    
    def train(self) -> bool:
        """Train the model."""
        if not self.model_data:
            log_error("No data prepared for training")
            return False
            
        try:
            self.model = self.create_model()
            self.model.fit(self.model_data['X_train_scaled'], self.model_data['y_train'])
            log_success(f"Model {self.model_type} trained successfully")
            return True
        except Exception as e:
            log_error(f"Training failed: {str(e)}")
            return False
    
    def evaluate(self) -> bool:
        """Evaluate the trained model."""
        if not self.model or not self.model_data:
            log_error("Model or data not available for evaluation")
            return False
            
        try:
            self.predictions = get_train_test_predictions(self.model, self.model_data)
            self.metrics = assess_model_metrics(self.predictions, self.model, self.model_data)
            log_success(f"Model {self.model_type} evaluated successfully")
            return True
        except Exception as e:
            log_error(f"Evaluation failed: {str(e)}")
            return False
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance if available."""
        if not self.model or not self.model_data:
            return pd.DataFrame()
            
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                importance = abs(self.model.coef_)
            else:
                return pd.DataFrame()
                
            feature_importance = pd.DataFrame({
                'feature': self.model_data['feature_cols'],
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return feature_importance
            
        except Exception as e:
            log_error(f"Feature importance extraction failed: {str(e)}")
            return pd.DataFrame()


@safe_run
def train_model_pipeline(
    symbol: str,
    model_type: str,
    model_factory_func: Callable,
    source_file: Optional[str] = None,
    target_days: int = 1,
    test_size: float = 0.2,
    save_model: bool = True,
    save_predictions: bool = True,
    **model_params
) -> str:
    """
    Universal model training pipeline that works with any model type.
    
    This function provides a completely generic approach to training machine learning
    models for stock price prediction. It's fully scalable and works with any
    scikit-learn compatible model without hard-coded model types.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'GOOGL', 'TSLA', 'MSFT')
        model_type: Model type identifier (e.g., 'xgboost', 'random_forest')
        model_factory_func: Function that creates and returns a model instance
        source_file: Enhanced CSV file with technical indicators (optional)
        target_days: Number of days ahead to predict
        test_size: Proportion of data for testing
        save_model: Whether to save the trained model
        save_predictions: Whether to save prediction results
        **model_params: Model-specific parameters
        
    Returns:
        Comprehensive training summary string
    """
    log_info(f"train_model_pipeline: Starting {model_type} training for {symbol}...")
    
    try:
        symbol = symbol.upper()
        
        # Prepare data
        model_data, error = prepare_model_data(symbol, source_file, target_days, test_size)
        if error:
            error_msg = f"train_model_pipeline: Data preparation failed: {error}"
            log_error(f"{error_msg}")
            return error_msg
        
        # Create model using factory function
        model = model_factory_func(**model_params)
        
        # Train model
        model.fit(model_data['X_train_scaled'], model_data['y_train'])
        
        # Generate predictions
        predictions_data = get_train_test_predictions(model, model_data)
        
        # Assess performance
        metrics = assess_model_metrics(predictions_data, model, model_data)
        
        # Get feature importance
        feature_importance = pd.DataFrame()
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = abs(model.coef_)
            else:
                importance = [1.0] * len(model_data['feature_cols'])  # Default equal importance
                
            feature_importance = pd.DataFrame({
                'feature': model_data['feature_cols'],
                'importance': importance
            }).sort_values('importance', ascending=False)
        except:
            # Fallback for models without feature importance
            feature_importance = pd.DataFrame({
                'feature': model_data['feature_cols'],
                'importance': [1.0] * len(model_data['feature_cols'])
            })
        
        # Save artifacts if requested
        filenames = {'model': None, 'results': None, 'predictions': None}
        if save_model or save_predictions:
            filenames = save_model_artifacts(
                model, model_data, predictions_data, metrics, feature_importance,
                symbol, model_type, model_params, target_days, save_model, save_predictions
            )
        
        # Generate summary
        summary = generate_model_summary(
            symbol, model_type, model_data, metrics, feature_importance,
            model_params, target_days, filenames
        )
        
        log_success(f"train_model_pipeline: Successfully trained {model_type} for {symbol}")
        return summary
        
    except Exception as e:
        error_msg = f"train_model_pipeline: Unexpected error training {model_type} for {symbol}: {str(e)}"
        log_error(f"{error_msg}")
        return error_msg


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
    """Save model artifacts with standardized naming conventions."""
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
        combined_df['Percentage_Error'] = (combined_df['Error'] / np.maximum(np.abs(combined_df['Actual']), 1e-8)) * 100
        
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
    """Generate comprehensive model training summary."""
    train_metrics = metrics['train_metrics']
    test_metrics = metrics['test_metrics']
    cv_metrics = metrics['cross_validation']
    
    # Model-specific configuration details
    model_display_name = model_type.replace('_', ' ').title()
    model_icons = {
        'xgboost': 'XGB',
        'random_forest': 'RF', 
        'svr': 'SVR',
        'gradient_boosting': 'GB',
        'ridge_regression': 'RR',
        'extra_trees': 'ET'
    }
    model_icon = model_icons.get(model_type, 'ML')
    
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

 COMPREHENSIVE PERFORMANCE METRICS:

 TRAIN SET PERFORMANCE:
- RMSE: ${train_metrics['rmse']:.3f}
- MAE: ${train_metrics['mae']:.3f}
- R²: {train_metrics['r2']:.3f}
- Information Ratio: {train_metrics['information_ratio']:.3f}
- MAPE: {train_metrics['mape']:.2f}%
- Directional Accuracy: {train_metrics['directional_accuracy']:.1f}%

 TEST SET PERFORMANCE:
- RMSE: ${test_metrics['rmse']:.3f}
- MAE: ${test_metrics['mae']:.3f}
- R²: {test_metrics['r2']:.3f}
- Information Ratio: {test_metrics['information_ratio']:.3f}
- MAPE: {test_metrics['mape']:.2f}%
- Directional Accuracy: {test_metrics['directional_accuracy']:.1f}%

 CROSS-VALIDATION SCORES:
- Mean R²: {cv_metrics['cv_r2_mean']:.3f}
- Std R²: {cv_metrics['cv_r2_std']:.3f}
- Completed Folds: {cv_metrics['cv_folds_completed']}/3
- Individual Scores: {[f"{score:.3f}" for score in cv_metrics['cv_scores']]}

 TOP 5 IMPORTANT FEATURES:
{chr(10).join([f"  {i+1}. {row['feature']}: {row['importance']:.3f}" for i, row in feature_importance.head().iterrows()])}

 FILES SAVED:
- Model: {filenames['model'] if filenames['model'] else 'Not saved'}
- Results: {filenames['results'] if filenames['results'] else 'Not saved'}
- Predictions: {filenames['predictions'] if filenames['predictions'] else 'Not saved'}

 MODEL INSIGHTS:
- Training Quality: {'Excellent' if train_metrics['r2'] > 0.8 else 'Good' if train_metrics['r2'] > 0.6 else 'Moderate' if train_metrics['r2'] > 0.3 else 'Poor'}
- Generalization: {'Good' if abs(train_metrics['r2'] - test_metrics['r2']) < 0.1 else 'Potential overfitting detected'}
- Directional Accuracy: {'Strong' if test_metrics['directional_accuracy'] > 60 else 'Moderate' if test_metrics['directional_accuracy'] > 50 else 'Weak'} trend prediction
- Recommendation: {'Model ready for deployment' if test_metrics['r2'] > 0.5 and test_metrics['directional_accuracy'] > 55 else 'Consider parameter tuning or more data'}
"""
    
    return summary