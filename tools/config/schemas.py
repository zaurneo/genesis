"""Parameter schemas and configuration for model training and analysis."""

from typing import Dict, Any, List, Union, Optional
from dataclasses import dataclass

# Parameter schemas for different model types
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

# Modeling contexts for different trading strategies
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
        "description": "Balanced approach for swing trading and medium-term positions"
    },
    "long_term_investing": {
        "target_days": [30, 60, 90],
        "preferred_models": ["random_forest", "ridge_regression"],
        "key_features": ["SMA_200", "Price_Trend", "Fundamental_Ratios", "Market_Beta"],
        "description": "Conservative approach for long-term investment decisions"
    },
    "high_volatility": {
        "target_days": [1, 3, 7],
        "preferred_models": ["xgboost", "svr"],
        "key_features": ["Volatility", "ATR", "Bollinger_Width", "Price_Acceleration"],
        "description": "Specialized for handling volatile market conditions"
    }
}

# Backtesting strategy schemas
BACKTESTING_STRATEGIES = {
    "threshold": {
        "parameters": ["threshold"],
        "defaults": {"threshold": 0.02},
        "description": "Buy if predicted return > threshold, sell if < -threshold"
    },
    "directional": {
        "parameters": [],
        "defaults": {},
        "description": "Buy if predicted price > current, sell if < current"
    },
    "percentile": {
        "parameters": ["buy_percentile", "sell_percentile"], 
        "defaults": {"buy_percentile": 0.8, "sell_percentile": 0.2},
        "description": "Buy/sell based on prediction percentiles"
    },
    "momentum": {
        "parameters": ["momentum_window", "momentum_threshold"],
        "defaults": {"momentum_window": 3, "momentum_threshold": 0.01},
        "description": "Trade based on predicted momentum signals"
    }
}

@dataclass
class ModelConfig:
    """Configuration class for model training."""
    model_type: str
    symbol: str
    target_days: int = 1
    test_size: float = 0.2
    parameters: Optional[Dict[str, Any]] = None
    save_model: bool = True
    save_predictions: bool = True
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = PARAMETER_SCHEMAS.get(self.model_type, {}).get("defaults", {})

@dataclass  
class BacktestConfig:
    """Configuration class for backtesting."""
    strategy_type: str = "directional"
    initial_capital: float = 10000.0
    transaction_cost: float = 0.001
    parameters: Optional[Dict[str, Any]] = None
    save_results: bool = True
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = BACKTESTING_STRATEGIES.get(self.strategy_type, {}).get("defaults", {})

# Validation schemas
VALIDATION_RULES = {
    "model_parameters": {
        "n_estimators": {"type": int, "min": 1, "max": 1000},
        "max_depth": {"type": [int, type(None)], "min": 1, "max": 50},
        "learning_rate": {"type": float, "min": 0.001, "max": 1.0},
        "test_size": {"type": float, "min": 0.05, "max": 0.5},
        "target_days": {"type": int, "min": 1, "max": 365}
    },
    "data_quality": {
        "min_rows": 30,
        "max_missing_percentage": 0.1,
        "required_columns": ["Open", "High", "Low", "Close", "Volume"]
    },
    "performance_metrics": {
        "min_r2": -1.0,
        "max_rmse_multiplier": 5.0,
        "min_directional_accuracy": 0.0
    }
}