"""Trading strategies for backtesting."""

import numpy as np
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class BaseTradingStrategy(ABC):
    """Abstract base class for trading strategies."""

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
    def safe_run(func): return func

    
    def __init__(self, **params):
        self.params = params
    
    @abstractmethod
    def generate_signal(self, current_price: float, predicted_price: float, context: Dict[str, Any]) -> int:
        """
        Generate trading signal based on current and predicted prices.
        
        Args:
            current_price: Current market price
            predicted_price: Model's predicted price
            context: Additional context (history, indicators, etc.)
            
        Returns:
            Trading signal: 1 (buy), -1 (sell), 0 (hold)
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get strategy description."""
        pass


class DirectionalStrategy(BaseTradingStrategy):
    """
    Simple directional strategy: buy if predicted > current, sell if predicted < current.
    
    This is the most straightforward strategy that directly uses the model's price prediction
    to make buy/sell decisions.
    """
    
    def generate_signal(self, current_price: float, predicted_price: float, context: Dict[str, Any]) -> int:
        """Generate signal based on price direction."""
        if predicted_price > current_price:
            return 1  # Buy
        elif predicted_price < current_price:
            return -1  # Sell
        else:
            return 0  # Hold
    
    def get_description(self) -> str:
        return "Directional Strategy: Buy if predicted > current, sell if predicted < current"


class ThresholdStrategy(BaseTradingStrategy):
    """
    Threshold-based strategy: only trade if predicted return exceeds threshold.
    
    This strategy adds a minimum threshold requirement to filter out weak signals
    and reduce transaction costs from excessive trading.
    """
    
    def __init__(self, threshold: float = 0.02, **params):
        super().__init__(**params)
        self.threshold = threshold
    
    def generate_signal(self, current_price: float, predicted_price: float, context: Dict[str, Any]) -> int:
        """Generate signal based on predicted return threshold."""
        predicted_return = (predicted_price - current_price) / current_price
        
        if predicted_return > self.threshold:
            return 1  # Buy
        elif predicted_return < -self.threshold:
            return -1  # Sell
        else:
            return 0  # Hold
    
    def get_description(self) -> str:
        return f"Threshold Strategy: Trade only if predicted return > {self.threshold:.1%}"


class PercentileStrategy(BaseTradingStrategy):
    """
    Percentile-based strategy: trade based on prediction percentiles over recent history.
    
    This strategy uses recent prediction history to determine buy/sell thresholds,
    making it adaptive to changing market conditions.
    """
    
    def __init__(self, buy_percentile: float = 75, sell_percentile: float = 25, lookback: int = 20, **params):
        super().__init__(**params)
        self.buy_percentile = buy_percentile
        self.sell_percentile = sell_percentile
        self.lookback = lookback
    
    def generate_signal(self, current_price: float, predicted_price: float, context: Dict[str, Any]) -> int:
        """Generate signal based on prediction percentiles."""
        prediction_history = context.get('prediction_history', [])
        
        if len(prediction_history) < self.lookback:
            return 0  # Not enough history
        
        recent_predictions = prediction_history[-self.lookback:]
        buy_threshold = np.percentile(recent_predictions, self.buy_percentile)
        sell_threshold = np.percentile(recent_predictions, self.sell_percentile)
        
        if predicted_price > buy_threshold:
            return 1  # Buy
        elif predicted_price < sell_threshold:
            return -1  # Sell
        else:
            return 0  # Hold
    
    def get_description(self) -> str:
        return f"Percentile Strategy: Buy at {self.buy_percentile}th percentile, sell at {self.sell_percentile}th percentile"


class MomentumStrategy(BaseTradingStrategy):
    """
    Momentum-based strategy: consider prediction momentum over multiple periods.
    
    This strategy looks at the trend in predictions to identify sustained
    directional moves rather than single-period predictions.
    """
    
    def __init__(self, momentum_window: int = 3, momentum_threshold: float = 0.01, **params):
        super().__init__(**params)
        self.momentum_window = momentum_window
        self.momentum_threshold = momentum_threshold
    
    def generate_signal(self, current_price: float, predicted_price: float, context: Dict[str, Any]) -> int:
        """Generate signal based on prediction momentum."""
        prediction_history = context.get('prediction_history', [])
        
        if len(prediction_history) < self.momentum_window:
            return 0  # Not enough history
        
        recent_predictions = prediction_history[-self.momentum_window:]
        
        # Calculate momentum as average change in predictions
        momentum = np.mean(np.diff(recent_predictions)) / current_price
        
        if momentum > self.momentum_threshold:
            return 1  # Buy (upward momentum)
        elif momentum < -self.momentum_threshold:
            return -1  # Sell (downward momentum)
        else:
            return 0  # Hold
    
    def get_description(self) -> str:
        return f"Momentum Strategy: Trade based on {self.momentum_window}-period prediction momentum"


class VolatilityAdjustedStrategy(BaseTradingStrategy):
    """
    Volatility-adjusted strategy: scale signals based on market volatility.
    
    This strategy adjusts position sizing and thresholds based on current
    market volatility to manage risk dynamically.
    """
    
    def __init__(self, base_threshold: float = 0.02, volatility_window: int = 20, **params):
        super().__init__(**params)
        self.base_threshold = base_threshold
        self.volatility_window = volatility_window
    
    def generate_signal(self, current_price: float, predicted_price: float, context: Dict[str, Any]) -> int:
        """Generate signal adjusted for market volatility."""
        price_history = context.get('price_history', [])
        
        if len(price_history) < self.volatility_window:
            # Fallback to simple directional strategy
            return 1 if predicted_price > current_price else -1 if predicted_price < current_price else 0
        
        recent_prices = price_history[-self.volatility_window:]
        returns = np.diff(recent_prices) / recent_prices[:-1]
        volatility = np.std(returns)
        
        # Adjust threshold based on volatility
        adjusted_threshold = self.base_threshold * (1 + volatility)
        predicted_return = (predicted_price - current_price) / current_price
        
        if predicted_return > adjusted_threshold:
            return 1  # Buy
        elif predicted_return < -adjusted_threshold:
            return -1  # Sell
        else:
            return 0  # Hold
    
    def get_description(self) -> str:
        return f"Volatility-Adjusted Strategy: Dynamic threshold based on {self.volatility_window}-period volatility"


class MeanReversionStrategy(BaseTradingStrategy):
    """
    Mean reversion strategy: trade against extreme predictions.
    
    This strategy assumes that extreme predictions are more likely to revert
    and trades against them rather than with them.
    """
    
    def __init__(self, reversion_threshold: float = 0.05, **params):
        super().__init__(**params)
        self.reversion_threshold = reversion_threshold
    
    def generate_signal(self, current_price: float, predicted_price: float, context: Dict[str, Any]) -> int:
        """Generate mean reversion signal."""
        predicted_return = (predicted_price - current_price) / current_price
        
        # Trade against extreme predictions
        if predicted_return > self.reversion_threshold:
            return -1  # Sell (expect reversion)
        elif predicted_return < -self.reversion_threshold:
            return 1  # Buy (expect reversion)
        else:
            return 0  # Hold
    
    def get_description(self) -> str:
        return f"Mean Reversion Strategy: Trade against predictions > {self.reversion_threshold:.1%}"


def get_strategy(strategy_type: str, **params) -> BaseTradingStrategy:
    """
    Factory function to create trading strategy instances.
    
    Args:
        strategy_type: Type of strategy to create
        **params: Strategy-specific parameters
        
    Returns:
        Trading strategy instance
    """
    strategies = {
        'directional': DirectionalStrategy,
        'threshold': ThresholdStrategy,
        'percentile': PercentileStrategy,
        'momentum': MomentumStrategy,
        'volatility_adjusted': VolatilityAdjustedStrategy,
        'mean_reversion': MeanReversionStrategy
    }
    
    if strategy_type not in strategies:
        raise ValueError(f"Unknown strategy type: {strategy_type}. Available: {list(strategies.keys())}")
    
    return strategies[strategy_type](**params)


def get_available_strategies() -> Dict[str, str]:
    """
    Get available trading strategies with descriptions.
    
    Returns:
        Dictionary mapping strategy names to descriptions
    """
    strategy_descriptions = {}
    
    for strategy_name in ['directional', 'threshold', 'percentile', 'momentum', 'volatility_adjusted', 'mean_reversion']:
        strategy = get_strategy(strategy_name)
        strategy_descriptions[strategy_name] = strategy.get_description()
    
    return strategy_descriptions