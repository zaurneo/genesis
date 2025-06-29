"""Backtesting engine for testing trading strategies with trained models."""

import os
import pickle
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, Literal

from ..config import OUTPUT_DIR, logger
from ..data import prepare_model_data


def backtest_model_strategy_impl(
    symbol: str,
    model_file: str,
    data_file: Optional[str] = None,
    initial_capital: float = 10000.0,
    strategy_type: Literal["threshold", "directional", "percentile"] = "directional",
    threshold: float = 0.02,
    transaction_cost: float = 0.001,
    save_results: bool = True
) -> str:
    """
    Backtest a trained model's predictions using various trading strategies.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')
        model_file: Trained model file (.pkl) to use for predictions
        data_file: Enhanced CSV file with technical indicators (if None, uses most recent)
        initial_capital: Starting capital for backtesting ($10,000 default)
        strategy_type: Trading strategy type:
                      - "threshold": Buy if predicted return > threshold, sell if < -threshold
                      - "directional": Buy if predicted price > current, sell if < current
                      - "percentile": Buy/sell based on prediction percentiles
        threshold: Threshold for buy/sell signals (used in threshold strategy)
        transaction_cost: Transaction cost as percentage (0.001 = 0.1%)
        save_results: Whether to save detailed backtest results
        
    Returns:
        String with comprehensive backtesting results and performance metrics
    """
    logger.info(f"backtest_model_strategy: Starting backtesting for {symbol.upper()} using {strategy_type} strategy...")
    
    try:
        symbol = symbol.upper()
        
        # Load trained model
        if not model_file.endswith('.pkl'):
            model_file += '.pkl'
        model_filepath = os.path.join(OUTPUT_DIR, model_file)
        
        if not os.path.exists(model_filepath):
            available_models = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('_model.pkl')]
            result = f"backtest_model_strategy: Model file '{model_file}' not found. Available models: {', '.join(available_models)}"
            logger.error(f"backtest_model_strategy: {result}")
            return result
        
        with open(model_filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        scaler = model_data['scaler']
        feature_cols = model_data['feature_cols']
        target_days = model_data.get('target_days', 1)
        
        # Load enhanced data
        if data_file:
            if not data_file.endswith('.csv'):
                data_file += '.csv'
            data_filepath = os.path.join(OUTPUT_DIR, data_file)
        else:
            # Find most recent enhanced data file
            enhanced_files = [f for f in os.listdir(OUTPUT_DIR) 
                            if f.startswith(f"apply_technical_indicators_and_transformations_{symbol}_") 
                            and f.endswith('.csv')]
            if not enhanced_files:
                result = f"backtest_model_strategy: No enhanced data files found for {symbol}"
                logger.error(f"backtest_model_strategy: {result}")
                return result
            latest_file = max(enhanced_files, key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)))
            data_filepath = os.path.join(OUTPUT_DIR, latest_file)
        
        if not os.path.exists(data_filepath):
            result = f"backtest_model_strategy: Data file not found: {data_filepath}"
            logger.error(f"backtest_model_strategy: {result}")
            return result
        
        # Load and prepare data
        data = pd.read_csv(data_filepath, index_col=0, parse_dates=True)
        
        # Ensure we have the required feature columns
        missing_features = [col for col in feature_cols if col not in data.columns]
        if missing_features:
            result = f"backtest_model_strategy: Missing features in data: {missing_features}"
            logger.error(f"backtest_model_strategy: {result}")
            return result
        
        # Prepare features for prediction
        features = data[feature_cols].dropna()
        prices = data['Close'].loc[features.index]
        
        if len(features) < 30:
            result = f"backtest_model_strategy: Insufficient data for backtesting: {len(features)} rows"
            logger.error(f"backtest_model_strategy: {result}")
            return result
        
        # Scale features and generate predictions
        features_scaled = scaler.transform(features)
        predictions = model.predict(features_scaled)
        
        # Create trading simulation
        portfolio = BacktestPortfolio(initial_capital, transaction_cost)
        signals = []
        
        for i in range(len(predictions)):
            current_price = prices.iloc[i]
            predicted_price = predictions[i]
            date = features.index[i]
            
            # Generate signal based on strategy
            if strategy_type == "directional":
                signal = 1 if predicted_price > current_price else -1
            elif strategy_type == "threshold":
                predicted_return = (predicted_price - current_price) / current_price
                if predicted_return > threshold:
                    signal = 1
                elif predicted_return < -threshold:
                    signal = -1
                else:
                    signal = 0
            elif strategy_type == "percentile":
                # Use percentile thresholds (top 25% buy, bottom 25% sell)
                if i < 20:  # Need some history for percentiles
                    signal = 0
                else:
                    recent_predictions = predictions[max(0, i-20):i]
                    buy_threshold = np.percentile(recent_predictions, 75)
                    sell_threshold = np.percentile(recent_predictions, 25)
                    
                    if predicted_price > buy_threshold:
                        signal = 1
                    elif predicted_price < sell_threshold:
                        signal = -1
                    else:
                        signal = 0
            else:
                signal = 0
            
            # Execute trade
            portfolio.process_signal(date, current_price, signal)
            signals.append({
                'date': date,
                'price': current_price,
                'predicted_price': predicted_price,
                'signal': signal,
                'portfolio_value': portfolio.get_total_value(current_price)
            })
        
        # Calculate performance metrics
        signals_df = pd.DataFrame(signals).set_index('date')
        final_value = portfolio.get_total_value(prices.iloc[-1])
        total_return = (final_value - initial_capital) / initial_capital * 100
        
        # Buy and hold benchmark
        buy_hold_return = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] * 100
        
        # Calculate additional metrics
        returns = signals_df['portfolio_value'].pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        max_drawdown = calculate_max_drawdown(signals_df['portfolio_value'])
        
        trades = portfolio.get_trade_history()
        win_rate = calculate_win_rate(trades) if trades else 0
        
        # Save results if requested
        results_file = None
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"backtest_{symbol}_{strategy_type}_{timestamp}.json"
            results_filepath = os.path.join(OUTPUT_DIR, results_file)
            
            results_data = {
                'symbol': symbol,
                'model_file': model_file,
                'strategy_type': strategy_type,
                'initial_capital': initial_capital,
                'final_value': final_value,
                'total_return': total_return,
                'buy_hold_return': buy_hold_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': len(trades),
                'transaction_cost': transaction_cost,
                'signals': signals_df.to_dict('records'),
                'trades': trades,
                'timestamp': timestamp
            }
            
            with open(results_filepath, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
        
        # Generate comprehensive summary
        summary = f"""backtest_model_strategy: Backtesting completed for {symbol} using {strategy_type} strategy:

 BACKTEST SUMMARY:
- Symbol: {symbol}
- Model: {model_file}
- Strategy: {strategy_type.title()}
- Period: {signals_df.index[0].strftime('%Y-%m-%d')} to {signals_df.index[-1].strftime('%Y-%m-%d')}
- Trading Days: {len(signals_df)}

 PERFORMANCE RESULTS:
- Initial Capital: ${initial_capital:,.2f}
- Final Portfolio Value: ${final_value:,.2f}
- Total Return: {total_return:+.2f}%
- Buy & Hold Return: {buy_hold_return:+.2f}%
- Excess Return: {total_return - buy_hold_return:+.2f}%

 RISK METRICS:
- Sharpe Ratio: {sharpe_ratio:.3f}
- Maximum Drawdown: {max_drawdown:.2f}%
- Win Rate: {win_rate:.1f}%

 TRADING ACTIVITY:
- Total Trades: {len(trades)}
- Transaction Costs: ${len(trades) * initial_capital * transaction_cost:,.2f}
- Average Trade: ${(final_value - initial_capital) / max(len(trades), 1):,.2f}

 RESULTS SAVED: {results_file if results_file else 'Not saved'}

 STRATEGY PERFORMANCE:
- Strategy Quality: {'Excellent' if total_return > buy_hold_return + 5 else 'Good' if total_return > buy_hold_return else 'Underperforming'}
- Risk Adjustment: {'Strong' if sharpe_ratio > 1.0 else 'Moderate' if sharpe_ratio > 0.5 else 'Weak'}
- Consistency: {'High' if win_rate > 60 else 'Medium' if win_rate > 50 else 'Low'}
"""
        
        logger.info(f"backtest_model_strategy: Completed backtesting for {symbol} with {total_return:+.2f}% return")
        return summary
        
    except Exception as e:
        error_msg = f"backtest_model_strategy: Error backtesting {symbol}: {str(e)}"
        logger.error(f"backtest_model_strategy: {error_msg}")
        return error_msg


class BacktestPortfolio:
    """Portfolio manager for backtesting simulations."""
    
    def __init__(self, initial_capital: float, transaction_cost: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.shares = 0
        self.transaction_cost = transaction_cost
        self.trades = []
        self.current_position = 0  # -1: short, 0: neutral, 1: long
        
    def process_signal(self, date, price: float, signal: int):
        """Process a trading signal."""
        if signal == self.current_position:
            return  # No change needed
        
        # Close current position if any
        if self.current_position != 0:
            self._close_position(date, price)
        
        # Open new position if signal is not neutral
        if signal != 0:
            self._open_position(date, price, signal)
    
    def _open_position(self, date, price: float, signal: int):
        """Open a new position."""
        available_capital = self.cash * (1 - self.transaction_cost)
        
        if signal == 1:  # Long position
            self.shares = available_capital / price
            self.cash = 0
        elif signal == -1:  # Short position (simplified)
            # For simplicity, treat short as inverse long
            self.shares = -available_capital / price
            self.cash = self.initial_capital * 2  # "Borrowed" cash
        
        self.current_position = signal
        self.trades.append({
            'date': date,
            'action': 'open',
            'position': 'long' if signal == 1 else 'short',
            'price': price,
            'shares': abs(self.shares),
            'cost': abs(self.shares) * price * self.transaction_cost
        })
    
    def _close_position(self, date, price: float):
        """Close current position."""
        if self.current_position == 1:  # Close long
            self.cash = self.shares * price * (1 - self.transaction_cost)
            self.shares = 0
        elif self.current_position == -1:  # Close short
            profit = -self.shares * price  # Negative shares, so profit is negative of shares * price
            self.cash = self.initial_capital + profit * (1 - self.transaction_cost)
            self.shares = 0
        
        self.trades.append({
            'date': date,
            'action': 'close',
            'position': 'long' if self.current_position == 1 else 'short',
            'price': price,
            'shares': abs(self.shares) if self.shares != 0 else 0,
            'cost': abs(self.shares) * price * self.transaction_cost if self.shares != 0 else 0
        })
        
        self.current_position = 0
    
    def get_total_value(self, current_price: float) -> float:
        """Get total portfolio value."""
        if self.current_position == 0:
            return self.cash
        elif self.current_position == 1:  # Long
            return self.shares * current_price
        else:  # Short
            return self.cash - abs(self.shares) * current_price
    
    def get_trade_history(self) -> list:
        """Get trade history."""
        return self.trades


def calculate_max_drawdown(portfolio_values: pd.Series) -> float:
    """Calculate maximum drawdown percentage."""
    peak = portfolio_values.expanding().max()
    drawdown = (portfolio_values - peak) / peak * 100
    return abs(drawdown.min())


def calculate_win_rate(trades: list) -> float:
    """Calculate win rate from trade history."""
    if len(trades) < 2:
        return 0
    
    wins = 0
    total_pairs = 0
    
    for i in range(0, len(trades) - 1, 2):
        if i + 1 < len(trades):
            open_trade = trades[i]
            close_trade = trades[i + 1]
            
            if open_trade['action'] == 'open' and close_trade['action'] == 'close':
                if open_trade['position'] == 'long':
                    profit = close_trade['price'] - open_trade['price']
                else:  # short
                    profit = open_trade['price'] - close_trade['price']
                
                if profit > 0:
                    wins += 1
                total_pairs += 1
    
    return (wins / total_pairs * 100) if total_pairs > 0 else 0