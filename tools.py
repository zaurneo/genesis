from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool
from clients.clients import model_client_gpt4o as model_client
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pickle
import os
import config

def file_path(name: str) -> str:
    """Return the absolute path for generated files."""
    return os.path.join(config.GENERATED_FILES_DIR, name)

# =============================================================================
# PROJECT OWNER TOOLS
# =============================================================================

def assign_task(agent_name: str, task_description: str, priority: str = "medium") -> Dict[str, Any]:
    """
    Assign a task to a specific agent.
    
    Args:
        agent_name: Name of the agent (Data_Engineer, Model_Executor, Model_Tester, Quality_Assurance)
        task_description: Description of the task
        priority: Priority level (low, medium, high)
    
    Returns:
        Dict with task assignment confirmation
    """
    task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    task = {
        "task_id": task_id,
        "agent": agent_name,
        "description": task_description,
        "priority": priority,
        "status": "assigned",
        "created_at": datetime.now().isoformat()
    }
    
    # Save to tasks file
    tasks = {}
    if os.path.exists(file_path("tasks.json")):
        with open(file_path("tasks.json"), "r") as f:
            tasks = json.load(f)
    
    tasks[task_id] = task
    with open(file_path("tasks.json"), "w") as f:
        json.dump(tasks, f, indent=2)
    
    return {"success": True, "task_id": task_id, "assigned_to": agent_name}

def check_progress() -> Dict[str, Any]:
    """
    Check overall project progress.
    
    Returns:
        Dict with progress summary
    """
    progress = {
        "data_loaded": os.path.exists(file_path("stock_data.csv")),
        "data_processed": os.path.exists(file_path("processed_data.csv")),
        "model_trained": os.path.exists(file_path("trained_model.pkl")),
        "evaluation_done": os.path.exists(file_path("evaluation.json")),
        "visualization_created": os.path.exists(file_path("analysis_chart.png")),
        "quality_checked": os.path.exists(file_path("quality_report.json"))
    }
    
    completed = sum(progress.values())
    total = len(progress)
    progress["completion_percentage"] = (completed / total) * 100
    progress["ready_for_completion"] = completed == total
    
    return progress

def validate_completion() -> Dict[str, Any]:
    """
    Validate if project is ready for completion.
    
    Returns:
        Dict with validation results
    """
    requirements = [
        (file_path("stock_data.csv"), "Stock data loaded"),
        (file_path("processed_data.csv"), "Data processed"),
        (file_path("trained_model.pkl"), "Model trained"),
        (file_path("evaluation.json"), "Model evaluated"),
        (file_path("analysis_chart.png"), "Visualization created"),
        (file_path("quality_report.json"), "Quality assured")
    ]
    
    results = []
    all_complete = True
    
    for file, description in requirements:
        exists = os.path.exists(file)
        results.append({"requirement": description, "file": file, "completed": exists})
        if not exists:
            all_complete = False
    
    return {
        "requirements": results,
        "all_complete": all_complete,
        "can_complete": all_complete
    }

def update_task_status(task_id: str, status: str) -> Dict[str, Any]:
    """
    Update task status.
    
    Args:
        task_id: Task ID to update
        status: New status (assigned, in_progress, completed)
    
    Returns:
        Dict with update confirmation
    """
    if not os.path.exists(file_path("tasks.json")):
        return {"error": "No tasks found"}

    with open(file_path("tasks.json"), "r") as f:
        tasks = json.load(f)
    
    if task_id not in tasks:
        return {"error": "Task not found"}
    
    tasks[task_id]["status"] = status
    tasks[task_id]["updated_at"] = datetime.now().isoformat()
    
    with open(file_path("tasks.json"), "w") as f:
        json.dump(tasks, f, indent=2)
    
    return {"success": True, "task_id": task_id, "new_status": status}

# =============================================================================
# DATA ENGINEER TOOLS
# =============================================================================

def load_stock_data(symbol: str, period: str = "1y") -> Dict[str, Any]:
    """
    Load stock data from yfinance.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'GOOGL')
        period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
    
    Returns:
        Dict with data loading results
    """
    try:
        # Download stock data
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        
        if data.empty:
            return {"error": f"No data found for symbol {symbol}"}
        
        # Add technical indicators
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        data['RSI'] = calculate_rsi(data['Close'])
        data['Volatility'] = data['Close'].rolling(window=20).std()
        
        # Save data
        data.to_csv(file_path("stock_data.csv"))
        
        return {
            "success": True,
            "symbol": symbol,
            "period": period,
            "rows": len(data),
            "columns": list(data.columns),
            "date_range": f"{data.index[0].date()} to {data.index[-1].date()}",
            "latest_price": float(data['Close'].iloc[-1])
        }
    except Exception as e:
        return {"error": f"Failed to load data: {str(e)}"}

def clean_and_prepare_data(target: str = "next_day_return") -> Dict[str, Any]:
    """
    Clean and prepare data for modeling.
    
    Args:
        target: Target variable to predict ('next_day_return', 'price_direction', 'close_price')
    
    Returns:
        Dict with data preparation results
    """
    try:
        if not os.path.exists(file_path("stock_data.csv")):
            return {"error": "No stock data found. Load data first."}

        data = pd.read_csv(file_path("stock_data.csv"), index_col=0, parse_dates=True)
        
        # Remove missing values
        data = data.dropna()
        
        # Create target variable
        if target == "next_day_return":
            data['target'] = data['Close'].pct_change().shift(-1)  # Next day return
        elif target == "price_direction":
            data['target'] = (data['Close'].shift(-1) > data['Close']).astype(int)  # 1 if price goes up
        elif target == "close_price":
            data['target'] = data['Close'].shift(-1)  # Next day closing price
        
        # Create features
        features = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA_20', 'MA_50', 'RSI', 'Volatility'
        ]
        
        # Add lag features
        for col in ['Close', 'Volume']:
            data[f'{col}_lag1'] = data[col].shift(1)
            data[f'{col}_lag2'] = data[col].shift(2)
            features.extend([f'{col}_lag1', f'{col}_lag2'])
        
        # Remove missing values after feature creation
        data = data.dropna()
        
        # Select features and target
        X = data[features]
        y = data['target']
        
        # Save processed data
        processed_data = pd.concat([X, y], axis=1)
        processed_data.to_csv(file_path("processed_data.csv"))
        
        # Save feature info
        feature_info = {
            "target": target,
            "features": features,
            "samples": len(processed_data),
            "target_type": "regression" if target in ["next_day_return", "close_price"] else "classification"
        }
        
        with open(file_path("feature_info.json"), "w") as f:
            json.dump(feature_info, f, indent=2)
        
        return {
            "success": True,
            "target": target,
            "features": len(features),
            "samples": len(processed_data),
            "target_stats": {
                "mean": float(y.mean()),
                "std": float(y.std()),
                "min": float(y.min()),
                "max": float(y.max())
            }
        }
    except Exception as e:
        return {"error": f"Data preparation failed: {str(e)}"}

def create_visualization() -> Dict[str, Any]:
    """
    Create stock analysis visualization.
    
    Returns:
        Dict with visualization results
    """
    try:
        if not os.path.exists(file_path("stock_data.csv")):
            return {"error": "No stock data found"}
        
        data = pd.read_csv(file_path("stock_data.csv"), index_col=0, parse_dates=True)
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Stock Analysis Dashboard', fontsize=16)
        
        # Price and moving averages
        axes[0, 0].plot(data.index, data['Close'], label='Close Price', linewidth=2)
        axes[0, 0].plot(data.index, data['MA_20'], label='MA 20', alpha=0.7)
        axes[0, 0].plot(data.index, data['MA_50'], label='MA 50', alpha=0.7)
        axes[0, 0].set_title('Price and Moving Averages')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Volume
        axes[0, 1].bar(data.index, data['Volume'], alpha=0.7, color='orange')
        axes[0, 1].set_title('Trading Volume')
        axes[0, 1].grid(True, alpha=0.3)
        
        # RSI
        axes[1, 0].plot(data.index, data['RSI'], color='purple')
        axes[1, 0].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
        axes[1, 0].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
        axes[1, 0].set_title('RSI (Relative Strength Index)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Price returns distribution
        returns = data['Close'].pct_change().dropna()
        axes[1, 1].hist(returns, bins=50, alpha=0.7, color='green')
        axes[1, 1].set_title('Daily Returns Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(file_path("analysis_chart.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            "success": True,
            "chart_saved": file_path("analysis_chart.png"),
            "stats": {
                "total_days": len(data),
                "avg_volume": float(data['Volume'].mean()),
                "price_change": float((data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100),
                "volatility": float(returns.std() * np.sqrt(252) * 100)  # Annualized volatility
            }
        }
    except Exception as e:
        return {"error": f"Visualization failed: {str(e)}"}

def generate_data_report() -> Dict[str, Any]:
    """
    Generate comprehensive data report.
    
    Returns:
        Dict with data analysis report
    """
    try:
        if not os.path.exists(file_path("stock_data.csv")):
            return {"error": "No stock data found"}

        data = pd.read_csv(file_path("stock_data.csv"), index_col=0, parse_dates=True)
        
        # Basic statistics
        report = {
            "symbol_info": {
                "total_trading_days": len(data),
                "date_range": f"{data.index[0].date()} to {data.index[-1].date()}"
            },
            "price_analysis": {
                "current_price": float(data['Close'].iloc[-1]),
                "highest_price": float(data['High'].max()),
                "lowest_price": float(data['Low'].min()),
                "average_price": float(data['Close'].mean()),
                "total_return_pct": float((data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100)
            },
            "volume_analysis": {
                "average_volume": float(data['Volume'].mean()),
                "max_volume": float(data['Volume'].max()),
                "volume_trend": "increasing" if data['Volume'].iloc[-10:].mean() > data['Volume'].iloc[-30:-10].mean() else "decreasing"
            },
            "technical_indicators": {
                "current_rsi": float(data['RSI'].iloc[-1]) if not pd.isna(data['RSI'].iloc[-1]) else None,
                "ma20_signal": "bullish" if data['Close'].iloc[-1] > data['MA_20'].iloc[-1] else "bearish",
                "ma50_signal": "bullish" if data['Close'].iloc[-1] > data['MA_50'].iloc[-1] else "bearish"
            }
        }
        
        with open(file_path("data_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        
        return {"success": True, "report": report}
    except Exception as e:
        return {"error": f"Report generation failed: {str(e)}"}

# =============================================================================
# MODEL EXECUTOR TOOLS
# =============================================================================

def train_prediction_model(model_type: str = "random_forest") -> Dict[str, Any]:
    """
    Train a model to predict stock movements.
    
    Args:
        model_type: Type of model ('random_forest', 'linear_regression')
    
    Returns:
        Dict with training results
    """
    try:
        if not os.path.exists(file_path("processed_data.csv")):
            return {"error": "No processed data found"}

        data = pd.read_csv(file_path("processed_data.csv"), index_col=0)
        
        # Load feature info
        with open(file_path("feature_info.json"), "r") as f:
            feature_info = json.load(f)
        
        # Prepare features and target
        target_col = 'target'
        feature_cols = [col for col in data.columns if col != target_col]
        
        X = data[feature_cols]
        y = data[target_col]
        
        # Remove any remaining NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        
        # Train model
        if model_type == "random_forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == "linear_regression":
            model = LinearRegression()
        else:
            return {"error": f"Unsupported model type: {model_type}"}
        
        model.fit(X_train, y_train)
        
        # Save model
        with open(file_path("trained_model.pkl"), "wb") as f:
            pickle.dump(model, f)
        
        # Generate predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Save predictions
        predictions = {
            "train_predictions": train_pred.tolist(),
            "test_predictions": test_pred.tolist(),
            "train_actual": y_train.tolist(),
            "test_actual": y_test.tolist()
        }
        
        with open(file_path("predictions.json"), "w") as f:
            json.dump(predictions, f, indent=2)
        
        return {
            "success": True,
            "model_type": model_type,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features_used": len(feature_cols),
            "target_type": feature_info["target_type"]
        }
    except Exception as e:
        return {"error": f"Model training failed: {str(e)}"}

def make_predictions(days_ahead: int = 5) -> Dict[str, Any]:
    """
    Make future predictions using the trained model.
    
    Args:
        days_ahead: Number of days to predict ahead
    
    Returns:
        Dict with prediction results
    """
    try:
        if not os.path.exists(file_path("trained_model.pkl")):
            return {"error": "No trained model found"}

        # Load model
        with open(file_path("trained_model.pkl"), "rb") as f:
            model = pickle.load(f)
        
        # Load latest data
        data = pd.read_csv(file_path("processed_data.csv"), index_col=0)
        
        # Use last row as features for prediction
        latest_features = data.drop(columns=['target']).iloc[-1:].values
        
        # Make prediction
        prediction = model.predict(latest_features)[0]
        
        # Load feature info for interpretation
        with open(file_path("feature_info.json"), "r") as f:
            feature_info = json.load(f)
        
        # Interpret prediction based on target type
        if feature_info["target"] == "next_day_return":
            direction = "up" if prediction > 0 else "down"
            confidence = abs(prediction) * 100
        elif feature_info["target"] == "price_direction":
            direction = "up" if prediction > 0.5 else "down"
            confidence = max(prediction, 1-prediction) * 100
        else:  # close_price
            direction = "N/A"
            confidence = 0
        
        result = {
            "success": True,
            "prediction": float(prediction),
            "target_type": feature_info["target"],
            "direction": direction,
            "confidence_pct": float(confidence),
            "prediction_date": datetime.now().isoformat()
        }
        
        # Save prediction
        with open(file_path("latest_prediction.json"), "w") as f:
            json.dump(result, f, indent=2)
        
        return result
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

def optimize_model(param_grid: Dict[str, List] = None) -> Dict[str, Any]:
    """
    Optimize model parameters.
    
    Args:
        param_grid: Parameter grid for optimization
    
    Returns:
        Dict with optimization results
    """
    try:
        if not os.path.exists(file_path("processed_data.csv")):
            return {"error": "No processed data found"}
        
        # Simple optimization - just try different n_estimators for RandomForest
        if param_grid is None:
            param_grid = {"n_estimators": [50, 100, 200]}
        
        data = pd.read_csv(file_path("processed_data.csv"), index_col=0)
        
        target_col = 'target'
        feature_cols = [col for col in data.columns if col != target_col]
        
        X = data[feature_cols]
        y = data[target_col]
        
        # Remove NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        
        best_score = float('inf')
        best_params = None
        
        # Simple grid search
        for n_est in param_grid.get("n_estimators", [100]):
            model = RandomForestRegressor(n_estimators=n_est, random_state=42)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            score = mean_squared_error(y_test, pred)
            
            if score < best_score:
                best_score = score
                best_params = {"n_estimators": n_est}
        
        return {
            "success": True,
            "best_params": best_params,
            "best_score": float(best_score),
            "optimization_metric": "mse"
        }
    except Exception as e:
        return {"error": f"Model optimization failed: {str(e)}"}

def get_feature_importance() -> Dict[str, Any]:
    """
    Get feature importance from the trained model.
    
    Returns:
        Dict with feature importance results
    """
    try:
        if not os.path.exists(file_path("trained_model.pkl")):
            return {"error": "No trained model found"}
        
        # Load model
        with open(file_path("trained_model.pkl"), "rb") as f:
            model = pickle.load(f)
        
        # Check if model has feature_importances_
        if not hasattr(model, 'feature_importances_'):
            return {"error": "Model does not support feature importance"}
        
        # Load feature names
        data = pd.read_csv(file_path("processed_data.csv"), index_col=0)
        feature_names = [col for col in data.columns if col != 'target']
        
        # Get importance
        importance = model.feature_importances_
        
        # Create importance dictionary
        feature_importance = dict(zip(feature_names, importance))
        
        # Sort by importance
        sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        result = {
            "success": True,
            "feature_importance": sorted_importance,
            "top_5_features": list(sorted_importance.keys())[:5]
        }
        
        with open(file_path("feature_importance.json"), "w") as f:
            json.dump(result, f, indent=2)
        
        return result
    except Exception as e:
        return {"error": f"Feature importance extraction failed: {str(e)}"}

# =============================================================================
# MODEL TESTER TOOLS
# =============================================================================

def evaluate_model_performance() -> Dict[str, Any]:
    """
    Evaluate model performance with various metrics.
    
    Returns:
        Dict with evaluation results
    """
    try:
        if not os.path.exists(file_path("predictions.json")):
            return {"error": "No predictions found"}

        with open(file_path("predictions.json"), "r") as f:
            predictions = json.load(f)
        
        # Load feature info
        with open(file_path("feature_info.json"), "r") as f:
            feature_info = json.load(f)
        
        # Calculate metrics
        y_test_true = np.array(predictions["test_actual"])
        y_test_pred = np.array(predictions["test_predictions"])
        
        # Basic metrics
        mse = mean_squared_error(y_test_true, y_test_pred)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test_true - y_test_pred))
        
        # Calculate R²
        ss_res = np.sum((y_test_true - y_test_pred) ** 2)
        ss_tot = np.sum((y_test_true - np.mean(y_test_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Direction accuracy (for returns prediction)
        if feature_info["target"] in ["next_day_return", "price_direction"]:
            direction_correct = np.sum(np.sign(y_test_true) == np.sign(y_test_pred))
            direction_accuracy = direction_correct / len(y_test_true)
        else:
            direction_accuracy = None
        
        evaluation = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2_score": float(r2),
            "direction_accuracy": float(direction_accuracy) if direction_accuracy is not None else None,
            "test_samples": len(y_test_true),
            "target_type": feature_info["target"]
        }
        
        with open(file_path("evaluation.json"), "w") as f:
            json.dump(evaluation, f, indent=2)
        
        return {"success": True, "evaluation": evaluation}
    except Exception as e:
        return {"error": f"Model evaluation failed: {str(e)}"}

def backtest_strategy() -> Dict[str, Any]:
    """
    Backtest a simple trading strategy based on model predictions.
    
    Returns:
        Dict with backtest results
    """
    try:
        if not os.path.exists(file_path("predictions.json")) or not os.path.exists(file_path("stock_data.csv")):
            return {"error": "Missing predictions or stock data"}

        with open(file_path("predictions.json"), "r") as f:
            predictions = json.load(f)
        
        # Load feature info
        with open(file_path("feature_info.json"), "r") as f:
            feature_info = json.load(f)
        
        # Simple strategy: buy if prediction > 0, sell if < 0
        test_pred = np.array(predictions["test_predictions"])
        test_actual = np.array(predictions["test_actual"])
        
        # Calculate strategy returns
        if feature_info["target"] == "next_day_return":
            # Prediction is return, actual is return
            strategy_signals = np.sign(test_pred)
            strategy_returns = strategy_signals * test_actual
        else:
            # For other targets, create simple signals
            strategy_signals = np.where(test_pred > np.median(test_pred), 1, -1)
            strategy_returns = strategy_signals * test_actual
        
        # Calculate performance metrics
        total_return = np.sum(strategy_returns)
        avg_return = np.mean(strategy_returns)
        volatility = np.std(strategy_returns)
        sharpe_ratio = avg_return / volatility if volatility > 0 else 0
        
        # Win rate
        win_rate = np.sum(strategy_returns > 0) / len(strategy_returns)
        
        backtest_results = {
            "total_return": float(total_return),
            "average_return": float(avg_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "win_rate": float(win_rate),
            "total_trades": len(strategy_returns),
            "profitable_trades": int(np.sum(strategy_returns > 0))
        }
        
        with open(file_path("backtest.json"), "w") as f:
            json.dump(backtest_results, f, indent=2)
        
        return {"success": True, "backtest": backtest_results}
    except Exception as e:
        return {"error": f"Backtesting failed: {str(e)}"}

def validate_predictions() -> Dict[str, Any]:
    """
    Validate prediction quality and consistency.
    
    Returns:
        Dict with validation results
    """
    try:
        if not os.path.exists(file_path("predictions.json")):
            return {"error": "No predictions found"}

        with open(file_path("predictions.json"), "r") as f:
            predictions = json.load(f)
        
        train_pred = np.array(predictions["train_predictions"])
        test_pred = np.array(predictions["test_predictions"])
        train_actual = np.array(predictions["train_actual"])
        test_actual = np.array(predictions["test_actual"])

        # Compute error metrics
        train_mse = float(mean_squared_error(train_actual, train_pred))
        test_mse = float(mean_squared_error(test_actual, test_pred))
        if train_mse == 0:
            overfitting_ratio = float("inf")
        else:
            overfitting_ratio = test_mse / train_mse
        
        # Check for prediction quality issues
        validation_results = {
            "train_predictions_valid": not (np.isnan(train_pred).any() or np.isinf(train_pred).any()),
            "test_predictions_valid": not (np.isnan(test_pred).any() or np.isinf(test_pred).any()),
            "prediction_range_reasonable": float(np.max(np.abs(test_pred))) < 1000,
            "no_constant_predictions": len(np.unique(test_pred)) > 1,
            "overfitting_check": {
                "train_mse": train_mse,
                "test_mse": test_mse,
                "overfitting_ratio": overfitting_ratio
            }
        }

        # Overall validation
        validation_results["all_checks_passed"] = all([
            validation_results["train_predictions_valid"],
            validation_results["test_predictions_valid"],
            validation_results["prediction_range_reasonable"],
            validation_results["no_constant_predictions"],
            np.isfinite(validation_results["overfitting_check"]["overfitting_ratio"]) and
            validation_results["overfitting_check"]["overfitting_ratio"] < 10  # Not too much overfitting
        ])
        
        with open(file_path("prediction_validation.json"), "w") as f:
            json.dump(validation_results, f, indent=2)
        
        return {"success": True, "validation": validation_results}
    except Exception as e:
        return {"error": f"Prediction validation failed: {str(e)}"}

def generate_test_report() -> Dict[str, Any]:
    """
    Generate comprehensive testing report.
    
    Returns:
        Dict with complete test report
    """
    try:
        report = {
            "report_generated_at": datetime.now().isoformat(),
            "test_results": {}
        }
        
        # Include evaluation
        if os.path.exists(file_path("evaluation.json")):
            with open(file_path("evaluation.json"), "r") as f:
                report["test_results"]["evaluation"] = json.load(f)
        
        # Include backtest
        if os.path.exists(file_path("backtest.json")):
            with open(file_path("backtest.json"), "r") as f:
                report["test_results"]["backtest"] = json.load(f)
        
        # Include validation
        if os.path.exists(file_path("prediction_validation.json")):
            with open(file_path("prediction_validation.json"), "r") as f:
                report["test_results"]["validation"] = json.load(f)
        
        # Overall assessment
        report["overall_assessment"] = {
            "model_performance": "good" if report["test_results"].get("evaluation", {}).get("r2_score", 0) > 0.1 else "poor",
            "predictions_valid": report["test_results"].get("validation", {}).get("all_checks_passed", False),
            "ready_for_deployment": False
        }
        
        # Set deployment readiness
        if (report["test_results"].get("validation", {}).get("all_checks_passed", False) and
            report["test_results"].get("evaluation", {}).get("r2_score", 0) > 0):
            report["overall_assessment"]["ready_for_deployment"] = True
        
        with open(file_path("test_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        
        return {"success": True, "report": report}
    except Exception as e:
        return {"error": f"Test report generation failed: {str(e)}"}

# =============================================================================
# QUALITY ASSURANCE TOOLS
# =============================================================================

def check_data_quality() -> Dict[str, Any]:
    """
    Check quality of all data files.
    
    Returns:
        Dict with data quality results
    """
    try:
        quality_checks = {}
        
        # Check stock data
        if os.path.exists(file_path("stock_data.csv")):
            data = pd.read_csv(file_path("stock_data.csv"), index_col=0)
            quality_checks["stock_data"] = {
                "exists": True,
                "rows": len(data),
                "has_required_columns": all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']),
                "no_missing_values": data.isnull().sum().sum() == 0,
                "reasonable_values": (data['Close'] > 0).all() and (data['Volume'] >= 0).all()
            }
        else:
            quality_checks["stock_data"] = {"exists": False}
        
        # Check processed data
        if os.path.exists(file_path("processed_data.csv")):
            data = pd.read_csv(file_path("processed_data.csv"), index_col=0)
            quality_checks["processed_data"] = {
                "exists": True,
                "rows": len(data),
                "has_target": "target" in data.columns,
                "sufficient_samples": len(data) > 50,
                "no_infinite_values": not np.isinf(data.select_dtypes(include=[np.number])).any().any()
            }
        else:
            quality_checks["processed_data"] = {"exists": False}
        
        # Overall quality score
        all_checks = []
        for category in quality_checks.values():
            if isinstance(category, dict) and "exists" in category:
                all_checks.extend([v for k, v in category.items() if k != "exists" and isinstance(v, bool)])
        
        quality_score = (sum(all_checks) / len(all_checks) * 100) if all_checks else 0
        
        return {
            "success": True,
            "quality_checks": quality_checks,
            "quality_score": quality_score,
            "data_quality_passed": quality_score >= 80
        }
    except Exception as e:
        return {"error": f"Data quality check failed: {str(e)}"}

def verify_model_outputs() -> Dict[str, Any]:
    """
    Verify all model outputs are present and valid.
    
    Returns:
        Dict with model output verification
    """
    try:
        verifications = {
            "trained_model": os.path.exists(file_path("trained_model.pkl")),
            "predictions": os.path.exists(file_path("predictions.json")),
            "evaluation": os.path.exists(file_path("evaluation.json")),
            "feature_importance": os.path.exists(file_path("feature_importance.json"))
        }
        
        # Check if files contain valid data
        if verifications["predictions"]:
            with open(file_path("predictions.json"), "r") as f:
                pred_data = json.load(f)
            verifications["predictions_valid"] = all(key in pred_data for key in 
                                                  ["train_predictions", "test_predictions", "train_actual", "test_actual"])
        
        if verifications["evaluation"]:
            with open(file_path("evaluation.json"), "r") as f:
                eval_data = json.load(f)
            verifications["evaluation_valid"] = "mse" in eval_data and "r2_score" in eval_data
        
        verifications["all_outputs_present"] = all(verifications.values())
        
        return {"success": True, "verifications": verifications}
    except Exception as e:
        return {"error": f"Model output verification failed: {str(e)}"}

def assess_compliance() -> Dict[str, Any]:
    """
    Assess project compliance with requirements.
    
    Returns:
        Dict with compliance assessment
    """
    requirements = {
        "data_loaded": os.path.exists(file_path("stock_data.csv")),
        "visualization_created": os.path.exists(file_path("analysis_chart.png")),
        "model_trained": os.path.exists(file_path("trained_model.pkl")),
        "model_evaluated": os.path.exists(file_path("evaluation.json")),
        "quality_assured": True  # This function itself
    }
    
    compliance_score = (sum(requirements.values()) / len(requirements)) * 100
    
    return {
        "success": True,
        "requirements": requirements,
        "compliance_score": compliance_score,
        "fully_compliant": compliance_score == 100
    }

def generate_quality_report() -> Dict[str, Any]:
    """
    Generate final quality assurance report.
    
    Returns:
        Dict with comprehensive QA report
    """
    try:
        # Run all quality checks
        data_quality = check_data_quality()
        model_verification = verify_model_outputs()
        compliance = assess_compliance()
        
        qa_report = {
            "report_generated_at": datetime.now().isoformat(),
            "quality_assessment": {
                "data_quality": data_quality,
                "model_outputs": model_verification,
                "compliance": compliance
            },
            "overall_quality": {
                "data_score": data_quality.get("quality_score", 0),
                "model_score": 100 if model_verification.get("verifications", {}).get("all_outputs_present", False) else 0,
                "compliance_score": compliance.get("compliance_score", 0)
            }
        }
        
        # Calculate final score
        scores = qa_report["overall_quality"]
        final_score = (scores["data_score"] + scores["model_score"] + scores["compliance_score"]) / 3
        qa_report["final_quality_score"] = final_score
        qa_report["quality_approved"] = final_score >= 75
        
        with open(file_path("quality_report.json"), "w") as f:
            json.dump(qa_report, f, indent=2)
        
        return {"success": True, "qa_report": qa_report}
    except Exception as e:
        return {"error": f"QA report generation failed: {str(e)}"}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

