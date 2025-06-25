"""
Complete stock analysis tools for multi-agent collaboration.
Includes data fetching, modeling, backtesting, visualization, and reporting tools.
All outputs are saved to appropriate folders with timestamps.
"""

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime, timedelta
import warnings
import markdown2
warnings.filterwarnings('ignore')

# Create output directories
OUTPUT_DIR = "output"
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
BACKTEST_DIR = os.path.join(OUTPUT_DIR, "backtests")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

for dir_path in [OUTPUT_DIR, MODELS_DIR, PLOTS_DIR, BACKTEST_DIR, REPORTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Original Tavily search tool
tavily_tool = TavilySearchResults(max_results=5)


def get_timestamp() -> str:
    """Get current timestamp for filename."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_dataframe(df: pd.DataFrame, filename: str) -> str:
    """Save DataFrame as CSV and return filepath."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath)
    return filepath


def save_json(data: dict, filename: str) -> str:
    """Save dictionary as JSON and return filepath."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    return filepath


def create_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create technical indicators as features for modeling."""
    data = df.copy()
    
    # Price-based features
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    
    # Price ratios
    data['Price_SMA5_Ratio'] = data['Close'] / data['SMA_5']
    data['Price_SMA20_Ratio'] = data['Close'] / data['SMA_20']
    
    # Volatility features
    data['Volatility_5'] = data['Close'].rolling(window=5).std()
    data['Volatility_20'] = data['Close'].rolling(window=20).std()
    
    # Volume features
    data['Volume_SMA_5'] = data['Volume'].rolling(window=5).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA_5']
    
    # Price changes
    data['Price_Change_1'] = data['Close'].pct_change(1)
    data['Price_Change_5'] = data['Close'].pct_change(5)
    
    # High-Low spread
    data['HL_Spread'] = (data['High'] - data['Low']) / data['Close']
    
    # RSI-like indicator
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    return data


# ================================
# STOCK DATA FETCHING TOOLS
# ================================

@tool
def get_stock_data(ticker: str, period: str = "1mo", interval: str = "1d") -> str:
    """
    Fetch stock OHLCV data for a given ticker and time period.
    Saves raw data as CSV and summary as JSON.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '1h', '1d', '1wk', '1mo')
    
    Returns:
        JSON string with stock data and key metrics, plus file save confirmation
    """
    try:
        stock = yf.Ticker(ticker.upper())
        data = stock.history(period=period, interval=interval)
        
        if data.empty:
            return f"No data available for ticker '{ticker}'"
        
        timestamp = get_timestamp()
        ticker_clean = ticker.upper()
        
        # Save raw OHLCV data as CSV
        csv_filename = f"{ticker_clean}_ohlcv_{period}_{interval}_{timestamp}.csv"
        csv_path = save_dataframe(data, csv_filename)
        
        # Calculate key metrics
        price_change = float(data['Close'].iloc[-1] - data['Close'].iloc[0])
        price_change_pct = float((price_change / data['Close'].iloc[0]) * 100)
        
        result = {
            "ticker": ticker_clean,
            "period": period,
            "interval": interval,
            "data_points": len(data),
            "current_price": float(data['Close'].iloc[-1]),
            "period_change": {
                "absolute": price_change,
                "percent": price_change_pct
            },
            "price_range": {
                "high": float(data['High'].max()),
                "low": float(data['Low'].min())
            },
            "volume": {
                "latest": int(data['Volume'].iloc[-1]),
                "average": int(data['Volume'].mean())
            },
            "recent_prices": [
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "close": float(row['Close']),
                    "volume": int(row['Volume'])
                }
                for idx, row in data.tail(5).iterrows()
            ],
            "files_saved": {
                "raw_data_csv": csv_path,
                "timestamp": timestamp
            }
        }
        
        # Save summary as JSON
        json_filename = f"{ticker_clean}_summary_{period}_{interval}_{timestamp}.json"
        json_path = save_json(result, json_filename)
        result["files_saved"]["summary_json"] = json_path
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return f"Error fetching data for {ticker}: {str(e)}"


@tool
def get_multiple_stocks(tickers_str: str, period: str = "1mo") -> str:
    """
    Fetch data for multiple stocks for comparison.
    Saves individual stock data as CSV files and comparison as JSON.
    
    Args:
        tickers_str: Comma-separated ticker symbols (e.g., 'AAPL,MSFT,GOOGL')
        period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
    
    Returns:
        JSON string with comparison data for all stocks, plus file save confirmation
    """
    try:
        tickers = [t.strip().upper() for t in tickers_str.split(',')]
        results = {}
        timestamp = get_timestamp()
        saved_files = []
        
        # Combined DataFrame for all stocks' closing prices
        combined_data = pd.DataFrame()
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(period=period)
                
                if not data.empty:
                    # Save individual stock data
                    csv_filename = f"{ticker}_ohlcv_{period}_{timestamp}.csv"
                    csv_path = save_dataframe(data, csv_filename)
                    saved_files.append(csv_path)
                    
                    # Add to combined closing prices
                    combined_data[ticker] = data['Close']
                    
                    price_change = float(data['Close'].iloc[-1] - data['Close'].iloc[0])
                    price_change_pct = float((price_change / data['Close'].iloc[0]) * 100)
                    
                    results[ticker] = {
                        "current_price": float(data['Close'].iloc[-1]),
                        "period_change_percent": price_change_pct,
                        "high": float(data['High'].max()),
                        "low": float(data['Low'].min()),
                        "average_volume": int(data['Volume'].mean()),
                        "data_file": csv_path
                    }
            except Exception as e:
                results[ticker] = f"Error fetching data: {str(e)}"
        
        # Save combined closing prices comparison
        if not combined_data.empty:
            combined_filename = f"comparison_closing_prices_{period}_{timestamp}.csv"
            combined_path = save_dataframe(combined_data, combined_filename)
            saved_files.append(combined_path)
        
        comparison = {
            "period": period,
            "stocks": results,
            "best_performer": max([k for k in results.keys() if isinstance(results[k], dict)], 
                                key=lambda k: results[k].get('period_change_percent', -999)) 
                                if any(isinstance(v, dict) for v in results.values()) else "N/A",
            "worst_performer": min([k for k in results.keys() if isinstance(results[k], dict)], 
                                 key=lambda k: results[k].get('period_change_percent', 999))
                                 if any(isinstance(v, dict) for v in results.values()) else "N/A",
            "files_saved": {
                "individual_data_files": saved_files,
                "combined_comparison": combined_path if not combined_data.empty else None,
                "timestamp": timestamp
            }
        }
        
        # Save comparison summary as JSON
        json_filename = f"stocks_comparison_{period}_{timestamp}.json"
        json_path = save_json(comparison, json_filename)
        comparison["files_saved"]["summary_json"] = json_path
        
        return json.dumps(comparison, indent=2)
        
    except Exception as e:
        return f"Error comparing stocks: {str(e)}"


@tool
def get_stock_returns(ticker: str, period: str = "1y") -> str:
    """
    Calculate returns and volatility for a stock.
    Saves returns data as CSV and analysis as JSON.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        period: Time period for calculation ('1mo', '3mo', '6mo', '1y', '2y', '5y')
    
    Returns:
        JSON string with return statistics, plus file save confirmation
    """
    try:
        stock = yf.Ticker(ticker.upper())
        data = stock.history(period=period)
        
        if data.empty:
            return f"No data available for ticker '{ticker}'"
        
        timestamp = get_timestamp()
        ticker_clean = ticker.upper()
        
        # Calculate daily returns
        daily_returns = data['Close'].pct_change().dropna()
        
        # Create returns DataFrame with additional metrics
        returns_df = pd.DataFrame({
            'Date': daily_returns.index,
            'Close_Price': data['Close'][1:],  # Align with returns
            'Daily_Return': daily_returns.values,
            'Cumulative_Return': (1 + daily_returns).cumprod() - 1
        })
        returns_df.set_index('Date', inplace=True)
        
        # Save returns data as CSV
        returns_filename = f"{ticker_clean}_returns_analysis_{period}_{timestamp}.csv"
        returns_path = save_dataframe(returns_df, returns_filename)
        
        result = {
            "ticker": ticker_clean,
            "period": period,
            "returns_analysis": {
                "total_return_percent": float(((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100),
                "average_daily_return": float(daily_returns.mean()),
                "volatility_daily": float(daily_returns.std()),
                "annualized_return": float(daily_returns.mean() * 252),
                "annualized_volatility": float(daily_returns.std() * (252 ** 0.5)),
                "best_day": float(daily_returns.max()),
                "worst_day": float(daily_returns.min())
            },
            "risk_metrics": {
                "sharpe_ratio_approx": float(daily_returns.mean() / daily_returns.std()) if daily_returns.std() > 0 else 0,
                "positive_days_percent": float((daily_returns > 0).mean() * 100)
            },
            "files_saved": {
                "returns_data_csv": returns_path,
                "timestamp": timestamp
            }
        }
        
        # Save analysis summary as JSON
        json_filename = f"{ticker_clean}_returns_summary_{period}_{timestamp}.json"
        json_path = save_json(result, json_filename)
        result["files_saved"]["analysis_json"] = json_path
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return f"Error calculating returns for {ticker}: {str(e)}"


@tool
def get_market_index(symbol: str = "^VIX", period: str = "1mo") -> str:
    """
    Get market index data (VIX, S&P 500, etc.).
    Saves index data as CSV and summary as JSON.
    
    Args:
        symbol: Index symbol ('^VIX', '^GSPC', '^DJI', '^IXIC')
        period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
    
    Returns:
        JSON string with index data, plus file save confirmation
    """
    try:
        index = yf.Ticker(symbol.upper())
        data = index.history(period=period)
        
        if data.empty:
            return f"No data available for index '{symbol}'"
        
        timestamp = get_timestamp()
        symbol_clean = symbol.replace("^", "").upper()
        
        # Save raw index data as CSV
        csv_filename = f"index_{symbol_clean}_data_{period}_{timestamp}.csv"
        csv_path = save_dataframe(data, csv_filename)
        
        change = float(data['Close'].iloc[-1] - data['Close'].iloc[0])
        change_pct = float((change / data['Close'].iloc[0]) * 100)
        
        result = {
            "symbol": symbol.upper(),
            "period": period,
            "current_level": float(data['Close'].iloc[-1]),
            "period_change": {
                "absolute": change,
                "percent": change_pct
            },
            "range": {
                "high": float(data['Close'].max()),
                "low": float(data['Close'].min())
            },
            "average": float(data['Close'].mean()),
            "volatility": float(data['Close'].std()),
            "files_saved": {
                "index_data_csv": csv_path,
                "timestamp": timestamp
            }
        }
        
        # Save summary as JSON
        json_filename = f"index_{symbol_clean}_summary_{period}_{timestamp}.json"
        json_path = save_json(result, json_filename)
        result["files_saved"]["summary_json"] = json_path
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return f"Error fetching index data for {symbol}: {str(e)}"


@tool
def get_company_info(ticker: str) -> str:
    """
    Get basic company information and key financial metrics.
    Saves company information as JSON.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
    
    Returns:
        JSON string with company information, plus file save confirmation
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info
        
        if not info:
            return f"No information available for ticker '{ticker}'"
        
        timestamp = get_timestamp()
        ticker_clean = ticker.upper()
        
        result = {
            "ticker": ticker_clean,
            "company": {
                "name": info.get('longName', 'N/A'),
                "sector": info.get('sector', 'N/A'),
                "industry": info.get('industry', 'N/A'),
                "country": info.get('country', 'N/A'),
                "website": info.get('website', 'N/A'),
                "business_summary": info.get('businessSummary', 'N/A')
            },
            "valuation": {
                "market_cap": info.get('marketCap', 'N/A'),
                "enterprise_value": info.get('enterpriseValue', 'N/A'),
                "pe_ratio": info.get('trailingPE', 'N/A'),
                "forward_pe": info.get('forwardPE', 'N/A'),
                "price_to_book": info.get('priceToBook', 'N/A'),
                "price_to_sales": info.get('priceToSalesTrailing12Months', 'N/A'),
                "dividend_yield": info.get('dividendYield', 'N/A')
            },
            "trading": {
                "current_price": info.get('currentPrice', 'N/A'),
                "day_high": info.get('dayHigh', 'N/A'),
                "day_low": info.get('dayLow', 'N/A'),
                "52_week_high": info.get('fiftyTwoWeekHigh', 'N/A'),
                "52_week_low": info.get('fiftyTwoWeekLow', 'N/A'),
                "volume": info.get('volume', 'N/A'),
                "average_volume": info.get('averageVolume', 'N/A')
            },
            "files_saved": {
                "timestamp": timestamp
            }
        }
        
        # Save company info as JSON
        json_filename = f"{ticker_clean}_company_info_{timestamp}.json"
        json_path = save_json(result, json_filename)
        result["files_saved"]["company_info_json"] = json_path
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return f"Error fetching company info for {ticker}: {str(e)}"


# ================================
# MACHINE LEARNING MODELING TOOLS
# ================================

@tool
def train_random_forest_model(ticker: str, csv_file_path: str, target_days: int = 1) -> str:
    """
    Train a Random Forest model to predict stock prices.
    
    Args:
        ticker: Stock ticker symbol
        csv_file_path: Path to the CSV file with stock data (from stock_data_fetcher)
        target_days: Number of days ahead to predict (default: 1)
    
    Returns:
        JSON string with model training results and file paths
    """
    try:
        # Load data
        if not os.path.exists(csv_file_path):
            return f"Error: CSV file not found at {csv_file_path}"
        
        data = pd.read_csv(csv_file_path, index_col=0, parse_dates=True)
        
        if len(data) < 50:
            return f"Error: Not enough data points. Need at least 50, got {len(data)}"
        
        # Create features
        data_with_features = create_technical_features(data)
        
        # Create target variable (future price)
        data_with_features[f'Target_{target_days}d'] = data_with_features['Close'].shift(-target_days)
        
        # Remove rows with NaN values
        clean_data = data_with_features.dropna()
        
        if len(clean_data) < 30:
            return f"Error: Not enough clean data after feature engineering. Got {len(clean_data)}"
        
        # Select features (exclude OHLCV and target)
        feature_columns = [col for col in clean_data.columns 
                          if col not in ['Open', 'High', 'Low', 'Close', 'Volume', f'Target_{target_days}d']]
        
        X = clean_data[feature_columns]
        y = clean_data[f'Target_{target_days}d']
        
        # Split data chronologically
        split_idx = int(len(clean_data) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest model
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_train = rf_model.predict(X_train_scaled)
        y_pred_test = rf_model.predict(X_test_scaled)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        timestamp = get_timestamp()
        ticker_clean = ticker.upper()
        
        # Save model and scaler
        model_filename = f"{ticker_clean}_rf_model_{target_days}d_{timestamp}.joblib"
        scaler_filename = f"{ticker_clean}_scaler_{target_days}d_{timestamp}.joblib"
        model_path = os.path.join(MODELS_DIR, model_filename)
        scaler_path = os.path.join(MODELS_DIR, scaler_filename)
        
        joblib.dump(rf_model, model_path)
        joblib.dump(scaler, scaler_path)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save feature importance
        importance_filename = f"{ticker_clean}_feature_importance_{target_days}d_{timestamp}.csv"
        importance_path = os.path.join(OUTPUT_DIR, importance_filename)
        feature_importance.to_csv(importance_path, index=False)
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'Date': X_test.index,
            'Actual': y_test.values,
            'Predicted': y_pred_test,
            'Error': y_test.values - y_pred_test
        })
        
        predictions_filename = f"{ticker_clean}_predictions_{target_days}d_{timestamp}.csv"
        predictions_path = os.path.join(OUTPUT_DIR, predictions_filename)
        predictions_df.to_csv(predictions_path, index=False)
        
        result = {
            "ticker": ticker_clean,
            "target_days": target_days,
            "model_type": "Random Forest",
            "data_info": {
                "total_samples": len(clean_data),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "features_used": feature_columns
            },
            "performance_metrics": {
                "train_rmse": float(train_rmse),
                "test_rmse": float(test_rmse),
                "train_mae": float(train_mae),
                "test_mae": float(test_mae),
                "train_r2": float(train_r2),
                "test_r2": float(test_r2)
            },
            "top_features": feature_importance.head(5).to_dict('records'),
            "files_saved": {
                "model_file": model_path,
                "scaler_file": scaler_path,
                "feature_importance": importance_path,
                "predictions": predictions_path,
                "timestamp": timestamp
            }
        }
        
        # Save model summary
        summary_filename = f"{ticker_clean}_model_summary_{target_days}d_{timestamp}.json"
        summary_path = os.path.join(OUTPUT_DIR, summary_filename)
        with open(summary_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        result["files_saved"]["model_summary"] = summary_path
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return f"Error training model for {ticker}: {str(e)}"


@tool
def make_stock_prediction(ticker: str, model_path: str, scaler_path: str, data_path: str, prediction_days: int = 5) -> str:
    """
    Make future stock price predictions using a trained model.
    
    Args:
        ticker: Stock ticker symbol
        model_path: Path to the saved model file
        scaler_path: Path to the saved scaler file
        data_path: Path to the latest stock data CSV
        prediction_days: Number of days to predict into the future
    
    Returns:
        JSON string with predictions and file paths
    """
    try:
        # Load model and scaler
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return f"Error: Model or scaler file not found"
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Load and prepare data
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        data_with_features = create_technical_features(data)
        clean_data = data_with_features.dropna()
        
        # Get feature columns (same as training)
        feature_columns = [col for col in clean_data.columns 
                          if col not in ['Open', 'High', 'Low', 'Close', 'Volume'] 
                          and not col.startswith('Target_')]
        
        # Use last available data point for prediction
        latest_features = clean_data[feature_columns].iloc[-1:].values
        latest_features_scaled = scaler.transform(latest_features)
        
        # Make prediction
        prediction = model.predict(latest_features_scaled)[0]
        
        timestamp = get_timestamp()
        ticker_clean = ticker.upper()
        
        # Create future predictions (simplified - using same features)
        current_price = clean_data['Close'].iloc[-1]
        predictions = []
        
        for i in range(prediction_days):
            pred_price = model.predict(latest_features_scaled)[0]
            future_date = clean_data.index[-1] + timedelta(days=i+1)
            
            predictions.append({
                "date": future_date.strftime("%Y-%m-%d"),
                "predicted_price": float(pred_price),
                "change_from_current": float(pred_price - current_price),
                "change_percent": float(((pred_price - current_price) / current_price) * 100)
            })
        
        result = {
            "ticker": ticker_clean,
            "prediction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "current_price": float(current_price),
            "model_used": model_path,
            "predictions": predictions,
            "summary": {
                "avg_predicted_price": float(np.mean([p["predicted_price"] for p in predictions])),
                "price_direction": "UP" if predictions[0]["predicted_price"] > current_price else "DOWN",
                "max_predicted_change": float(max([abs(p["change_percent"]) for p in predictions]))
            }
        }
        
        # Save predictions
        predictions_filename = f"{ticker_clean}_future_predictions_{prediction_days}d_{timestamp}.json"
        predictions_path = os.path.join(OUTPUT_DIR, predictions_filename)
        with open(predictions_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        result["files_saved"] = {
            "predictions_file": predictions_path,
            "timestamp": timestamp
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return f"Error making predictions for {ticker}: {str(e)}"


# ================================
# BACKTESTING AND VISUALIZATION TOOLS
# ================================

@tool
def run_backtest(ticker: str, model_path: str, scaler_path: str, test_data_path: str, initial_capital: float = 10000) -> str:
    """
    Run backtesting on a trained model with trading simulation.
    
    Args:
        ticker: Stock ticker symbol
        model_path: Path to the saved model file
        scaler_path: Path to the saved scaler file
        test_data_path: Path to the test data CSV
        initial_capital: Starting capital for backtesting (default: $10,000)
    
    Returns:
        JSON string with backtest results and file paths
    """
    try:
        # Load model and scaler
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Load test data
        data = pd.read_csv(test_data_path, index_col=0, parse_dates=True)
        data_with_features = create_technical_features(data)
        clean_data = data_with_features.dropna()
        
        # Prepare features
        feature_columns = [col for col in clean_data.columns 
                          if col not in ['Open', 'High', 'Low', 'Close', 'Volume'] 
                          and not col.startswith('Target_')]
        
        X = clean_data[feature_columns]
        X_scaled = scaler.transform(X)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        
        # Simple trading strategy: Buy when prediction > current price, Sell otherwise
        portfolio_value = [initial_capital]
        positions = []  # 1 for long, 0 for cash
        cash = initial_capital
        shares = 0
        
        backtest_data = []
        
        for i in range(1, len(clean_data)):
            current_price = clean_data['Close'].iloc[i]
            predicted_price = predictions[i-1] if i-1 < len(predictions) else current_price
            
            # Trading decision
            if predicted_price > current_price * 1.01:  # Buy if prediction is 1% higher
                if cash > 0:  # Buy
                    shares = cash / current_price
                    cash = 0
                    action = "BUY"
                else:
                    action = "HOLD_LONG"
            elif predicted_price < current_price * 0.99:  # Sell if prediction is 1% lower
                if shares > 0:  # Sell
                    cash = shares * current_price
                    shares = 0
                    action = "SELL"
                else:
                    action = "HOLD_CASH"
            else:
                action = "HOLD"
            
            # Calculate portfolio value
            current_portfolio_value = cash + (shares * current_price)
            portfolio_value.append(current_portfolio_value)
            
            backtest_data.append({
                "date": clean_data.index[i].strftime("%Y-%m-%d"),
                "price": float(current_price),
                "predicted_price": float(predicted_price),
                "action": action,
                "portfolio_value": float(current_portfolio_value),
                "cash": float(cash),
                "shares": float(shares)
            })
        
        # Calculate performance metrics
        final_value = portfolio_value[-1]
        total_return = (final_value - initial_capital) / initial_capital * 100
        
        # Buy and hold comparison
        buy_hold_return = (clean_data['Close'].iloc[-1] - clean_data['Close'].iloc[0]) / clean_data['Close'].iloc[0] * 100
        
        # Calculate Sharpe ratio (simplified)
        returns = pd.Series(portfolio_value).pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        timestamp = get_timestamp()
        ticker_clean = ticker.upper()
        
        # Save backtest data
        backtest_df = pd.DataFrame(backtest_data)
        backtest_filename = f"{ticker_clean}_backtest_results_{timestamp}.csv"
        backtest_path = os.path.join(BACKTEST_DIR, backtest_filename)
        backtest_df.to_csv(backtest_path, index=False)
        
        result = {
            "ticker": ticker_clean,
            "backtest_period": {
                "start_date": clean_data.index[0].strftime("%Y-%m-%d"),
                "end_date": clean_data.index[-1].strftime("%Y-%m-%d"),
                "total_days": len(clean_data)
            },
            "performance": {
                "initial_capital": float(initial_capital),
                "final_value": float(final_value),
                "total_return_percent": float(total_return),
                "buy_hold_return_percent": float(buy_hold_return),
                "excess_return": float(total_return - buy_hold_return),
                "sharpe_ratio": float(sharpe_ratio)
            },
            "trading_stats": {
                "total_trades": len([d for d in backtest_data if d["action"] in ["BUY", "SELL"]]),
                "buy_trades": len([d for d in backtest_data if d["action"] == "BUY"]),
                "sell_trades": len([d for d in backtest_data if d["action"] == "SELL"])
            },
            "files_saved": {
                "backtest_data": backtest_path,
                "timestamp": timestamp
            }
        }
        
        # Save backtest summary
        summary_filename = f"{ticker_clean}_backtest_summary_{timestamp}.json"
        summary_path = os.path.join(BACKTEST_DIR, summary_filename)
        with open(summary_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        result["files_saved"]["backtest_summary"] = summary_path
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return f"Error running backtest for {ticker}: {str(e)}"


@tool
def create_backtest_visualization(ticker: str, backtest_csv_path: str) -> str:
    """
    Create visualizations for backtest results.
    
    Args:
        ticker: Stock ticker symbol
        backtest_csv_path: Path to the backtest results CSV file
    
    Returns:
        JSON string with visualization file paths
    """
    try:
        # Load backtest data
        if not os.path.exists(backtest_csv_path):
            return f"Error: Backtest CSV file not found at {backtest_csv_path}"
        
        data = pd.read_csv(backtest_csv_path, parse_dates=['date'])
        data.set_index('date', inplace=True)
        
        timestamp = get_timestamp()
        ticker_clean = ticker.upper()
        
        # Create visualization plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{ticker_clean} Backtest Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Price vs Predictions
        ax1.plot(data.index, data['price'], label='Actual Price', linewidth=2)
        ax1.plot(data.index, data['predicted_price'], label='Predicted Price', alpha=0.7, linewidth=1)
        ax1.set_title('Actual vs Predicted Prices')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Portfolio Value Over Time
        ax2.plot(data.index, data['portfolio_value'], label='Portfolio Value', color='green', linewidth=2)
        # Calculate buy and hold for comparison
        initial_price = data['price'].iloc[0]
        initial_capital = data['portfolio_value'].iloc[0]
        shares_buy_hold = initial_capital / initial_price
        buy_hold_value = shares_buy_hold * data['price']
        ax2.plot(data.index, buy_hold_value, label='Buy & Hold', color='blue', alpha=0.7, linestyle='--')
        ax2.set_title('Portfolio Performance')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Trading Actions
        buy_signals = data[data['action'] == 'BUY']
        sell_signals = data[data['action'] == 'SELL']
        
        ax3.plot(data.index, data['price'], color='black', alpha=0.6, linewidth=1)
        if not buy_signals.empty:
            ax3.scatter(buy_signals.index, buy_signals['price'], color='green', marker='^', s=100, label='BUY', alpha=0.8)
        if not sell_signals.empty:
            ax3.scatter(sell_signals.index, sell_signals['price'], color='red', marker='v', s=100, label='SELL', alpha=0.8)
        ax3.set_title('Trading Signals')
        ax3.set_ylabel('Price ($)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Prediction Accuracy
        prediction_error = data['price'] - data['predicted_price']
        ax4.hist(prediction_error, bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax4.set_title('Prediction Error Distribution')
        ax4.set_xlabel('Prediction Error ($)')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        # Adjust layout and save
        plt.tight_layout()
        plot_filename = f"{ticker_clean}_backtest_visualization_{timestamp}.png"
        plot_path = os.path.join(PLOTS_DIR, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create performance metrics plot
        fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Calculate cumulative returns
        portfolio_returns = data['portfolio_value'].pct_change().fillna(0)
        price_returns = data['price'].pct_change().fillna(0)
        
        cum_portfolio_returns = (1 + portfolio_returns).cumprod() - 1
        cum_price_returns = (1 + price_returns).cumprod() - 1
        
        ax.plot(data.index, cum_portfolio_returns * 100, label='Strategy Returns', linewidth=2, color='green')
        ax.plot(data.index, cum_price_returns * 100, label='Buy & Hold Returns', linewidth=2, color='blue', alpha=0.7)
        ax.set_title(f'{ticker_clean} Cumulative Returns Comparison')
        ax.set_ylabel('Cumulative Returns (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        returns_plot_filename = f"{ticker_clean}_returns_comparison_{timestamp}.png"
        returns_plot_path = os.path.join(PLOTS_DIR, returns_plot_filename)
        plt.savefig(returns_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        result = {
            "ticker": ticker_clean,
            "visualizations_created": [
                "Price vs Predictions",
                "Portfolio Performance", 
                "Trading Signals",
                "Prediction Error Distribution",
                "Cumulative Returns Comparison"
            ],
            "files_saved": {
                "main_visualization": plot_path,
                "returns_comparison": returns_plot_path,
                "timestamp": timestamp
            },
            "analysis_summary": {
                "total_trades": len(data[data['action'].isin(['BUY', 'SELL'])]),
                "prediction_mae": float(abs(prediction_error).mean()),
                "prediction_rmse": float(np.sqrt((prediction_error ** 2).mean()))
            }
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return f"Error creating visualizations for {ticker}: {str(e)}"


# ================================
# REPORTING TOOLS
# ================================

@tool
def save_investment_report(ticker: str, report_title: str, report_content: str, report_format: str = "markdown") -> str:
    """
    Save a comprehensive investment report in the specified format.
    
    Args:
        ticker: Stock ticker symbol
        report_title: Title of the report
        report_content: Full report content (markdown formatted)
        report_format: Format to save ('markdown', 'html', 'txt')
    
    Returns:
        JSON string with file paths and confirmation
    """
    try:
        timestamp = get_timestamp()
        ticker_clean = ticker.upper()
        
        # Clean title for filename
        safe_title = "".join(c for c in report_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title.replace(' ', '_')[:50]  # Limit length
        
        saved_files = {}
        
        # Save as Markdown
        if report_format in ['markdown', 'md']:
            md_filename = f"{ticker_clean}_{safe_title}_{timestamp}.md"
            md_path = os.path.join(REPORTS_DIR, md_filename)
            
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(f"# {report_title}\n\n")
                f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Ticker:** {ticker_clean}\n\n")
                f.write("---\n\n")
                f.write(report_content)
            
            saved_files["markdown"] = md_path
        
        # Save as HTML
        if report_format in ['html', 'htm']:
            html_content = markdown2.markdown(report_content, extras=['tables', 'fenced-code-blocks'])
            
            html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{report_title}</title>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .header {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .highlight {{ background-color: #fff3cd; padding: 10px; border-left: 4px solid #ffc107; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{report_title}</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Ticker:</strong> {ticker_clean}</p>
    </div>
    {html_content}
</body>
</html>"""
            
            html_filename = f"{ticker_clean}_{safe_title}_{timestamp}.html"
            html_path = os.path.join(REPORTS_DIR, html_filename)
            
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_template)
            
            saved_files["html"] = html_path
        
        # Save as plain text
        if report_format in ['txt', 'text']:
            txt_filename = f"{ticker_clean}_{safe_title}_{timestamp}.txt"
            txt_path = os.path.join(REPORTS_DIR, txt_filename)
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"{report_title}\n")
                f.write("=" * len(report_title) + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Ticker: {ticker_clean}\n\n")
                f.write("-" * 50 + "\n\n")
                # Remove markdown formatting for plain text
                clean_content = report_content.replace('**', '').replace('*', '').replace('#', '')
                f.write(clean_content)
            
            saved_files["text"] = txt_path
        
        result = {
            "ticker": ticker_clean,
            "report_title": report_title,
            "report_format": report_format,
            "timestamp": timestamp,
            "files_saved": saved_files,
            "success": True
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return f"Error saving report for {ticker}: {str(e)}"


@tool
def create_executive_summary(ticker: str, key_findings: str, recommendation: str, risk_factors: str) -> str:
    """
    Create a structured executive summary for investment reports.
    
    Args:
        ticker: Stock ticker symbol
        key_findings: Key findings from analysis
        recommendation: Investment recommendation (BUY/HOLD/SELL)
        risk_factors: Main risk factors to consider
    
    Returns:
        JSON string with formatted executive summary
    """
    try:
        timestamp = get_timestamp()
        ticker_clean = ticker.upper()
        
        # Create structured executive summary
        executive_summary = f"""# Executive Summary: {ticker_clean}

## 🎯 Investment Recommendation: **{recommendation.upper()}**

### Key Findings
{key_findings}

### Risk Assessment
{risk_factors}

### Bottom Line
Based on our comprehensive analysis including machine learning predictions and backtesting results, 
we recommend a **{recommendation.upper()}** position in {ticker_clean}.

---
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # Save executive summary
        summary_filename = f"{ticker_clean}_executive_summary_{timestamp}.md"
        summary_path = os.path.join(REPORTS_DIR, summary_filename)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(executive_summary)
        
        result = {
            "ticker": ticker_clean,
            "recommendation": recommendation.upper(),
            "executive_summary": executive_summary,
            "files_saved": {
                "executive_summary": summary_path,
                "timestamp": timestamp
            }
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return f"Error creating executive summary for {ticker}: {str(e)}"


# ================================
# TOOL LISTS FOR EASY IMPORT
# ================================

# Stock data tools
stock_tools = [
    get_stock_data,
    get_multiple_stocks,
    get_stock_returns,
    get_market_index,
    get_company_info
]

# Machine learning tools
modeling_tools = [
    train_random_forest_model,
    make_stock_prediction,
    run_backtest,
    create_backtest_visualization
]

# Reporting tools
reporting_tools = [
    save_investment_report,
    create_executive_summary
]

# All tools combined
all_tools = [tavily_tool] + stock_tools + modeling_tools + reporting_tools