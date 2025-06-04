from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool
from clients import model_client_gpt4o as model_client
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
from autogen_agentchat.agents import AssistantAgent
from agents import code_execution_agent

# Additional imports for file management and knowledge retrieval
from pprint import pprint
from dotenv import load_dotenv
load_dotenv()

from prompts import ARCHIVE_AGENT_MATCH_DOMAIN_PROMPT
from utils.misc import light_gpt4_wrapper_autogen
from utils.rag_tools import get_informed_answer
from utils.search_tools import find_relevant_github_repo


# Reference to the active group chat so tools can control the conversation
TEAM_CONTEXT = None

# Default directories for various agent operations

# File path used by search utilities
SEARCH_RESULTS_FILE = f"{COMM_DIR}/search_results.json"


# =============================================================================
# GENERAL PURPOSE TOOLS
# =============================================================================

# Metadata describing the signature and purpose of each general purpose tool.
agent_functions = [
    {
        "name": "read_file",
        "description": "Reads a file and returns its contents.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": f"Path to the file relative to {WORK_DIR}.",
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "read_multiple_files",
        "description": "Reads multiple files and returns their contents.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": f"List of paths relative to {WORK_DIR}.",
                },
            },
            "required": ["file_paths"],
        },
    },
    {
        "name": "read_directory_contents",
        "description": "Return file names contained in a directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "directory_path": {
                    "type": "string",
                    "description": f"Directory relative to {WORK_DIR}.",
                },
            },
            "required": ["directory_path"],
        },
    },
    {
        "name": "save_file",
        "description": "Save a file to disk without overwriting existing files.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": f"Destination path relative to {WORK_DIR}.",
                },
                "file_contents": {
                    "type": "string",
                    "description": "Contents to write to the file.",
                },
            },
            "required": ["file_path", "file_contents"],
        },
    },
    {
        "name": "save_multiple_files",
        "description": "Save multiple files in one call without overwriting.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": f"List of paths relative to {WORK_DIR}.",
                },
                "file_contents": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file contents matching file_paths.",
                },
            },
            "required": ["file_paths", "file_contents"],
        },
    },
    {
        "name": "execute_code_block",
        "description": "Execute a Python or bash code block and capture output.",
        "parameters": {
            "type": "object",
            "properties": {
                "lang": {"type": "string", "description": "Language of the code"},
                "code_block": {
                    "type": "string",
                    "description": "Code block to execute. If first line is '# filename: <name>' it will be saved.",
                },
            },
            "required": ["lang", "code_block"],
        },
    },
    {
        "name": "consult_archive_agent",
        "description": "Query the archive agent for domain specific information.",
        "parameters": {
            "type": "object",
            "properties": {
                "domain_description": {
                    "type": "string",
                    "description": "Description of the target knowledge domain.",
                },
                "question": {
                    "type": "string",
                    "description": "Detailed question to ask the archive agent.",
                },
            },
            "required": ["domain_description", "question"],
        },
    },
]


def read_file(file_path: str) -> str:
    """Return the contents of ``file_path`` relative to ``WORK_DIR``."""
    resolved_path = os.path.abspath(os.path.normpath(f"{WORK_DIR}/{file_path}"))
    with open(resolved_path, "r") as f:
        return f.read()


def read_directory_contents(directory_path: str) -> List[str]:
    """List all files in ``directory_path`` relative to ``WORK_DIR``."""
    resolved_path = os.path.abspath(os.path.normpath(f"{WORK_DIR}/{directory_path}"))
    return os.listdir(resolved_path)


def read_multiple_files(file_paths: List[str]) -> List[str]:
    """Read and return a list of file contents for each path provided."""
    resolved_paths = [
        os.path.abspath(os.path.normpath(f"{WORK_DIR}/{path}")) for path in file_paths
    ]
    contents = []
    for path in resolved_paths:
        with open(path, "r") as f:
            contents.append(f.read())
    return contents


def save_file(file_path: str, file_contents: str) -> str:
    """Create ``file_path`` relative to ``WORK_DIR`` with ``file_contents``."""
    resolved_path = os.path.abspath(os.path.normpath(f"{WORK_DIR}/{file_path}"))
    if os.path.exists(resolved_path):
        raise Exception(f"File already exists at {resolved_path}.")

    directory = os.path.dirname(resolved_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(resolved_path, "w") as f:
        f.write(file_contents)

    return f"File saved to {resolved_path}."


def save_multiple_files(file_paths: List[str], file_contents: List[str]) -> str:
    """Save each item in ``file_contents`` to the matching path."""
    resolved_paths = [
        os.path.abspath(os.path.normpath(f"{WORK_DIR}/{path}")) for path in file_paths
    ]

    for path in resolved_paths:
        if os.path.exists(path):
            raise Exception(f"File already exists at {path}.")

    for idx, path in enumerate(resolved_paths):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(path, "w") as f:
            f.write(file_contents[idx])

    return f"Files saved to {resolved_paths}."


# Agent dedicated to executing arbitrary code blocks

def execute_code_block(lang: str, code_block: str) -> str:
    """Execute a code block using ``code_execution_agent`` and return logs."""
    code_execution_agent._code_execution_config.pop("last_n_messages", None)
    exitcode, logs = code_execution_agent.execute_code_blocks([(lang, code_block)])
    status = "execution succeeded" if exitcode == 0 else "execution failed"
    return f"exitcode: {exitcode} ({status})\nCode output: {logs}"


def consult_archive_agent(domain_description: str, question: str) -> str:
    """Consult the archive agent for a domain-specific answer."""
    domain_descriptions = []
    for root, dirs, files in os.walk(DOMAIN_KNOWLEDGE_DOCS_DIR):
        for dir in dirs:
            for file in os.listdir(os.path.join(root, dir)):
                if file == "domain_description.txt":
                    with open(os.path.join(root, dir, file), "r") as f:
                        domain_descriptions.append({
                            "domain_name": dir,
                            "domain_description": f.read(),
                        })
        break

    str_desc = ""
    for desc in domain_descriptions:
        str_desc += (
            f"Domain: {desc['domain_name']}\n\nDescription:\n{'*' * 50}\n{desc['domain_description']}\n{'*' * 50}\n\n"
        )

    find_domain_query = ARCHIVE_AGENT_MATCH_DOMAIN_PROMPT.format(
        domain_description=domain_description,
        available_domains=str_desc,
    )

    domain_response = light_gpt4_wrapper_autogen(find_domain_query, return_json=True)
    domain_response = domain_response.get("items", [])
    domain_response = sorted(domain_response, key=lambda x: int(x.get("rating", 0)), reverse=True)

    top_domain = domain_response[0] if domain_response else {"rating": 0}

    DOMAIN_RESPONSE_THRESHOLD = 5
    if top_domain.get("rating", 0) < DOMAIN_RESPONSE_THRESHOLD:
        domain, domain_description = find_relevant_github_repo(domain_description)
    else:
        domain = top_domain.get("domain")
        domain_description = top_domain.get("domain_description")

    return get_informed_answer(
        domain=domain,
        domain_description=domain_description,
        question=question,
        docs_dir=DOMAIN_KNOWLEDGE_DOCS_DIR,
        storage_dir=DOMAIN_KNOWLEDGE_STORAGE_DIR,
        vector_top_k=80,
        reranker_top_n=20,
        rerank=True,
        fusion=True,
    )




















def register_team(team) -> None:
    """Register the running team for coordination tools."""
    global TEAM_CONTEXT
    TEAM_CONTEXT = team

def start_report_phase() -> Dict[str, Any]:
    """Signal the team to switch to the final report phase.

    The phase can only be started when all required outputs exist and every
    task recorded in ``tasks.json`` is marked as completed. If the project is
    not ready, an error is returned instead of switching phases.
    """
    if not (TEAM_CONTEXT and hasattr(TEAM_CONTEXT, "start_report_phase")):
        return {"error": "Team context not initialized"}

    completion = validate_completion()
    tasks_status = all_tasks_completed()
    if not completion.get("can_complete") or not tasks_status["all_tasks_completed"]:
        return {
            "error": "Project not ready for report phase",
            "requirements_met": completion.get("can_complete"),
            "all_tasks_completed": tasks_status["all_tasks_completed"],
        }

    TEAM_CONTEXT.start_report_phase()
    return {"success": True}

def file_path(name: str) -> str:
    """Return the absolute path for generated files."""
    return os.path.join(config.GENERATED_FILES_DIR, name)

# Mapping of JSON files to the functions that generate them. This helps
# pinpoint where a malformed JSON file might originate.
JSON_FILE_GENERATORS = {
    "tasks.json": "assign_task/update_task_status",
    "feature_info.json": "clean_and_prepare_data",
    "predictions.json": "train_prediction_model",
    "latest_prediction.json": "make_predictions",
    "evaluation.json": "evaluate_model_performance",
    "backtest.json": "backtest_strategy",
    "prediction_validation.json": "validate_predictions",
    "test_report.json": "generate_test_report",
    "quality_report.json": "generate_quality_report",
    "data_report.json": "generate_data_report",
}

def _convert_to_python_types(obj: Any) -> Any:
    """Recursively convert numpy scalar types to native Python types."""
    if isinstance(obj, dict):
        return {k: _convert_to_python_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_to_python_types(v) for v in obj]
    # numpy scalars have an ``item`` method that returns the corresponding
    # Python scalar (e.g. ``np.bool_`` -> ``bool``)
    try:
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass
    return obj

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


def all_tasks_completed() -> Dict[str, Any]:
    """Check whether every task in ``tasks.json`` is marked as completed."""
    if not os.path.exists(file_path("tasks.json")):
        return {"all_tasks_completed": False, "incomplete_tasks": []}

    with open(file_path("tasks.json"), "r") as f:
        tasks = json.load(f)

    incomplete = [tid for tid, t in tasks.items() if t.get("status") != "completed"]
    return {"all_tasks_completed": len(incomplete) == 0, "incomplete_tasks": incomplete}


def validate_json_file(file_name: str) -> Dict[str, Any]:
    """Validate that a JSON file is well formed and report errors."""
    path = file_path(file_name)
    if not os.path.exists(path):
        return {"error": "File not found", "file": path}

    with open(path, "r") as f:
        content = f.read()

    try:
        json.loads(content)
    except json.JSONDecodeError as e:
        generator = JSON_FILE_GENERATORS.get(file_name)
        return {
            "error": "Invalid JSON",
            "file": path,
            "generator": generator,
            "line": e.lineno,
            "column": e.colno,
            "message": e.msg,
        }

    stripped = content.rstrip()
    if stripped and stripped[-1] not in ('}', ']'):  # catch trailing characters
        generator = JSON_FILE_GENERATORS.get(file_name)
        return {
            "error": "JSON does not end with a closing bracket",
            "file": path,
            "generator": generator,
        }

    return {"success": True, "file": path}

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
        
        qa_report = _convert_to_python_types(qa_report)

        with open(file_path("quality_report.json"), "w") as f:
            json.dump(qa_report, f, indent=2)
        
        return {"success": True, "qa_report": qa_report}
    except Exception as e:
        return {"error": f"QA report generation failed: {str(e)}"}

def generate_html_report() -> Dict[str, Any]:
    """Create an investor-friendly HTML summary using available reports."""
    try:
        if not (TEAM_CONTEXT and getattr(TEAM_CONTEXT, "report_phase", False)):
            return {"error": "Report phase not started"}

        parts = ["<html><head><title>Investor Report</title></head><body>",
                 "<h1>Project Results</h1>"]
        if os.path.exists(file_path("data_report.json")):
            with open(file_path("data_report.json"), "r") as f:
                parts.append("<h2>Data Report</h2><pre>" + json.dumps(json.load(f), indent=2) + "</pre>")
        if os.path.exists(file_path("evaluation.json")):
            with open(file_path("evaluation.json"), "r") as f:
                parts.append("<h2>Model Evaluation</h2><pre>" + json.dumps(json.load(f), indent=2) + "</pre>")
        if os.path.exists(file_path("quality_report.json")):
            with open(file_path("quality_report.json"), "r") as f:
                parts.append("<h2>Quality Assurance</h2><pre>" + json.dumps(json.load(f), indent=2) + "</pre>")
        if os.path.exists(file_path("analysis_chart.png")):
            import base64
            with open(file_path("analysis_chart.png"), "rb") as f:
                img = base64.b64encode(f.read()).decode("utf-8")
            parts.append(f"<img src='data:image/png;base64,{img}' alt='Chart'>")
        parts.append("</body></html>")
        output = file_path("investor_report.html")
        with open(output, "w") as f:
            f.write("\n".join(parts))
        return {"success": True, "file": output}
    except Exception as e:
        return {"error": f"HTML report generation failed: {str(e)}"}


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

