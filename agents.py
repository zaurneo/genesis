from langgraph.prebuilt import create_react_agent
from langgraph.graph import END
from models import model_gpt_4o_mini, model_gpt_4_1
from tools.logs.logging_helpers import log_info, log_success, log_error, get_logger
# Import external tools
try:
    from langchain_community.tools.tavily_search import TavilySearchResults
    tavily_tool = TavilySearchResults(max_results=5)
except ImportError:
    tavily_tool = None

# Import tools from refactored modules
from tools.data import (
    fetch_yahoo_finance_data_impl as fetch_yahoo_finance_data,
    get_available_stock_periods_and_intervals_impl as get_available_stock_periods_and_intervals,
    read_csv_data_impl as read_csv_data,
    apply_technical_indicators_and_transformations_impl as apply_technical_indicators_and_transformations
)

from tools.models.base import train_model_pipeline

from tools.backtesting import (
    backtest_model_strategy_impl as backtest_model_strategy,
    backtest_multiple_models_impl as backtest_multiple_models
)

from tools.visualization import (
    visualize_stock_data_impl as visualize_stock_data,
    visualize_backtesting_results_impl as visualize_backtesting_results,
    visualize_model_comparison_backtesting_impl as visualize_model_comparison_backtesting,
    generate_comprehensive_html_report_impl as generate_comprehensive_html_report
)

from tools.utils import (
    list_saved_stock_files_impl as list_saved_stock_files,
    save_text_to_file_impl as save_text_to_file,
    debug_file_system_impl as debug_file_system,
    validate_model_parameters_impl as validate_model_parameters,
    get_model_selection_guide_impl as get_model_selection_guide
)

# Create ML training tool functions
def train_xgboost_price_predictor(symbol: str, source_file=None, target_days=1, **params):
    """Train XGBoost price predictor using the universal pipeline."""
    try:
        import xgboost as xgb
        def xgb_factory(**kwargs):
            return xgb.XGBRegressor(**kwargs)
        return train_model_pipeline(symbol, "xgboost", xgb_factory, source_file, target_days, **params)
    except ImportError:
        return "Error: XGBoost not available. Install with: pip install xgboost"

def train_random_forest_price_predictor(symbol: str, source_file=None, target_days=1, **params):
    """Train Random Forest price predictor using the universal pipeline."""
    try:
        from sklearn.ensemble import RandomForestRegressor
        def rf_factory(**kwargs):
            return RandomForestRegressor(**kwargs)
        return train_model_pipeline(symbol, "random_forest", rf_factory, source_file, target_days, **params)
    except ImportError:
        return "Error: Scikit-learn not available. Install with: pip install scikit-learn"

model_gpt = model_gpt_4_1

# Helper function to get tool names safely
def get_tool_name(tool):
    """Get the name of a tool, handling both functions and class instances."""
    if tool is None:
        return "None"
    if hasattr(tool, '__name__'):
        return tool.__name__
    elif hasattr(tool, 'name'):
        return tool.name
    elif hasattr(tool, '__class__'):
        return tool.__class__.__name__
    else:
        return str(tool)

# Initialize logger for agents module
logger = get_logger(__name__)
log_info("Agents module initialized")


from prompts import (
    make_system_prompt_with_handoffs,
    STOCK_DATA_FETCHER_PROMPT,
    STOCK_ANALYZER_PROMPT,
    STOCK_REPORTER_PROMPT
)


# 🔹 Difference between Python variable name and the `name` argument:
# - The Python variable name (e.g., `stock_data_node`) is just for referencing the node in your code.
#   It has no effect on the execution, memory, or message labeling.
# 
# - The `name` argument (e.g., `name="stock_data_fetcher"`) is used to label the agent's messages
#   within the chat history. It identifies **who** said what, and is essential for memory,
#   logging, and handoffs between agents.
#   Best practice: keep both names aligned for clarity and debugging.

# Stock Data Fetcher agent and node
stock_data_fetcher_tools = [
    tavily_tool, 
    fetch_yahoo_finance_data, 
    get_available_stock_periods_and_intervals,
    list_saved_stock_files,
    apply_technical_indicators_and_transformations  
]

log_info("Creating stock_data_agent with tools: " + ", ".join([get_tool_name(tool) for tool in stock_data_fetcher_tools]))
stock_data_agent = create_react_agent(
    model = model_gpt,
    tools=stock_data_fetcher_tools,
    name = "stock_data_agent",
    prompt=make_system_prompt_with_handoffs(
        STOCK_DATA_FETCHER_PROMPT([get_tool_name(tool) for tool in stock_data_fetcher_tools])
        # ["stock_analyzer", "stock_reporter"]
    ),
)
log_success("stock_data_agent created successfully")


# Stock Analyzer agent and node - ENHANCED with new scalable ML tools and multi-model backtesting
stock_analyzer_tools = [
    list_saved_stock_files,
    # Core enhanced ML training tools (using new pipeline)
    train_xgboost_price_predictor,
    train_random_forest_price_predictor,
    # AI-assisted parameter decision and validation tools
    validate_model_parameters,
    get_model_selection_guide,
    # Backtesting and analysis
    backtest_model_strategy,
    # Multi-model backtesting and comparison
    backtest_multiple_models,
    debug_file_system
]

log_info("Creating stock_analyzer_agent with enhanced ML tools: " + ", ".join([get_tool_name(tool) for tool in stock_analyzer_tools]))
stock_analyzer_agent = create_react_agent(
    model = model_gpt,
    tools = stock_analyzer_tools,
    name = "stock_analyzer_agent",
    prompt=make_system_prompt_with_handoffs(
        STOCK_ANALYZER_PROMPT([get_tool_name(tool) for tool in stock_analyzer_tools])
        # ["stock_data_fetcher", "stock_reporter"]
    ),
)
log_success("stock_analyzer_agent created successfully with enhanced ML capabilities")


# Stock Reporter agent and node - ENHANCED with multi-model visualization
stock_reporter_tools = [
    list_saved_stock_files,
    read_csv_data,
    save_text_to_file,
    visualize_stock_data,
    visualize_backtesting_results,
    # NEW: Multi-model comparison visualization
    visualize_model_comparison_backtesting,
    generate_comprehensive_html_report
]

log_info("Creating stock_reporter_agent with visualization tools: " + ", ".join([get_tool_name(tool) for tool in stock_reporter_tools]))
stock_reporter_agent = create_react_agent(
    model = model_gpt,
    tools = stock_reporter_tools,
    name = "stock_reporter_agent",
    prompt=make_system_prompt_with_handoffs(
        STOCK_REPORTER_PROMPT([get_tool_name(tool) for tool in stock_reporter_tools])
        # ["stock_data_fetcher", "stock_analyzer"]
    ),
)
log_success("stock_reporter_agent created successfully with multi-model visualization")