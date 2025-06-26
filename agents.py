from langgraph.prebuilt import create_react_agent
from langgraph.graph import END
from models import model_gpt_4o_mini, model_gpt_4_1
from tools import (
    tavily_tool, 
    fetch_yahoo_finance_data, 
    get_available_stock_periods_and_intervals,
    visualize_stock_data,
    list_saved_stock_files,
    save_text_to_file,
    read_csv_data,
    apply_technical_indicators_and_transformations,
    train_xgboost_price_predictor,  # NEW: Add these imports
    train_random_forest_price_predictor,  # NEW
    backtest_model_strategy  # NEW
)

model_gpt = model_gpt_4_1


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

stock_data_agent = create_react_agent(
    model = model_gpt,
    tools=stock_data_fetcher_tools,
    name = "stock_data_agent",
    prompt=make_system_prompt_with_handoffs(
        STOCK_DATA_FETCHER_PROMPT([tool.name for tool in stock_data_fetcher_tools])
        # ["stock_analyzer", "stock_reporter"]
    ),
)



# Stock Analyzer agent and node
stock_analyzer_tools = [
    list_saved_stock_files,
    train_xgboost_price_predictor,  # NEW: Add ML modeling tools
    train_random_forest_price_predictor,  # NEW
    backtest_model_strategy  # NEW: Add backtesting tool
]

stock_analyzer_agent = create_react_agent(
    model = model_gpt,
    tools = stock_analyzer_tools,
    name = "stock_analyzer_agent",
    prompt=make_system_prompt_with_handoffs(
        STOCK_ANALYZER_PROMPT([tool.name for tool in stock_analyzer_tools])
        # ["stock_data_fetcher", "stock_reporter"]
    ),
)



# Stock Reporter agent and node
stock_reporter_tools = [
    list_saved_stock_files,
    read_csv_data,
    save_text_to_file,
    visualize_stock_data
]

stock_reporter_agent = create_react_agent(
    model = model_gpt,
    tools = stock_reporter_tools,
    name = "stock_reporter_agent",
    prompt=make_system_prompt_with_handoffs(
        STOCK_REPORTER_PROMPT([tool.name for tool in stock_reporter_tools])
        # ["stock_data_fetcher", "stock_analyzer"]
    ),
)
