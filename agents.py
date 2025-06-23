from langgraph.prebuilt import create_react_agent
from langgraph.graph import END
from utils import make_node_with_multiple_routes_and_memory
from models import model_gpt_4o_mini
from tools import tavily_tool
from prompts import make_system_prompt_with_handoffs


# 🔹 Difference between Python variable name and the `name` argument:
# - The Python variable name (e.g., `stock_data_node`) is just for referencing the node in your code.
#   It has no effect on the execution, memory, or message labeling.
# 
# - The `name` argument (e.g., `name="stock_data_fetcher"`) is used to label the agent's messages
#   within the chat history. It identifies **who** said what, and is essential for memory,
#   logging, and handoffs between agents.
#   Best practice: keep both names aligned for clarity and debugging.

# Stock Data Fetcher agent and node
stock_data_agent = create_react_agent(
    model = model_gpt_4o_mini,
    tools=[tavily_tool],
    prompt=make_system_prompt_with_handoffs(
        "You fetch stock data and prices. You are working with stock analyzer and stock reporter colleagues.",
        ["stock_analyzer", "stock_reporter"]
    ),
)

stock_data_node = make_node_with_multiple_routes_and_memory(
    agent=stock_data_agent,
    next_nodes=["stock_analyzer", "stock_reporter", END],
    name="stock_data_fetcher"
)


# Stock Analyzer agent and node
stock_analyzer_agent = create_react_agent(
    model = model_gpt_4o_mini,
    tools = [],
    prompt=make_system_prompt_with_handoffs(
        "You analyze stock data and provide insights. You are working with stock data fetcher and stock reporter colleagues.",
        ["stock_data_fetcher", "stock_reporter"]
    ),
)

stock_analyzer_node = make_node_with_multiple_routes_and_memory(
    agent=stock_analyzer_agent,
    next_nodes=["stock_data_fetcher", "stock_reporter", END],
    name="stock_analyzer"
)


# Stock Reporter agent and node
stock_reporter_agent = create_react_agent(
    model = model_gpt_4o_mini,
    tools = [],
    prompt=make_system_prompt_with_handoffs(
        "You create stock analysis reports and summaries. You are working with stock data fetcher and stock analyzer colleagues.",
        ["stock_data_fetcher", "stock_analyzer"]
    ),
)

stock_reporter_node = make_node_with_multiple_routes_and_memory(
    agent=stock_reporter_agent,
    next_nodes=["stock_data_fetcher", "stock_analyzer", END],
    name="stock_reporter"
)