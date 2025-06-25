from langgraph.prebuilt import create_react_agent
from langgraph.graph import END
from utils import make_node_with_multiple_routes_and_memory
from models import model_gpt_4o_mini
from tools import tavily_tool, stock_tools, modeling_tools, reporting_tools, all_tools
from prompts import make_system_prompt_with_handoffs


# 🔹 Difference between Python variable name and the `name` argument:
# - The Python variable name (e.g., `stock_data_node`) is just for referencing the node in your code.
#   It has no effect on the execution, memory, or message labeling.
# 
# - The `name` argument (e.g., `name="stock_data_fetcher"`) is used to label the agent's messages
#   within the chat history. It identifies **who** said what, and is essential for memory,
#   logging, and handoffs between agents.
#   Best practice: keep both names aligned for clarity and debugging.

# Stock Data Fetcher agent and node - fetches stock data
stock_data_agent = create_react_agent(
    model = model_gpt_4o_mini,
    tools=stock_tools + [tavily_tool],  # All stock data tools plus web search for additional context
    prompt=make_system_prompt_with_handoffs(
        """You fetch stock data and prices using specialized financial data tools. 
        You have access to:
        - Real-time and historical stock price data (OHLCV)
        - Multiple stock comparison capabilities
        - Market index data (VIX, S&P 500, etc.)
        - Stock return calculations
        - Company fundamental information
        - Web search for additional financial news and context
        
        Always use the appropriate financial data tools first, then supplement with web search if needed for context or news.
        Your main job is to gather and save stock data that other agents can use for modeling and testing.
        You are working with stock modeller, stock tester, and stock reporter colleagues.""",
        ["stock_modeller", "stock_tester", "stock_reporter"]
    ),
)

stock_data_node = make_node_with_multiple_routes_and_memory(
    agent=stock_data_agent,
    next_nodes=["stock_modeller", "stock_tester", "stock_reporter", END],
    name="stock_data_fetcher"
)


# Stock Modeller agent and node - creates predictive models
stock_modeller_agent = create_react_agent(
    model = model_gpt_4o_mini,
    tools = modeling_tools,
    prompt=make_system_prompt_with_handoffs(
        """You are a stock price modelling specialist who creates predictive models using machine learning.
        You have access to:
        - Random Forest model training for stock price prediction
        - Feature engineering with technical indicators
        - Model evaluation and performance metrics
        - Future price prediction capabilities
        
        Your workflow:
        1. Use data files (CSV) provided by the stock data fetcher
        2. Train Random Forest models with technical features
        3. Evaluate model performance with proper metrics
        4. Make future price predictions
        5. Save all models and results for the tester to use
        
        Always use the CSV file paths provided by the data fetcher as input for your modeling tools.
        You are working with stock data fetcher, stock tester, and stock reporter colleagues.""",
        ["stock_data_fetcher", "stock_tester", "stock_reporter"]
    ),
)

stock_modeller_node = make_node_with_multiple_routes_and_memory(
    agent=stock_modeller_agent,
    next_nodes=["stock_data_fetcher", "stock_tester", "stock_reporter", END],
    name="stock_modeller"
)


# Stock Tester agent and node - performs backtesting and visualization
stock_tester_agent = create_react_agent(
    model = model_gpt_4o_mini,
    tools = modeling_tools,  # Uses same tools but focuses on backtesting functions
    prompt=make_system_prompt_with_handoffs(
        """You are a backtesting and visualization specialist who evaluates trading strategies.
        You have access to:
        - Backtesting framework with trading simulation
        - Performance metrics calculation (returns, Sharpe ratio, etc.)
        - Comprehensive visualization tools
        - Trading strategy evaluation
        
        Your workflow:
        1. Use trained models and data provided by the modeller and data fetcher
        2. Run backtests with realistic trading scenarios
        3. Calculate performance metrics and compare to buy-and-hold
        4. Create detailed visualizations of results
        5. Analyze model prediction accuracy
        
        Always use model files (.joblib) and data files (CSV) provided by other agents.
        Focus on creating clear, insightful visualizations and performance reports.
        You are working with stock data fetcher, stock modeller, and stock reporter colleagues.""",
        ["stock_data_fetcher", "stock_modeller", "stock_reporter"]
    ),
)

stock_tester_node = make_node_with_multiple_routes_and_memory(
    agent=stock_tester_agent,
    next_nodes=["stock_data_fetcher", "stock_modeller", "stock_reporter", END],
    name="stock_tester"
)


# Stock Reporter agent and node - creates comprehensive reports
stock_reporter_agent = create_react_agent(
    model = model_gpt_4o_mini,
    tools = reporting_tools,  # Reporting tools for saving formatted reports
    prompt=make_system_prompt_with_handoffs(
        """You are a financial reporting specialist who creates comprehensive investment reports and summaries.
        
        Your expertise includes:
        - Synthesizing data from multiple sources (stock data, ML models, backtest results)
        - Creating executive summaries with key insights
        - Formatting professional financial reports (Markdown, HTML, PDF)
        - Highlighting investment recommendations and risk factors
        - Interpreting model performance and trading strategy results
        
        You have access to:
        - Report formatting and saving tools
        - Executive summary creation tools
        
        Your workflow:
        1. Review outputs from data fetcher (stock data and market context)
        2. Analyze model performance and predictions from the modeller
        3. Examine backtest results and visualizations from the tester
        4. Synthesize all information into clear, actionable reports
        5. Use your tools to save professional formatted reports
        6. Provide investment recommendations with proper risk disclosures
        
        You work with data files, model summaries, and backtest reports provided by your colleagues.
        Focus on creating clear, professional reports that non-technical stakeholders can understand.
        You are working with stock data fetcher, stock modeller, and stock tester colleagues.""",
        ["stock_data_fetcher", "stock_modeller", "stock_tester"]
    ),
)

stock_reporter_node = make_node_with_multiple_routes_and_memory(
    agent=stock_reporter_agent,
    next_nodes=["stock_data_fetcher", "stock_modeller", "stock_tester", END],
    name="stock_reporter"
)