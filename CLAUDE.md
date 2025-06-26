# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Genesis is a multi-agent stock analysis system built using LangChain and LangGraph. It uses a supervisor pattern to coordinate three specialized AI agents for comprehensive stock market analysis, including data fetching, technical analysis, machine learning predictions, and report generation.

## Common Commands

### Running the Application
```bash
# Set required environment variables
export OPENAI_API_KEY="your-openai-api-key"
export TAVILY_API_KEY="your-tavily-api-key"

# Install dependencies
pip install -r requirements.txt

# Run the main application
python3 main.py
```

### Development Tasks
Since no test framework or linting tools are configured, basic Python commands:
```bash
# Check syntax errors
python3 -m py_compile *.py

# Run a specific agent or tool individually
python3 -c "from agents import stock_data_agent; # test specific functionality"
```

## Architecture

### Core Components

1. **Supervisor System** (`main.py`): 
   - Entry point that creates a LangGraph supervisor workflow
   - Coordinates message passing between agents
   - Uses InMemorySaver for checkpointing and InMemoryStore for data persistence
   - Streams real-time updates from agent execution

2. **Agent System** (`agents.py`):
   - **stock_data_agent**: Fetches Yahoo Finance data and computes technical indicators
   - **stock_analyzer_agent**: Trains ML models, performs backtesting
   - **stock_reporter_agent**: Creates visualizations and generates comprehensive analysis reports

3. **Tool Library** (`tools.py`):
   - Data fetching: `get_yahoo_finance_data`
   - Technical indicators: `calculate_sma`, `calculate_ema`, `calculate_rsi`, `calculate_macd`, `calculate_bollinger_bands`
   - ML models: `train_xgboost_model`, `train_random_forest_model`
   - Backtesting: `backtest_strategy`
   - Visualization: `create_line_chart`, `create_candlestick_chart`, `create_volume_chart`
   - File operations: `save_to_csv`, `read_csv`, `save_visualization`, `save_model`, `load_model`

4. **Model Configuration** (`models.py`):
   - Uses OpenAI GPT-4o-mini as the base LLM
   - Configures both text generation and tool-calling models

### Key Architectural Patterns

- **Supervisor Pattern**: The supervisor agent manages workflow and delegates tasks to specialized agents
- **Tool Binding**: Each agent has specific tools bound to it for its specialized tasks
- **State Management**: Uses LangGraph's state management for tracking conversation and data flow
- **Output Persistence**: All outputs (data, models, visualizations, reports) saved to `output/` directory

### Data Flow

```
User Query → Supervisor → Stock Data Agent → Enhanced CSV with indicators
                     ↓
              Stock Analyzer Agent → ML Models (PKL) + Backtest Results (JSON)
                     ↓
              Stock Reporter Agent → Visualizations (HTML) + Final Report (MD/TXT)
```

### Important Notes

1. **Missing Module**: The `langgraph_supervisor` module is imported but not included. This needs to be available for the system to work.

2. **Hardcoded Query**: The main.py contains a hardcoded query for Apple stock. Modify line 40 in main.py to analyze different stocks or strategies.

3. **Output Directory**: The system automatically creates an `output/` directory for all generated files. Each tool that saves files uses this directory.

4. **Environment Variables**: The system requires `OPENAI_API_KEY` and `TAVILY_API_KEY` to be set. These are loaded via python-dotenv from the .env file.

5. **Agent Communication**: Agents communicate through the supervisor using a standardized message format. Each agent returns results that can be used by subsequent agents in the workflow.

6. **Error Handling**: Tools include validation for data quality, file operations, and API responses. Failed operations are reported back through the agent system.

## File Structure Conventions

- All generated outputs go to `output/` directory
- CSV files: Stock data with suffix indicating content (e.g., `AAPL_data_with_indicators.csv`)
- HTML files: Interactive visualizations
- PKL files: Serialized ML models
- JSON files: Structured results (model predictions, backtest metrics)
- MD/TXT files: Human-readable reports