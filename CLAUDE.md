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

3. **Tool Library** (`tools/` package):
   - Data fetching: `fetch_yahoo_finance_data`, `apply_technical_indicators_and_transformations`
   - ML models: `train_xgboost_price_predictor`, `train_random_forest_price_predictor`
   - Backtesting: `backtest_model_strategy`, `backtest_multiple_models`
   - Visualization: `visualize_stock_data`, `visualize_backtesting_results`, `visualize_model_comparison_backtesting`
   - File operations: `read_csv_data`, `save_text_to_file`, `list_saved_stock_files`
   - Utilities: `validate_model_parameters`, `get_model_selection_guide`, `debug_file_system`

4. **Model Configuration** (`models.py`):
   - Uses OpenAI GPT-4o-mini as the base LLM
   - Configures both text generation and tool-calling models

### Key Architectural Patterns

- **Supervisor Pattern**: The supervisor agent manages workflow and delegates tasks to specialized agents
- **Modular Tool Architecture**: Tools are organized in a refactored package structure with clear separation of concerns
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

## Modular Tools Architecture

The tools are organized in a modular package structure for better maintainability:

```
tools/
├── __init__.py          # Package initialization and exports
├── tools.py             # Facade with @tool decorators for LangChain
├── config/              # Configuration, constants, and schemas
├── data/                # Data fetching and processing
├── models/              # ML model training pipeline
├── backtesting/         # Trading strategy backtesting
├── visualization/       # Charts and reports
└── utils/               # File management and utilities
```

This architecture provides:
- **Separation of Concerns**: Each module has a focused responsibility
- **Reusable Components**: Universal training pipeline, base classes
- **Maintainability**: Easy to extend with new models and features
- **Backward Compatibility**: All original @tool functions preserved

## File Structure Conventions

- All generated outputs go to `output/` directory
- CSV files: Stock data with suffix indicating content (e.g., `AAPL_data_with_indicators.csv`)
- HTML files: Interactive visualizations
- PKL files: Serialized ML models
- JSON files: Structured results (model predictions, backtest metrics)
- MD/TXT files: Human-readable reports