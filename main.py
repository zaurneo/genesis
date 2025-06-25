#!/usr/bin/env python3
"""
Multi-agent collaboration main execution script with supervisor.
"""

from langgraph_supervisor import create_supervisor
from agents import *
from models import model_gpt_4o_mini

# Create supervisor workflow
workflow = create_supervisor(
    # Pass the individual agents (not the nodes)
    [stock_data_agent, stock_analyzer_agent, stock_reporter_agent],
    model=model_gpt_4o_mini,
    output_mode="full_history",
    prompt=(
        "You are a team supervisor managing a specialized stock analysis team with three experts:\n\n"
        "1. **stock_data_fetcher**: Fetches real-time and historical stock data from Yahoo Finance, "
        "saves data to CSV files, and provides market information. Use for data collection tasks.\n\n"
        "2. **stock_analyzer**: Analyzes stock data and creates visualizations (line charts, candlestick charts, "
        "volume charts, combined charts). Use for technical analysis and chart creation.\n\n"
        "3. **stock_reporter**: Creates comprehensive stock analysis reports and summaries, "
        "combining data and analysis into professional documentation. Use for final report generation.\n\n"
        "**SUPERVISION STRATEGY:**\n"
        "- For data fetching requests: assign to stock_data_fetcher\n"
        "- For analysis and visualization requests: assign to stock_analyzer\n"
        "- For report creation requests: assign to stock_reporter\n"
        "- For comprehensive requests: coordinate the workflow (data → analysis → report)\n\n"
        "Always ensure the right specialist handles each task for optimal results."
    )
)

# Compile the workflow
graph = workflow.compile()

result = graph.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Processing Query: 'Get Apple stock data, analyze its performance, and create a summary report.'\n"
        }
    ]
})

for m in result["messages"]:
    m.pretty_print()

