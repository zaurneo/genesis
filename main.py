#!/usr/bin/env python3
"""
Multi-agent collaboration main execution script.
"""


from typing import Annotated, Literal
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_experimental.utilities import PythonREPL
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState, END, StateGraph, START
from langgraph.types import Command
from agents import *


"""Create and compile the multi-agent graph."""
workflow = StateGraph(MessagesState)
workflow.add_node("stock_data_fetcher", stock_data_node)
workflow.add_node("stock_analyzer", stock_analyzer_node)
workflow.add_node("stock_reporter", stock_reporter_node)

workflow.add_edge(START, "stock_data_fetcher")
graph = workflow.compile()


# Example query
events = graph.stream(
    {
        "messages": [
            (
                "user",
                "Get Apple stock data, analyze its performance, and create a summary report.",
            )
        ],
    },
    # Maximum number of steps to take in the graph
    {"recursion_limit": 150},
)

# Print results
for s in events:
    print(s)
    print("----")