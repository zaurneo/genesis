#!/usr/bin/env python3
"""
Multi-agent collaboration main execution script.
"""

import uuid
from typing import Annotated, Literal
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_experimental.utilities import PythonREPL
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState, END, StateGraph, START
from langgraph.types import Command
from agents import *
from utils import pretty_print_session_info, pretty_print_step_separator

"""Create and compile the multi-agent graph."""
workflow = StateGraph(MessagesState)
workflow.add_node("stock_data_fetcher", stock_data_node)
workflow.add_node("stock_analyzer", stock_analyzer_node)
workflow.add_node("stock_reporter", stock_reporter_node)

workflow.add_edge(START, "stock_data_fetcher")
graph = workflow.compile()

# Create a unique session ID for this conversation
session_id = uuid.uuid4()
config = {
    "configurable": {"session_id": session_id},
    "recursion_limit": 150
}

# Print session information
pretty_print_session_info(session_id)

# Example query
print("🚀 Processing Query: 'Get Apple stock data, analyze its performance, and create a summary report.'\n")

events = graph.stream(
    {
        "messages": [
            HumanMessage(
                content="Get Apple stock data, analyze its performance, and create a summary report."
            )
        ],
    },
    config=config,
    stream_mode="values"
)

# Process and pretty print results
step_count = 0
for event in events:
    step_count += 1
    print(f"📝 Step {step_count}:")
    
    # Pretty print the last message in the event
    if "messages" in event and event["messages"]:
        if isinstance(event["messages"], list):
            # If it's a list, get the last message
            event["messages"][-1].pretty_print()
        else:
            # If it's a single message
            event["messages"].pretty_print()
    
    pretty_print_step_separator()

print("✅ Multi-Agent Analysis Complete!")
print(f"💾 Session {session_id} saved to memory")
print("=" * 50)