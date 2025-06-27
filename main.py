#!/usr/bin/env python3
"""
Multi-agent collaboration main execution script with supervisor.
"""
import uuid
from supervisor import create_supervisor
from agents import *
from models import model_gpt_4o_mini
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from prompts import SUPERVISOR_PROMPT

checkpointer = InMemorySaver()
store = InMemoryStore()

session_id = uuid.uuid4()
config = {
    "configurable": {"thread_id": str(session_id)},
    "recursion_limit": 150
}

# Create supervisor workflow
workflow = create_supervisor(
    [stock_data_agent, stock_analyzer_agent, stock_reporter_agent],
    model=model_gpt_4o_mini,
    output_mode="full_history",
    prompt=SUPERVISOR_PROMPT
)

# Compile the workflow
graph = workflow.compile(
    checkpointer = InMemorySaver(),
    store = InMemoryStore()
    )

for chunk in graph.stream({
    "messages": [
        {
            "role": "user", 
            "content": "Processing Query: 'Get Apple stock data, create technical indicators, tain random forest model, backtest its results, analyze its performance, and create a summary report.'"
        }         
    ]
    }, config=config):
    
    print(f"\n🤖 Agent Update:")
    for node_name, messages in chunk.items():
        print(f"📍 Node: {node_name}")
        if isinstance(messages, dict) and "messages" in messages:
            for msg in messages["messages"]:
                if hasattr(msg, 'pretty_print'):
                    msg.pretty_print()
                else:
                    print(f"  {msg}")
        print("-" * 50)


