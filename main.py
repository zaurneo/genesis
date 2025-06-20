from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.types import Command
from prompts import *
from tools import *
from handoffs import *
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()
from agents import *


# Define multi-agent graph
multi_agent_graph = (
    StateGraph(MessagesState)
    .add_node(market_analyzer)
    .add_node(portfolio_manager)
    .add_edge(START, "market_analyzer")
    .compile()
)

# Run the multi-agent graph
for chunk in multi_agent_graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "analyze AAPL stock and recommend a balanced portfolio allocation for moderate risk tolerance"
            }
        ]
    }
):
    print(chunk)
    print("\n")