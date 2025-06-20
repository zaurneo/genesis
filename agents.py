"""
Multi-agent collaboration agents and graph definition.
"""

from typing import Annotated, Literal
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState, END, StateGraph, START
from langgraph.types import Command
from prompts import *
from langgraph.prebuilt import create_react_agent, InjectedState
from models import *
from tools import *


from typing import Callable
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
from langgraph.types import Command

import re
from typing import List
from langgraph.graph import END
from langgraph.graph.message import BaseMessage

def get_next_node(last_message: BaseMessage, candidates: List[str]) -> str:
    content = last_message.content

    # Priority 1: If FINAL ANSWER is in content, end the graph
    if "FINAL ANSWER" in content.upper():
        return END

    # Priority 2: Regex-based handoff pattern
    handoff_match = re.search(r"Handoff to (\w+)", content, re.IGNORECASE)
    if handoff_match:
        mentioned_agent = handoff_match.group(1).lower()
        for candidate in candidates:
            if mentioned_agent in candidate.lower():
                return candidate

    # Priority 3: Fallback to first option
    return candidates[0]


def make_node_with_multiple_routes(
    agent: Callable[[MessagesState], dict],
    next_nodes: List[str],    # e.g., ["chart_generator", "data_enricher"]
    name: str
    ) -> Callable[[MessagesState], Command[str]]:
    def node(state: MessagesState) -> Command[str]:
        result = agent.invoke(state)

        # Decide next hop from message content
        goto = get_next_node(result["messages"][-1], next_nodes)

        # Wrap last message as "HumanMessage"
        result["messages"][-1] = HumanMessage(
            content=result["messages"][-1].content,
            name=name
        )

        return Command(update={"messages": result["messages"]}, goto=goto)

    return node


# Stock Data Fetcher agent and node
stock_data_agent = create_react_agent(
    model = model_gpt_4o_mini,
    tools=[tavily_tool],
    prompt=make_system_prompt_with_handoffs(
        "You fetch stock data and prices. You are working with stock analyzer and stock reporter colleagues.",
        ["stock_analyzer", "stock_reporter"]
    ),
)

stock_data_node = make_node_with_multiple_routes(
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

stock_analyzer_node = make_node_with_multiple_routes(
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

stock_reporter_node = make_node_with_multiple_routes(
    agent=stock_reporter_agent,
    next_nodes=["stock_data_fetcher", "stock_analyzer", END],
    name="stock_reporter"
)