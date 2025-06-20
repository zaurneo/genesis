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
from langgraph.graph.message import HumanMessage
from langgraph.graph import MessagesState, Command

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







# Research agent and node
research_agent = create_react_agent(
    model = model_gpt_4o_mini,
    tools=[tavily_tool],
    prompt=make_system_prompt(
        "You can only do research. You are working with a chart generator colleague."
    ),
)

research_node = make_node_with_multiple_routes(
    agent=research_agent,
    next_nodes=["chart_generator", "data_enricher", END],
    name="researcher"
)



chart_agent = create_react_agent(
    model = model_gpt_4o_mini,
    tools = [],
    prompt=make_system_prompt(
        "You can only generate charts. You are working with a researcher colleague."
    ),
)





# Old original code
# def research_node(state: MessagesState,) -> Command[Literal["chart_generator", END]]:
#     # Call the research agent with the current conversation state
#     result = research_agent.invoke(state)

#     # Check the agent's last message to decide whether to continue or stop
#     goto = get_next_node(result["messages"][-1], "chart_generator")

#     # Some LLMs require the last message in the prompt to be from a human,
#     # so we change the role of the last message to "HumanMessage"
#     result["messages"][-1] = HumanMessage(
#         content=result["messages"][-1].content,  # Keep the same text content
#         name="researcher"  # Give it a name/role of "researcher"
#     )

#     # Return a Command that updates the shared message state and routes to the next step
#     return Command(
#         update={
#             # Pass along the full conversation so other agents can use it
#             "messages": result["messages"],
#         },
#         # Go to the next node (either "chart_generator" or END)
#         goto=goto,
#     )

