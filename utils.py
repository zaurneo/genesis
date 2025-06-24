from typing import Annotated, Literal
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState, END, StateGraph, START
from langgraph.types import Command
from prompts import *
from langgraph.prebuilt import create_react_agent, InjectedState
from models import *
from tools import *
import uuid

from typing import Callable
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
from langgraph.types import Command

import re
from typing import List
from langgraph.graph import END
from langgraph.graph.message import BaseMessage

# Memory management functions
from langchain_core.chat_history import InMemoryChatMessageHistory



"""
Multi-agent collaboration agents and graph definition.
"""

chats_by_session_id = {}

def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    """Get or create chat history for a session."""
    chat_history = chats_by_session_id.get(session_id)
    if chat_history is None:
        chat_history = InMemoryChatMessageHistory()
        chats_by_session_id[session_id] = chat_history
    return chat_history

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


"""
SIMPLE VERSION SHOWN BELOW
llm = ChatOpenAI(model="gpt-4o")

search_agent = create_react_agent(llm, tools=[tavily_tool])


def search_node(state: State) -> Command[Literal["supervisor"]]:
    result = search_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="search")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )
"""




def make_node_with_multiple_routes_and_memory(
    agent: Callable[[MessagesState], dict],
    next_nodes: List[str],    # e.g., ["chart_generator", "data_enricher"]
    name: str
    ) -> Callable[[MessagesState, RunnableConfig], Command[str]]:
    def node(state: MessagesState, config: RunnableConfig) -> Command[str]:
        # Memory management - validate session ID is present
        if "configurable" not in config or "session_id" not in config["configurable"]:
            raise ValueError(
                "Make sure that the config includes the following information: {'configurable': {'session_id': 'some_value'}}"
            )
        
        # Fetch the history of messages and append to it any new messages
        chat_history = get_chat_history(config["configurable"]["session_id"])
        full_messages = list(chat_history.messages) + state["messages"]
        
        # Create a new state with the full message history
        state_with_history = {"messages": full_messages}
        
        # Invoke the agent with the full message history
        result = agent.invoke(state_with_history)

        # Decide next hop from message content
        goto = get_next_node(result["messages"][-1], next_nodes)

        # Update the message with agent name but preserve message type
        last_message = result["messages"][-1]
        if hasattr(last_message, 'name'):
            last_message.name = name

        # Update chat history with new messages
        chat_history.add_messages(state["messages"] + [last_message])

        return Command(update={"messages": last_message}, goto=goto)

    return node



"""
Session
"""

def pretty_print_session_info(session_id: uuid.UUID):
    """Print session information in a structured way."""
    print("=" * 50)
    print(f"🔄 Starting Multi-Agent Stock Analysis Session")
    print(f"📊 Session ID: {session_id}")
    print("=" * 50)

def pretty_print_step_separator():
    """Print a separator between processing steps."""
    print("\n" + "-" * 50 + "\n")
