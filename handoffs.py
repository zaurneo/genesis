from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.graph import MessagesState
from langgraph.types import Command


def create_handoff_tool(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Transfer to {agent_name}"

    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[MessagesState, InjectedState], 
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }
        return Command(  
            goto=agent_name,  
            update={"messages": state["messages"] + [tool_message]},  
            graph=Command.PARENT,  
        )
    return handoff_tool


# Handoffs
transfer_to_tech_lead = create_handoff_tool(
    agent_name="tech_lead",
    description="Transfer user to the technical lead for code review, guidance, and task assignment.",
)
transfer_to_writer = create_handoff_tool(
    agent_name="writer",
    description="Transfer user to the code writer for implementing main application code.",
)
transfer_to_executor = create_handoff_tool(
    agent_name="executor",
    description="Transfer user to the executor for code testing and execution.",
)