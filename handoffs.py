

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
transfer_to_hotel_assistant = create_handoff_tool(
    agent_name="hotel_assistant",
    description="Transfer user to the hotel-booking assistant.",
)
transfer_to_flight_assistant = create_handoff_tool(
    agent_name="flight_assistant",
    description="Transfer user to the flight-booking assistant.",
)

# def create_handoff_tool(*, agent_name: str, description: str | None = None):
#     """Create a handoff tool following official LangGraph patterns"""
#     name = f"transfer_to_{agent_name}"
#     description = description or f"Transfer to {agent_name}"
    
#     @tool(name, description=description)
#     def handoff_tool(
#         state: Annotated[MessagesState, InjectedState],
#         tool_call_id: Annotated[str, InjectedToolCallId],
#         reason: str = ""  # Keep your useful reason parameter
#     ) -> Command:
#         """Transfer control to the specified agent using official Command pattern"""
        
#         # Create tool message as per official pattern
#         content = f"Successfully transferred to {agent_name}"
#         if reason:
#             content += f". Reason: {reason}"
            
#         tool_message = {
#             "role": "tool",
#             "content": content,
#             "name": name,
#             "tool_call_id": tool_call_id,
#         }
        
#         # Return Command object for proper routing and state management
#         return Command(
#             goto=agent_name,
#             update={"messages": state["messages"] + [tool_message]},
#             graph=Command.PARENT,  # Important for multi-agent systems
#         )
    
#     return handoff_tool