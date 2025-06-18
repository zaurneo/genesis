# tools.py - Fixed handoff tools for LangGraph compatibility
from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.graph import MessagesState
from langgraph.types import Command

def create_handoff_tool(*, agent_name: str, description: str | None = None):
    """Create a handoff tool for transferring between agents with improved error handling"""
    name = f"transfer_to_{agent_name}"
    description = description or f"Transfer to {agent_name}"
    
    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[MessagesState, InjectedState], 
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """
        Transfer control to the specified agent.
        This tool should only be called once per response to avoid coordination errors.
        """
        try:
            # Create the tool message to add to state
            tool_message = {
                "role": "tool",
                "content": f"Successfully transferred to {agent_name}",
                "name": name,
                "tool_call_id": tool_call_id,
            }
            
            # Create the command to transfer to the target agent
            # FIXED: Removed graph=Command.PARENT which was causing ParentCommand error
            return Command(
                goto=agent_name,
                update={"messages": state["messages"] + [tool_message]}
            )
            
        except Exception as e:
            # Fallback: return a tool message indicating the error
            error_message = {
                "role": "tool",
                "content": f"Transfer to {agent_name} failed: {str(e)}",
                "name": name,
                "tool_call_id": tool_call_id,
            }
            
            # Return a command that stays in the current state but adds the error message
            return Command(
                update={"messages": state["messages"] + [error_message]}
            )
    
    return handoff_tool