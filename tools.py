# tools.py - Simplified handoff tools for agent communication
"""
Handoff tools for agent-to-agent communication in the hierarchical 6-agent system.
Task management tools have been moved to code_tools.py for better organization.
"""

from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.graph import MessagesState
from langgraph.types import Command

def create_handoff_tool(*, agent_name: str, description: str | None = None):
    """Create a handoff tool for transferring between agents"""
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
            tool_message = {
                "role": "tool",
                "content": f"Successfully transferred to {agent_name}",
                "name": name,
                "tool_call_id": tool_call_id,
            }
            
            # Create the command to transfer to the target agent
            command = Command(  
                goto=agent_name,  
                update={"messages": state["messages"] + [tool_message]},  
                graph=Command.PARENT,  
            )
            
            return command
            
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
                update={"messages": state["messages"] + [error_message]},
                graph=Command.PARENT,
            )
    
    return handoff_tool