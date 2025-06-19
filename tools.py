# Improved tools.py with better error handling

from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.graph import MessagesState
from langgraph.types import Command

def create_handoff_tool(*, agent_name: str, description: str | None = None):
    """Create a handoff tool for transferring between agents"""
    name = f"transfer_to_{agent_name}"
    description = description or f"Transfer control to {agent_name} agent"
    
    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[MessagesState, InjectedState], 
        tool_call_id: Annotated[str, InjectedToolCallId],
        reason: str = ""
    ) -> str:
        """
        Transfer control to the specified agent.
        
        Args:
            reason: Optional reason for the handoff
            
        Returns:
            Success message as string (not Command object)
        """
        try:
            # Simply return a success message
            # The graph's conditional edges will handle the actual routing
            if reason:
                return f"Successfully transferred to {agent_name}. Reason: {reason}"
            else:
                return f"Successfully transferred to {agent_name}"
            
        except Exception as e:
            # Return error message instead of raising
            return f"Error transferring to {agent_name}: {str(e)}"
    
    return handoff_tool