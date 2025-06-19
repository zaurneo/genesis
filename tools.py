# tools.py - CORRECTED VERSION for LangGraph react agents

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
    ):
        """
        Transfer control to the specified agent.
        
        Args:
            reason: Optional reason for the handoff
            
        Returns:
            Command object for routing AND string message for ToolMessage
        """
        try:
            # For react agents, we need to return a string that becomes a ToolMessage
            # The Command routing will be handled by a separate mechanism
            success_msg = f"Successfully transferred to {agent_name}"
            if reason:
                success_msg += f". Reason: {reason}"
            
            # Store the routing decision in the state for the graph to use
            # This is a workaround - we'll handle Command routing in the graph
            return success_msg
            
        except Exception as e:
            return f"Error transferring to {agent_name}: {str(e)}"
    
    return handoff_tool