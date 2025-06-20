"""
System prompts and prompt-related functions for multi-agent collaboration.
"""
from typing import List

def make_system_prompt_with_handoffs(role_description: str, possible_handoffs: List[str]) -> str:
    """
    Create a system prompt that includes handoff instructions.
    
    Args:
        role_description: Description of this agent's role
        possible_handoffs: List of agents this agent can hand off to
        
    Returns:
        System prompt with handoff instructions
    """
    base_prompt = (
        "You are a helpful AI assistant, collaborating with other assistants. "
        "Use the provided tools to progress towards answering the question. "
        "If you are unable to fully answer, that's OK, another assistant with different tools "
        "will help where you left off. Execute what you can to make progress."
        f"\n{role_description}"
    )
    
    if possible_handoffs:
        agent_list = ", ".join(possible_handoffs)
        handoff_instructions = (
            f"\n\nWhen you need to hand off to another agent, write 'Handoff to [AGENT_NAME]' "
            f"where AGENT_NAME is one of: {agent_list}. "
            f"If your work is complete, end your response with 'FINAL ANSWER'."
        )
    else:
        handoff_instructions = "\n\nIf your work is complete, end your response with 'FINAL ANSWER'."
    
    return f"{base_prompt}\n{role_description}{handoff_instructions}"