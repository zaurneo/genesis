"""
System prompts and prompt-related functions for multi-agent collaboration.
"""

def make_system_prompt(suffix: str) -> str:
    """Create a system prompt for each agent with a custom suffix.
    
    Args:
        suffix: Additional instructions specific to the agent's role
        
    Returns:
        Complete system prompt string
    """
    return (
        "You are a helpful AI assistant, collaborating with other assistants."
        " Use the provided tools to progress towards answering the question."
        " If you are unable to fully answer, that's OK, another assistant with different tools "
        " will help where you left off. Execute what you can to make progress."
        " If you or any of the other assistants have the final answer or deliverable,"
        " prefix your response with FINAL ANSWER so the team knows to stop."
        f"\n{suffix}"
    )