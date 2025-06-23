"""
System prompts and prompt-related functions for multi-agent collaboration.
"""
from typing import List

# General description of all agents in the system
AGENTS_DESCRIPTION = """
TEAM OVERVIEW:
You are part of a specialized stock analysis team with three distinct roles:

1. stock_data_node:
   - Primary responsibility: Fetch current stock data, prices, and financial information
   - Tools available: Tavily search tool for real-time data retrieval
   - Focus: Data collection and initial information gathering
   - Does NOT: Perform analysis or create reports

2. STOCK_ANALYZER: 
   - Primary responsibility: Analyze stock data and provide insights, trends, and interpretations
   - Tools available: None (works with data provided by data fetcher)
   - Focus: Technical analysis, performance evaluation, trend identification
   - Does NOT: Fetch raw data or create formatted reports

3. STOCK_REPORTER:
   - Primary responsibility: Create comprehensive stock analysis reports and summaries
   - Tools available: None (works with analysis provided by analyzer)
   - Focus: Report formatting, executive summaries, final documentation
   - Does NOT: Fetch data or perform technical analysis

IMPORTANT: Each agent should ONLY perform their designated role. Do not attempt to do another agent's job - instead, hand off to the appropriate specialist when needed.
"""

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
        "will help where you left off. "
        # "Execute what you can to make progress."
    )
    
    # Add the comprehensive team description
    team_context = f"\n\n{AGENTS_DESCRIPTION}"
    
    # Add specific role description
    role_context = f"\n\nYOUR SPECIFIC ROLE:\n{role_description}"
    
    if possible_handoffs:
        agent_list = ", ".join(possible_handoffs)
        handoff_instructions = (
            f"\n\nWhen you need to hand off to another agent, write 'Handoff to [AGENT_NAME]' "
            f"where AGENT_NAME is one of: {agent_list}. "
            f"If your work is complete, end your response with 'FINAL'."
        )
    else:
        handoff_instructions = "\n\nIf your work is complete, end your response with 'FINAL'."
    
    return f"{base_prompt}{team_context}{role_context}{handoff_instructions}"