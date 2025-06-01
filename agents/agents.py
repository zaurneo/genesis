from autogen_agentchat.agents import AssistantAgent
from tools.google_search import google_search
from tools.stock_analysis import analyze_stock
from autogen_core.tools import FunctionTool
from clients.clients import model_client_gpt4o as model_client

google_search_tool = FunctionTool(google_search, description="Search Google for company info")
stock_analysis_tool = FunctionTool(analyze_stock, description="Analyze stock and create plot")


search_agent = AssistantAgent(
    name="Google_Search_Agent",
    model_client=model_client,
    tools=[google_search_tool],
    system_message="You are a helpful assistant who searches for company info."
)

stock_agent = AssistantAgent(
    name="Stock_Analysis_Agent",
    model_client=model_client,
    tools=[stock_analysis_tool],
    system_message="You are a helpful assistant who analyzes company stock."
)

report_agent = AssistantAgent(
    name="Report_Agent",
    model_client=model_client,
    system_message="You generate a company research report based on the other agents' results."
)














planner_agent = AssistantAgent(
    name = "planner",
    model_client=model_client,
    system_message=(
        "You are the Planner and Moderator for this project. You are the ONLY agent authorized to declare project completion. "
        "Your role is to coordinate tasks, ensure alignment between agents, and track overall progress. "
        "Ensure all agents collaborate effectively, resolve conflicts if they arise, and validate task completions. "
        "Provide updates and summarize progress periodically by stating 'UPDATE' and 'SUMMARY' appropirately."
        
        "Begin by outlining a clear project plan for ..."
        "Only declare 'PROJECT COMPLETED' when you have verified ... are successfully finalized and integrated ..."
    ),
)





# Create the scientist/engineer agent.
scientist_agent = AssistantAgent(
    "scientist",
    model_client=model_client,
    system_message=(
        "You are a meticulous scientist tasked with creating a clear, actionable plan for identifying and ranking citations related to CRISPR-Cas9 gene therapy. "
        "Once your plan is ready, clearly indicate 'PLAN COMPLETE' to signal that your part is done. Avoid any unnecessary follow-up after submitting the plan."
    ),
)

# Create the researcher agent.
researcher_agent = AssistantAgent(
    "researcher",
    model_client=model_client,
    system_message=(
        "You are a focused researcher responsible for executing the plan provided by the scientist agent. "
        "Locate, reference, and compile citations in a structured manner. "
        "Once you have completed the task, clearly indicate 'SEARCH COMPLETE' to signal your part is done and stop further interactions."
    ),
)

# Create the critic agent.
critic_agent = AssistantAgent(
    "critic",
    model_client=model_client,
    system_message=(
        "You are a strict reviewer validating the accuracy and relevance of citations provided by the researcher. "
        "Ensure all citations meet the highest scientific standards. Once validation is complete, clearly indicate 'VALIDATION COMPLETE'. "
        "Avoid unnecessary follow-up once the task is finalized."
    ),
)











# Sample
agent = AssistantAgent(
    name="weather_agent",
    model_client=model_client,
    tools=[],
    system_message="You are a helpful assistant.",
    reflect_on_tool_use=True,
    model_client_stream=True,  # Enable streaming tokens from the model client.
)