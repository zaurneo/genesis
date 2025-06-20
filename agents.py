from prompts import *
from tools import *
from handoffs import *
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()
from langgraph.prebuilt import create_react_agent, InjectedState

gpt_api_key = os.environ.get("gpt_api_key", "")
model = ChatOpenAI(model="gpt-4o-mini", api_key=gpt_api_key)



# Define agents
tech_lead = create_react_agent(
    model=model,
    tools=[review_code, assign_task, transfer_to_writer, transfer_to_executor],
    prompt=PROMPT_TECH_LEAD,
    name="tech_lead"
)
writer = create_react_agent(
    model=model,
    tools=[write_code, refactor_code, transfer_to_tech_lead, transfer_to_executor],
    prompt=PROMPT_WRITER,
    name="writer"
)
executor = create_react_agent(
    model=model,
    tools=[execute_code, write_test, transfer_to_tech_lead, transfer_to_writer],
    prompt=PROMPT_EXECUTOR,
    name="executor"
)