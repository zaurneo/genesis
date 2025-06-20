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
flight_assistant = create_react_agent(
    model=model,
    tools=[book_flight, transfer_to_hotel_assistant],
    prompt=PROMPT_A,
    name="flight_assistant"
)
hotel_assistant = create_react_agent(
    model=model,
    tools=[book_hotel, transfer_to_flight_assistant],
    prompt=PROMPT_B,
    name="hotel_assistant"
)