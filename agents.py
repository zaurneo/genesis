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
market_analyzer = create_react_agent(
    model=model,
    tools=[analyze_stock, get_market_trends, transfer_to_portfolio_manager],
    prompt=PROMPT_A,
    name="market_analyzer"
)
portfolio_manager = create_react_agent(
    model=model,
    tools=[calculate_portfolio_risk, recommend_portfolio_allocation, transfer_to_market_analyzer],
    prompt=PROMPT_B,
    name="portfolio_manager"
)