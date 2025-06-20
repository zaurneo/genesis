import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()

gpt_api_key = os.environ.get("OPENAI_API_KEY", "")
model_gpt_4o_mini = ChatOpenAI(model="gpt-4o-mini", api_key=gpt_api_key)