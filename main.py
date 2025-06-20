import getpass
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, StateGraph, START
from agents import GraphState, generate, code_check, reflect, code
from handoffs import decide_to_finish
from prompts import code_gen_prompt, code_gen_prompt_claude
from tools import check_claude_output, insert_errors, parse_output
from rag import concatenated_content
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()


gpt_api_key = os.environ.get("gpt_api_key", "")
# model = ChatOpenAI(model="gpt-4o-mini", api_key=gpt_api_key)

# LLM setup
expt_llm = "gpt-4o-mini"
llm = ChatOpenAI(temperature=0, model=expt_llm, api_key=gpt_api_key)
code_gen_chain_oai = code_gen_prompt | llm.with_structured_output(code)

# Anthropic setup
# expt_llm = "claude-3-opus-20240229"
# llm = ChatAnthropic(
#     model=expt_llm,
#     default_headers={"anthropic-beta": "tools-2024-04-04"},
# )
structured_llm_claude = llm.with_structured_output(code, include_raw=True)

# Chain with output check
code_chain_claude_raw = (
    code_gen_prompt_claude | structured_llm_claude | check_claude_output
)

# This will be run as a fallback chain
fallback_chain = insert_errors | code_chain_claude_raw
N = 3  # Max re-tries
code_gen_chain_re_try = code_chain_claude_raw.with_fallbacks(
    fallbacks=[fallback_chain] * N, exception_key="error"
)

# No re-try
code_gen_chain = code_gen_prompt_claude | structured_llm_claude | parse_output

# Make these available to agents
import agents
agents.code_gen_chain = code_gen_chain
agents.concatenated_content = concatenated_content

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("generate", generate)  # generation solution
workflow.add_node("check_code", code_check)  # check code
workflow.add_node("reflect", reflect)  # reflect

# Build graph
workflow.add_edge(START, "generate")
workflow.add_edge("generate", "check_code")
workflow.add_conditional_edges(
    "check_code",
    decide_to_finish,
    {
        "end": END,
        "reflect": "reflect", 
        "generate": "generate",
    },
)
workflow.add_edge("reflect", "generate")

app = workflow.compile()

# Example usage
question = "How can I directly pass a string to a runnable and use it to construct the input needed for my prompt?"
solution = app.invoke({"messages": [("user", question)], "iterations": 0, "error": ""})
print(solution["generation"])