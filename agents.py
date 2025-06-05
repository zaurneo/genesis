from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_core.tools import FunctionTool
from clients import model_client_gpt4o as model_client
from autogen_agentchat.base import Handoff
from genesis.utils.tools import *
import config
from prompts import *
from utils.common import *

# assign_task, check_progress, validate_completion, update_task_status, 
# generate_html_report, load_stock_data, clean_and_prepare_data, 
# create_visualization, generate_data_report, train_prediction_model, 
# make_predictions, optimize_model, get_feature_importance, evaluate_model_performance, 
# backtest_strategy, validate_predictions, generate_test_report, check_data_quality, 
# verify_model_outputs, assess_compliance, generate_quality_report

code_execution_agent = AssistantAgent(
    name="CodeExecutionAgent",
    model_client=model_client,
    tools = [],
    system_message="THIS AGENT IS ONLY USED FOR EXECUTING CODE. DO NOT USE THIS AGENT FOR ANYTHING ELSE.",
    code_execution_config={"work_dir": config.WORK_DIR},
)

function_calling_agent = AssistantAgent(
    name="FunctionCallingAgent",
    system_message=FUNCTION_CALLING_AGENT_SYSTEM_PROMPT,
    tools=[read_file,
        read_multiple_files,
        read_directory_contents,
        save_file,
        save_multiple_files,
        execute_code_block,
        consult_archive_agent
    ],
    model_client=model_client,
)


user_proxy = UserProxyAgent(
    name = "user_proxy", 
    input_func=input)

user_handoff_agent = AssistantAgent(
    name = "user_handoff_agent", 
    system_message=USER_PROXY_SYSTEM_PROMPT,
    handoffs=[Handoff(target="user", message="Transfer to user.")],
    model_client=model_client,)


code_reviewer = AssistantAgent(
    name="CodeReviewer",
    system_message="""You are an expert at reviewing code and suggesting improvements. 
    Pay particluar attention to any potential syntax errors. 
    Also, remind the Coding agent that they should always provide FULL and COMPLILABLE code and not shorten code blocks with comments such as '# Other class and method definitions remain unchanged...' or '# ... (previous code remains unchanged)'.""",
    model_client=model_client,
)

agent_awareness_expert = AssistantAgent(
    name="AgentAwarenessExpert",
    system_message=AGENT_AWARENESS_SYSTEM_PROMPT,
    model_client=model_client,
)

python_expert = AssistantAgent(
    name="PythonExpert",
    model_client=model_client,
    system_message=PYTHON_EXPERT_SYSTEM_PROMPT,
)


innovative_thinker_agent = AssistantAgent(
    name="InnovativeThinkerAgent",
    system_message=INNOVATIVE_THINKER_SYSTEM_PROMPT,
    model_client=model_client,
)

agi_gestalt_agent = AssistantAgent(
    name="AGIGestaltAgent",
    system_message=AGI_GESTALT_SYSTEM_PROMPT,
    model_client=model_client,
)

project_strategy_manager_agent = AssistantAgent(
    name="ProjectStrategyManagerAgent",
    system_message=PROJECT_STRATEGY_MANAGER_SYSTEM_PROMPT,
    model_client=model_client,
)

first_principles_thinker_agent = AssistantAgent(
    name="FirstPrinciplesThinkerAgent",
    system_message=FIRST_PRINCIPLES_THINKER_SYSTEM_PROMPT,
    model_client=model_client,
)


emotional_intelligence_expert_agent = AssistantAgent(
    name="EmotionalIntelligenceExpertAgent",
    system_message=EMOTIONAL_INTELLIGENCE_EXPERT_SYSTEM_PROMPT,
    model_client=model_client,
)

efficiency_optimizer_agent = AssistantAgent(
    name="EfficiencyOptimizerAgent",
    system_message=EFFICIENCY_OPTIMIZER_SYSTEM_PROMPT,
    model_client=model_client,
)

task_history_review_agent = AssistantAgent(
    name="TaskHistoryReviewAgent",
    system_message=TASK_HISTORY_REVIEW_AGENT_SYSTEM_PROMPT,
    model_client=model_client,
)

task_comprehension_agent = AssistantAgent(
    name="TaskComprehensionAgent",
    system_message=TASK_COMPREHENSION_AGENT_SYSTEM_PROMPT,
    model_client=model_client,
)



project_owner = AssistantAgent(
    name="Project_Owner",
    model_client=model_client,
    tools=[
        FunctionTool(assign_task, description="Assign tasks to specific agents"),
        FunctionTool(check_progress, description="Check overall project progress"),
        FunctionTool(validate_completion, description="Validate if project is ready for completion"),
        FunctionTool(update_task_status, description="Update task status"),
        FunctionTool(start_report_phase, description="Begin final report phase"),
    ],
    system_message=PROJECT_OWNER_PROMPT
)

data_engineer = AssistantAgent(
    name="Data_Engineer",
    model_client=model_client,
    tools=[
        FunctionTool(load_stock_data, description="Load stock data from yfinance"),
        FunctionTool(clean_and_prepare_data, description="Clean and prepare data for modeling"),
        FunctionTool(create_visualization, description="Create stock analysis charts"),
        FunctionTool(generate_data_report, description="Generate data analysis report")
    ],
    system_message=DATA_ENGINEER_PROMPT
)

model_executor = AssistantAgent(
    name="Model_Executor",
    model_client=model_client,
    tools=[
        FunctionTool(train_prediction_model, description="Train stock prediction models"),
        FunctionTool(make_predictions, description="Make future stock predictions"),
        FunctionTool(optimize_model, description="Optimize model parameters"),
        FunctionTool(get_feature_importance, description="Extract feature importance from models")
    ],
    system_message=MODEL_EXECUTOR_PROMPT
)

model_tester = AssistantAgent(
    name="Model_Tester",
    model_client=model_client,
    tools=[
        FunctionTool(evaluate_model_performance, description="Evaluate model with metrics"),
        FunctionTool(backtest_strategy, description="Backtest trading strategy based on predictions"),
        FunctionTool(validate_predictions, description="Validate prediction quality"),
        FunctionTool(generate_test_report, description="Generate comprehensive test report")
    ],
    system_message=MODEL_TESTER_PROMPT
)



quality_assurance = AssistantAgent(
    name="Quality_Assurance",
    model_client=model_client,
    tools=[
        FunctionTool(check_data_quality, description="Check quality of all data files"),
        FunctionTool(verify_model_outputs, description="Verify model outputs are valid"),
        FunctionTool(assess_compliance, description="Assess compliance with requirements"),
        FunctionTool(generate_quality_report, description="Generate final QA report")
    ],
    system_message=QUALITY_ASSURANCE_PROMPT
)

report_insight_generator = AssistantAgent(
    name="Report_Insight_Generator",
    model_client=model_client,
    tools=[
        FunctionTool(generate_html_report, description="Compile final investor HTML report"),
    ],
    system_message=REPORT_INSIGHT_GENERATOR_PROMPT
    )



