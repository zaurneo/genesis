from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_core.tools import FunctionTool
from clients import model_client_gpt4o as model_client
from autogen_agentchat.base import Handoff
# from utils.tools import *
import config
from prompts import *
# from utils.common import *

# assign_task, check_progress, validate_completion, update_task_status, 
# generate_html_report, load_stock_data, clean_and_prepare_data, 
# create_visualization, generate_data_report, train_prediction_model, 
# make_predictions, optimize_model, get_feature_importance, evaluate_model_performance, 
# backtest_strategy, validate_predictions, generate_test_report, check_data_quality, 
# verify_model_outputs, assess_compliance, generate_quality_report

# Executes arbitrary code blocks on behalf of other agents.
code_execution_agent = AssistantAgent(
    name="CodeExecutionAgent",
    description="Executes provided code blocks and returns the output",
    model_client=model_client,
    tools=[],
    system_message="THIS AGENT IS ONLY USED FOR EXECUTING CODE. DO NOT USE THIS AGENT FOR ANYTHING ELSE.",
)

# Handles autogen function calls for the team.
function_calling_agent = AssistantAgent(
    name="FunctionCallingAgent",
    description="Invokes registered functions when requested by other agents",
    system_message=FUNCTION_CALLING_AGENT_SYSTEM_PROMPT,
    tools=[
        # read_file,
        # read_multiple_files,
        # read_directory_contents,
        # save_file,
        # save_multiple_files,
        # execute_code_block,
        # consult_archive_agent
    ],
    model_client=model_client,
)


# Represents the human user in the conversation.
user_proxy = UserProxyAgent(
    name="user_proxy",
    description="Proxy agent that forwards prompts to the actual user",
    input_func=input)

# Routes messages directly to the user when required.
user_handoff_agent = AssistantAgent(
    name="user_handoff_agent",
    description="Handles handoffs from agents to the human user",
    system_message=USER_PROXY_SYSTEM_PROMPT,
    handoffs=[Handoff(target="user", message="Transfer to user.")],
    model_client=model_client,)


# Provides feedback on code quality and correctness.
code_reviewer = AssistantAgent(
    name="CodeReviewer",
    description="Reviews code snippets and suggests improvements",
    system_message="""You are an expert at reviewing code and suggesting improvements.
    Pay particluar attention to any potential syntax errors.
    Also, remind the Coding agent that they should always provide FULL and COMPLILABLE code and not shorten code blocks with comments such as '# Other class and method definitions remain unchanged...' or '# ... (previous code remains unchanged)'.""",
    model_client=model_client,
)

# Advises agents on team capabilities and available tools.
agent_awareness_expert = AssistantAgent(
    name="AgentAwarenessExpert",
    description="Guides agents on team capabilities and processes",
    system_message=AGENT_AWARENESS_SYSTEM_PROMPT,
    model_client=model_client,
)

# Provides guidance on Python programming tasks.
python_expert = AssistantAgent(
    name="PythonExpert",
    description="Expert consultant for Python development",
    model_client=model_client,
    system_message=PYTHON_EXPERT_SYSTEM_PROMPT,
)


# Generates creative solutions and approaches.
innovative_thinker_agent = AssistantAgent(
    name="InnovativeThinkerAgent",
    description="Agent focused on brainstorming novel ideas",
    system_message=INNOVATIVE_THINKER_SYSTEM_PROMPT,
    model_client=model_client,
)

# Synthesizes insights from multiple agents.
agi_gestalt_agent = AssistantAgent(
    name="AGIGestaltAgent",
    description="Combines results from various agents for holistic reasoning",
    system_message=AGI_GESTALT_SYSTEM_PROMPT,
    model_client=model_client,
)

# Oversees overall project planning and strategy.
project_strategy_manager_agent = AssistantAgent(
    name="ProjectStrategyManagerAgent",
    description="Coordinates project strategy and milestones",
    system_message=PROJECT_STRATEGY_MANAGER_SYSTEM_PROMPT,
    model_client=model_client,
)

# Breaks down problems using first-principles reasoning.
first_principles_thinker_agent = AssistantAgent(
    name="FirstPrinciplesThinkerAgent",
    description="Applies first-principles thinking to complex tasks",
    system_message=FIRST_PRINCIPLES_THINKER_SYSTEM_PROMPT,
    model_client=model_client,
)


# Monitors team communication tone and empathy.
emotional_intelligence_expert_agent = AssistantAgent(
    name="EmotionalIntelligenceExpertAgent",
    description="Ensures communications demonstrate emotional intelligence",
    system_message=EMOTIONAL_INTELLIGENCE_EXPERT_SYSTEM_PROMPT,
    model_client=model_client,
)

# Suggests workflow and code optimizations.
efficiency_optimizer_agent = AssistantAgent(
    name="EfficiencyOptimizerAgent",
    description="Optimizes team processes and resource usage",
    system_message=EFFICIENCY_OPTIMIZER_SYSTEM_PROMPT,
    model_client=model_client,
)

# Reviews past task assignments and outcomes.
task_history_review_agent = AssistantAgent(
    name="TaskHistoryReviewAgent",
    description="Analyzes previous tasks for lessons learned",
    system_message=TASK_HISTORY_REVIEW_AGENT_SYSTEM_PROMPT,
    model_client=model_client,
)

# Clarifies task requirements and objectives.
task_comprehension_agent = AssistantAgent(
    name="TaskComprehensionAgent",
    description="Ensures tasks are fully understood before execution",
    system_message=TASK_COMPREHENSION_AGENT_SYSTEM_PROMPT,
    model_client=model_client,
)



project_owner = AssistantAgent(
    # Central coordinator assigning work to other agents.
    name="Project_Owner",
    description="Manages tasks and monitors overall progress",
    model_client=model_client,
    tools=[
        # FunctionTool(assign_task, description="Assign tasks to specific agents"),
        # FunctionTool(check_progress, description="Check overall project progress"),
        # FunctionTool(validate_completion, description="Validate if project is ready for completion"),
        # FunctionTool(update_task_status, description="Update task status"),
        # FunctionTool(start_report_phase, description="Begin final report phase"),
    ],
    system_message=PROJECT_OWNER_PROMPT
)

data_engineer = AssistantAgent(
    # Handles data collection and preprocessing tasks.
    name="Data_Engineer",
    description="Prepares datasets for modeling and analysis",
    model_client=model_client,
    tools=[
        # FunctionTool(load_stock_data, description="Load stock data from yfinance"),
        # FunctionTool(clean_and_prepare_data, description="Clean and prepare data for modeling"),
        # FunctionTool(create_visualization, description="Create stock analysis charts"),
        # FunctionTool(generate_data_report, description="Generate data analysis report")
    ],
    system_message=DATA_ENGINEER_PROMPT
)

model_executor = AssistantAgent(
    # Runs predictive models on prepared data.
    name="Model_Executor",
    description="Executes statistical or ML models",
    model_client=model_client,
    tools=[
        # FunctionTool(train_prediction_model, description="Train stock prediction models"),
        # FunctionTool(make_predictions, description="Make future stock predictions"),
        # FunctionTool(optimize_model, description="Optimize model parameters"),
        # FunctionTool(get_feature_importance, description="Extract feature importance from models")
    ],
    system_message=MODEL_EXECUTOR_PROMPT
)

model_tester = AssistantAgent(
    # Validates models and checks prediction quality.
    name="Model_Tester",
    description="Tests and evaluates trained models",
    model_client=model_client,
    tools=[
        # FunctionTool(evaluate_model_performance, description="Evaluate model with metrics"),
        # FunctionTool(backtest_strategy, description="Backtest trading strategy based on predictions"),
        # FunctionTool(validate_predictions, description="Validate prediction quality"),
        # FunctionTool(generate_test_report, description="Generate comprehensive test report")
    ],
    system_message=MODEL_TESTER_PROMPT
)



quality_assurance = AssistantAgent(
    # Performs cross-agent quality checks.
    name="Quality_Assurance",
    description="Ensures outputs meet required standards",
    model_client=model_client,
    tools=[
        # FunctionTool(check_data_quality, description="Check quality of all data files"),
        # FunctionTool(verify_model_outputs, description="Verify model outputs are valid"),
        # FunctionTool(assess_compliance, description="Assess compliance with requirements"),
        # FunctionTool(generate_quality_report, description="Generate final QA report")
    ],
    system_message=QUALITY_ASSURANCE_PROMPT
)

report_insight_generator = AssistantAgent(
    # Generates the final investor report.
    name="Report_Insight_Generator",
    description="Compiles analysis results into an HTML report",
    model_client=model_client,
    tools=[
        # FunctionTool(generate_html_report, description="Compile final investor HTML report"),
    ],
    system_message=REPORT_INSIGHT_GENERATOR_PROMPT
    )



