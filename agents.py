from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool
from clients import model_client_gpt4o as model_client
from tools import *
from prompts import (
    PROJECT_OWNER_PROMPT,
    DATA_ENGINEER_PROMPT,
    MODEL_EXECUTOR_PROMPT,
    MODEL_TESTER_PROMPT,
    QUALITY_ASSURANCE_PROMPT,
    REPORT_INSIGHT_GENERATOR_PROMPT,
)
# assign_task, check_progress, validate_completion, update_task_status, 
# generate_html_report, load_stock_data, clean_and_prepare_data, 
# create_visualization, generate_data_report, train_prediction_model, 
# make_predictions, optimize_model, get_feature_importance, evaluate_model_performance, 
# backtest_strategy, validate_predictions, generate_test_report, check_data_quality, 
# verify_model_outputs, assess_compliance, generate_quality_report

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



