from autogen_agentchat.teams import RoundRobinGroupChat
from teams import OwnerMediationGroupChat
from autogen_agentchat.ui import Console
import asyncio
from agents.agents import (
    project_owner,
    data_engineer,
    model_executor,
    model_tester,
    quality_assurance,
    report_insight_generator,
)
from clients.clients import model_client_gpt4o as model_client
from autogen_agentchat.conditions import TextMentionTermination
import config
from tools import register_team

project_path = config.GENERATED_FILES_DIR

task = (
    f"Write a financial report on American Airlines."
    f"Save the final investment report in HTML file with clear explanations and visualizations for investors."
    f"Investor report should be clear for investor to understand with good explanations of results, analysis and visualizations."
    f"All new files should be saved in `{project_path}`."
)


async def main():
    text_termination = TextMentionTermination("GENESIS COMPLETED")
    team = OwnerMediationGroupChat(
        project_owner,
        [data_engineer, model_executor, model_tester, quality_assurance],
        report_agent=report_insight_generator,
        termination_condition=text_termination,
    )
    register_team(team)
    stream = team.run_stream(task=task)
    await Console(stream)
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
