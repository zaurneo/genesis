from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
import asyncio
from agents.agents import project_owner, data_engineer, model_executor, model_tester, quality_assurance
from clients.clients import model_client_gpt4o as model_client
from autogen_agentchat.conditions import TextMentionTermination
import config

project_path = config.GENERATED_FILES_DIR

task = (
    f"Write a financial report on American Airlines."
    f"Save the final report in HTML file with clear explanations and visualizations for investors."
    f"All new files should be saved in `{project_path}`."
)


async def main():
    text_termination = TextMentionTermination("GENESIS COMPLETED")
    team = RoundRobinGroupChat([project_owner, data_engineer, model_executor, 
                                # model_tester, quality_assurance
                                ], 
                               # max_turns=3, 
                               termination_condition=text_termination
                           )
    stream = team.run_stream(task=task)
    await Console(stream)
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
