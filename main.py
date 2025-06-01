from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
import asyncio
from agents.agents import stock_agent, report_agent
from clients.clients import model_client_gpt4o as model_client
from autogen_agentchat.conditions import TextMentionTermination



async def main():
    # text_termination = TextMentionTermination("PROJECT COMPLETED")
    team = RoundRobinGroupChat([stock_agent, report_agent], max_turns=3, 
    #                           termination_condition=text_termination
                           )
    stream = team.run_stream(task="Write a financial report on American Airlines")
    await Console(stream)
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
