from autogen_agentchat.teams import SelectorGroupChat
from teams import FlexibleHandoffGroupChat
from autogen_agentchat.ui import Console
import asyncio
from agents import *
from clients import model_client_gpt4o as model_client
from autogen_agentchat.conditions import TextMentionTermination, HandoffTermination
import config


# from utils.tools import register_team
from utils.common import log_stream


project_path = config.GENERATED_FILES_DIR

# task = (
#     f"Write a financial report on American Airlines."
#     f"Save the final investment report in HTML file with clear explanations and visualizations for investors."
#     f"Investor report should be clear for investor to understand with good explanations of results, analysis and visualizations."
#     f"All new files should be saved in `{project_path}`."
# )

task = (
    f"Write a report about American Airlines"
)


selected_agents = [
        data_engineer,
        python_expert,
        innovative_thinker_agent,
        model_executor
    ]

async def main():
    import logging
    logging.getLogger("autogen").setLevel(logging.WARNING)
    logging.getLogger("autogen_agentchat").setLevel(logging.WARNING)
    
    text_termination = TextMentionTermination("GENESIS COMPLETED")
    handoff_termination = HandoffTermination(target="user")
    
    print("\n" + "="*60)
    print("🎯 DIRECT AUTOGEN SELECTOR CHAT")
    print("="*60)
    print("Using built-in autogen classes only - no custom wrappers!")
    print("="*60 + "\n")


    print("👑 Owner:", getattr(project_owner, 'name', str(project_owner)))
    print("👥 Agents:")
    for i, agent in enumerate(selected_agents, 1):
        print(f"   {i}. {getattr(agent, 'name', str(agent))}")
    print()

    participants = [project_owner] + selected_agents


    selector_prompt = """Select an agent to perform task.

    {roles}

    Current conversation context:
    {history}

    Read the above conversation, then select an agent from {participants} to perform the next task.
    Make sure the planner agent has assigned tasks before other agents start working.
    Only select one agent.
    """

  
    team = SelectorGroupChat(
            participants=participants,
            model_client=model_client,
            termination_condition=text_termination,
            selector_prompt=selector_prompt,
            allow_repeated_speaker=True,  # Allow an agent to speak multiple turns in a row.
        )





    # register_team(team)
    stream = team.run_stream(task=task)
    logged_stream = log_stream(stream)
    await Console(logged_stream)
    await model_client.close()


    team.export_turn_logs(f"{project_path}/turn_logs.txt")

    # Print summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(team.get_turn_logs_summary())
    print(f"\nTotal owner interventions: {team.owner_intervention_count}")
    print(f"Turn logs saved to: {project_path}/turn_logs.txt")
    print(f"Conversation log saved to: {project_path}/conversation_log.jsonl")



if __name__ == "__main__":
    asyncio.run(main())
