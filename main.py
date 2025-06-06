from autogen_agentchat.teams import RoundRobinGroupChat
from teams import FlexibleHandoffGroupChat
from autogen_agentchat.ui import Console
import asyncio
from agents import *
from clients import model_client_gpt4o as model_client
from autogen_agentchat.conditions import TextMentionTermination, HandoffTermination
import config

# from utils.tools import register_team
# from utils.common import log_stream


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


agents = [
                                
                                data_engineer, 
                                model_executor, 
                                model_tester, 
                                quality_assurance,
                                user_proxy,
                                user_handoff_agent,
                                code_reviewer,
                                agent_awareness_expert, 
                                python_expert,
                                innovative_thinker_agent,
                                agi_gestalt_agent,
                                project_strategy_manager_agent,
                                first_principles_thinker_agent,
                                emotional_intelligence_expert_agent,
                                efficiency_optimizer_agent,
                                task_history_review_agent,
                                task_comprehension_agent,
                                report_insight_generator]

async def main():
    import logging
    logging.getLogger("autogen").setLevel(logging.WARNING)
    logging.getLogger("autogen_agentchat").setLevel(logging.WARNING)
    
    text_termination = TextMentionTermination("GENESIS COMPLETED")
    handoff_termination = HandoffTermination(target="user")
    
    # Show output mode options
    print("\n" + "="*60)
    print("🎨 OUTPUT MODE OPTIONS:")
    print("="*60)
    print("1. 'full'      - Full formatted agent messages")
    print("2. 'formatted' - Nicely boxed agent messages (default)")
    print("3. 'summary'   - One-line summaries of agent work")
    print("4. 'minimal'   - Only show turn tracking")
    print("="*60 + "\n")

    owner = project_owner

    
    team = FlexibleHandoffGroupChat(
        owner=owner,
        agents=agents,
        max_agent_turns=4,  # Owner intervenes after 4 agent exchanges
        # report_agent=report_insight_generator,  # Optional: designate report agent
        tasks=task,  # Pass the tasks
        format_agent_output=True,  # Enable formatted output
        show_agent_messages=True,  # Show agent messages
        termination_condition=text_termination,
    )

    

    team.set_output_mode('summary') # or formatted



    # register_team(team)
    stream = team.run_stream(task=task)
    # logged_stream = log_stream(stream)
    # await Console(logged_stream)
    await Console(stream)
    await model_client.close()


    team.export_turn_logs(f"{project_path}/turn_logs.txt")

    # Print summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(team.get_turn_logs_summary())
    print(f"\nTotal owner interventions: {team.owner_intervention_count}")
    print(f"Turn logs saved to: {project_path}/turn_logs.txt")



if __name__ == "__main__":
    asyncio.run(main())
