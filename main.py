#!/usr/bin/env python3
"""
Multi-agent collaboration main execution script with supervisor and human-in-the-loop.
"""
import uuid
from supervisor.supervisor import create_supervisor
from agents import *
from models import model_gpt_4o_mini
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from prompts import SUPERVISOR_PROMPT

def main():
    """Main execution function with human-in-the-loop support."""
    
    # Initialize components
    checkpointer = InMemorySaver()
    store = InMemoryStore()
    
    session_id = uuid.uuid4()
    config = {
        "configurable": {"thread_id": str(session_id)},
        "recursion_limit": 150
    }
    
    # Create supervisor workflow
    workflow = create_supervisor(
        [stock_data_agent, stock_analyzer_agent, stock_reporter_agent],
        model=model_gpt_4o_mini,
        output_mode="full_history",
        prompt=SUPERVISOR_PROMPT,
        human_proxy="human"
    )
    
    # Compile the workflow with human interruption
    graph = workflow.compile(
        checkpointer=checkpointer,
        store=store,
        interrupt_before=["human"]
    )
    
    print("🤖 Genesis Multi-Agent Stock Analysis System")
    print("=" * 50)
    print("Welcome! I can help you analyze stocks using multiple AI agents.")
    print("Type 'exit', 'quit', or 'bye' to end the session.")
    print("=" * 50)
    
    # Initial query to start the system
    initial_query = """Get Apple stock data, create technical indicators, tain random forest model, backtest its results, analyze its performance, and create a summary report.
    then transfer_to_human for more questions"""
    print(f"\n🚀 Starting with initial query: '{initial_query}'")
    
    # Initialize the conversation
    inputs = {
        "messages": [
            {
                "role": "user", 
                "content": f"Processing Query: '{initial_query}'"
            }         
        ]
    }
    
    while True:
        try:
            # Run the graph until interruption or completion
            print("\n" + "="*50)
            print("🔄 Running agent workflow...")
            print("="*50)
            
            for chunk in graph.stream(inputs, config=config, stream_mode="updates"):
                print(f"\n🤖 Agent Update:")
                for node_name, messages in chunk.items():
                    print(f"📍 Node: {node_name}")
                    if isinstance(messages, dict) and "messages" in messages:
                        for msg in messages["messages"]:
                            if hasattr(msg, 'pretty_print'):
                                msg.pretty_print()
                            else:
                                print(f"  {msg}")
                    print("-" * 30)
            
            # Check if we've reached an interruption point
            current_state = graph.get_state(config)
            
            if current_state.next == ("human",):
                # We're at the human node - get user input
                print("\n" + "="*50)
                print("💬 HUMAN INPUT REQUIRED")
                print("="*50)
                print("The system is ready for your questions or instructions.")
                print("You can ask about:")
                print("  • Stock analysis for any symbol")
                print("  • Technical indicators and charts")
                print("  • Machine learning predictions")
                print("  • Backtesting strategies")
                print("  • Comprehensive reports")
                print("Type 'exit', 'quit', or 'bye' to end the session.")
                print("-" * 50)
                
                # Get user input
                try:
                    user_input = input("\n👤 Your question or request: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n\n👋 Session ended by user. Goodbye!")
                    break
                
                # Check for exit commands
                if user_input.lower() in ['exit', 'quit', 'bye', '']:
                    print("\n👋 Thank you for using Genesis! Goodbye!")
                    break
                
                # Continue the conversation with user input
                print(f"\n📝 Processing your request: '{user_input}'")
                
                # Update the existing state with human response
                graph.update_state(config, {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_input
                        }
                    ]
                })
                
                # Continue with existing conversation state
                inputs = None
                
            else:
                # No interruption - workflow completed
                print("\n" + "="*50)
                print("✅ WORKFLOW COMPLETED")
                print("="*50)
                print("The analysis is complete. You can ask additional questions or exit.")
                
                # Get user input for next action
                try:
                    user_input = input("\n👤 Next question or request (or 'exit' to quit): ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n\n👋 Session ended by user. Goodbye!")
                    break
                
                # Check for exit commands
                if user_input.lower() in ['exit', 'quit', 'bye', '']:
                    print("\n👋 Thank you for using Genesis! Goodbye!")
                    break
                
                # Continue in same session but restart workflow with full context
                print(f"\n📝 Processing your new request: '{user_input}'")
                
                # Get current conversation history and add new message
                current_state = graph.get_state(config)
                all_messages = current_state.values.get("messages", [])
                all_messages.append({
                    "role": "user",
                    "content": user_input
                })
                
                # Restart with full conversation history
                inputs = {
                    "messages": all_messages
                }
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Process interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")
            print("You can try asking another question or type 'exit' to quit.")
            
            # Get user input to continue or exit
            try:
                user_input = input("\n👤 Try again or type 'exit': ").strip()
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    break
                
                # Continue with user input
                inputs = {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_input
                        }
                    ]
                }
            except (EOFError, KeyboardInterrupt):
                break

if __name__ == "__main__":
    main()