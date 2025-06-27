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
    
    # ENHANCED initial query showcasing new scalable ML capabilities
    initial_query = """Get Apple stock data, apply technical indicators, then use the enhanced scalable ML system to:

1. Get AI-assisted model parameter recommendations for short-term trading
2. Train multiple models (XGBoost, Random Forest, SVR) using the zero-duplication pipeline
3. Compare model performance and select the best approach
4. Backtest the top models with different trading strategies
5. Create comprehensive visualizations and analysis report

Then transfer_to_human for more questions about the enhanced ML capabilities."""
    
    print(f"\n🚀 Starting with enhanced ML showcase: '{initial_query}'")
    
    # Initialize the conversation
    inputs = {
        "messages": [
            {
                "role": "user", 
                "content": f"Processing Enhanced ML Query: '{initial_query}'"
            }         
        ]
    }
    
    while True:
        try:
            # Run the graph until interruption or completion
            print("\n" + "="*50)
            print("🔄 Running enhanced agent workflow...")
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
                print("The enhanced ML system is ready for your questions or instructions.")
                print("You can ask about:")
                print("  • Enhanced scalable ML model training")
                print("  • AI-assisted parameter selection and validation")
                print("  • Multi-model comparison and ensemble analysis")
                print("  • Advanced backtesting with multiple strategies")
                print("  • Model selection guides and recommendations")
                print("  • Zero-duplication architecture benefits")
                print("  • Any stock analysis for any symbol")
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
                    print("\n👋 Thank you for using Genesis Enhanced ML System! Goodbye!")
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
                print("✅ ENHANCED ML WORKFLOW COMPLETED")
                print("="*50)
                print("The enhanced analysis is complete. You can ask additional questions about:")
                print("  • Model performance comparisons")
                print("  • Parameter optimization insights")
                print("  • Backtesting strategy effectiveness")
                print("  • Adding new model types (SVR, Gradient Boosting, etc.)")
                print("  • Scalable architecture benefits")
                
                # Get user input for next action
                try:
                    user_input = input("\n👤 Next question or request (or 'exit' to quit): ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n\n👋 Session ended by user. Goodbye!")
                    break
                
                # Check for exit commands
                if user_input.lower() in ['exit', 'quit', 'bye', '']:
                    print("\n👋 Thank you for using Genesis Enhanced ML System! Goodbye!")
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