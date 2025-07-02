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
from tools.logs.logging_helpers import (
    setup_logging, log_info, log_success, log_warning, 
    log_error, log_progress, safe_run
)

# Import event tracking for debugging
try:
    from events import diagnose_missing_sections, get_event_stream, Event, EventType
    _events_available = True
except ImportError:
    _events_available = False

@safe_run
def main():
    """Main execution function with human-in-the-loop support."""
    
    # Initialize logging
    setup_logging(level="INFO")
    log_info("Starting Genesis Multi-Agent Stock Analysis System")
    
    # Initialize components
    checkpointer = InMemorySaver()
    store = InMemoryStore()
    
    session_id = uuid.uuid4()
    session_id_str = str(session_id)
    config = {
        "configurable": {"thread_id": session_id_str},
        "recursion_limit": 150
    }
    
    # Publish supervisor start event
    if _events_available:
        event_stream = get_event_stream()
        event_stream.publish(Event(
            type=EventType.AGENT_STARTED,
            agent_id='supervisor',
            session_id=session_id_str,
            data={'workflow': 'Genesis Multi-Agent Stock Analysis'}
        ))
    
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
    
    # Keep UI prints for user interaction
    print("🤖 Genesis Multi-Agent Stock Analysis System")
    print("=" * 50)
    print("Welcome! I can help you analyze stocks using multiple AI agents.")
    print("Type 'exit', 'quit', or 'bye' to end the session.")
    print(f"Session ID: {session_id}")
    print("=" * 50)
    
    log_info("UI initialized and ready for user interaction")
    
    # ENHANCED initial query showcasing new multi-model backtesting capabilities
    initial_query = """Get Apple stock data, apply technical indicators, then use the enhanced scalable ML system with multi-model capabilities to:

1. Get AI-assisted model parameter recommendations using get_model_selection_guide for short-term trading
2. Train multiple models using different algorithms (XGBoost, Random Forest) with various parameter configurations
3. Use the new backtest_multiple_models tool to compare all models simultaneously
4. Identify best performers across different criteria (return, Sharpe ratio, drawdown)
5. Create comprehensive multi-model comparison visualizations showing:
   - Performance rankings across all models
   - Parameter sensitivity analysis
   - Risk-return scatter plots
   - Model type effectiveness comparison
6. Generate a comprehensive HTML report with all findings and model comparison insights

Then transfer_to_human for more questions about the enhanced multi-model ML capabilities."""
    
    print(f"\n🚀 Starting with enhanced multi-model ML showcase: '{initial_query}'")
    log_info(f"Processing initial query: {initial_query[:100]}...")
    
    # Initialize the conversation
    inputs = {
        "messages": [
            {
                "role": "user", 
                "content": f"Processing Enhanced Multi-Model ML Query: '{initial_query}'"
            }         
        ]
    }
    
    while True:
        try:
            # Run the graph until interruption or completion
            print("\n" + "="*50)
            print(" Running enhanced multi-model agent workflow...")
            print("="*50)
            log_progress("Running agent workflow")
            
            for chunk in graph.stream(inputs, config=config, stream_mode="updates"):
                print(f"\n🤖 Agent Update:")
                for node_name, messages in chunk.items():
                    print(f"📍 Node: {node_name}")
                    log_info(f"Processing node: {node_name}")
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
                print("The enhanced multi-model ML system is ready for your questions or instructions.")
                print("You can ask about:")
                print("  • Enhanced scalable ML model training with XGBoost and Random Forest")
                print("  • Multi-model backtesting and performance comparison")
                print("  • AI-assisted parameter selection and validation")
                print("  • Model ranking and selection based on different criteria")
                print("  • Parameter sensitivity analysis across model types")
                print("  • Risk-return optimization and model effectiveness")
                print("  • Advanced visualization of model comparison results")
                print("  • Zero-duplication architecture benefits and scalability")
                print("  • Best model identification for different trading strategies")
                print("  • Any stock analysis for any symbol with multi-model approach")
                print("Type 'exit', 'quit', or 'bye' to end the session.")
                print("-" * 50)
                
                # Get user input
                try:
                    user_input = input("\n👤 Your question or request: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n\n👋 Session ended by user. Goodbye!")
                    log_info("Session ended by user (EOF/KeyboardInterrupt)")
                    break
                
                # Check for exit commands
                if user_input.lower() in ['exit', 'quit', 'bye', '']:
                    print("\n👋 Thank you for using Genesis Enhanced Multi-Model ML System! Goodbye!")
                    log_info("User exited normally")
                    break
                
                # Continue the conversation with user input
                print(f"\n📝 Processing your request: '{user_input}'")
                log_info(f"User input received: {user_input}")
                
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
                print("✅ ENHANCED MULTI-MODEL ML WORKFLOW COMPLETED")
                print("="*50)
                log_success("Workflow completed successfully")
                
                # Publish supervisor completion event
                if _events_available:
                    event_stream.publish(Event(
                        type=EventType.AGENT_COMPLETED,
                        agent_id='supervisor',
                        session_id=session_id_str,
                        data={'status': 'workflow_completed', 'result': 'success'}
                    ))
                print("The enhanced multi-model analysis is complete. You can ask additional questions about:")
                print("  • Model performance comparisons and rankings")
                print("  • Parameter optimization insights across model types")
                print("  • Backtesting strategy effectiveness analysis")
                print("  • Model performance analysis and parameter optimization")
                print("  • Multi-model ensemble analysis and consensus insights")
                print("  • Risk-return optimization for different trading goals")
                print("  • Scalable architecture benefits and model comparison efficiency")
                print("  • Best model selection for specific market conditions")
                
                # Get user input for next action
                try:
                    user_input = input("\n👤 Next question or request (or 'exit' to quit): ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n\n👋 Session ended by user. Goodbye!")
                    log_info("Session ended by user (EOF/KeyboardInterrupt)")
                    break
                
                # Check for exit commands
                if user_input.lower() in ['exit', 'quit', 'bye', '']:
                    print("\n👋 Thank you for using Genesis Enhanced Multi-Model ML System! Goodbye!")
                    log_info("User exited normally")
                    break
                
                # Continue in same session but restart workflow with full context
                print(f"\n📝 Processing your new request: '{user_input}'")
                log_info(f"Processing new request: {user_input}")
                
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
            print("\n\n  Process interrupted by user. Goodbye!")
            log_info("Session interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")
            print("You can try asking another question or type 'exit' to quit.")
            log_error(f"Error in main workflow: {str(e)}", exc_info=True)
            
            # Publish supervisor failure event
            if _events_available:
                event_stream.publish(Event(
                    type=EventType.AGENT_FAILED,
                    agent_id='supervisor',
                    session_id=session_id_str,
                    error=str(e),
                    data={'status': 'workflow_failed'}
                ))
            
            # Show diagnostic information if available
            if _events_available:
                session_id = str(config.get('configurable', {}).get('thread_id', ''))
                if session_id:
                    print("\n🔍 DEBUG: Workflow Diagnostic Report")
                    print("-" * 40)
                    print(diagnose_missing_sections(session_id))
                    print("-" * 40)
            
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
                log_info("Session ended during error recovery")
                break

if __name__ == "__main__":
    main()