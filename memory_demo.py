#!/usr/bin/env python3
"""
Demo script showing memory handling across multiple sessions.
"""

import uuid
from langchain_core.messages import HumanMessage
from agents import *
from langgraph.graph import MessagesState, END, StateGraph, START

def run_session_demo():
    """Demonstrate memory handling with multiple sessions."""
    
    # Create the workflow
    workflow = StateGraph(MessagesState)
    workflow.add_node("stock_data_fetcher", stock_data_node)
    workflow.add_node("stock_analyzer", stock_analyzer_node)
    workflow.add_node("stock_reporter", stock_reporter_node)
    workflow.add_edge(START, "stock_data_fetcher")
    graph = workflow.compile()
    
    # Session 1: Apple analysis
    print("🟢 SESSION 1: Apple Stock Analysis")
    print("=" * 50)
    session1_id = uuid.uuid4()
    config1 = {
        "configurable": {"session_id": session1_id},
        "recursion_limit": 50
    }
    
    events1 = graph.stream(
        {"messages": [HumanMessage(content="Analyze Apple stock performance")]},
        config=config1,
        stream_mode="values"
    )
    
    step_count = 0
    for event in events1:
        step_count += 1
        print(f"📝 Step {step_count}:")
        if "messages" in event and event["messages"]:
            if isinstance(event["messages"], list):
                event["messages"][-1].pretty_print()
            else:
                event["messages"].pretty_print()
        print("\n" + "-" * 30 + "\n")
    
    print("\n🟡 SESSION 2: Tesla Stock Analysis")
    print("=" * 50)
    session2_id = uuid.uuid4()
    config2 = {
        "configurable": {"session_id": session2_id},
        "recursion_limit": 50
    }
    
    events2 = graph.stream(
        {"messages": [HumanMessage(content="Analyze Tesla stock performance")]},
        config=config2,
        stream_mode="values"
    )
    
    step_count = 0
    for event in events2:
        step_count += 1
        print(f"📝 Step {step_count}:")
        if "messages" in event and event["messages"]:
            if isinstance(event["messages"], list):
                event["messages"][-1].pretty_print()
            else:
                event["messages"].pretty_print()
        print("\n" + "-" * 30 + "\n")
    
    # Demonstrate memory persistence
    print("\n🔵 SESSION 1 CONTINUED: Follow-up question")
    print("=" * 50)
    
    # Continue session 1 - memory should remember previous Apple analysis
    followup_config = {
        "configurable": {"session_id": session1_id},
        "recursion_limit": 30
    }
    
    followup_events = graph.stream(
        {"messages": [HumanMessage(content="What was the previous stock I asked you to analyze?")]},
        config=followup_config,
        stream_mode="values"
    )
    
    step_count = 0
    for event in followup_events:
        step_count += 1
        print(f"📝 Step {step_count}:")
        if "messages" in event and event["messages"]:
            if isinstance(event["messages"], list):
                event["messages"][-1].pretty_print()
            else:
                event["messages"].pretty_print()
        print("\n" + "-" * 30 + "\n")
    
    print("✅ Demo Complete!")
    print(f"💾 Session 1 ID: {session1_id}")
    print(f"💾 Session 2 ID: {session2_id}")
    print(f"📊 Total sessions in memory: {len(chats_by_session_id)}")

if __name__ == "__main__":
    run_session_demo()