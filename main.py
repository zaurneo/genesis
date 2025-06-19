# main.py - FIXED VERSION with proper graph routing

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END, MessagesState

import functools
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END

# Import enhanced conversation viewer
from conversation_viewer import CodeDevelopmentViewer

# Import agents
try:
    from agents import architect, writer, executor, technical_lead, task_manager, docs, finalizer
    print("✅ All agents imported successfully")
except ImportError as e:
    print(f"❌ Error importing agents: {e}")
    sys.exit(1)

load_dotenv()

# Define state properly
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]

def agent_node(state: AgentState, agent, name: str):
    """Wrapper function for agents to work as nodes - FIXED VERSION"""
    try:
        result = agent.invoke(state)
        
        # Handle different return types
        if isinstance(result, dict):
            # If it's already a dict with messages, return it
            if "messages" in result:
                return {"messages": result["messages"]}
            # If it's a dict but not the right format, wrap it
            return {"messages": state["messages"]}
        
        # If agent returns something else, preserve existing messages
        return {"messages": state["messages"]}
        
    except Exception as e:
        print(f"❌ Error in agent {name}: {e}")
        # Return the state unchanged if there's an error
        return {"messages": state["messages"]}

class HierarchicalCodeDevelopmentSystem:
    """Enterprise-grade code development system with Technical Lead authority and task tracking"""
    
    def __init__(self):
        self.graph = None
        self.viewer = CodeDevelopmentViewer()
        self.setup_directories()
        self.setup_graph()
    
    def setup_directories(self):
        """Create comprehensive development workspace"""
        directories = [
            'workspace',           # Main development workspace
            'workspace/projects',  # Individual projects
            'workspace/backups',   # Code backups
            'workspace/tasks',     # Task tracking files
            'logs',               # Development session logs
            'templates',          # Code templates
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        print("🏗️ Enterprise development workspace initialized")
        print(f"📁 Workspace: {Path('workspace').absolute()}")
    
    def setup_graph(self):
        """Initialize the 6-agent hierarchical development graph - CORRECTED VERSION"""
        print("🔧 Building hierarchical 6-agent development system...")
        
        try:
            # Build the graph with proper state
            workflow = StateGraph(AgentState)
            
            # Create node functions for each agent with better error handling
            architect_node = functools.partial(agent_node, agent=architect, name="architect")
            writer_node = functools.partial(agent_node, agent=writer, name="writer")
            executor_node = functools.partial(agent_node, agent=executor, name="executor")
            technical_lead_node = functools.partial(agent_node, agent=technical_lead, name="technical_lead")
            task_manager_node = functools.partial(agent_node, agent=task_manager, name="task_manager")
            docs_node = functools.partial(agent_node, agent=docs, name="docs")
            finalizer_node = functools.partial(agent_node, agent=finalizer, name="finalizer")
            
            # Add nodes
            workflow.add_node("architect", architect_node)
            workflow.add_node("writer", writer_node)
            workflow.add_node("executor", executor_node)
            workflow.add_node("technical_lead", technical_lead_node)
            workflow.add_node("task_manager", task_manager_node)
            workflow.add_node("docs", docs_node)
            workflow.add_node("finalizer", finalizer_node)
            
            # Set entry point
            workflow.add_edge(START, "architect")
            
            # Add conditional routing based on tool calls
            def route_based_on_last_message(state: AgentState) -> str:
                """Route to next agent based on the last tool call in the conversation"""
                try:
                    messages = state["messages"]
                    if not messages:
                        return "architect"
                    
                    # Look at the last few messages for tool calls
                    for msg in reversed(messages[-3:]):  # Check last 3 messages
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            for tool_call in msg.tool_calls:
                                tool_name = tool_call.get('name', '')
                                if tool_name.startswith('transfer_to_'):
                                    target_agent = tool_name.replace('transfer_to_', '')
                                    if target_agent in ["architect", "writer", "executor", "technical_lead", "task_manager", "docs", "finalizer"]:
                                        return target_agent
                        
                        # Also check ToolMessages for transfer confirmations
                        if hasattr(msg, 'name') and hasattr(msg, 'content'):
                            if msg.name and msg.name.startswith('transfer_to_'):
                                target_agent = msg.name.replace('transfer_to_', '')
                                if target_agent in ["architect", "writer", "executor", "technical_lead", "task_manager", "docs", "finalizer"]:
                                    return target_agent
                    
                    # Default fallback
                    return "technical_lead"
                    
                except Exception as e:
                    print(f"❌ Routing error: {e}")
                    return "technical_lead"
            
            # Add conditional edges from each agent (except finalizer)
            for agent_name in ["architect", "writer", "executor", "technical_lead", "task_manager", "docs"]:
                workflow.add_conditional_edges(
                    agent_name,
                    route_based_on_last_message,
                    {
                        "architect": "architect",
                        "writer": "writer", 
                        "executor": "executor",
                        "technical_lead": "technical_lead",
                        "task_manager": "task_manager",
                        "docs": "docs",
                        "finalizer": "finalizer"
                    }
                )
            
            # End point
            workflow.add_edge("finalizer", END)
            
            # Compile the workflow
            self.graph = workflow.compile()
            
            print("✅ Graph created successfully with conditional edge routing!")
            
        except Exception as e:
            print(f"❌ Error creating graph: {e}")
            import traceback
            traceback.print_exc()
            raise

    def run_quick_development(self, request: str):
        """Run quick development for simple requests"""
        print(f"⚡ Quick Hierarchical Development Mode:")
        print(f"Request: {request}")
        print("=" * 60)
        
        # FIXED: Better error handling and debugging
        try:
            self.viewer.run(self.graph, request)
        except Exception as e:
            print(f"❌ Development session error: {e}")
            import traceback
            traceback.print_exc()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = self.viewer.save_log(f"quick_hierarchical_{timestamp}.txt")
        
        self.show_results(log_path)
    
    def show_results(self, log_path=None):
        """Display comprehensive development results"""
        print("\n" + "=" * 80)
        print("🎉 HIERARCHICAL ENTERPRISE DEVELOPMENT SESSION COMPLETED!")
        print("=" * 80)
        
        # Show generated files and projects
        print(f"\n📁 Generated Content:")
        if log_path:
            print(f"📋 Development Log: {log_path}")
        
        workspace = Path("workspace")
        if workspace.exists():
            # Show projects
            projects_dir = workspace / "projects"
            if projects_dir.exists():
                projects = list(projects_dir.glob("*"))
                if projects:
                    print(f"🏗️ Projects Created: {len(projects)}")
                    for project in projects[:3]:
                        print(f"  📦 {project.name}/")
            
            # Show individual files
            files = list(workspace.rglob("*.py"))
            if files:
                print(f"📄 Python Files: {len(files)}")
                for file in files[:5]:
                    size = file.stat().st_size
                    print(f"  🐍 {file.relative_to(workspace)} ({size} bytes)")
        
        print("\n💡 The 7-agent hierarchical system delivered enterprise-grade results!")
    
    def show_system_status(self):
        """Show current system status and capabilities"""
        print("📊 HIERARCHICAL ENTERPRISE DEVELOPMENT SYSTEM STATUS")
        print("=" * 60)
        
        # Check workspace
        workspace = Path("workspace")
        if workspace.exists():
            files = list(workspace.rglob("*"))
            files = [f for f in files if f.is_file()]
            print(f"📁 Workspace files: {len(files)}")
        
        print("\n⚠️ Key System Rules:")
        print("  • Agents can only make ONE tool call at a time")
        print("  • Technical Lead has authority over all agents")
        print("  • Only Writer can create/modify code")
        print("  • Task Manager updates only per Technical Lead directive")
        print("  • All agents must wait for tool results before next action")

    def get_demo_projects(self):
        """Get comprehensive demo project examples"""
        return [
            """Create a professional web scraper with error handling, rate limiting, data validation, 
            comprehensive testing, and proper task tracking. Include security best practices and 
            performance monitoring.""",
            
            """Build a complete REST API with Flask that includes user authentication, data validation, 
            error handling, logging, comprehensive testing, and detailed documentation. Follow enterprise 
            standards with proper task management.""",
            
            """Develop a data analysis pipeline that reads CSV files, performs statistical analysis, 
            generates visualizations, and exports results. Include comprehensive error handling, testing, 
            performance optimization, and task tracking throughout development.""",
            
            """Create a command-line tool for file organization with features like duplicate detection, 
            batch operations, logging, and configuration management. Include comprehensive testing, 
            documentation, and proper project management with task tracking.""",
            
            """Build a machine learning model training system with data preprocessing, model selection, 
            evaluation metrics, and prediction API. Include proper project structure, comprehensive testing, 
            documentation, and task management with Technical Lead oversight.""",
            
            """Develop a monitoring dashboard that tracks system metrics, sends alerts, and provides 
            real-time visualization. Include database integration, security features, testing, and 
            comprehensive task tracking with Technical Lead validation.""",
            
            """Create a complete Python package/library with proper setup.py, comprehensive documentation, 
            testing, CI/CD configuration, and publishing preparation. Follow all enterprise best practices 
            with detailed task management and Technical Lead oversight."""
        ]
    
    def run_interactive_demo(self):
        """Run interactive development demonstration"""
        print("🚀 HIERARCHICAL ENTERPRISE CODE DEVELOPMENT SYSTEM")
        print("=" * 80)
        print("🧑‍💼 TECHNICAL LEAD AUTHORITY: Experienced leader guides and validates all work")
        print("📝 SINGLE CODE WRITER: Only Writer creates and fixes ALL code")
        print("📊 TASK TRACKING: Task Manager maintains organized progress tables")
        print("🔄 HIERARCHICAL HANDOFFS: All agents report to Technical Lead")
        print("🏁 FINALIZER: Completes session when all work is done")
        print("⚠️ SINGLE TOOL CALLS: Agents make one tool call at a time")
        print("=" * 80)
        
        demo_projects = self.get_demo_projects()
        
        print("\n🎯 Available Enterprise Demo Projects:")
        for i, project in enumerate(demo_projects, 1):
            clean_desc = ' '.join(project.split()[:15]) + "..."
            print(f"{i}. {clean_desc}")
        
        # Get user choice
        try:
            choice = input(f"\nSelect a project (1-{len(demo_projects)}) or press Enter for custom: ").strip()
            
            if not choice:
                custom_request = input("Describe your development project: ").strip()
                if custom_request:
                    selected_project = custom_request
                else:
                    selected_project = demo_projects[0]
            else:
                project_index = int(choice) - 1
                if 0 <= project_index < len(demo_projects):
                    selected_project = demo_projects[project_index]
                else:
                    print("Invalid choice, using default...")
                    selected_project = demo_projects[0]
                    
        except (ValueError, KeyboardInterrupt):
            print("Using default project...")
            selected_project = demo_projects[0]
        
        print(f"\n🎮 Starting Hierarchical Enterprise Development Project:")
        print("=" * 80)
        print(f"📋 {selected_project}")
        print("=" * 80)
        
        # Run the development process with better error handling
        try:
            self.viewer.run(self.graph, selected_project)
        except Exception as e:
            print(f"❌ Development session error: {e}")
            import traceback
            traceback.print_exc()
        
        # Save development log
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = self.viewer.save_log(f"hierarchical_dev_{timestamp}.txt")
        
        self.show_results(log_path)

def main():
    """Main entry point for hierarchical development system"""
    print("🚀 Hierarchical Enterprise Code Development System")
    print("=" * 70)
    
    # Create system with better error handling
    try:
        system = HierarchicalCodeDevelopmentSystem()
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "--demo":
            system.run_interactive_demo()
            return
        
        elif command == "--status":
            system.show_system_status()
            return
        
        elif command == "--quick":
            if len(sys.argv) > 2:
                request = " ".join(sys.argv[2:])
                system.run_quick_development(request)
            else:
                print("❌ Please provide a request after --quick")
            return
        
        else:
            # Treat as development request
            request = " ".join(sys.argv[1:])
            system.run_quick_development(request)
            return
    
    # Default - run interactive demo
    try:
        system.run_interactive_demo()
    except KeyboardInterrupt:
        print("\n👋 Development session interrupted by user")
    except Exception as e:
        print(f"\n❌ Development error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()