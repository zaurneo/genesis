# main.py - 6-AGENT HIERARCHICAL SYSTEM with Technical Lead Authority
"""
Streamlined 6-Agent Enterprise Code Development System:

🏗️ Architect → 🧑‍💼 Technical Lead → ✍️ Writer → ⚡ Executor → 🧑‍💼 Technical Lead → 📚 Docs
                     ↕️                                          ↕️
                 📊 Task Manager                            📊 Task Manager

HIERARCHICAL: Technical Lead has authority and oversight over all agents
STREAMLINED: Writer handles both creation and fixing, removing redundant agents
ORGANIZED: Task Manager tracks all work in table format per Technical Lead directives
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END, MessagesState

# Import enhanced conversation viewer
from conversation_viewer import CodeDevelopmentViewer
from agents import architect, writer, executor, technical_lead, task_manager, docs

load_dotenv()

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
        """Initialize the 6-agent hierarchical development graph"""
        print("🔧 Building hierarchical 6-agent development system with Technical Lead authority...")
        
        try:
            # Build hierarchical graph - Technical Lead has oversight
            workflow = StateGraph(MessagesState)
            
            # Add all agent nodes
            workflow.add_node("architect", architect)
            workflow.add_node("writer", writer)
            workflow.add_node("executor", executor)
            workflow.add_node("technical_lead", technical_lead)
            workflow.add_node("task_manager", task_manager)
            workflow.add_node("docs", docs)
            
            # Set entry point - always start with architect
            workflow.add_edge(START, "architect")
            
            # Simple edges - agents use handoff tools to control routing
            # Technical Lead can redirect any workflow
            workflow.add_edge("architect", END)
            workflow.add_edge("writer", END) 
            workflow.add_edge("executor", END)
            workflow.add_edge("technical_lead", END)
            workflow.add_edge("task_manager", END)
            workflow.add_edge("docs", END)
            
            # Compile the workflow
            self.graph = workflow.compile()
            
            print("✅ 6-agent hierarchical development graph created successfully!")
            print("🧑‍💼 Technical Lead has authority over all workflow decisions!")
            
        except Exception as e:
            print(f"❌ Error creating graph: {e}")
            import traceback
            traceback.print_exc()
            raise
    
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
        print("=" * 80)
        print("This system demonstrates:")
        print("1. 🏗️ Professional project architecture and structure planning")
        print("2. 🧑‍💼 Technical Lead oversight with authority and experience")
        print("3. ✍️ Unified code creation and fixing by expert Writer")
        print("4. ⚡ Comprehensive code execution and performance testing")
        print("5. 📊 Organized task tracking and status management")
        print("6. 📚 Complete documentation generation and validation")
        print("7. 🔒 Security scanning and quality assurance")
        print("8. 📈 Performance monitoring and optimization")
        print("9. 🧪 Intelligent test creation and execution")
        print("10. 🎯 Authoritative guidance and decision-making")
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
        
        # Run the development process
        self.viewer.run(self.graph, selected_project)
        
        # Save development log
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = self.viewer.save_log(f"hierarchical_dev_{timestamp}.txt")
        
        self.show_results(log_path)
    
    def run_quick_development(self, request: str):
        """Run quick development for simple requests"""
        print(f"⚡ Quick Hierarchical Development Mode:")
        print(f"Request: {request}")
        print("=" * 60)
        
        self.viewer.run(self.graph, request)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = self.viewer.save_log(f"quick_hierarchical_{timestamp}.txt")
        
        self.show_results(log_path)
    
    def show_results(self, log_path=None):
        """Display comprehensive development results"""
        print("\n" + "=" * 80)
        print("🎉 HIERARCHICAL ENTERPRISE DEVELOPMENT SESSION COMPLETED!")
        print("=" * 80)
        print("The system has demonstrated:")
        print("✅ Technical Lead authority and oversight throughout development")
        print("✅ Professional project architecture and design")
        print("✅ Unified code creation and fixing by expert Writer")
        print("✅ Comprehensive testing and performance monitoring")
        print("✅ Organized task tracking and status management")
        print("✅ Quality assurance and security scanning")
        print("✅ Complete documentation generation")
        print("✅ Hierarchical workflow with clear authority structure")
        
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
                    for project in projects[:3]:  # Show first 3
                        print(f"  📦 {project.name}/")
            
            # Show task files
            tasks_dir = workspace / "tasks"
            if tasks_dir.exists():
                task_files = list(tasks_dir.glob("*.md"))
                if task_files:
                    print(f"📊 Task Tracking Files: {len(task_files)}")
                    for task_file in task_files[:3]:
                        print(f"  📋 {task_file.name}")
            
            # Show individual files
            files = list(workspace.glob("*.py"))
            if files:
                print(f"📄 Python Files: {len(files)}")
                for file in files[:5]:  # Show first 5
                    size = file.stat().st_size
                    print(f"  🐍 {file.name} ({size} bytes)")
            
            # Show backups
            backups_dir = workspace / "backups"
            if backups_dir.exists():
                backups = list(backups_dir.glob("*.backup"))
                if backups:
                    print(f"💾 Code Backups: {len(backups)}")
        
        print("\n💡 The 6-agent hierarchical system with Technical Lead authority delivered enterprise-grade results!")
    
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
        
        # Show agent capabilities
        print("\n🤖 Agent Capabilities & Hierarchy:")
        agents = {
            "🏗️ Architect": "Project design, structure planning → reports to Technical Lead",
            "🧑‍💼 Technical Lead": "AUTHORITY over all agents, guides, validates, and makes decisions",
            "✍️ Writer": "ALL code creation and fixing (only code writer) → reports to Technical Lead",
            "⚡ Executor": "Code execution, performance monitoring → reports to Technical Lead",
            "📊 Task Manager": "Task tracking in table format, updates only per Technical Lead directive",
            "📚 Docs": "Documentation generation → reports to Technical Lead"
        }
        
        for agent, capability in agents.items():
            print(f"  {agent}: {capability}")
        
        # Show available tools
        print(f"\n🛠️ Available Tools:")
        tools = [
            "Code quality analysis", "Security vulnerability scanning",
            "Intelligent testing", "Performance monitoring", 
            "Dependency management", "Project structure creation",
            "Code backup/versioning", "Documentation generation",
            "Task tracking and management", "Technical Lead oversight"
        ]
        
        for tool in tools:
            print(f"  • {tool}")
        
        print(f"\n📝 Role Separation:")
        print(f"  ✅ Can write code: Writer ONLY")
        print(f"  🧑‍💼 Authority: Technical Lead has oversight over ALL agents")
        print(f"  📊 Task updates: Only Task Manager per Technical Lead directive")
        print(f"  📄 Documentation: Docs agent")
        print(f"  ❌ Read-only: Architect, Executor")
        
        print(f"\n🏛️ Hierarchical System:")
        print(f"  🧑‍💼 Technical Lead: AUTHORITY and oversight over all workflow")
        print(f"  📐 Architect → Technical Lead validation required")
        print(f"  📝 Writer → Technical Lead guidance and approval")
        print(f"  ⚡ Executor → Technical Lead evaluation of results")
        print(f"  📊 Task Manager → Updates only per Technical Lead directive")
        print(f"  📚 Docs → Technical Lead validation required")
        print(f"  🎯 All major decisions flow through Technical Lead")

def main():
    """Main entry point for hierarchical development system"""
    print("🚀 Hierarchical Enterprise Code Development System")
    print("=" * 70)
    print("🧑‍💼 Technical Lead Authority: Experienced leader with oversight")
    print("✍️ Single Code Writer: Writer handles ALL code creation and fixing")
    print("📊 Task Management: Organized tracking per Technical Lead directives")
    print("🔄 Hierarchical Handoffs: All agents report to Technical Lead")
    print("=" * 70)
    
    # Create system
    system = HierarchicalCodeDevelopmentSystem()
    
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
                print("Example: python main.py --quick 'create a calculator with tests'")
            return
        
        elif command == "--help":
            print("🤖 HIERARCHICAL ENTERPRISE DEVELOPMENT SYSTEM - HELP")
            print("=" * 60)
            print("\nAvailable commands:")
            print("  --demo     : Run full interactive demo")
            print("  --quick    : Quick development mode")
            print("  --status   : Show system status")
            print("  --help     : Show this help")
            print("\nExamples:")
            print("  python main.py --demo")
            print("  python main.py --quick 'create a web scraper'")
            print("  python main.py --status")
            print("\nHierarchical Structure:")
            print("  🧑‍💼 Technical Lead: Authority over all agents")
            print("  📝 Writer: Only agent who can write/fix code")
            print("  📊 Task Manager: Tracks tasks per Technical Lead directives")
            print("  🔍 Others: Report to Technical Lead for guidance")
            print("\nKey Features:")
            print("  🎯 Technical Lead oversight and decision-making")
            print("  📋 Organized task tracking in table format")
            print("  🔄 Hierarchical handoffs with clear authority")
            print("  ✅ Enterprise-grade quality and standards")
            return
        
        else:
            # Treat unknown command as development request
            request = " ".join(sys.argv[1:])
            system.run_quick_development(request)
            return
    
    # Default action - run interactive demo
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