# main.py - FINAL FIXED VERSION with proper graph routing
"""
Enhanced Multi-Agent Enterprise Code Development System with 7 Specialized Agents:

🏗️ Architect → ✍️ Writer → ⚡ Executor → 🔍 Analyzer → 🔧 Fixer → ✅ Quality → 📚 Docs

FINAL FIX: Proper LangGraph routing with conditional edges for handoffs
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
from agents import architect, writer, executor, analyzer, fixer, quality, docs

load_dotenv()

def route_after_architect(state: MessagesState):
    """Route after architect based on last message"""
    messages = state.get("messages", [])
    if not messages:
        return "writer"
    
    last_message = messages[-1]
    # Check if architect wants to transfer to writer
    if hasattr(last_message, 'content') and last_message.content:
        content = str(last_message.content).lower()
        if "writer" in content or "passed to the writer" in content or "transfer" in content:
            return "writer"
    
    # Check for tool calls that transfer to writer
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            if tool_call.get('name') == 'transfer_to_writer':
                return "writer"
    
    return "writer"  # Default to writer

def route_after_writer(state: MessagesState):
    """Route after writer - should go to executor"""
    return "executor"

def route_after_executor(state: MessagesState):
    """Route after executor based on results"""
    messages = state.get("messages", [])
    if not messages:
        return "quality"
    
    last_message = messages[-1]
    if hasattr(last_message, 'content') and last_message.content:
        content = str(last_message.content).lower()
        if "error" in content or "failed" in content or "missing" in content:
            if "missing" in content and "test" in content:
                return "writer"  # Missing tests -> back to writer
            else:
                return "analyzer"  # Errors -> analyzer
        elif "success" in content or "passed" in content:
            return "quality"  # Success -> quality
    
    return "quality"  # Default to quality

def route_after_analyzer(state: MessagesState):
    """Route after analyzer - should go to fixer"""
    return "fixer"

def route_after_fixer(state: MessagesState):
    """Route after fixer - back to executor for testing"""
    return "executor"

def route_after_quality(state: MessagesState):
    """Route after quality - should go to docs if good, or fixer if issues"""
    messages = state.get("messages", [])
    if not messages:
        return "docs"
    
    last_message = messages[-1]
    if hasattr(last_message, 'content') and last_message.content:
        content = str(last_message.content).lower()
        if "issue" in content or "problem" in content or "failed" in content:
            return "fixer"  # Issues -> fixer
        else:
            return "docs"  # Good -> docs
    
    return "docs"  # Default to docs

def route_after_docs(state: MessagesState):
    """Route after docs - end the workflow"""
    return END

class EnhancedCodeDevelopmentSystem:
    """Enterprise-grade code development system with 7 specialized agents"""
    
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
            'workspace/tests',     # Test files
            'logs',               # Development session logs
            'templates',          # Code templates
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        print("🏗️ Enterprise development workspace initialized")
        print(f"📁 Workspace: {Path('workspace').absolute()}")
    
    def setup_graph(self):
        """Initialize the 7-agent development graph with PROPER ROUTING"""
        print("🔧 Building enterprise 7-agent development system with conditional routing...")
        
        try:
            # Build graph with conditional edges for proper handoffs
            workflow = StateGraph(MessagesState)
            
            # Add all agent nodes
            workflow.add_node("architect", architect)
            workflow.add_node("writer", writer)
            workflow.add_node("executor", executor)
            workflow.add_node("analyzer", analyzer)
            workflow.add_node("fixer", fixer)
            workflow.add_node("quality", quality)
            workflow.add_node("docs", docs)
            
            # Set entry point
            workflow.add_edge(START, "architect")
            
            # Add conditional edges for proper routing
            workflow.add_conditional_edges(
                "architect",
                route_after_architect,
                {"writer": "writer"}
            )
            
            workflow.add_conditional_edges(
                "writer", 
                route_after_writer,
                {"executor": "executor"}
            )
            
            workflow.add_conditional_edges(
                "executor",
                route_after_executor,
                {
                    "writer": "writer",      # Missing tests -> writer
                    "analyzer": "analyzer",  # Errors -> analyzer
                    "quality": "quality"     # Success -> quality
                }
            )
            
            workflow.add_conditional_edges(
                "analyzer",
                route_after_analyzer,
                {"fixer": "fixer"}
            )
            
            workflow.add_conditional_edges(
                "fixer",
                route_after_fixer,
                {"executor": "executor"}  # Back to executor for testing
            )
            
            workflow.add_conditional_edges(
                "quality",
                route_after_quality,
                {
                    "fixer": "fixer",    # Issues -> fixer
                    "docs": "docs"       # Good -> docs
                }
            )
            
            workflow.add_conditional_edges(
                "docs",
                route_after_docs,
                {END: END}
            )
            
            # Compile the workflow
            self.graph = workflow.compile()
            
            print("✅ 7-agent development graph with conditional routing created successfully!")
            
        except Exception as e:
            print(f"❌ Error creating graph: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def get_demo_projects(self):
        """Get comprehensive demo project examples"""
        return [
            """Create a professional web scraper with error handling, rate limiting, data validation, 
            and comprehensive testing. Include proper project structure, documentation, and security checks.""",
            
            """Build a complete REST API with Flask that includes user authentication, data validation, 
            error handling, logging, testing, and comprehensive documentation. Use proper project structure 
            and follow security best practices.""",
            
            """Develop a data analysis pipeline that reads CSV files, performs statistical analysis, 
            generates visualizations, and exports results. Include error handling, testing, 
            performance optimization, and complete documentation.""",
            
            """Create a command-line tool for file organization with features like duplicate detection, 
            batch operations, logging, and configuration management. Include comprehensive testing, 
            documentation, and proper error handling.""",
            
            """Build a machine learning model training system with data preprocessing, model selection, 
            evaluation metrics, and prediction API. Include proper project structure, testing, 
            documentation, and performance monitoring.""",
            
            """Develop a monitoring dashboard that tracks system metrics, sends alerts, and provides 
            visualization. Include database integration, real-time updates, testing, security, 
            and comprehensive documentation.""",
            
            """Create a complete package/library with proper setup.py, documentation, testing, 
            CI/CD configuration, and publishing preparation. Follow all Python packaging best practices."""
        ]
    
    def run_interactive_demo(self):
        """Run interactive development demonstration"""
        print("🚀 ENTERPRISE CODE DEVELOPMENT SYSTEM")
        print("=" * 80)
        print("🎯 POWERED BY: 7 Specialized AI Agents + Conditional Routing")
        print("📝 ROLE SEPARATION: Only Writer & Fixer can write code")
        print("🔄 SMART ROUTING: Agents hand off based on context")
        print("=" * 80)
        print("This system will:")
        print("1. 🏗️ Design proper project architecture and structure")
        print("2. ✍️ Write clean, professional, enterprise-grade code")
        print("3. ⚡ Execute code with performance monitoring")
        print("4. 🔍 Analyze any errors with root cause analysis")
        print("5. 🔧 Fix errors and improve code quality automatically")
        print("6. ✅ Ensure code quality, security, and best practices")
        print("7. 📚 Generate comprehensive documentation")
        print("8. 🧪 Create and run tests intelligently")
        print("9. 📊 Monitor performance and resource usage")
        print("10. 🔒 Perform security vulnerability scanning")
        print("=" * 80)
        
        demo_projects = self.get_demo_projects()
        
        print("\n🎯 Available Enterprise Demo Projects:")
        for i, project in enumerate(demo_projects, 1):
            clean_desc = ' '.join(project.split()[:12]) + "..."
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
        
        print(f"\n🎮 Starting Enterprise Development Project:")
        print("=" * 80)
        print(f"📋 {selected_project}")
        print("=" * 80)
        
        # Run the development process
        self.viewer.run(self.graph, selected_project)
        
        # Save development log
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = self.viewer.save_log(f"enterprise_dev_{timestamp}.txt")
        
        self.show_results(log_path)
    
    def run_quick_development(self, request: str):
        """Run quick development for simple requests"""
        print(f"⚡ Quick Development Mode:")
        print(f"Request: {request}")
        print("=" * 60)
        
        self.viewer.run(self.graph, request)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = self.viewer.save_log(f"quick_dev_{timestamp}.txt")
        
        self.show_results(log_path)
    
    def show_results(self, log_path=None):
        """Display comprehensive development results"""
        print("\n" + "=" * 80)
        print("🎉 ENTERPRISE DEVELOPMENT SESSION COMPLETED!")
        print("=" * 80)
        print("The system has demonstrated:")
        print("✅ Professional project architecture and design")
        print("✅ Clean, enterprise-grade code generation")
        print("✅ Comprehensive error detection and fixing")
        print("✅ Code quality assurance and security scanning")
        print("✅ Automated testing and validation")
        print("✅ Performance monitoring and optimization")
        print("✅ Complete documentation generation")
        print("✅ Professional project structure")
        
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
            
            # Show tests
            tests_dir = workspace / "tests"
            if tests_dir.exists():
                tests = list(tests_dir.glob("test_*.py"))
                if tests:
                    print(f"🧪 Test Files: {len(tests)}")
        
        print("\n💡 The 7-agent system collaborated to deliver enterprise-grade code!")
    
    def show_system_status(self):
        """Show current system status and capabilities"""
        print("📊 ENTERPRISE DEVELOPMENT SYSTEM STATUS")
        print("=" * 60)
        
        # Check workspace
        workspace = Path("workspace")
        if workspace.exists():
            files = list(workspace.rglob("*"))
            files = [f for f in files if f.is_file()]
            print(f"📁 Workspace files: {len(files)}")
        
        # Show agent capabilities
        print("\n🤖 Agent Capabilities:")
        agents = {
            "🏗️ Architect": "Project design, structure planning, requirements analysis (read-only)",
            "✍️ Writer": "Code generation, implementation, clean coding practices (writes code)",
            "⚡ Executor": "Code execution, performance monitoring, dependency management (read-only)",
            "🔍 Analyzer": "Error diagnosis, root cause analysis, issue categorization (read-only)",
            "🔧 Fixer": "Error correction, code improvement, backup management (writes code)",
            "✅ Quality": "Quality assurance, security scanning, testing, standards (read-only)",
            "📚 Docs": "Documentation generation, guides, API docs, examples (docs only)"
        }
        
        for agent, capability in agents.items():
            print(f"  {agent}: {capability}")
        
        # Show available tools
        print(f"\n🛠️ Available Tools:")
        tools = [
            "Code quality analysis", "Security vulnerability scanning",
            "Intelligent testing", "Performance monitoring", 
            "Dependency management", "Project structure creation",
            "Code backup/versioning", "Documentation generation"
        ]
        
        for tool in tools:
            print(f"  • {tool}")
        
        print(f"\n📝 Role Separation:")
        print(f"  ✅ Can write code: Writer, Fixer")
        print(f"  ❌ Read-only: Architect, Executor, Analyzer, Quality")
        print(f"  📄 Docs only: Docs")
        
        print(f"\n🔄 Smart Routing:")
        print(f"  📐 Architect → Writer (always)")
        print(f"  📝 Writer → Executor (for testing)")
        print(f"  ⚡ Executor → Writer (missing tests) | Analyzer (errors) | Quality (success)")
        print(f"  🔍 Analyzer → Fixer (for fixes)")
        print(f"  🔧 Fixer → Executor (for re-testing)")
        print(f"  ✅ Quality → Fixer (issues) | Docs (success)")
        print(f"  📚 Docs → END (completion)")

def main():
    """Main entry point for enhanced development system"""
    print("🚀 Enterprise Code Development System with 7 AI Agents")
    print("=" * 70)
    print("🏗️ Architect → ✍️ Writer → ⚡ Executor → 🔍 Analyzer → 🔧 Fixer → ✅ Quality → 📚 Docs")
    print("📝 Role Separation: Only Writer & Fixer can write code")
    print("🔄 Smart Conditional Routing: Context-based handoffs")
    print("=" * 70)
    
    # Create system
    system = EnhancedCodeDevelopmentSystem()
    
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
                print("Example: python main.py --quick 'create a calculator'")
            return
        
        elif command == "--help":
            print("🤖 ENTERPRISE DEVELOPMENT SYSTEM - HELP")
            print("=" * 50)
            print("\nAvailable commands:")
            print("  --demo     : Run full interactive demo")
            print("  --quick    : Quick development mode")
            print("  --status   : Show system status")
            print("  --help     : Show this help")
            print("\nExamples:")
            print("  python main.py --demo")
            print("  python main.py --quick 'create a web scraper'")
            print("  python main.py --status")
            print("\nRole Separation:")
            print("  📝 Writer: Creates new code and files")
            print("  🔧 Fixer: Fixes existing code")
            print("  🔍 Others: Read-only, request code via smart routing")
            print("\nSmart Routing:")
            print("  🔄 Context-aware handoffs between agents")
            print("  📐 Conditional edges based on agent outputs")
            print("  🎯 Automatic workflow progression")
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