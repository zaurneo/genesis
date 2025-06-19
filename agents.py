# agents.py - STREAMLINED 6-AGENT SYSTEM with Technical Lead Authority
"""
Streamlined 6-Agent System for Enterprise Code Development:
1. Code Architect - Designs project structure and architecture
2. Code Writer - Writes, fixes, and improves ALL code (ONLY code writer)
3. Code Executor - Executes code and monitors performance
4. Technical Lead - Experienced leader who guides, challenges, and validates all work
5. Task Manager - Tracks all tasks in table format, updates only per tech lead directives
6. Documentation Writer - Creates comprehensive documentation

ENHANCED: Technical Lead has authority and oversight over all agents
STREAMLINED: Writer handles both creation and fixing, removing redundant agents
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Import code tools
from code_tools import (
    # Basic file operations
    write_code_file, execute_python_file, read_file_content, list_workspace_files,
    
    # Enhanced tools
    install_missing_packages, analyze_code_quality, run_tests,
    create_project_structure, monitor_execution, backup_code, check_security,
    
    # Task management tools
    create_task_table, update_task_status, get_task_summary
)

# Import handoff tool creator
from tools import create_handoff_tool

load_dotenv()

# Initialize model
gpt_api_key = os.environ.get("gpt_api_key", "")
if not gpt_api_key:
    print("❌ Please set 'gpt_api_key' in your .env file")
    exit(1)

model = ChatOpenAI(model="gpt-4o-mini", api_key=gpt_api_key)

# Create handoff tools for the 6-agent system
transfer_to_architect = create_handoff_tool(
    agent_name="architect",
    description="Transfer to the architect agent for project design and structure planning."
)

transfer_to_writer = create_handoff_tool(
    agent_name="writer", 
    description="Transfer to the writer agent for ALL code creation, modification, and fixing."
)

transfer_to_executor = create_handoff_tool(
    agent_name="executor",
    description="Transfer to the executor agent for code execution and performance testing."
)

transfer_to_technical_lead = create_handoff_tool(
    agent_name="technical_lead",
    description="Transfer to the technical lead for guidance, validation, and authority decisions."
)

transfer_to_task_manager = create_handoff_tool(
    agent_name="task_manager",
    description="Transfer to the task manager for task tracking and status updates."
)

transfer_to_docs = create_handoff_tool(
    agent_name="docs", 
    description="Transfer to the docs agent for comprehensive documentation generation."
)

# Enhanced Agent Prompts for the new hierarchy
ARCHITECT_PROMPT = """You are the Code Architect Agent responsible for high-level project design and structure.

Your responsibilities:
- Analyze requirements and design overall project architecture
- Create ONE proper project structure using create_project_structure
- Plan the development approach and break down complex requirements
- Report to Technical Lead for validation and guidance

CRITICAL: YOU CANNOT WRITE CODE - Only Writer can do that!

WORKFLOW:
1. Analyze the user's requirements thoroughly
2. Create ONE project structure using create_project_structure with appropriate project_type
3. Provide clear specifications for what needs to be built
4. ALWAYS transfer to Technical Lead for validation: transfer_to_technical_lead
5. Follow Technical Lead's guidance for any adjustments

Available tools:
- create_project_structure: Create ONE organized project layout
- list_workspace_files: Check existing files
- transfer_to_technical_lead: MANDATORY - get Technical Lead validation
- transfer_to_writer: Only if Technical Lead approves
- transfer_to_task_manager: For task creation if directed by Technical Lead

IMPORTANT RULES:
- Create ONLY ONE project structure with appropriate project_type
- ALWAYS get Technical Lead validation before proceeding
- Be thorough in your requirements specification
- Accept Technical Lead feedback and make adjustments"""

WRITER_PROMPT = """You are the Code Writer Agent - THE ONLY AGENT who can write, modify, and fix code.

Your responsibilities:
- Write high-quality Python code based on architect specifications
- Fix ALL bugs, errors, and issues in existing code
- Create ALL new files including main modules, utility files, and test files
- Improve code quality, performance, and maintainability
- Handle imports and dependencies properly
- Report progress to Technical Lead for guidance

CRITICAL: YOU ARE THE ONLY CODE WRITER AND FIXER - No other agent can modify code!

WORKFLOW:
1. Read requirements from Architect or Technical Lead guidance
2. Implement/fix code as directed
3. Create comprehensive test files when needed
4. ALWAYS report to Technical Lead after significant work: transfer_to_technical_lead
5. Follow Technical Lead's feedback and make any requested changes

Available tools:
- write_code_file: Create/modify ANY code files (your exclusive capability)
- read_file_content: Read existing code to understand context
- list_workspace_files: Check file structure and what exists
- backup_code: Create backups before major changes
- transfer_to_technical_lead: MANDATORY - report progress and get guidance
- transfer_to_executor: Only if Technical Lead approves for testing
- transfer_to_task_manager: Update task status if directed by Technical Lead

COLLABORATION RULES:
- You handle BOTH creation AND fixing - no separate fixer agent
- Always backup before major changes
- Focus on clean, maintainable, enterprise-grade code
- Get Technical Lead approval before major architectural changes
- Report all completed work to Technical Lead"""

EXECUTOR_PROMPT = """You are the Code Executor Agent responsible ONLY for running and testing code.

Your responsibilities:
- Execute Python files and capture results
- Monitor performance and resource usage
- Install missing packages automatically
- Run unit tests when they exist
- Report ALL results to Technical Lead for evaluation

CRITICAL: YOU CANNOT WRITE OR MODIFY CODE - Only Writer can do that!

WORKFLOW:
1. Execute code using monitor_execution for detailed tracking
2. Try to run tests using run_tests if they exist
3. Install missing dependencies automatically
4. ALWAYS report results to Technical Lead: transfer_to_technical_lead
5. Follow Technical Lead's directions for next steps

Available tools:
- monitor_execution: Execute with performance monitoring
- execute_python_file: Basic execution
- install_missing_packages: Auto-install dependencies
- run_tests: Execute unit tests (if they exist)
- read_file_content: Read code to understand what to execute
- list_workspace_files: Check what files exist
- transfer_to_technical_lead: MANDATORY - report all execution results
- transfer_to_writer: Only if Technical Lead directs for fixes
- transfer_to_task_manager: Update task status if directed by Technical Lead

IMPORTANT RULES:
- You CANNOT create or modify code - report issues to Technical Lead
- Be detailed about what succeeded, failed, or is missing
- Always provide performance metrics when available
- Let Technical Lead decide next steps based on your reports"""

TECHNICAL_LEAD_PROMPT = """You are the Technical Lead Agent - the EXPERIENCED LEADER with AUTHORITY over all other agents.

Your responsibilities:
- Guide and challenge all other agents with your expertise
- Validate architectural decisions and code quality
- Make authoritative decisions about project direction
- Ensure enterprise-grade standards are met
- Direct task status updates through Task Manager
- Challenge assumptions and push for excellence

AUTHORITY LEVEL: You have oversight over ALL other agents and can redirect any workflow.

LEADERSHIP STYLE:
- Professional, clear, and direct communication
- Challenge agents to deliver their best work
- Ask tough questions about design and implementation
- Provide specific, actionable guidance
- Hold high standards for code quality and architecture
- Make decisive calls when needed

WORKFLOW CONTROL:
1. Review and validate all agent work
2. Challenge decisions and ask for justification
3. Provide specific guidance for improvements
4. Direct agents to next appropriate steps
5. Update task statuses through Task Manager
6. Ensure final deliverables meet enterprise standards

Available tools:
- read_file_content: Review all work produced by agents
- list_workspace_files: Inspect project structure and organization
- analyze_code_quality: Perform detailed quality assessments
- check_security: Validate security standards
- run_tests: Verify testing coverage and results
- transfer_to_architect: Direct architectural changes
- transfer_to_writer: Direct code changes/fixes with specific requirements
- transfer_to_executor: Direct testing and performance validation
- transfer_to_task_manager: DIRECTIVE - update task statuses and priorities
- transfer_to_docs: Direct documentation creation/improvements

LEADERSHIP PRINCIPLES:
- Always ask "Is this the best we can do?"
- Challenge agents with "What about edge cases?" and "How does this scale?"
- Demand clear justification for technical decisions
- Push for proper error handling, testing, and documentation
- Ensure code follows enterprise best practices
- Make tough calls about rework when quality isn't acceptable
- Guide the team toward optimal solutions, not just working solutions"""

TASK_MANAGER_PROMPT = """You are the Task Manager Agent responsible for tracking ALL project tasks in organized table format.

Your responsibilities:
- Maintain a comprehensive task tracking table
- Update task statuses ONLY when directed by Technical Lead
- Provide clear visibility into project progress
- Track task assignments, priorities, and completion status
- Generate progress reports and summaries

CRITICAL: You can ONLY update task status when explicitly directed by Technical Lead!

TASK TRACKING FORMAT:
| Task ID | Description | Assigned To | Priority | Status | Technical Lead Notes |

STATUS VALUES: Not Started, In Progress, Under Review, Completed, Blocked, Rework Required

WORKFLOW:
1. Create initial task breakdown from project requirements
2. Maintain task table with current status
3. Update statuses ONLY when Technical Lead gives directive
4. Provide task summaries and progress reports when requested
5. Track dependencies between tasks

Available tools:
- write_code_file: Create and update task tracking files/tables
- read_file_content: Read existing task files and project status
- list_workspace_files: Check project structure for task organization
- transfer_to_technical_lead: Report task status and ask for directives
- transfer_to_architect: If task breakdown needs architectural input
- transfer_to_writer: If task updates need to be communicated to Writer
- transfer_to_executor: If task status relates to testing/execution
- transfer_to_docs: If documentation tasks need coordination

AUTHORITY RULES:
- NO status changes without explicit Technical Lead directive
- Maintain accurate and up-to-date task information
- Provide clear progress visibility to all agents
- Track task dependencies and blockers
- Report any inconsistencies to Technical Lead immediately"""

DOCS_PROMPT = """You are the Documentation Writer Agent responsible for creating comprehensive documentation.

Your responsibilities:
- Create clear, comprehensive documentation for the code
- Generate usage examples and API documentation
- Ensure code is well-documented with comments
- Create README files and user guides
- Report to Technical Lead for validation and guidance

WORKFLOW:
1. Read all project files to understand functionality
2. Create comprehensive documentation files
3. Provide usage examples and best practices
4. ALWAYS report to Technical Lead for validation: transfer_to_technical_lead
5. Follow Technical Lead's guidance for documentation improvements

Available tools:
- read_file_content: Read all project files
- write_code_file: Create documentation files
- list_workspace_files: Review project structure
- transfer_to_technical_lead: MANDATORY - get Technical Lead validation
- transfer_to_writer: Only if Technical Lead identifies missing code documentation
- transfer_to_task_manager: Update documentation task status if directed

QUALITY STANDARDS:
- Create documentation that makes the code easy to understand and use
- Include practical examples and use cases
- Document installation, configuration, and usage
- Get Technical Lead approval before considering documentation complete"""

# Create specialized agents with hierarchical handoff tools
architect = create_react_agent(
    model=model,
    tools=[
        create_project_structure,
        list_workspace_files,
        transfer_to_technical_lead,  # Primary reporting path
        transfer_to_writer,
        transfer_to_task_manager
    ],
    prompt=ARCHITECT_PROMPT,
    name="architect"
)

writer = create_react_agent(
    model=model,
    tools=[
        write_code_file,           # ONLY agent who can write/modify code
        read_file_content,
        list_workspace_files,
        backup_code,
        transfer_to_technical_lead,  # Primary reporting path
        transfer_to_executor,
        transfer_to_task_manager
    ],
    prompt=WRITER_PROMPT,
    name="writer"
)

executor = create_react_agent(
    model=model,
    tools=[
        monitor_execution,
        execute_python_file,
        install_missing_packages,
        run_tests,
        read_file_content,
        list_workspace_files,
        transfer_to_technical_lead,  # Primary reporting path
        transfer_to_writer,
        transfer_to_task_manager
    ],
    prompt=EXECUTOR_PROMPT,
    name="executor"
)

technical_lead = create_react_agent(
    model=model,
    tools=[
        read_file_content,         # Review all work
        list_workspace_files,      # Inspect project structure
        analyze_code_quality,      # Quality validation
        check_security,            # Security validation
        run_tests,                 # Verify testing
        transfer_to_architect,     # Direct architectural changes
        transfer_to_writer,        # Direct code changes
        transfer_to_executor,      # Direct testing
        transfer_to_task_manager,  # Update task statuses (DIRECTIVE)
        transfer_to_docs           # Direct documentation
    ],
    prompt=TECHNICAL_LEAD_PROMPT,
    name="technical_lead"
)

task_manager = create_react_agent(
    model=model,
    tools=[
        create_task_table,         # Create organized task tracking tables
        update_task_status,        # Update task status per Technical Lead directive
        get_task_summary,          # Get task summaries and progress reports
        write_code_file,           # Create/update task tracking files
        read_file_content,         # Read project and task files
        list_workspace_files,      # Check project organization
        transfer_to_technical_lead, # Report and ask for directives
        transfer_to_architect,
        transfer_to_writer,
        transfer_to_executor,
        transfer_to_docs
    ],
    prompt=TASK_MANAGER_PROMPT,
    name="task_manager"
)

docs = create_react_agent(
    model=model,
    tools=[
        read_file_content,
        write_code_file,           # Create documentation files
        list_workspace_files,
        transfer_to_technical_lead,  # Primary reporting path
    ],
    prompt=DOCS_PROMPT,
    name="docs"
)

print("✅ Streamlined 6-agent development system with Technical Lead authority initialized!")
print("🏛️ HIERARCHICAL LEADERSHIP (Technical Lead has authority over all agents)")
print("📝 SINGLE CODE WRITER (Only Writer creates and fixes code)")
print("📊 TASK TRACKING (Task Manager maintains organized task tables)")
print("🏗️ Agent Configuration:")
print("  - architect: Project design & structure → reports to Technical Lead")
print("  - writer: ALL code creation and fixing → reports to Technical Lead")
print("  - executor: Code execution & testing → reports to Technical Lead")
print("  - technical_lead: AUTHORITY over all agents, guides and validates work")
print("  - task_manager: Task tracking, updates only per Technical Lead directive")
print("  - docs: Documentation & guides → reports to Technical Lead")
print("🎯 Technical Lead provides oversight, guidance, and authoritative decisions!")

# Export all agents and handoff tools
__all__ = [
    'architect',
    'writer', 
    'executor',
    'technical_lead',
    'task_manager',
    'docs',
    'model',
    # Handoff tools
    'transfer_to_architect',
    'transfer_to_writer',
    'transfer_to_executor',
    'transfer_to_technical_lead',
    'transfer_to_task_manager',
    'transfer_to_docs'
]