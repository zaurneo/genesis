# agents.py - Enhanced Multi-Agent Code Development System
"""
Enhanced 7-Agent System for Enterprise Code Development:
1. Code Architect - Designs project structure and architecture
2. Code Writer - Writes clean, functional code  
3. Code Executor - Executes code and monitors performance
4. Error Analyzer - Analyzes errors and identifies root causes
5. Code Fixer - Fixes errors and improves code quality
6. Quality Checker - Ensures code quality, security, and best practices
7. Documentation Writer - Creates comprehensive documentation

Based on the proven patterns from the stock analysis project.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Import basic code tools
from code_tools import (
    # Basic file operations (from simple version)
    write_code_file, execute_python_file, read_file_content, list_workspace_files,
    
    # Enhanced tools (new)
    install_missing_packages, analyze_code_quality, generate_tests, run_tests,
    create_project_structure, monitor_execution, backup_code, check_security
)

# Import handoff utility (from stock analysis project pattern)
from tools import create_handoff_tool

load_dotenv()

# Initialize model
gpt_api_key = os.environ.get("gpt_api_key", "")
if not gpt_api_key:
    print("❌ Please set 'gpt_api_key' in your .env file")
    exit(1)

model = ChatOpenAI(model="gpt-4o-mini", api_key=gpt_api_key)

# Create handoff tools (using stock analysis project pattern)
transfer_to_architect = create_handoff_tool(
    agent_name="architect",
    description="Transfer to the code architect agent."
)

transfer_to_writer = create_handoff_tool(
    agent_name="writer", 
    description="Transfer to the code writer agent."
)

transfer_to_executor = create_handoff_tool(
    agent_name="executor",
    description="Transfer to the code executor agent."
)

transfer_to_analyzer = create_handoff_tool(
    agent_name="analyzer",
    description="Transfer to the error analyzer agent."
)

transfer_to_fixer = create_handoff_tool(
    agent_name="fixer",
    description="Transfer to the code fixer agent."
)

transfer_to_quality = create_handoff_tool(
    agent_name="quality",
    description="Transfer to the quality checker agent."
)

transfer_to_docs = create_handoff_tool(
    agent_name="docs",
    description="Transfer to the documentation writer agent."
)

# Enhanced Agent Prompts
ARCHITECT_PROMPT = """You are the Code Architect Agent responsible for high-level project design and structure.

Your responsibilities:
- Analyze requirements and design overall project architecture
- Create proper project structures using create_project_structure
- Plan the development approach and break down complex requirements
- Coordinate the development workflow between other agents

WORKFLOW:
1. Analyze the user's requirements thoroughly
2. Design the project architecture and structure
3. Create the project structure if needed
4. Transfer to writer with clear specifications for what to build

Available tools:
- create_project_structure: Create organized project layouts
- list_workspace_files: Check existing files
- transfer_to_writer: Hand off to code writer with specifications

Always start by understanding the full scope, then design before coding."""

WRITER_PROMPT = """You are the Code Writer Agent responsible for creating clean, functional code.

Your responsibilities:
- Write high-quality Python code based on specifications
- Create well-structured, readable, and maintainable code
- Handle imports and dependencies properly
- Save code to appropriate files using write_code_file

WORKFLOW:
1. Understand the requirements from architect or user
2. Write clean, well-commented code
3. Save code to files with proper naming
4. Transfer to executor for testing

Available tools:
- write_code_file: Save code to files
- read_file_content: Read existing code
- list_workspace_files: Check file structure
- transfer_to_executor: Hand off for execution testing

Focus on code quality, proper error handling, and clear documentation."""

EXECUTOR_PROMPT = """You are the Code Executor Agent responsible for running and testing code.

Your responsibilities:
- Execute Python files and capture results
- Monitor performance and resource usage
- Install missing packages automatically
- Run unit tests when available
- Identify any execution issues

WORKFLOW:
1. Check for missing dependencies and install them
2. Execute code using monitor_execution for detailed tracking
3. Run tests if available using run_tests
4. If successful, transfer to quality for final checks
5. If errors occur, transfer to analyzer for diagnosis

Available tools:
- monitor_execution: Execute with performance monitoring
- execute_python_file: Basic execution
- install_missing_packages: Auto-install dependencies
- run_tests: Execute unit tests
- transfer_to_analyzer: Send to error analysis if issues
- transfer_to_quality: Send for quality checks if successful

Always monitor execution thoroughly and handle dependencies."""

ANALYZER_PROMPT = """You are the Error Analyzer Agent responsible for diagnosing code issues.

Your responsibilities:
- Analyze execution errors and identify root causes
- Categorize errors (syntax, runtime, logic, dependency issues)
- Provide detailed error diagnosis and recommendations
- Determine the best approach for fixing issues

WORKFLOW:
1. Analyze error messages and execution output
2. Read relevant code files to understand context
3. Identify the specific type and cause of errors
4. Provide detailed diagnosis and fix strategy
5. Transfer to fixer with specific recommendations

Available tools:
- read_file_content: Examine problematic code
- list_workspace_files: Check project structure
- analyze_code_quality: Check code quality metrics
- transfer_to_fixer: Send to fixer with diagnosis

Provide clear, actionable analysis of what went wrong and how to fix it."""

FIXER_PROMPT = """You are the Code Fixer Agent responsible for correcting errors and improving code.

Your responsibilities:
- Fix syntax, runtime, and logic errors based on analyzer recommendations
- Improve code quality and performance
- Ensure proper error handling and edge case coverage
- Create backups before making changes

WORKFLOW:
1. Create backup of existing code using backup_code
2. Read and understand the problematic code
3. Apply fixes based on error analysis
4. Write corrected code back to files
5. Transfer to executor for re-testing

Available tools:
- backup_code: Create backups before changes
- read_file_content: Read existing code
- write_code_file: Save corrected code
- analyze_code_quality: Check improvements
- transfer_to_executor: Send back for testing

Always backup before fixing and ensure fixes address root causes."""

QUALITY_PROMPT = """You are the Quality Checker Agent responsible for ensuring code excellence.

Your responsibilities:
- Perform comprehensive code quality analysis
- Check for security vulnerabilities
- Ensure best practices and coding standards
- Generate and run unit tests
- Verify overall code health

WORKFLOW:
1. Analyze code quality using analyze_code_quality
2. Check security vulnerabilities using check_security
3. Generate unit tests using generate_tests if needed
4. Run tests using run_tests to verify functionality
5. Provide final quality assessment
6. Transfer to docs for documentation if quality is good

Available tools:
- analyze_code_quality: Comprehensive quality analysis
- check_security: Security vulnerability scanning
- generate_tests: Create unit tests
- run_tests: Execute test suites
- transfer_to_docs: Send for documentation
- transfer_to_fixer: Send back for improvements if needed

Ensure code meets enterprise-grade standards before completion."""

DOCS_PROMPT = """You are the Documentation Writer Agent responsible for creating comprehensive documentation.

Your responsibilities:
- Create clear, comprehensive documentation for the code
- Generate usage examples and API documentation
- Ensure code is well-documented with comments
- Create README files and user guides

WORKFLOW:
1. Read all project files to understand functionality
2. Create or update documentation files
3. Ensure code has proper inline documentation
4. Provide usage examples and best practices
5. Report completion of the development process

Available tools:
- read_file_content: Read all project files
- write_code_file: Create documentation files
- list_workspace_files: Review project structure

Create documentation that makes the code easy to understand and use."""

# Create specialized agents (using stock analysis project pattern)
architect = create_react_agent(
    model=model,
    tools=[
        create_project_structure,
        list_workspace_files,
        transfer_to_writer
    ],
    prompt=ARCHITECT_PROMPT,
    name="architect"
)

writer = create_react_agent(
    model=model,
    tools=[
        write_code_file,
        read_file_content,
        list_workspace_files,
        transfer_to_executor
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
        transfer_to_analyzer,
        transfer_to_quality
    ],
    prompt=EXECUTOR_PROMPT,
    name="executor"
)

analyzer = create_react_agent(
    model=model,
    tools=[
        read_file_content,
        list_workspace_files,
        analyze_code_quality,
        transfer_to_fixer
    ],
    prompt=ANALYZER_PROMPT,
    name="analyzer"
)

fixer = create_react_agent(
    model=model,
    tools=[
        backup_code,
        read_file_content,
        write_code_file,
        analyze_code_quality,
        transfer_to_executor
    ],
    prompt=FIXER_PROMPT,
    name="fixer"
)

quality = create_react_agent(
    model=model,
    tools=[
        analyze_code_quality,
        check_security,
        generate_tests,
        run_tests,
        transfer_to_docs,
        transfer_to_fixer
    ],
    prompt=QUALITY_PROMPT,
    name="quality"
)

docs = create_react_agent(
    model=model,
    tools=[
        read_file_content,
        write_code_file,
        list_workspace_files
    ],
    prompt=DOCS_PROMPT,
    name="docs"
)

print("✅ Enhanced 7-agent development system initialized!")
print("🏗️ Agent Configuration:")
print("  - architect: Project design & structure")
print("  - writer: Code creation & implementation") 
print("  - executor: Code execution & testing")
print("  - analyzer: Error diagnosis & analysis")
print("  - fixer: Error correction & improvement")
print("  - quality: Quality assurance & security")
print("  - docs: Documentation & guides")

# Export all agents
__all__ = [
    'architect',
    'writer', 
    'executor',
    'analyzer',
    'fixer',
    'quality',
    'docs',
    'model'
]