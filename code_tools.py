# code_tools.py - FINAL VERSION with Task Management for Hierarchical 6-Agent System
"""
Enhanced tools for comprehensive code development, testing, quality assurance, and task management.
Includes task management tools for the hierarchical 6-agent system with Technical Lead authority.
"""

import os
import sys
import subprocess
import ast
import importlib.util
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool

@tool
def create_task_table(project_name: str, initial_tasks: str = "") -> str:
    """Create an organized task tracking table for project management
    
    Args:
        project_name: Name of the project for task tracking
        initial_tasks: Comma-separated list of initial tasks (optional)
        
    Returns:
        Status of task table creation
    """
    try:
        if not project_name or not project_name.strip():
            return "❌ Error: Project name cannot be empty"
        
        workspace = Path("workspace")
        tasks_dir = workspace / "tasks"
        tasks_dir.mkdir(exist_ok=True)
        
        # Clean project name
        project_name = project_name.strip().replace(" ", "_").lower()
        task_file = tasks_dir / f"{project_name}_tasks.md"
        
        # Create task table structure
        task_content = f"""# Task Tracking: {project_name.replace('_', ' ').title()}

**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Technical Lead:** Authorized to update task statuses  
**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Task Status Legend
- **Not Started**: Task has been identified but not begun
- **In Progress**: Task is currently being worked on
- **Under Review**: Task completed, awaiting Technical Lead review
- **Completed**: Task finished and approved by Technical Lead
- **Blocked**: Task cannot proceed due to dependencies
- **Rework Required**: Task needs to be redone per Technical Lead directive

## Task Table

| Task ID | Description | Assigned To | Priority | Status | Technical Lead Notes | Last Updated |
|---------|-------------|-------------|----------|--------|---------------------|--------------|
"""
        
        # Add initial tasks if provided
        if initial_tasks and initial_tasks.strip():
            task_list = [task.strip() for task in initial_tasks.split(',') if task.strip()]
            for i, task in enumerate(task_list, 1):
                task_content += f"| T{i:03d} | {task} | Writer | Medium | Not Started | - | {datetime.now().strftime('%Y-%m-%d %H:%M')} |\n"
        else:
            # Add default initial tasks
            default_tasks = [
                "Project structure creation",
                "Core functionality implementation", 
                "Unit test development",
                "Integration testing",
                "Code quality review",
                "Documentation creation"
            ]
            for i, task in enumerate(default_tasks, 1):
                task_content += f"| T{i:03d} | {task} | TBD | Medium | Not Started | - | {datetime.now().strftime('%Y-%m-%d %H:%M')} |\n"
        
        task_content += f"""
## Task Statistics
- **Total Tasks**: {len(initial_tasks.split(',')) if initial_tasks else 6}
- **Completed**: 0
- **In Progress**: 0
- **Blocked**: 0

## Notes
- Only Technical Lead can authorize status changes
- All task updates must include Technical Lead approval
- Task assignments can be modified by Technical Lead directive
"""
        
        # Write task file
        with open(task_file, 'w', encoding='utf-8') as f:
            f.write(task_content)
        
        return f"""
✅ Task tracking table created successfully!

📊 Task Table Details:
- Project: {project_name.replace('_', ' ').title()}
- File: {task_file.relative_to(workspace)}
- Initial Tasks: {len(initial_tasks.split(',')) if initial_tasks else 6}
- Authority: Technical Lead controls all status updates

🎯 Next Steps:
1. Technical Lead reviews and adjusts task breakdown
2. Task assignments made per Technical Lead directive
3. Status updates only with Technical Lead authorization
        """.strip()
        
    except Exception as e:
        return f"❌ Error: Cannot create task table: {str(e)}"

@tool
def update_task_status(project_name: str, task_id: str, new_status: str, tech_lead_notes: str = "", assigned_to: str = "") -> str:
    """Update task status - ONLY to be used with Technical Lead authorization
    
    Args:
        project_name: Name of the project
        task_id: Task ID (e.g., T001)
        new_status: New status (Not Started, In Progress, Under Review, Completed, Blocked, Rework Required)
        tech_lead_notes: Technical Lead notes for this update
        assigned_to: Who the task is assigned to (optional)
        
    Returns:
        Status of task update
    """
    try:
        if not all([project_name, task_id, new_status]):
            return "❌ Error: Project name, task ID, and new status are required"
        
        workspace = Path("workspace")
        tasks_dir = workspace / "tasks"
        
        # Clean inputs
        project_name = project_name.strip().replace(" ", "_").lower()
        task_file = tasks_dir / f"{project_name}_tasks.md"
        
        if not task_file.exists():
            return f"❌ Error: Task file for project '{project_name}' not found"
        
        # Valid statuses
        valid_statuses = ["Not Started", "In Progress", "Under Review", "Completed", "Blocked", "Rework Required"]
        if new_status not in valid_statuses:
            return f"❌ Error: Invalid status. Valid options: {', '.join(valid_statuses)}"
        
        # Read current task file
        with open(task_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find and update the specific task
        lines = content.split('\n')
        updated = False
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        for i, line in enumerate(lines):
            if f"| {task_id} |" in line:
                # Parse the existing task line
                parts = [part.strip() for part in line.split('|')]
                if len(parts) >= 8:
                    # Update the task line: | Task ID | Description | Assigned To | Priority | Status | Technical Lead Notes | Last Updated |
                    parts[5] = new_status  # Status
                    parts[6] = tech_lead_notes or parts[6]  # Technical Lead Notes
                    parts[7] = current_time  # Last Updated
                    
                    if assigned_to:
                        parts[3] = assigned_to  # Assigned To
                    
                    lines[i] = " | ".join(parts) + " |"
                    updated = True
                    break
        
        if not updated:
            return f"❌ Error: Task {task_id} not found in project {project_name}"
        
        # Update the "Last Updated" timestamp in header
        for i, line in enumerate(lines):
            if "**Last Updated:**" in line:
                lines[i] = f"**Last Updated:** {current_time}"
                break
        
        # Update task statistics
        status_counts = {"Not Started": 0, "In Progress": 0, "Under Review": 0, 
                        "Completed": 0, "Blocked": 0, "Rework Required": 0}
        total_tasks = 0
        
        for line in lines:
            if line.startswith("| T") and "|" in line:
                parts = [part.strip() for part in line.split('|')]
                if len(parts) >= 6:
                    total_tasks += 1
                    status = parts[5].strip()
                    if status in status_counts:
                        status_counts[status] += 1
        
        # Update statistics section
        for i, line in enumerate(lines):
            if "- **Total Tasks**:" in line:
                lines[i] = f"- **Total Tasks**: {total_tasks}"
            elif "- **Completed**:" in line:
                lines[i] = f"- **Completed**: {status_counts['Completed']}"
            elif "- **In Progress**:" in line:
                lines[i] = f"- **In Progress**: {status_counts['In Progress']}"
            elif "- **Blocked**:" in line:
                lines[i] = f"- **Blocked**: {status_counts['Blocked']}"
        
        # Write updated content
        with open(task_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        return f"""
✅ Task status updated successfully!

📊 Task Update Details:
- Project: {project_name.replace('_', ' ').title()}
- Task ID: {task_id}
- New Status: {new_status}
- Technical Lead Notes: {tech_lead_notes or 'No additional notes'}
- Updated: {current_time}

📈 Current Statistics:
- Total Tasks: {total_tasks}
- Completed: {status_counts['Completed']}
- In Progress: {status_counts['In Progress']}
- Blocked: {status_counts['Blocked']}
        """.strip()
        
    except Exception as e:
        return f"❌ Error: Cannot update task status: {str(e)}"

@tool
def get_task_summary(project_name: str) -> str:
    """Get a summary of current task status for a project
    
    Args:
        project_name: Name of the project
        
    Returns:
        Task summary and statistics
    """
    try:
        if not project_name or not project_name.strip():
            return "❌ Error: Project name cannot be empty"
        
        workspace = Path("workspace")
        tasks_dir = workspace / "tasks"
        
        # Clean project name
        project_name = project_name.strip().replace(" ", "_").lower()
        task_file = tasks_dir / f"{project_name}_tasks.md"
        
        if not task_file.exists():
            return f"❌ Error: Task file for project '{project_name}' not found"
        
        # Read task file
        with open(task_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse tasks and statistics
        lines = content.split('\n')
        tasks = []
        status_counts = {"Not Started": 0, "In Progress": 0, "Under Review": 0, 
                        "Completed": 0, "Blocked": 0, "Rework Required": 0}
        
        for line in lines:
            if line.startswith("| T") and "|" in line:
                parts = [part.strip() for part in line.split('|')]
                if len(parts) >= 8:
                    task_info = {
                        'id': parts[1],
                        'description': parts[2],
                        'assigned_to': parts[3],
                        'priority': parts[4],
                        'status': parts[5],
                        'notes': parts[6],
                        'updated': parts[7]
                    }
                    tasks.append(task_info)
                    if task_info['status'] in status_counts:
                        status_counts[task_info['status']] += 1
        
        # Build summary
        summary = f"""
📊 Task Summary for {project_name.replace('_', ' ').title()}:
{'=' * 50}

📈 Statistics:
- Total Tasks: {len(tasks)}
- Completed: {status_counts['Completed']}
- In Progress: {status_counts['In Progress']}
- Under Review: {status_counts['Under Review']}
- Not Started: {status_counts['Not Started']}
- Blocked: {status_counts['Blocked']}
- Rework Required: {status_counts['Rework Required']}

🎯 Progress: {(status_counts['Completed'] / max(1, len(tasks)) * 100):.1f}% Complete

📋 Active Tasks:
"""
        
        # Show active tasks (not completed)
        active_tasks = [t for t in tasks if t['status'] != 'Completed']
        if active_tasks:
            for task in active_tasks[:10]:  # Show up to 10 active tasks
                summary += f"- {task['id']}: {task['description']} [{task['status']}]\n"
            if len(active_tasks) > 10:
                summary += f"... and {len(active_tasks) - 10} more active tasks\n"
        else:
            summary += "All tasks completed!\n"
        
        return summary.strip()
        
    except Exception as e:
        return f"❌ Error: Cannot get task summary: {str(e)}"

@tool
def run_tests(test_path: str) -> str:
    """Run unit tests and return results with FIXED path handling
    
    Args:
        test_path: Path to test file or directory (relative to workspace)
        
    Returns:
        Test execution results
    """
    try:
        if not test_path or not test_path.strip():
            return "❌ Error: Test path cannot be empty"
        
        workspace = Path("workspace")
        if not workspace.exists():
            return "❌ Error: Workspace directory does not exist"
        
        # Clean and resolve the test path
        test_path = test_path.strip()
        
        # Handle different path formats
        if test_path.startswith("workspace/") or test_path.startswith("workspace\\"):
            # Remove workspace prefix to avoid double workspace
            test_path = test_path.replace("workspace/", "").replace("workspace\\", "")
        
        # Convert to Path object
        full_test_path = workspace / test_path
        
        # Check if it's a directory or file
        if full_test_path.is_dir():
            # Look for test files in the directory
            test_files = list(full_test_path.glob("test_*.py"))
            if not test_files:
                return f"❌ Error: No test files (test_*.py) found in directory {test_path}"
            
            # Run all test files
            results = []
            for test_file in test_files:
                try:
                    result = subprocess.run(
                        [sys.executable, str(test_file)],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        cwd=workspace
                    )
                    
                    if result.returncode == 0:
                        results.append(f"✅ {test_file.name}: PASSED")
                    else:
                        results.append(f"❌ {test_file.name}: FAILED")
                        if result.stderr:
                            results.append(f"   Error: {result.stderr.strip()[:200]}")
                            
                except subprocess.TimeoutExpired:
                    results.append(f"⏰ {test_file.name}: TIMED OUT")
                except Exception as e:
                    results.append(f"❌ {test_file.name}: ERROR - {str(e)}")
            
            return "🧪 TEST RESULTS:\n" + "\n".join(results)
            
        elif full_test_path.is_file():
            # Run single test file
            try:
                result = subprocess.run(
                    [sys.executable, str(full_test_path)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=workspace
                )
                
                output = f"🧪 TEST RESULTS for {test_path}:\n"
                if result.stdout:
                    output += f"📤 OUTPUT:\n{result.stdout.strip()}\n"
                
                if result.stderr:
                    output += f"❌ ERRORS:\n{result.stderr.strip()}\n"
                
                if result.returncode == 0:
                    output += "✅ All tests passed successfully!"
                else:
                    output += f"❌ Tests failed with exit code: {result.returncode}"
                
                return output
                
            except subprocess.TimeoutExpired:
                return f"❌ Error: Test execution timed out after 30 seconds"
            except FileNotFoundError:
                return f"❌ Error: Python executable not found"
                
        else:
            return f"❌ Error: Test path '{test_path}' not found. Available files:\n{_list_workspace_contents()}"
        
    except Exception as e:
        return f"❌ Error: Cannot run tests for '{test_path}': {str(e)}\nWorkspace contents:\n{_list_workspace_contents()}"

def _list_workspace_contents():
    """Helper function to list workspace contents for debugging"""
    try:
        workspace = Path("workspace")
        if not workspace.exists():
            return "Workspace does not exist"
        
        contents = []
        for item in workspace.rglob("*"):
            if item.is_file():
                rel_path = item.relative_to(workspace)
                contents.append(f"  📄 {rel_path}")
        
        return "\n".join(contents[:10]) + (f"\n... ({len(contents)-10} more files)" if len(contents) > 10 else "")
    except Exception:
        return "Could not list workspace contents"

@tool
def list_workspace_files() -> str:
    """List all files in the workspace with IMPROVED formatting"""
    try:
        workspace = Path("workspace")
        if not workspace.exists():
            workspace.mkdir(exist_ok=True)
            return "📁 Workspace created (was empty)"
        
        files = list(workspace.rglob("*"))
        files = [f for f in files if f.is_file()]
        
        if not files:
            return "📁 Workspace is empty (no files found)"
        
        # Group files by project/directory
        file_groups = {}
        for file in files:
            rel_path = file.relative_to(workspace)
            if "/" in str(rel_path) or "\\" in str(rel_path):
                # File in subdirectory
                parts = rel_path.parts
                project = parts[0]
                if project not in file_groups:
                    file_groups[project] = []
                file_groups[project].append(rel_path)
            else:
                # File in root workspace
                if "root" not in file_groups:
                    file_groups["root"] = []
                file_groups["root"].append(rel_path)
        
        file_list = "📁 Workspace structure:\n"
        for project, project_files in sorted(file_groups.items()):
            if project == "root":
                file_list += f"📁 Root:\n"
            else:
                file_list += f"📁 {project}/:\n"
            
            for file_path in sorted(project_files):
                try:
                    full_path = workspace / file_path
                    size = full_path.stat().st_size
                    file_list += f"  📄 {file_path} ({size} bytes)\n"
                except Exception:
                    file_list += f"  📄 {file_path} (size unknown)\n"
        
        return file_list.strip()
        
    except Exception as e:
        return f"❌ Error: Cannot list workspace files: {str(e)}"

@tool
def read_file_content(filename: str) -> str:
    """Read the content of a file with IMPROVED path handling"""
    try:
        if not filename or not filename.strip():
            return "❌ Error: Filename cannot be empty"
        
        workspace = Path("workspace")
        
        # Clean filename and handle different path formats
        filename = filename.strip()
        if filename.startswith("workspace/") or filename.startswith("workspace\\"):
            filename = filename.replace("workspace/", "").replace("workspace\\", "")
        
        file_path = workspace / filename
        
        if not file_path.exists():
            # Try to find the file in subdirectories
            possible_files = list(workspace.rglob(Path(filename).name))
            if possible_files:
                file_path = possible_files[0]
                actual_path = file_path.relative_to(workspace)
                return f"📄 Content of {actual_path} (found at different location):\n{file_path.read_text(encoding='utf-8')}"
            else:
                return f"❌ Error: File '{filename}' not found in workspace.\nAvailable files:\n{_list_workspace_contents()}"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return f"📄 Content of {filename}:\n{content}"
        
    except PermissionError:
        return f"❌ Error: Permission denied reading '{filename}'"
    except UnicodeDecodeError:
        return f"❌ Error: Cannot decode '{filename}' as UTF-8"
    except Exception as e:
        return f"❌ Error: Cannot read file '{filename}': {str(e)}"

@tool
def write_code_file(filename: str, content: str) -> str:
    """Write code to a file in the workspace with IMPROVED path handling"""
    try:
        if not filename or not filename.strip():
            return "❌ Error: Filename cannot be empty"
        
        if not content:
            return "❌ Error: Content cannot be empty"
        
        workspace = Path("workspace")
        workspace.mkdir(exist_ok=True)
        
        # Clean filename and handle different path formats
        filename = filename.strip()
        if filename.startswith("workspace/") or filename.startswith("workspace\\"):
            filename = filename.replace("workspace/", "").replace("workspace\\", "")
        
        file_path = workspace / filename
        
        # Ensure parent directories exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        file_size = file_path.stat().st_size
        return f"✅ SUCCESS: Code written to {filename} ({file_size} bytes)"
        
    except PermissionError:
        return f"❌ Error: Permission denied writing to {filename}"
    except Exception as e:
        return f"❌ Error: Cannot write file '{filename}': {str(e)}"

@tool
def create_project_structure(project_name: str, project_type: str = "basic") -> str:
    """Create a proper project structure with improved error handling"""
    try:
        # Validate inputs
        if not project_name or not project_name.strip():
            return "❌ Error: Project name cannot be empty"
        
        # Clean project name - remove path separators and clean it up
        project_name = project_name.strip().replace("/", "_").replace("\\", "_").replace(" ", "_").lower()
        
        # Remove common prefixes that might cause confusion
        if project_name.startswith("my_"):
            project_name = project_name[3:]
        
        workspace = Path("workspace")
        workspace.mkdir(exist_ok=True)  # Ensure workspace exists
        project_dir = workspace / project_name
        
        # If project exists, return info about existing project instead of error
        if project_dir.exists():
            try:
                existing_files = list(project_dir.rglob("*"))
                existing_files = [f for f in existing_files if f.is_file()]
                file_list = [str(f.relative_to(project_dir)) for f in existing_files[:10]]
                
                return f"""
✅ Project '{project_name}' already exists and ready for development!

📁 Existing project structure:
{chr(10).join(f"  📄 {file}" for file in sorted(file_list))}
{f"  ... ({len(existing_files)-10} more files)" if len(existing_files) > 10 else ""}

🚀 Project is ready - proceed with development!
                """.strip()
                
            except Exception:
                return f"✅ Project '{project_name}' already exists and is ready for development!"
        
        # Create project directory structure
        try:
            project_dir.mkdir(parents=True, exist_ok=False)
        except Exception as e:
            return f"❌ Error: Cannot create project directory: {str(e)}"
        
        # Basic structure for all projects
        try:
            (project_dir / "src").mkdir(exist_ok=True)
            (project_dir / "tests").mkdir(exist_ok=True)
            (project_dir / "docs").mkdir(exist_ok=True)
            
            # Create __init__.py files
            (project_dir / "src" / "__init__.py").touch()
            (project_dir / "tests" / "__init__.py").touch()
        except Exception as e:
            return f"❌ Error: Cannot create project structure: {str(e)}"
        
        # Create main module
        main_file = project_dir / "src" / "main.py"
        try:
            with open(main_file, 'w', encoding='utf-8') as f:
                f.write(f'''"""
{project_name} - Main module
"""

def main():
    """Main entry point"""
    print("Hello from {project_name}!")

if __name__ == "__main__":
    main()
''')
        except Exception as e:
            return f"❌ Error: Cannot create main.py: {str(e)}"
        
        # Create README
        readme_file = project_dir / "README.md"
        try:
            with open(readme_file, 'w', encoding='utf-8') as f:
                f.write(f'''# {project_name}

Description of your project.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/main.py
```

## Development

```bash
python -m pytest tests/
```
''')
        except Exception as e:
            return f"❌ Error: Cannot create README.md: {str(e)}"
        
        # Create requirements.txt
        requirements_file = project_dir / "requirements.txt"
        try:
            with open(requirements_file, 'w', encoding='utf-8') as f:
                if project_type == "web":
                    f.write("flask>=2.0.0\nrequests>=2.25.0\nbeautifulsoup4>=4.9.0\n")
                elif project_type == "cli":
                    f.write("click>=8.0.0\nargparse\n")
                elif project_type == "package":
                    f.write("setuptools>=65.0\nwheel>=0.37.0\n")
                else:
                    f.write("# Add your dependencies here\n")
        except Exception as e:
            return f"❌ Error: Cannot create requirements.txt: {str(e)}"
        
        # Project-specific files
        if project_type == "web":
            try:
                (project_dir / "templates").mkdir(exist_ok=True)
                (project_dir / "static").mkdir(exist_ok=True)
                app_file = project_dir / "src" / "app.py"
                with open(app_file, 'w', encoding='utf-8') as f:
                    f.write('''from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, World!"

if __name__ == "__main__":
    app.run(debug=True)
''')
            except Exception as e:
                return f"❌ Error: Cannot create web app files: {str(e)}"
        
        elif project_type == "package":
            try:
                setup_file = project_dir / "setup.py"
                with open(setup_file, 'w', encoding='utf-8') as f:
                    f.write(f'''from setuptools import setup, find_packages

setup(
    name="{project_name}",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],
    author="Your Name",
    description="A short description of {project_name}",
)
''')
            except Exception as e:
                return f"❌ Error: Cannot create setup.py: {str(e)}"
        
        # Create .gitignore
        gitignore_file = project_dir / ".gitignore"
        try:
            with open(gitignore_file, 'w', encoding='utf-8') as f:
                f.write('''__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.mypy_cache
.pytest_cache
.hypothesis
''')
        except Exception as e:
            return f"❌ Error: Cannot create .gitignore: {str(e)}"
        
        # Count created files
        files_created = []
        try:
            for root, dirs, files in os.walk(project_dir):
                for file in files:
                    file_path = Path(root) / file
                    rel_path = file_path.relative_to(project_dir)
                    files_created.append(str(rel_path))
        except Exception as e:
            files_created = ["(could not enumerate files)"]
        
        return f"""
✅ SUCCESS: Project '{project_name}' created successfully!

📁 Project structure ({project_type} type):
{chr(10).join(f"  📄 {file}" for file in sorted(files_created))}

🚀 Next steps:
1. cd workspace/{project_name}
2. Edit src/main.py with your code
3. Add dependencies to requirements.txt
4. Run: python src/main.py

💡 Project ready for development!
        """.strip()
        
    except Exception as e:
        return f"❌ CRITICAL ERROR: Failed to create project structure: {str(e)}"

@tool
def install_missing_packages(packages: str) -> str:
    """Automatically install missing Python packages"""
    try:
        if not packages or not packages.strip():
            return "❌ Error: No packages specified"
            
        package_list = [pkg.strip() for pkg in packages.split(',') if pkg.strip()]
        if not package_list:
            return "❌ Error: No valid packages found"
            
        results = []
        
        for package in package_list:
            try:
                result = subprocess.run(
                    [sys.executable, '-m', 'pip', 'install', package],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    results.append(f"✅ {package}: Installed successfully")
                else:
                    error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                    results.append(f"❌ {package}: {error_msg}")
                    
            except subprocess.TimeoutExpired:
                results.append(f"⏰ {package}: Installation timed out (60s)")
            except Exception as e:
                results.append(f"❌ {package}: {str(e)}")
        
        return "📦 Package Installation Results:\n" + "\n".join(results)
        
    except Exception as e:
        return f"❌ Error: Package installation failed: {str(e)}"

@tool 
def analyze_code_quality(filename: str) -> str:
    """Analyze code quality with error handling"""
    try:
        if not filename or not filename.strip():
            return "❌ Error: Filename cannot be empty"
            
        workspace = Path("workspace")
        
        # Clean filename
        filename = filename.strip()
        if filename.startswith("workspace/") or filename.startswith("workspace\\"):
            filename = filename.replace("workspace/", "").replace("workspace\\", "")
            
        file_path = workspace / filename
        
        if not file_path.exists():
            return f"❌ Error: File '{filename}' not found"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        if not code.strip():
            return f"⚠️ Warning: File '{filename}' is empty"
        
        # Parse the code
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return f"❌ Syntax Error in '{filename}': {e}"
        
        # Basic quality metrics
        lines = code.split('\n')
        total_lines = len(lines)
        code_lines = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        comment_lines = len([line for line in lines if line.strip().startswith('#')])
        blank_lines = total_lines - code_lines - comment_lines
        
        # Function and class analysis
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        # Calculate complexity (simplified)
        complexity_score = 1  # Base complexity
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity_score += 1
            elif isinstance(node, ast.BoolOp):
                complexity_score += len(node.values) - 1
        
        # Import analysis
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        
        # Generate quality score
        comment_ratio = comment_lines / total_lines if total_lines > 0 else 0
        quality_score = min(100, max(0, 100 - (complexity_score * 2) + (comment_ratio * 20)))
        
        # Recommendations
        recommendations = []
        if quality_score < 60:
            recommendations.append("• Consider refactoring to improve overall quality")
        if comment_ratio < 0.1:
            recommendations.append("• Add more comments to explain complex logic")
        if complexity_score > 20:
            recommendations.append("• Break down complex functions into smaller ones")
        if len(functions) == 0:
            recommendations.append("• Consider organizing code into functions")
        if quality_score >= 80:
            recommendations.append("• Great code quality! Keep it up!")
        
        report = f"""
📊 Code Quality Analysis for {filename}:
{'=' * 50}

📏 Size Metrics:
- Total Lines: {total_lines}
- Code Lines: {code_lines}
- Comment Lines: {comment_lines}
- Blank Lines: {blank_lines}
- Comment Ratio: {comment_ratio:.1%}

🏗️ Structure:
- Functions: {len(functions)}
- Classes: {len(classes)}
- Imports: {len(imports)}

🧮 Complexity:
- Complexity Score: {complexity_score}
- Average per Function: {complexity_score / max(1, len(functions)):.1f}

📈 Quality Score: {quality_score:.0f}/100

💡 Recommendations:
{chr(10).join(recommendations) if recommendations else "• Code quality looks good!"}
        """
        
        return report.strip()
        
    except Exception as e:
        return f"❌ Error: Cannot analyze code quality for '{filename}': {str(e)}"

@tool
def check_security(filename: str) -> str:
    """Basic security vulnerability check with error handling"""
    try:
        if not filename or not filename.strip():
            return "❌ Error: Filename cannot be empty"
        
        workspace = Path("workspace")
        
        # Clean filename
        filename = filename.strip()
        if filename.startswith("workspace/") or filename.startswith("workspace\\"):
            filename = filename.replace("workspace/", "").replace("workspace\\", "")
            
        file_path = workspace / filename
        
        if not file_path.exists():
            return f"❌ Error: File '{filename}' not found"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        if not code.strip():
            return f"⚠️ Warning: File '{filename}' is empty"
        
        # Security checks
        security_issues = []
        
        # Check for dangerous functions
        dangerous_patterns = [
            ("eval(", "Use of eval() can execute arbitrary code"),
            ("exec(", "Use of exec() can execute arbitrary code"),
            ("__import__", "Dynamic imports can be dangerous"),
            ("subprocess.call", "Consider using subprocess.run() with shell=False"),
            ("os.system", "Use subprocess instead of os.system"),
            ("input(", "Be careful with user input, validate thoroughly"),
            ("pickle.load", "pickle.load can execute arbitrary code"),
            ("yaml.load", "Use yaml.safe_load instead of yaml.load"),
        ]
        
        for pattern, warning in dangerous_patterns:
            if pattern in code:
                security_issues.append(f"⚠️ {warning}")
        
        # Check for hardcoded secrets (basic patterns)
        secret_patterns = [
            ("password", "Possible hardcoded password"),
            ("api_key", "Possible hardcoded API key"),
            ("secret", "Possible hardcoded secret"),
            ("token", "Possible hardcoded token"),
        ]
        
        for pattern, warning in secret_patterns:
            if pattern.lower() in code.lower() and "=" in code:
                security_issues.append(f"🔑 {warning}")
        
        # Check for SQL injection potential
        if "sql" in code.lower() and ("%" in code or ".format(" in code):
            security_issues.append("🗄️ Potential SQL injection vulnerability - use parameterized queries")
        
        # Generate security score
        security_score = max(0, 100 - (len(security_issues) * 15))
        
        report = f"""
🔒 Security Analysis for {filename}:
{'=' * 50}

🎯 Security Score: {security_score}/100

{'🚨 Security Issues Found:' if security_issues else '✅ No major security issues detected'}
{chr(10).join(security_issues) if security_issues else ''}

💡 General Recommendations:
• Validate all user inputs
• Use environment variables for secrets
• Keep dependencies updated
• Use parameterized queries for databases
• Avoid eval() and exec() functions
• Use subprocess.run() instead of os.system()
        """
        
        return report.strip()
        
    except Exception as e:
        return f"❌ Error: Cannot check security for '{filename}': {str(e)}"

@tool
def execute_python_file(filename: str) -> str:
    """Execute a Python file and capture output/errors with better error handling"""
    try:
        if not filename or not filename.strip():
            return "❌ Error: Filename cannot be empty"
        
        workspace = Path("workspace")
        
        # Clean filename
        filename = filename.strip()
        if filename.startswith("workspace/") or filename.startswith("workspace\\"):
            filename = filename.replace("workspace/", "").replace("workspace\\", "")
            
        file_path = workspace / filename
        
        if not file_path.exists():
            return f"❌ Error: File '{filename}' not found in workspace"
        
        # Run the Python file with timeout
        try:
            result = subprocess.run(
                [sys.executable, str(file_path)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=workspace
            )
        except subprocess.TimeoutExpired:
            return f"❌ Error: Execution timed out after 30 seconds"
        except FileNotFoundError:
            return f"❌ Error: Python executable not found"
        
        output = ""
        if result.stdout:
            output += f"📤 OUTPUT:\n{result.stdout.strip()}\n"
        
        if result.stderr:
            output += f"❌ ERRORS:\n{result.stderr.strip()}\n"
        
        if result.returncode == 0:
            output += "✅ Execution successful!"
        else:
            output += f"❌ Exit code: {result.returncode}"
        
        return output
        
    except Exception as e:
        return f"❌ Error: Cannot execute '{filename}': {str(e)}"

@tool
def monitor_execution(filename: str, timeout: int = 10) -> str:
    """Monitor code execution with resource tracking and error handling"""
    try:
        if not filename or not filename.strip():
            return "❌ Error: Filename cannot be empty"
        
        workspace = Path("workspace")
        
        # Clean filename
        filename = filename.strip()
        if filename.startswith("workspace/") or filename.startswith("workspace\\"):
            filename = filename.replace("workspace/", "").replace("workspace\\", "")
            
        file_path = workspace / filename
        
        if not file_path.exists():
            return f"❌ Error: File '{filename}' not found"
        
        start_time = time.time()
        
        # Start the process
        try:
            process = subprocess.Popen(
                [sys.executable, str(file_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=workspace
            )
        except FileNotFoundError:
            return f"❌ Error: Python executable not found"
        
        # Try to monitor with psutil if available
        try:
            import psutil
            ps_process = psutil.Process(process.pid)
            max_memory = 0
            cpu_samples = []
            
            while process.poll() is None:
                try:
                    memory_mb = ps_process.memory_info().rss / 1024 / 1024
                    cpu_percent = ps_process.cpu_percent()
                    
                    max_memory = max(max_memory, memory_mb)
                    cpu_samples.append(cpu_percent)
                    
                    time.sleep(0.1)
                    
                    # Check timeout
                    if time.time() - start_time > timeout:
                        process.terminate()
                        return f"⏰ Execution timed out after {timeout} seconds"
                        
                except psutil.NoSuchProcess:
                    break
            
            # Get final results
            stdout, stderr = process.communicate()
            execution_time = time.time() - start_time
            
            avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
            
            result = f"""
📊 Execution Monitoring Results for {filename}:
{'=' * 50}

⏱️ Performance:
- Execution Time: {execution_time:.2f} seconds
- Max Memory Usage: {max_memory:.2f} MB
- Average CPU Usage: {avg_cpu:.1f}%

📤 Output:
{stdout.strip() if stdout else '(no output)'}

{'❌ Errors:' + chr(10) + stderr.strip() if stderr else '✅ No errors'}

🎯 Exit Code: {process.returncode}
            """
            
            return result.strip()
            
        except ImportError:
            # Fallback without psutil
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                execution_time = time.time() - start_time
                
                return f"""
📊 Basic Execution Results for {filename}:
⏱️ Execution Time: {execution_time:.2f} seconds
📤 Output: {stdout.strip() if stdout else '(no output)'}
{'❌ Errors: ' + stderr.strip() if stderr else '✅ No errors'}
🎯 Exit Code: {process.returncode}

💡 Install psutil for detailed resource monitoring: pip install psutil
                """.strip()
            except subprocess.TimeoutExpired:
                process.kill()
                return f"⏰ Execution timed out after {timeout} seconds"
            
    except Exception as e:
        return f"❌ Error: Cannot monitor execution for '{filename}': {str(e)}"

@tool
def backup_code(filename: str) -> str:
    """Create a backup of code file with version control and error handling"""
    try:
        if not filename or not filename.strip():
            return "❌ Error: Filename cannot be empty"
        
        workspace = Path("workspace")
        
        # Clean filename
        filename = filename.strip()
        if filename.startswith("workspace/") or filename.startswith("workspace\\"):
            filename = filename.replace("workspace/", "").replace("workspace\\", "")
            
        file_path = workspace / filename
        
        if not file_path.exists():
            return f"❌ Error: File '{filename}' not found"
        
        # Create backups directory
        backups_dir = workspace / "backups"
        backups_dir.mkdir(exist_ok=True)
        
        # Generate backup filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f"{filename.replace('/', '_').replace('\\', '_')}_{timestamp}.backup"
        backup_path = backups_dir / backup_filename
        
        # Copy file
        import shutil
        try:
            shutil.copy2(file_path, backup_path)
        except Exception as e:
            return f"❌ Error: Cannot create backup: {str(e)}"
        
        # Update backup index
        index_file = backups_dir / "backup_index.json"
        try:
            if index_file.exists():
                with open(index_file, 'r', encoding='utf-8') as f:
                    index = json.load(f)
            else:
                index = {}
        except Exception:
            index = {}
        
        if filename not in index:
            index[filename] = []
        
        index[filename].append({
            "backup_file": backup_filename,
            "timestamp": timestamp,
            "size": backup_path.stat().st_size,
            "created": datetime.now().isoformat()
        })
        
        # Keep only last 10 backups per file
        if len(index[filename]) > 10:
            old_backup = index[filename].pop(0)
            old_file = backups_dir / old_backup["backup_file"]
            if old_file.exists():
                try:
                    old_file.unlink()
                except Exception:
                    pass  # Ignore cleanup errors
        
        # Save updated index
        try:
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index, f, indent=2)
        except Exception:
            pass  # Ignore index update errors
        
        return f"""
✅ Backup created successfully!

📁 Backup Details:
- Original: {filename}
- Backup: {backup_filename}
- Size: {backup_path.stat().st_size} bytes
- Location: workspace/backups/

📚 Total backups for {filename}: {len(index[filename])}
        """.strip()
        
    except Exception as e:
        return f"❌ Error: Cannot create backup for '{filename}': {str(e)}"

# Export all tools including task management
__all__ = [
    # Task management tools for hierarchical system
    'create_task_table',
    'update_task_status', 
    'get_task_summary',
    
    # Basic tools
    'write_code_file',
    'execute_python_file', 
    'read_file_content',
    'list_workspace_files',
    
    # Enhanced tools
    'install_missing_packages',
    'analyze_code_quality',
    'run_tests',
    'create_project_structure',
    'monitor_execution',
    'backup_code',
    'check_security'
]