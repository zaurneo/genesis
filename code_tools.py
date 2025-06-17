# code_tools.py - Enhanced Code Development Tools (Missing from Simple Version)
"""
Enhanced tools for comprehensive code development, testing, and quality assurance.
Based on patterns from the stock analysis project.
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
def install_missing_packages(packages: str) -> str:
    """Automatically install missing Python packages
    
    Args:
        packages: Comma-separated list of package names
        
    Returns:
        Installation results
    """
    try:
        package_list = [pkg.strip() for pkg in packages.split(',')]
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
                    results.append(f"❌ {package}: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                results.append(f"⏰ {package}: Installation timed out")
            except Exception as e:
                results.append(f"❌ {package}: {str(e)}")
        
        return "📦 Package Installation Results:\n" + "\n".join(results)
        
    except Exception as e:
        return f"❌ Error installing packages: {e}"

@tool
def analyze_code_quality(filename: str) -> str:
    """Analyze code quality, style, and complexity
    
    Args:
        filename: Python file to analyze
        
    Returns:
        Code quality report
    """
    try:
        workspace = Path("workspace")
        file_path = workspace / filename
        
        if not file_path.exists():
            return f"❌ File {filename} not found"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Parse the code
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return f"❌ Syntax Error: {e}"
        
        # Basic quality metrics
        lines = code.split('\n')
        total_lines = len(lines)
        code_lines = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        comment_lines = len([line for line in lines if line.strip().startswith('#')])
        blank_lines = total_lines - code_lines - comment_lines
        
        # Function and class analysis
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        # Complexity analysis (simplified McCabe)
        complexity_score = _calculate_complexity(tree)
        
        # Import analysis
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        
        # Generate quality score
        comment_ratio = comment_lines / total_lines if total_lines > 0 else 0
        quality_score = min(100, max(0, 
            100 - (complexity_score * 2) + (comment_ratio * 20)
        ))
        
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
{_get_quality_recommendations(quality_score, comment_ratio, complexity_score, len(functions))}
        """
        
        return report
        
    except Exception as e:
        return f"❌ Error analyzing code quality: {e}"

def _calculate_complexity(tree) -> int:
    """Calculate simplified complexity score"""
    complexity = 1  # Base complexity
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            complexity += len(node.values) - 1
    
    return complexity

def _get_quality_recommendations(score: float, comment_ratio: float, complexity: int, num_functions: int) -> str:
    """Generate quality improvement recommendations"""
    recommendations = []
    
    if score < 60:
        recommendations.append("• Consider refactoring to improve overall quality")
    if comment_ratio < 0.1:
        recommendations.append("• Add more comments to explain complex logic")
    if complexity > 20:
        recommendations.append("• Break down complex functions into smaller ones")
    if num_functions == 0:
        recommendations.append("• Consider organizing code into functions")
    if score >= 80:
        recommendations.append("• Great code quality! Keep it up!")
    
    return "\n".join(recommendations) if recommendations else "• Code quality looks good!"

@tool
def generate_tests(filename: str) -> str:
    """Generate unit tests for Python code
    
    Args:
        filename: Python file to generate tests for
        
    Returns:
        Generated test file content
    """
    try:
        workspace = Path("workspace")
        file_path = workspace / filename
        
        if not file_path.exists():
            return f"❌ File {filename} not found"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Parse to find functions
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return f"❌ Cannot parse {filename}: {e}"
        
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        if not functions:
            return f"⚠️ No functions found in {filename} to test"
        
        # Generate test file
        test_filename = f"test_{filename}"
        module_name = filename.replace('.py', '')
        
        test_content = f'''import unittest
import sys
from pathlib import Path

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the module to test
try:
    import {module_name}
except ImportError as e:
    print(f"Cannot import {module_name}: {{e}}")
    sys.exit(1)

class Test{module_name.title()}(unittest.TestCase):
    """Test cases for {module_name}.py"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        pass
    
    def tearDown(self):
        """Clean up after each test method."""
        pass
'''
        
        # Generate test methods for each function
        for func in functions:
            func_name = func.name
            if not func_name.startswith('_'):  # Skip private functions
                test_content += f'''
    def test_{func_name}(self):
        """Test {func_name} function"""
        # TODO: Add test cases for {func_name}
        # Example:
        # result = {module_name}.{func_name}()
        # self.assertEqual(result, expected_value)
        self.assertTrue(hasattr({module_name}, '{func_name}'), "Function {func_name} should exist")
        
    def test_{func_name}_edge_cases(self):
        """Test {func_name} edge cases"""
        # TODO: Add edge case tests for {func_name}
        pass
'''
        
        test_content += '''

if __name__ == '__main__':
    unittest.main(verbosity=2)
'''
        
        # Save test file
        test_file_path = workspace / test_filename
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        return f"""
✅ Test file generated: {test_filename}

📋 Generated tests for {len(functions)} functions:
{', '.join([f.name for f in functions if not f.name.startswith('_')])}

🧪 To run tests:
python {test_filename}

💡 Remember to fill in the TODO sections with actual test cases!
        """
        
    except Exception as e:
        return f"❌ Error generating tests: {e}"

@tool
def run_tests(test_filename: str) -> str:
    """Run unit tests and return results
    
    Args:
        test_filename: Test file to run
        
    Returns:
        Test execution results
    """
    try:
        workspace = Path("workspace")
        test_file_path = workspace / test_filename
        
        if not test_file_path.exists():
            return f"❌ Test file {test_filename} not found"
        
        # Run the tests
        result = subprocess.run(
            [sys.executable, str(test_file_path)],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=workspace
        )
        
        output = ""
        if result.stdout:
            output += f"📤 TEST OUTPUT:\n{result.stdout}\n"
        
        if result.stderr:
            output += f"❌ TEST ERRORS:\n{result.stderr}\n"
        
        if result.returncode == 0:
            output += "✅ All tests passed!"
        else:
            output += f"❌ Tests failed with exit code: {result.returncode}"
        
        return output
        
    except subprocess.TimeoutExpired:
        return "❌ Tests timed out (30 seconds)"
    except Exception as e:
        return f"❌ Error running tests: {e}"

@tool
def create_project_structure(project_name: str, project_type: str = "basic") -> str:
    """Create a proper project structure
    
    Args:
        project_name: Name of the project
        project_type: Type of project (basic, web, cli, package)
        
    Returns:
        Created project structure info
    """
    try:
        workspace = Path("workspace")
        project_dir = workspace / project_name
        
        if project_dir.exists():
            return f"❌ Project {project_name} already exists"
        
        project_dir.mkdir(parents=True)
        
        # Basic structure for all projects
        (project_dir / "src").mkdir()
        (project_dir / "tests").mkdir()
        (project_dir / "docs").mkdir()
        
        # Create __init__.py files
        (project_dir / "src" / "__init__.py").touch()
        (project_dir / "tests" / "__init__.py").touch()
        
        # Create main module
        main_file = project_dir / "src" / "main.py"
        with open(main_file, 'w') as f:
            f.write(f'''"""
{project_name} - Main module
"""

def main():
    """Main entry point"""
    print("Hello from {project_name}!")

if __name__ == "__main__":
    main()
''')
        
        # Create README
        readme_file = project_dir / "README.md"
        with open(readme_file, 'w') as f:
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
        
        # Create requirements.txt
        requirements_file = project_dir / "requirements.txt"
        with open(requirements_file, 'w') as f:
            if project_type == "web":
                f.write("flask\nrequests\n")
            elif project_type == "cli":
                f.write("click\nargparse\n")
            elif project_type == "package":
                f.write("setuptools\nwheel\n")
            else:
                f.write("# Add your dependencies here\n")
        
        # Project-specific files
        if project_type == "web":
            (project_dir / "templates").mkdir()
            (project_dir / "static").mkdir()
            app_file = project_dir / "src" / "app.py"
            with open(app_file, 'w') as f:
                f.write('''from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, World!"

if __name__ == "__main__":
    app.run(debug=True)
''')
        
        elif project_type == "package":
            setup_file = project_dir / "setup.py"
            with open(setup_file, 'w') as f:
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
        
        # Create .gitignore
        gitignore_file = project_dir / ".gitignore"
        with open(gitignore_file, 'w') as f:
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
        
        files_created = []
        for root, dirs, files in os.walk(project_dir):
            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(project_dir)
                files_created.append(str(rel_path))
        
        return f"""
✅ Project '{project_name}' created successfully!

📁 Project structure ({project_type} type):
{chr(10).join(f"  📄 {file}" for file in sorted(files_created))}

🚀 Next steps:
1. cd workspace/{project_name}
2. Edit src/main.py with your code
3. Add dependencies to requirements.txt
4. Run: python src/main.py
        """
        
    except Exception as e:
        return f"❌ Error creating project: {e}"

@tool
def monitor_execution(filename: str, timeout: int = 10) -> str:
    """Monitor code execution with resource tracking
    
    Args:
        filename: Python file to execute
        timeout: Maximum execution time in seconds
        
    Returns:
        Execution monitoring results
    """
    try:
        import psutil
        workspace = Path("workspace")
        file_path = workspace / filename
        
        if not file_path.exists():
            return f"❌ File {filename} not found"
        
        start_time = time.time()
        
        # Start the process
        process = subprocess.Popen(
            [sys.executable, str(file_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=workspace
        )
        
        # Monitor resource usage
        try:
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
{stdout if stdout else '(no output)'}

{'❌ Errors:' + chr(10) + stderr if stderr else '✅ No errors'}

🎯 Exit Code: {process.returncode}
            """
            
            return result
            
        except ImportError:
            # Fallback without psutil
            stdout, stderr = process.communicate(timeout=timeout)
            execution_time = time.time() - start_time
            
            return f"""
📊 Basic Execution Results for {filename}:
⏱️ Execution Time: {execution_time:.2f} seconds
📤 Output: {stdout if stdout else '(no output)'}
{'❌ Errors: ' + stderr if stderr else '✅ No errors'}
🎯 Exit Code: {process.returncode}

💡 Install psutil for detailed resource monitoring: pip install psutil
            """
            
    except subprocess.TimeoutExpired:
        return f"⏰ Execution timed out after {timeout} seconds"
    except Exception as e:
        return f"❌ Error monitoring execution: {e}"

@tool
def backup_code(filename: str) -> str:
    """Create a backup of code file with version control
    
    Args:
        filename: File to backup
        
    Returns:
        Backup information
    """
    try:
        workspace = Path("workspace")
        file_path = workspace / filename
        
        if not file_path.exists():
            return f"❌ File {filename} not found"
        
        # Create backups directory
        backups_dir = workspace / "backups"
        backups_dir.mkdir(exist_ok=True)
        
        # Generate backup filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f"{filename}_{timestamp}.backup"
        backup_path = backups_dir / backup_filename
        
        # Copy file
        import shutil
        shutil.copy2(file_path, backup_path)
        
        # Update backup index
        index_file = backups_dir / "backup_index.json"
        if index_file.exists():
            with open(index_file, 'r') as f:
                index = json.load(f)
        else:
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
                old_file.unlink()
        
        # Save updated index
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)
        
        return f"""
✅ Backup created successfully!

📁 Backup Details:
- Original: {filename}
- Backup: {backup_filename}
- Size: {backup_path.stat().st_size} bytes
- Location: workspace/backups/

📚 Total backups for {filename}: {len(index[filename])}
        """
        
    except Exception as e:
        return f"❌ Error creating backup: {e}"

@tool
def check_security(filename: str) -> str:
    """Basic security vulnerability check
    
    Args:
        filename: Python file to check
        
    Returns:
        Security analysis report
    """
    try:
        workspace = Path("workspace")
        file_path = workspace / filename
        
        if not file_path.exists():
            return f"❌ File {filename} not found"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
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
        
        return report
        
    except Exception as e:
        return f"❌ Error checking security: {e}"

# Basic code operations (from simple version)
@tool
def write_code_file(filename: str, content: str) -> str:
    """Write code to a file in the workspace"""
    try:
        workspace = Path("workspace")
        workspace.mkdir(exist_ok=True)
        
        file_path = workspace / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"✅ Code written to {file_path}"
    except Exception as e:
        return f"❌ Error writing file: {e}"

@tool
def execute_python_file(filename: str) -> str:
    """Execute a Python file and capture output/errors"""
    try:
        workspace = Path("workspace")
        file_path = workspace / filename
        
        if not file_path.exists():
            return f"❌ File {filename} not found"
        
        # Run the Python file
        result = subprocess.run(
            [sys.executable, str(file_path)],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=workspace
        )
        
        output = ""
        if result.stdout:
            output += f"📤 OUTPUT:\n{result.stdout}\n"
        
        if result.stderr:
            output += f"❌ ERRORS:\n{result.stderr}\n"
        
        if result.returncode == 0:
            output += "✅ Execution successful!"
        else:
            output += f"❌ Exit code: {result.returncode}"
        
        return output
        
    except subprocess.TimeoutExpired:
        return "❌ Execution timed out (30 seconds)"
    except Exception as e:
        return f"❌ Execution error: {e}"

@tool
def read_file_content(filename: str) -> str:
    """Read the content of a file"""
    try:
        workspace = Path("workspace")
        file_path = workspace / filename
        
        if not file_path.exists():
            return f"❌ File {filename} not found"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return f"📄 Content of {filename}:\n{content}"
    except Exception as e:
        return f"❌ Error reading file: {e}"

@tool
def list_workspace_files() -> str:
    """List all files in the workspace"""
    try:
        workspace = Path("workspace")
        if not workspace.exists():
            return "📁 Workspace is empty"
        
        files = list(workspace.rglob("*"))
        files = [f for f in files if f.is_file()]
        
        if not files:
            return "📁 Workspace is empty"
        
        file_list = "📁 Workspace files:\n"
        for file in sorted(files):
            size = file.stat().st_size
            rel_path = file.relative_to(workspace)
            file_list += f"  📄 {rel_path} ({size} bytes)\n"
        
        return file_list
    except Exception as e:
        return f"❌ Error listing files: {e}"

# Export all tools (basic + enhanced)
__all__ = [
    # Basic tools
    'write_code_file',
    'execute_python_file', 
    'read_file_content',
    'list_workspace_files',
    
    # Enhanced tools
    'install_missing_packages',
    'analyze_code_quality',
    'generate_tests',
    'run_tests',
    'create_project_structure',
    'monitor_execution',
    'backup_code',
    'check_security'
]