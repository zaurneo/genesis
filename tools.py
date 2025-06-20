# tools.py - Tools for Tech Lead, Writer, and Executor agents

# Tech Lead tools
def review_code(code: str, review_criteria: str = "general"):
    """Review code for quality, best practices, and architectural concerns"""
    return f"Code review completed for {review_criteria} criteria. Reviewed {len(code.split())} words of code. Review findings and recommendations have been analyzed."

def assign_task(task_description: str, assignee: str):
    """Assign a specific task to a team member (writer or executor)"""
    return f"Task assigned to {assignee}: {task_description}. Task has been logged and prioritized."

# Writer tools  
def write_code(requirements: str, language: str = "python"):
    """Write main application code based on requirements"""
    return f"Main application code written in {language} based on requirements: {requirements}. Code structure and implementation completed."

def refactor_code(existing_code: str, refactor_goals: str):
    """Refactor existing code to improve quality and maintainability"""
    return f"Code refactored successfully. Goals addressed: {refactor_goals}. Improved code structure and readability."

# Executor/Tester tools
def execute_code(code: str, execution_context: str = "test"):
    """Execute code and report results"""
    return f"Code executed in {execution_context} environment. Execution completed with results captured and analyzed."

def write_test(test_type: str, target_functionality: str):
    """Write test code for specific functionality"""
    return f"Test code written for {test_type} testing of {target_functionality}. Test cases cover expected scenarios and edge cases."