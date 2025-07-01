#!/usr/bin/env python3
"""
Fix self-referencing fallback functions in all tools files.
"""

import os
import re
from pathlib import Path

def fix_fallback_functions(file_path):
    """Fix self-referencing fallback functions in a file."""
    print(f"Fixing {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix the self-referencing fallback functions
    fixes = [
        (r'def log_info\(msg, \*\*kwargs\): log_info\(msg\)', 'def log_info(msg, **kwargs): logger.info(msg)'),
        (r'def log_success\(msg, \*\*kwargs\): log_info\(msg\)', 'def log_success(msg, **kwargs): logger.info(msg)'),
        (r'def log_warning\(msg, \*\*kwargs\): log_warning\(msg\)', 'def log_warning(msg, **kwargs): logger.warning(msg)'),
        (r'def log_error\(msg, \*\*kwargs\): log_error\(msg\)', 'def log_error(msg, **kwargs): logger.error(msg)'),
        (r'def log_progress\(msg, \*\*kwargs\): log_info\(msg\)', 'def log_progress(msg, **kwargs): logger.info(msg)'),
        (r'def log_debug\(msg, \*\*kwargs\): log_debug\(msg\)', 'def log_debug(msg, **kwargs): logger.debug(msg)'),
        (r'def log_critical\(msg, \*\*kwargs\): log_critical\(msg\)', 'def log_critical(msg, **kwargs): logger.critical(msg)'),
    ]
    
    original_content = content
    for pattern, replacement in fixes:
        content = re.sub(pattern, replacement, content)
    
    # Check for malformed import placement (inside function docstrings)
    if '"""' in content and '# Import logging helpers' in content:
        # Find misplaced imports
        lines = content.split('\n')
        new_lines = []
        in_docstring = False
        skip_until_end_docstring = False
        
        for i, line in enumerate(lines):
            # Skip misplaced imports inside docstrings/functions
            if 'def ' in line and '"""' in lines[i+1:i+5]:
                # Found function definition, check next few lines for docstring
                in_function = True
                new_lines.append(line)
                continue
            
            if skip_until_end_docstring:
                if line.strip().endswith('"""'):
                    skip_until_end_docstring = False
                continue
                
            if (line.strip().startswith('# Import logging helpers') or 
                line.strip().startswith('import sys') or
                line.strip().startswith('from pathlib import Path') or
                line.strip().startswith('parent_dir = Path') or
                'from tools.logs.logging_helpers import' in line or
                'def log_info(msg' in line or
                'def log_success(msg' in line):
                
                # Check if we're inside a docstring
                before_lines = '\n'.join(lines[:i])
                docstring_count = before_lines.count('"""')
                if docstring_count % 2 == 1:  # Inside docstring
                    skip_until_end_docstring = True
                    continue
            
            new_lines.append(line)
        
        content = '\n'.join(new_lines)
    
    # Write back if changed
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✅ Fixed {file_path.name}")
    else:
        print(f"  ⚪ No fixes needed for {file_path.name}")

def main():
    """Fix all Python files in tools directory."""
    tools_dir = Path('./tools')
    
    if not tools_dir.exists():
        print("tools/ directory not found!")
        return
    
    # Find all Python files that were problematic
    python_files = list(tools_dir.rglob('*.py'))
    
    # Also check specific files that had issues
    additional_files = [
        Path('./tools/data/fetchers.py'),
        Path('./tools/models/base.py'),
        Path('./tools/backtesting/engine.py'),
        Path('./tools/data/processors.py'),
        Path('./tools/data/indicators.py'),
        Path('./tools/utils/file_manager.py'),
        Path('./tools/utils/parameter_optimizer.py'),
        Path('./tools/visualization/charts.py'),
        Path('./tools/visualization/comparisons.py'),
        Path('./tools/visualization/reports.py'),
        Path('./tools/backtesting/analyzers.py'),
    ]
    
    all_files = list(set(python_files + additional_files))
    
    print(f"Fixing {len(all_files)} Python files")
    print("=" * 50)
    
    for file_path in all_files:
        if file_path.exists():
            try:
                fix_fallback_functions(file_path)
            except Exception as e:
                print(f"  ❌ Error fixing {file_path}: {e}")
    
    print("=" * 50)
    print("✅ All files processed!")

if __name__ == "__main__":
    main()