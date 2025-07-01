"""File management utilities for the stock analyzer package."""

import os
import json
import pickle
import sys
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

# Import logging helpers
try:
    from ..logs.logging_helpers import log_info, log_success, log_warning, log_error, log_progress, safe_run
    _logging_helpers_available = True
except ImportError:
    _logging_helpers_available = False
    # Fallback to regular logger if logging_helpers not available
    def log_info(msg, **kwargs): logger.info(msg)
    def log_success(msg, **kwargs): logger.info(msg)
    def log_warning(msg, **kwargs): logger.warning(msg) 
    def log_error(msg, **kwargs): logger.error(msg)
    def log_progress(msg, **kwargs): logger.info(msg)
    def safe_run(func): return func

from ..config import OUTPUT_DIR, logger


@safe_run
def list_saved_stock_files_impl() -> str:
    """
    List all saved stock data files and charts in the output directory.
    
    This function provides a comprehensive overview of all generated files,
    categorized by type, with file sizes and modification timestamps.
    
    Returns:
        String with detailed file listing and statistics
    """
    log_success(f"list_saved_stock_files: Listing all files in output directory...")
    
    try:
        if not os.path.exists(OUTPUT_DIR):
            result = f"Output directory '{OUTPUT_DIR}' does not exist."
            log_warning(f"list_saved_stock_files: {result}")
            return result
        
        files = os.listdir(OUTPUT_DIR)
        if not files:
            result = f"No files found in output directory '{OUTPUT_DIR}'."
            log_success(f"list_saved_stock_files: {result}")
            return result
        
        # Categorize files
        csv_files = [f for f in files if f.endswith('.csv')]
        html_files = [f for f in files if f.endswith('.html')]
        pkl_files = [f for f in files if f.endswith('.pkl')]
        json_files = [f for f in files if f.endswith('.json')]
        other_files = [f for f in files if not any(f.endswith(ext) for ext in ['.csv', '.html', '.pkl', '.json'])]
        
        # Calculate total directory size
        total_size = sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in files if os.path.isfile(os.path.join(OUTPUT_DIR, f)))
        total_size_mb = total_size / (1024 * 1024)
        
        summary = f"""list_saved_stock_files: File inventory for output directory:

 DIRECTORY OVERVIEW:
- Location: {OUTPUT_DIR}
- Total Files: {len(files)}
- Total Size: {total_size_mb:.2f} MB

 CSV DATA FILES ({len(csv_files)}):
{chr(10).join([f"  • {f}" for f in csv_files[:10]])}
{'  • ... and more' if len(csv_files) > 10 else ''}

 HTML CHARTS ({len(html_files)}):
{chr(10).join([f"  • {f}" for f in html_files[:10]])}
{'  • ... and more' if len(html_files) > 10 else ''}

🤖 MODEL FILES ({len(pkl_files)}):
{chr(10).join([f"  • {f}" for f in pkl_files[:10]])}
{'  • ... and more' if len(pkl_files) > 10 else ''}

 RESULT FILES ({len(json_files)}):
{chr(10).join([f"  • {f}" for f in json_files[:5]])}
{'  • ... and more' if len(json_files) > 5 else ''}

📂 OTHER FILES ({len(other_files)}):
{chr(10).join([f"  • {f}" for f in other_files[:5]])}
{'  • ... and more' if len(other_files) > 5 else ''}

 FILE TYPE BREAKDOWN:
- Data Files (CSV): {len(csv_files)} files - Raw and processed stock data
- Visualizations (HTML): {len(html_files)} files - Interactive charts and reports
- Models (PKL): {len(pkl_files)} files - Trained machine learning models
- Results (JSON): {len(json_files)} files - Analysis results and metadata
- Other: {len(other_files)} files - Miscellaneous outputs

📍 USAGE NOTES:
- CSV files contain stock data with technical indicators
- HTML files are interactive charts viewable in any browser
- PKL files are trained models for prediction and analysis
- JSON files contain structured results and performance metrics
- All files are timestamped for version control
"""
        
        log_success(f"list_saved_stock_files: Listed {len(files)} files in output directory")
        return summary
        
    except Exception as e:
        error_msg = f"list_saved_stock_files: Error listing files: {str(e)}"
        log_error(f"list_saved_stock_files: {error_msg}")
        return error_msg


def save_text_to_file_impl(
    content: str,
    filename: str,
    file_format: str = "txt",
    custom_header: Optional[str] = None
) -> str:
    """
    Save text content to files in various formats.
    
    This function provides a flexible way to save analysis results, reports,
    or any text content to files with proper formatting and metadata.
    
    Args:
        content: Text content to save
        filename: Base filename (without extension)
        file_format: File format ("txt", "md", "csv", "json")
        custom_header: Optional header to prepend to the content
        
    Returns:
        String with save confirmation and file location
    """
    log_info(f" save_text_to_file: Saving content to {filename}.{file_format}...")
    
    try:
        # Ensure filename doesn't have extension
        if '.' in filename:
            filename = filename.split('.')[0]
        
        # Add timestamp to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_filename = f"{filename}_{timestamp}.{file_format}"
        filepath = os.path.join(OUTPUT_DIR, full_filename)
        
        # Prepare content with header
        if custom_header:
            final_content = f"{custom_header}\\n\\n{content}"
        else:
            # Add default header
            default_header = f"Generated by Stock Analyzer\\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n{'='*50}"
            final_content = f"{default_header}\\n\\n{content}"
        
        # Save based on format
        if file_format.lower() == "json":
            # Try to parse as JSON, otherwise save as text
            try:
                json_data = json.loads(content)
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2)
            except json.JSONDecodeError:
                # Save as text if not valid JSON
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(final_content)
        else:
            # Save as text
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(final_content)
        
        # Get file size
        file_size = os.path.getsize(filepath)
        file_size_kb = file_size / 1024
        
        summary = f"""save_text_to_file: Successfully saved content to file:

 FILE DETAILS:
- Filename: {full_filename}
- Format: {file_format.upper()}
- Size: {file_size_kb:.2f} KB
- Location: {filepath}

📝 CONTENT SUMMARY:
- Lines: {len(final_content.split(chr(10)))}
- Characters: {len(final_content):,}
- Header: {'Custom' if custom_header else 'Default'}

 USAGE NOTES:
- File saved with timestamp for version control
- Content includes metadata header for context
- File is ready for sharing or further processing
- Compatible with standard text editors and viewers
"""
        
        log_success(f"save_text_to_file: Successfully saved {full_filename}")
        return summary
        
    except Exception as e:
        error_msg = f"save_text_to_file: Error saving file '{filename}': {str(e)}"
        log_error(f"save_text_to_file: {error_msg}")
        return error_msg


def debug_file_system_impl(
    symbol: Optional[str] = None,
    show_content: bool = False
) -> str:
    """
    Debug tool to check file system status and help troubleshoot file-related issues.
    
    Args:
        symbol: Stock symbol to check files for (optional)
        show_content: Whether to show sample content from files
        
    Returns:
        String with detailed file system information
    """
    log_info(f" debug_file_system: Starting file system analysis{' for ' + symbol.upper() if symbol else ''}...")
    
    try:
        # Check if output directory exists
        if not os.path.exists(OUTPUT_DIR):
            result = f"debug_file_system: Output directory '{OUTPUT_DIR}' does not exist. Creating it now..."
            log_warning(f"debug_file_system: {result}")
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            return result + "\\nDirectory created successfully."
        
        # Get all files in output directory
        try:
            all_files = os.listdir(OUTPUT_DIR)
        except Exception as e:
            result = f"debug_file_system: Error reading output directory: {str(e)}"
            log_error(f"debug_file_system: {result}")
            return result
        
        if not all_files:
            result = f"debug_file_system: Output directory '{OUTPUT_DIR}' is empty. No files found."
            log_info(f"debug_file_system: {result}")
            return result
        
        # Categorize files
        csv_files = [f for f in all_files if f.endswith('.csv')]
        pkl_files = [f for f in all_files if f.endswith('.pkl')]
        json_files = [f for f in all_files if f.endswith('.json')]
        html_files = [f for f in all_files if f.endswith('.html')]
        other_files = [f for f in all_files if not any(f.endswith(ext) for ext in ['.csv', '.pkl', '.json', '.html'])]
        
        # If symbol specified, filter for that symbol
        if symbol:
            symbol = symbol.upper()
            symbol_csv = [f for f in csv_files if symbol in f.upper()]
            symbol_pkl = [f for f in pkl_files if symbol in f.upper()]
            symbol_json = [f for f in json_files if symbol in f.upper()]
            symbol_html = [f for f in html_files if symbol in f.upper()]
        else:
            symbol_csv = csv_files
            symbol_pkl = pkl_files
            symbol_json = json_files
            symbol_html = html_files
        
        # Calculate sizes
        total_size = 0
        file_details = []
        
        for file in all_files:
            try:
                filepath = os.path.join(OUTPUT_DIR, file)
                if os.path.isfile(filepath):
                    size = os.path.getsize(filepath)
                    total_size += size
                    modified = datetime.fromtimestamp(os.path.getmtime(filepath))
                    
                    file_details.append({
                        'name': file,
                        'size': size,
                        'size_mb': size / (1024 * 1024),
                        'modified': modified,
                        'extension': file.split('.')[-1] if '.' in file else 'none'
                    })
            except Exception as e:
                log_warning(f"Warning: Could not get info for file {file}: {str(e)}")
        
        # Sort by modification time (newest first)
        file_details.sort(key=lambda x: x['modified'], reverse=True)
        
        # Generate summary
        symbol_filter = f" for {symbol}" if symbol else ""
        
        summary = f"""debug_file_system: File system analysis{symbol_filter}:

 DIRECTORY STATUS:
- Path: {OUTPUT_DIR}
- Exists: Yes
- Readable: Yes
- Total Files: {len(all_files)}
- Total Size: {total_size / (1024 * 1024):.2f} MB

 FILE BREAKDOWN{symbol_filter}:
- CSV Files: {len(symbol_csv)} ({', '.join(symbol_csv[:3])}{'...' if len(symbol_csv) > 3 else ''})
- Model Files (PKL): {len(symbol_pkl)} ({', '.join(symbol_pkl[:3])}{'...' if len(symbol_pkl) > 3 else ''})
- Result Files (JSON): {len(symbol_json)} ({', '.join(symbol_json[:3])}{'...' if len(symbol_json) > 3 else ''})
- Chart Files (HTML): {len(symbol_html)} ({', '.join(symbol_html[:3])}{'...' if len(symbol_html) > 3 else ''})
- Other Files: {len(other_files)} ({', '.join(other_files[:3])}{'...' if len(other_files) > 3 else ''})

 RECENT FILES (Top 10):
{chr(10).join([f"  {i+1}. {f['name']} ({f['size_mb']:.2f} MB, {f['modified'].strftime('%Y-%m-%d %H:%M')})" for i, f in enumerate(file_details[:10])])}

 FILE HEALTH CHECK:
- Largest File: {max(file_details, key=lambda x: x['size'])['name'] if file_details else 'None'} ({max(file_details, key=lambda x: x['size'])['size_mb']:.2f} MB if file_details else 0)
- Newest File: {file_details[0]['name'] if file_details else 'None'} ({file_details[0]['modified'].strftime('%Y-%m-%d %H:%M') if file_details else 'N/A'})
- File Extensions: {', '.join(set(f['extension'] for f in file_details)) if file_details else 'None'}

 RECOMMENDATIONS:
- Directory Status: {'Healthy' if len(all_files) > 0 else 'Empty directory'}
- File Organization: {'Good variety of file types' if len(set(f['extension'] for f in file_details)) > 2 else 'Limited file types'}
- Storage Usage: {'Normal' if total_size < 100*1024*1024 else 'Large directory size'}
"""
        
        # Add content preview if requested
        if show_content and file_details:
            sample_file = file_details[0]
            sample_path = os.path.join(OUTPUT_DIR, sample_file['name'])
            
            try:
                if sample_file['extension'] == 'csv':
                    df = pd.read_csv(sample_path, nrows=5)
                    content_preview = f"CSV Sample (first 5 rows):\\n{df.to_string()}"
                elif sample_file['extension'] == 'json':
                    with open(sample_path, 'r') as f:
                        data = json.load(f)
                    content_preview = f"JSON Sample:\\n{json.dumps(data, indent=2)[:500]}..."
                elif sample_file['extension'] in ['txt', 'md']:
                    with open(sample_path, 'r') as f:
                        content = f.read(500)
                    content_preview = f"Text Sample:\\n{content}..."
                else:
                    content_preview = f"Binary file - content preview not available"
                
                summary += f"\\n\\n📄 CONTENT PREVIEW ({sample_file['name']}):\\n{content_preview}"
                
            except Exception as e:
                summary += f"\\n\\n📄 CONTENT PREVIEW: Error reading {sample_file['name']}: {str(e)}"
        
        log_info(f"debug_file_system: Analyzed {len(all_files)} files in output directory")
        return summary
        
    except Exception as e:
        error_msg = f"debug_file_system: Unexpected error: {str(e)}"
        log_error(f"debug_file_system: {error_msg}")
        return error_msg


def discover_files_by_pattern(pattern: str, file_type: Optional[str] = None) -> List[str]:
    """
    Discover files matching a specific pattern.
    
    Args:
        pattern: Pattern to match in filename
        file_type: File extension to filter by (optional)
        
    Returns:
        List of matching filenames
    """
    try:
        if not os.path.exists(OUTPUT_DIR):
            return []
        
        all_files = os.listdir(OUTPUT_DIR)
        
        # Filter by pattern
        matching_files = [f for f in all_files if pattern.lower() in f.lower()]
        
        # Filter by file type if specified
        if file_type:
            if not file_type.startswith('.'):
                file_type = '.' + file_type
            matching_files = [f for f in matching_files if f.endswith(file_type)]
        
        # Sort by modification time (newest first)
        matching_files.sort(
            key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)),
            reverse=True
        )
        
        return matching_files
        
    except Exception as e:
        log_error(f"Error discovering files: {str(e)}")
        return []


def get_file_info(filename: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific file.
    
    Args:
        filename: Name of the file
        
    Returns:
        Dictionary with file information
    """
    try:
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        if not os.path.exists(filepath):
            return {'error': 'File not found'}
        
        stat = os.stat(filepath)
        
        info = {
            'filename': filename,
            'size': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'modified': datetime.fromtimestamp(stat.st_mtime),
            'created': datetime.fromtimestamp(stat.st_ctime),
            'extension': filename.split('.')[-1] if '.' in filename else None,
            'is_readable': os.access(filepath, os.R_OK),
            'is_writable': os.access(filepath, os.W_OK)
        }
        
        return info
        
    except Exception as e:
        return {'error': str(e)}


def cleanup_old_files(days_old: int = 30, file_pattern: Optional[str] = None) -> str:
    """
    Clean up old files from the output directory.
    
    Args:
        days_old: Remove files older than this many days
        file_pattern: Optional pattern to match for deletion
        
    Returns:
        Summary of cleanup operation
    """
    try:
        if not os.path.exists(OUTPUT_DIR):
            return "Output directory does not exist"
        
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        files_removed = []
        total_size_freed = 0
        
        all_files = os.listdir(OUTPUT_DIR)
        
        for filename in all_files:
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            # Check if file matches pattern (if specified)
            if file_pattern and file_pattern.lower() not in filename.lower():
                continue
            
            # Check if file is old enough
            if os.path.getmtime(filepath) < cutoff_time:
                try:
                    file_size = os.path.getsize(filepath)
                    os.remove(filepath)
                    files_removed.append(filename)
                    total_size_freed += file_size
                    log_info(f"Removed old file: {filename}")
                except Exception as e:
                    log_warning(f"Could not remove {filename}: {str(e)}")
        
        summary = f"""File cleanup completed:
- Files removed: {len(files_removed)}
- Space freed: {total_size_freed / (1024 * 1024):.2f} MB
- Cutoff: Files older than {days_old} days
- Pattern filter: {file_pattern or 'None'}
"""
        
        if files_removed:
            summary += f"\\nRemoved files:\\n" + "\\n".join(f"  - {f}" for f in files_removed)
        
        return summary
        
    except Exception as e:
        error_msg = f"Error during cleanup: {str(e)}"
        log_error(error_msg)
        return error_msg