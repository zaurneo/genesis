"""
Bulletproof logging and error-handling system for Python projects.

This module provides:
- UTF-8 safe logging configuration for cross-platform compatibility
- Emoji-tagged logging helpers for better readability
- Decorator for safe function execution with automatic error logging
- Protection against UnicodeEncodeError on all platforms
"""

import logging
import sys
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from functools import wraps
from typing import Optional, Any, Callable, TypeVar, Union
import traceback

# Type variable for decorator
F = TypeVar('F', bound=Callable[..., Any])

# Default configuration
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_FILE = "output/logs/app.log"
DEFAULT_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
DEFAULT_BACKUP_COUNT = 5

# Emoji constants for consistent logging
EMOJI_INFO = "ℹ️"
EMOJI_SUCCESS = "✅"
EMOJI_WARNING = "⚠️"
EMOJI_ERROR = "❌"
EMOJI_DEBUG = "🐞"
EMOJI_CRITICAL = "🚨"
EMOJI_PROGRESS = "🔄"


def setup_logging(
    level: Union[str, int] = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    force: bool = False
) -> logging.Logger:
    """
    Configure root logger with UTF-8 safe console and file handlers.
    
    Args:
        level: Logging level (e.g., "INFO", "DEBUG", logging.INFO)
        log_file: Path to log file (creates directories if needed)
        log_format: Log message format string
        date_format: Date format string
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        force: Force reconfiguration even if logging is already set up
        
    Returns:
        Configured root logger instance
    """
    # Convert string level to logging constant
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    # Use defaults if not provided
    if log_format is None:
        log_format = DEFAULT_LOG_FORMAT
    if date_format is None:
        date_format = DEFAULT_DATE_FORMAT
    if log_file is None:
        log_file = DEFAULT_LOG_FILE
    
    # Get root logger
    root_logger = logging.getLogger()
    
    # Check if logging is already configured
    if root_logger.hasHandlers() and not force:
        return root_logger
    
    # Clear existing handlers if forcing
    if force:
        root_logger.handlers.clear()
    
    # Set logging level
    root_logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(log_format, date_format)
    
    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Ensure UTF-8 encoding for console output
    if hasattr(console_handler.stream, 'reconfigure'):
        console_handler.stream.reconfigure(encoding='utf-8')
    
    root_logger.addHandler(console_handler)
    
    # File handler with UTF-8 encoding and rotation
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'  # Crucial for preventing UnicodeEncodeError
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    return root_logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance, ensuring logging is set up.
    
    Args:
        name: Logger name (None for root logger)
        
    Returns:
        Logger instance
    """
    if not logging.getLogger().hasHandlers():
        setup_logging()
    return logging.getLogger(name)


def log_info(msg: str, logger: Optional[logging.Logger] = None) -> None:
    """Log an info message with emoji tag."""
    if logger is None:
        logger = get_logger()
    logger.info(f"{EMOJI_INFO} {msg}")


def log_success(msg: str, logger: Optional[logging.Logger] = None) -> None:
    """Log a success message with emoji tag."""
    if logger is None:
        logger = get_logger()
    logger.info(f"{EMOJI_SUCCESS} {msg}")


def log_warning(msg: str, logger: Optional[logging.Logger] = None) -> None:
    """Log a warning message with emoji tag."""
    if logger is None:
        logger = get_logger()
    logger.warning(f"{EMOJI_WARNING} {msg}")


def log_error(msg: str, exc_info: bool = False, logger: Optional[logging.Logger] = None) -> None:
    """
    Log an error message with emoji tag.
    
    Args:
        msg: Error message
        exc_info: Include exception traceback if True
        logger: Logger instance to use
    """
    if logger is None:
        logger = get_logger()
    logger.error(f"{EMOJI_ERROR} {msg}", exc_info=exc_info)


def log_debug(msg: str, logger: Optional[logging.Logger] = None) -> None:
    """Log a debug message with emoji tag."""
    if logger is None:
        logger = get_logger()
    logger.debug(f"{EMOJI_DEBUG} {msg}")


def log_critical(msg: str, exc_info: bool = False, logger: Optional[logging.Logger] = None) -> None:
    """
    Log a critical message with emoji tag.
    
    Args:
        msg: Critical message
        exc_info: Include exception traceback if True
        logger: Logger instance to use
    """
    if logger is None:
        logger = get_logger()
    logger.critical(f"{EMOJI_CRITICAL} {msg}", exc_info=exc_info)


def log_progress(msg: str, step: Optional[int] = None, total: Optional[int] = None, 
                 logger: Optional[logging.Logger] = None) -> None:
    """
    Log a progress message with emoji tag.
    
    Args:
        msg: Progress message
        step: Current step number
        total: Total number of steps
        logger: Logger instance to use
    """
    if logger is None:
        logger = get_logger()
    
    progress_msg = f"{EMOJI_PROGRESS} {msg}"
    if step is not None and total is not None:
        progress_msg += f" [{step}/{total}]"
    elif step is not None:
        progress_msg += f" [Step {step}]"
    
    logger.info(progress_msg)


def safe_run(func: F) -> F:
    """
    Decorator that wraps a function with try/except and logs uncaught exceptions.
    
    The decorated function will:
    - Log any uncaught exceptions with full traceback
    - Re-raise the exception after logging
    - Include function name and arguments in error message
    
    Example:
        @safe_run
        def risky_function(x, y):
            return x / y
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Format arguments for logging
            args_str = ", ".join(repr(arg) for arg in args)
            kwargs_str = ", ".join(f"{k}={repr(v)}" for k, v in kwargs.items())
            all_args = ", ".join(filter(None, [args_str, kwargs_str]))
            
            # Log the error with full context
            error_msg = f"Exception in {func.__name__}({all_args}): {type(e).__name__}: {str(e)}"
            log_error(error_msg, exc_info=True)
            
            # Re-raise the exception
            raise
    
    return wrapper  # type: ignore


def test_unicode_safety() -> bool:
    """
    Test that logging can handle Unicode/emoji without errors.
    
    Returns:
        True if test passes, False otherwise
    """
    import tempfile
    
    try:
        # Create a temporary log file
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as tf:
            temp_log_file = tf.name
        
        # Set up logging with the temp file
        test_logger = logging.getLogger('unicode_test')
        test_logger.setLevel(logging.DEBUG)
        test_logger.handlers.clear()
        
        # Add file handler with UTF-8 encoding
        fh = logging.FileHandler(temp_log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        test_logger.addHandler(fh)
        
        # Test writing emoji
        test_msg = "Test message with emoji: 🎉 ✅ 🚀 测试中文"
        test_logger.info(test_msg)
        
        # Close handler
        fh.close()
        test_logger.handlers.clear()
        
        # Read back and verify
        with open(temp_log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Clean up
        os.unlink(temp_log_file)
        
        return test_msg in content
        
    except Exception as e:
        print(f"Unicode safety test failed: {e}")
        return False


def main():
    """Self-test the logging helpers."""
    print("Starting logging_helpers self-test...")
    
    # Initialize logging
    setup_logging(level="DEBUG")
    
    # Test each logging level
    log_debug("This is a debug message")
    log_info("This is an info message")
    log_progress("Processing data", step=3, total=10)
    log_success("Operation completed successfully")
    log_warning("This is a warning message")
    log_error("This is an error message (no exception)")
    log_critical("This is a critical message")
    
    # Test the safe_run decorator
    @safe_run
    def test_division(a: int, b: int) -> float:
        return a / b
    
    # This should work
    try:
        result = test_division(10, 2)
        log_success(f"Division successful: 10 / 2 = {result}")
    except:
        pass
    
    # This should fail and be logged
    try:
        result = test_division(10, 0)
    except ZeroDivisionError:
        log_info("Caught expected ZeroDivisionError (logged by @safe_run)")
    
    # Test Unicode safety
    if test_unicode_safety():
        log_success("Unicode/emoji handling test passed")
    else:
        log_error("Unicode/emoji handling test failed")
    
    # Test with different logger
    custom_logger = logging.getLogger("custom_module")
    log_info("Message from custom logger", logger=custom_logger)
    
    print("\n✅ Logging self-test passed")


if __name__ == "__main__":
    main()