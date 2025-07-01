"""Tools - A modular architecture for stock analysis and prediction.

This package provides tools for:
- Fetching and processing stock data
- Training machine learning models
- Backtesting trading strategies
- Generating visualizations and reports
"""

__version__ = "1.0.0"
__author__ = "Genesis Tools Team"

# Import logging helpers
import sys
from pathlib import Path

# Add parent directory to path to import logging_helpers
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

try:
    from tools.logs.logging_helpers import log_info, log_warning, log_error, get_logger
    _logging_helpers_available = True
except ImportError:
    _logging_helpers_available = False
    # Fallback to print if logging_helpers not available
    def log_info(msg, **kwargs): print(f"INFO: {msg}")
    def log_warning(msg, **kwargs): print(f"WARNING: {msg}")
    def log_error(msg, **kwargs): print(f"ERROR: {msg}")
    def get_logger(name=None): return None

# Get logger for this module
logger = get_logger(__name__)

# Import core modules with error handling
try:
    from . import config
    _config_loaded = True
except ImportError as e:
    log_warning(f"Config module failed to load: {e}")
    _config_loaded = False

try:
    from . import data
    _data_loaded = True
except ImportError as e:
    log_warning(f"Data module failed to load: {e}")
    _data_loaded = False

try:
    from . import models
    _models_loaded = True
except ImportError as e:
    log_warning(f"Models module failed to load: {e}")
    _models_loaded = False

# Try to import tools with graceful error handling
try:
    from .tools import *
    _tools_loaded = True
except ImportError as e:
    log_warning(f"Some dependencies not available: {e}")
    log_info("Core modules (config, data, models) are still available")
    _tools_loaded = False

# Build __all__ list based on successfully loaded modules
__all__ = []
if _config_loaded:
    __all__.append("config")
if _data_loaded:
    __all__.append("data")
if _models_loaded:
    __all__.append("models")

if _tools_loaded:
    # Add tool functions to __all__ if successfully imported
    try:
        from .tools import __all__ as tools_all
        __all__.extend(tools_all)
    except:
        pass

log_info(f"Genesis Tools v{__version__} initialized")
log_info(f"Available modules: {', '.join(__all__)}")
