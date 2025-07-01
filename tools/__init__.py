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

# Try to import submodules with graceful error handling
_backtesting_loaded = False
_visualization_loaded = False
_utils_loaded = False

try:
    from . import backtesting
    _backtesting_loaded = True
except ImportError as e:
    log_warning(f"Backtesting module failed to load: {e}")

try:
    from . import visualization
    _visualization_loaded = True
except ImportError as e:
    log_warning(f"Visualization module failed to load: {e}")

try:
    from . import utils
    _utils_loaded = True
except ImportError as e:
    log_warning(f"Utils module failed to load: {e}")

# Build __all__ list based on successfully loaded modules
__all__ = []
if _config_loaded:
    __all__.append("config")
if _data_loaded:
    __all__.append("data")
if _models_loaded:
    __all__.append("models")
if _backtesting_loaded:
    __all__.append("backtesting")
if _visualization_loaded:
    __all__.append("visualization")
if _utils_loaded:
    __all__.append("utils")

# Import specific functions from submodules for convenience
if _data_loaded:
    try:
        from .data import (
            fetch_yahoo_finance_data_impl,
            get_available_stock_periods_and_intervals_impl,
            read_csv_data_impl,
            apply_technical_indicators_and_transformations_impl
        )
        __all__.extend([
            "fetch_yahoo_finance_data_impl",
            "get_available_stock_periods_and_intervals_impl",
            "read_csv_data_impl",
            "apply_technical_indicators_and_transformations_impl"
        ])
    except ImportError:
        pass

if _models_loaded:
    try:
        from .models.base import train_model_pipeline
        __all__.append("train_model_pipeline")
    except ImportError:
        pass

if _backtesting_loaded:
    try:
        from .backtesting import (
            backtest_model_strategy_impl,
            backtest_multiple_models_impl
        )
        __all__.extend([
            "backtest_model_strategy_impl",
            "backtest_multiple_models_impl"
        ])
    except ImportError:
        pass

if _visualization_loaded:
    try:
        from .visualization import (
            visualize_stock_data_impl,
            visualize_backtesting_results_impl,
            visualize_model_comparison_backtesting_impl,
            generate_comprehensive_html_report_impl
        )
        __all__.extend([
            "visualize_stock_data_impl",
            "visualize_backtesting_results_impl",
            "visualize_model_comparison_backtesting_impl",
            "generate_comprehensive_html_report_impl"
        ])
    except ImportError:
        pass

if _utils_loaded:
    try:
        from .utils import (
            list_saved_stock_files_impl,
            save_text_to_file_impl,
            debug_file_system_impl,
            validate_model_parameters_impl,
            get_model_selection_guide_impl
        )
        __all__.extend([
            "list_saved_stock_files_impl",
            "save_text_to_file_impl",
            "debug_file_system_impl",
            "validate_model_parameters_impl",
            "get_model_selection_guide_impl"
        ])
    except ImportError:
        pass

log_info(f"Genesis Tools v{__version__} initialized")
log_info(f"Available modules: {', '.join(__all__)}")
