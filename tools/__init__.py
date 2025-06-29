"""Tools - A modular architecture for stock analysis and prediction.

This package provides tools for:
- Fetching and processing stock data
- Training machine learning models
- Backtesting trading strategies
- Generating visualizations and reports
"""

__version__ = "1.0.0"
__author__ = "Genesis Tools Team"

# Import core modules with error handling
try:
    from . import config
    _config_loaded = True
except ImportError as e:
    print(f"Warning: Config module failed to load: {e}")
    _config_loaded = False

try:
    from . import data
    _data_loaded = True
except ImportError as e:
    print(f"Warning: Data module failed to load: {e}")
    _data_loaded = False

try:
    from . import models
    _models_loaded = True
except ImportError as e:
    print(f"Warning: Models module failed to load: {e}")
    _models_loaded = False

# Try to import tools with graceful error handling
try:
    from .tools import *
    _tools_loaded = True
except ImportError as e:
    print(f"Warning: Some dependencies not available: {e}")
    print("Core modules (config, data, models) are still available")
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

print(f"Genesis Tools v{__version__} initialized")
print(f"Available modules: {', '.join(__all__)}")
