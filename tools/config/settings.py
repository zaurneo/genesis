"""Configuration management for the stock analyzer package."""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from .constants import OUTPUT_DIR, LOG_FORMAT, LOG_DATE_FORMAT


@dataclass
class StockAnalyzerSettings:
    """Global settings for the stock analyzer package."""
    
    # Directory settings
    output_dir: Path = OUTPUT_DIR
    log_dir: Optional[Path] = None
    cache_dir: Optional[Path] = None
    
    # Logging settings
    log_level: str = "INFO"
    log_format: str = LOG_FORMAT
    log_date_format: str = LOG_DATE_FORMAT
    enable_file_logging: bool = True
    enable_console_logging: bool = True
    
    # Performance settings
    enable_caching: bool = True
    cache_expiry_minutes: int = 15
    max_workers: int = -1  # -1 means use all available cores
    
    # API settings
    yahoo_finance_timeout: int = 30
    tavily_max_results: int = 5
    
    # Data processing settings
    default_test_size: float = 0.2
    default_target_days: int = 1
    random_state: int = 42
    
    # File size limits
    max_csv_size_mb: int = 100
    max_model_size_mb: int = 500
    
    # Advanced settings
    advanced_features: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize derived settings."""
        # Ensure directories exist
        self.output_dir.mkdir(exist_ok=True)
        
        if self.log_dir is None:
            self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        if self.cache_dir is None:
            self.cache_dir = self.output_dir / "cache"
        if self.enable_caching:
            self.cache_dir.mkdir(exist_ok=True)
    
    def setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        handlers = []
        
        # Console handler
        if self.enable_console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.log_level)
            console_formatter = logging.Formatter(self.log_format, self.log_date_format)
            console_handler.setFormatter(console_formatter)
            handlers.append(console_handler)
        
        # File handler
        if self.enable_file_logging:
            log_file = self.log_dir / "stock_analyzer.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(self.log_level)
            file_formatter = logging.Formatter(self.log_format, self.log_date_format)
            file_handler.setFormatter(file_formatter)
            handlers.append(file_handler)
        
        # Configure root logger
        logging.basicConfig(
            level=self.log_level,
            handlers=handlers,
            force=True  # Override any existing configuration
        )
        
        return logging.getLogger("stock_analyzer")
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get model-specific configuration."""
        from .schemas import PARAMETER_SCHEMAS
        return PARAMETER_SCHEMAS.get(model_type, {}).get("defaults", {})
    
    def get_backtesting_config(self, strategy_type: str) -> Dict[str, Any]:
        """Get backtesting strategy configuration."""
        from .schemas import BACKTESTING_STRATEGIES
        return BACKTESTING_STRATEGIES.get(strategy_type, {}).get("defaults", {})
    
    @classmethod
    def from_env(cls) -> "StockAnalyzerSettings":
        """Create settings from environment variables."""
        return cls(
            output_dir=Path(os.getenv("STOCK_ANALYZER_OUTPUT_DIR", OUTPUT_DIR)),
            log_level=os.getenv("STOCK_ANALYZER_LOG_LEVEL", "INFO"),
            enable_caching=os.getenv("STOCK_ANALYZER_ENABLE_CACHING", "true").lower() == "true",
            cache_expiry_minutes=int(os.getenv("STOCK_ANALYZER_CACHE_EXPIRY", "15")),
            yahoo_finance_timeout=int(os.getenv("YAHOO_FINANCE_TIMEOUT", "30")),
            random_state=int(os.getenv("STOCK_ANALYZER_RANDOM_STATE", "42"))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            "output_dir": str(self.output_dir),
            "log_dir": str(self.log_dir) if self.log_dir else None,
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            "log_level": self.log_level,
            "enable_caching": self.enable_caching,
            "cache_expiry_minutes": self.cache_expiry_minutes,
            "max_workers": self.max_workers,
            "yahoo_finance_timeout": self.yahoo_finance_timeout,
            "default_test_size": self.default_test_size,
            "default_target_days": self.default_target_days,
            "random_state": self.random_state,
            "advanced_features": self.advanced_features
        }


# Global settings instance
settings = StockAnalyzerSettings.from_env()

# Setup logging on import
logger = settings.setup_logging()


def update_settings(**kwargs) -> None:
    """Update global settings."""
    global settings
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
        else:
            settings.advanced_features[key] = value


def get_setting(key: str, default: Any = None) -> Any:
    """Get a setting value."""
    if hasattr(settings, key):
        return getattr(settings, key)
    return settings.advanced_features.get(key, default)


def reset_settings() -> None:
    """Reset settings to defaults."""
    global settings
    settings = StockAnalyzerSettings.from_env()