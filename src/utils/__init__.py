"""
Utilities Module for Intelligent Review Analytics Platform

This module provides essential utility functions and configurations used
throughout the project including configuration management, logging setup,
visualization tools, and general helper functions.

Key Components:
    config: Centralized configuration management
    helpers: Common utility functions and file operations
    visualization: Plotting and chart creation utilities
    logger: Logging configuration and setup

Example:
    from src.utils import Config, setup_logging
    from src.utils.visualization import create_performance_chart
    
    # Initialize configuration and logging
    config = Config()
    logger = setup_logging()
    
    # Use configuration
    data_path = config.get_data_path('processed')
"""

from .config import Config
from .helpers import (
    setup_logging,
    save_json,
    load_json,
    save_model,
    load_model,
    ensure_directory_exists,
    validate_dataframe_structure,
    create_summary_statistics,
    format_large_number,
    generate_timestamp,
    create_progress_tracker,
    update_progress_tracker
)

# Module metadata
__version__ = "1.0.0"
__author__ = "[Your Name]"

# Public API
__all__ = [
    "Config",
    "setup_logging",
    "save_json",
    "load_json", 
    "save_model",
    "load_model",
    "ensure_directory_exists",
    "validate_dataframe_structure",
    "create_summary_statistics",
    "format_large_number",
    "generate_timestamp",
    "create_progress_tracker",
    "update_progress_tracker"
]

# Default configuration constants
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_CV_FOLDS = 5

# Project constants
PROJECT_NAME = "Intelligent Review Analytics Platform"
PROJECT_VERSION = "1.0.0"
SUPPORTED_ALGORITHMS = ['logistic_regression', 'svm', 'random_forest', 'xgboost', 'bert']