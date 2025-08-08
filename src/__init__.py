"""
Intelligent Review Analytics Platform - Core Package

A comprehensive sentiment analysis platform demonstrating progressive ML algorithms,
business intelligence generation, and production deployment capabilities.

Modules:
    data: Data loading, preprocessing, and feature engineering
    models: ML algorithm implementations (Logistic Regression → BERT)
    business: Business intelligence analysis and ROI calculation
    evaluation: Model evaluation, comparison, and validation
    utils: Utility functions, configuration, and visualization

"""

__version__ = "1.0.0"
__author__ = "[Your Name]"
__email__ = "[Your Email]"
__description__ = "Production-ready sentiment analysis platform with progressive ML algorithms"

# Package-level imports for convenience
from .utils.config import Config
from .utils.helpers import setup_logging

# Initialize logging when package is imported
setup_logging()

# Package metadata
__all__ = [
    "data",
    "models", 
    "business",
    "evaluation",
    "utils",
    "Config"
]