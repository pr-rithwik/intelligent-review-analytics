"""
Data Processing Module for Intelligent Review Analytics Platform

This module provides comprehensive data handling capabilities including:
- Robust TSV file loading with validation
- Text preprocessing and normalization
- Feature engineering for ML algorithms
- Data quality assessment and reporting

Key Components:
    loader: ReviewDataLoader for TSV file handling and validation
    preprocessor: Text cleaning and normalization utilities
    feature_engineer: Feature extraction and vectorization
    validator: Data quality validation and reporting

Example:
    from src.data import ReviewDataLoader
    
    loader = ReviewDataLoader("data/raw/")
    datasets = loader.load_all_datasets()
    checkpoint = loader.get_validation_checkpoint()
"""

from .loader import ReviewDataLoader, load_and_validate_datasets


# Public API
__all__ = [
    "ReviewDataLoader",
    "load_and_validate_datasets"
]

# Module-level constants
SUPPORTED_FORMATS = ['.tsv', '.csv']
REQUIRED_COLUMNS = ['sentiment', 'text']
OPTIONAL_COLUMNS = ['productId', 'userId', 'summary', 'helpfulnessNumerator', 'helpfulnessDenominator']

# Default configuration
DEFAULT_CONFIG = {
    'encoding': 'utf-8',
    'separator': '\t',
    'na_values': ['', 'NULL', 'null', 'NaN', 'nan'],
    'quality_threshold': 90.0,
    'min_text_length': 1,
    'max_text_length': 10000
}