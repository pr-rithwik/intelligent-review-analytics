"""
Data Validation Package for Intelligent Review Analytics Platform

Comprehensive data validation framework providing structure validation,
quality checks, business rules validation, and phase completion checkpoints.

Public API:
    DataQualityValidator: Main validator orchestrating all components
    validate_dataset_quality: Convenience function for single dataset validation
    validate_multiple_datasets: Convenience function for multi-dataset validation
    
Individual Components:
    DatasetStructureValidator: Structure and schema validation
    MissingValueChecker: Missing value analysis
    TextQualityChecker: Text content quality validation
    SentimentValidator: Sentiment value validation
    DataConsistencyChecker: Logical consistency checks
    
Configuration:
    get_default_validation_config: Get default configuration
    create_custom_validation_config: Create custom configuration
    
Example Usage:
    from src.data.validation import DataQualityValidator
    
    validator = DataQualityValidator()
    results = validator.validate_dataset_quality(df, "training_data")
    
    print(f"Overall Score: {results['overall_assessment']['overall_score']}")
    print(f"Status: {results['overall_assessment']['status']}")
"""

# Core validation components
from .core_validator import (
    DataQualityValidator,
    validate_dataset_quality,
    validate_multiple_datasets
)

# Structure validation
from .structure_validator import DatasetStructureValidator

# Quality checks
from .quality_checks import (
    MissingValueChecker,
    TextQualityChecker,
    SentimentValidator,
    DataConsistencyChecker,
    run_all_quality_checks,
    create_quality_check_suite,
    validate_single_aspect
)

# Configuration management
from .config import (
    get_default_validation_config,
    create_custom_validation_config,
    get_validation_config_for_environment,
    validate_config,
    QUALITY_THRESHOLDS,
    CRITICAL_COLUMNS,
    IMPORTANT_COLUMNS,
    OPTIONAL_COLUMNS,
    VALID_SENTIMENT_VALUES
)

# Public API
__all__ = [
    # Core validator
    'DataQualityValidator',
    'validate_dataset_quality',
    'validate_multiple_datasets',
    
    # Individual validators
    'DatasetStructureValidator',
    'MissingValueChecker',
    'TextQualityChecker',
    'SentimentValidator',
    'DataConsistencyChecker',
    
    # Quality check utilities
    'run_all_quality_checks',
    'create_quality_check_suite',
    'validate_single_aspect',
    
    # Configuration
    'get_default_validation_config',
    'create_custom_validation_config',
    'get_validation_config_for_environment',
    'validate_config',
    
    # Constants
    'QUALITY_THRESHOLDS',
    'CRITICAL_COLUMNS',
    'IMPORTANT_COLUMNS',
    'OPTIONAL_COLUMNS',
    'VALID_SENTIMENT_VALUES'
]

# Version information
__version__ = "1.0.0"

# Package metadata
VALIDATION_FRAMEWORK_INFO = {
    'name': 'Intelligent Review Analytics - Data Validation Framework',
    'version': __version__,
    'components': [
        'Structure Validation',
        'Quality Checks (Missing Values, Text Quality, Sentiment, Consistency)',
        'Business Rules Validation',
        'Phase Completion Checkpoints'
    ],
    'supported_scopes': ['basic', 'standard', 'comprehensive'],
    'output_formats': ['detailed_dict', 'summary_dict', 'json_export']
}

def get_framework_info() -> dict:
    """Get information about the validation framework."""
    return VALIDATION_FRAMEWORK_INFO.copy()

def get_available_validators() -> dict:
    """Get list of available validators and their descriptions."""
    return {
        'DataQualityValidator': 'Main orchestrator for comprehensive validation',
        'DatasetStructureValidator': 'Structure, schema, and basic property validation',
        'MissingValueChecker': 'Missing value analysis with business impact scoring',
        'TextQualityChecker': 'Text content quality including length and diversity',
        'SentimentValidator': 'Sentiment value validation and class balance analysis',
        'DataConsistencyChecker': 'Logical consistency and integrity constraints'
    }

def quick_validate(df, dataset_name="dataset", scope="standard"):
    """
    Quick validation function for immediate feedback.
    
    Args:
        df: DataFrame to validate
        dataset_name: Name for the dataset
        scope: Validation scope ('basic', 'standard', 'comprehensive')
        
    Returns:
        Dict with validation results
    """
    validator = DataQualityValidator()
    return validator.validate_dataset_quality(df, dataset_name, scope)

def validation_health_check():
    """
    Perform a health check of the validation framework.
    
    Returns:
        Dict with framework status
    """
    try:
        # Test imports
        from .core_validator import DataQualityValidator
        from .structure_validator import DatasetStructureValidator
        from .quality_checks import MissingValueChecker
        from .config import get_default_validation_config
        
        # Test configuration loading
        config = get_default_validation_config()
        
        # Test validator initialization
        validator = DataQualityValidator(config)
        
        return {
            'status': 'HEALTHY',
            'framework_version': __version__,
            'components_loaded': len(__all__),
            'config_loaded': len(config) > 0,
            'validators_available': len(get_available_validators())
        }
        
    except Exception as e:
        return {
            'status': 'UNHEALTHY',
            'error': str(e),
            'framework_version': __version__
        }

# Initialize logging for the validation framework
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())