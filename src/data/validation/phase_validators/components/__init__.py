"""
Components Package for Phase Validators

File: src/data/validation/phase_validators/components/__init__.py

Individual component validators for specific aspects of phase validation.
Each component focuses on a single validation responsibility.
"""

from .dataset_validator import DatasetRequirementsValidator
from .baseline_validator import BaselinePerformanceValidator

# Public API
__all__ = [
    'DatasetRequirementsValidator',
    'BaselinePerformanceValidator'
]

# Component registry for dynamic loading
COMPONENT_VALIDATORS = {
    'dataset_requirements': DatasetRequirementsValidator,
    'baseline_performance': BaselinePerformanceValidator,
    # Future components will be added here:
    # 'business_insights': BusinessInsightsValidator,
    # 'preprocessing_pipeline': PreprocessingPipelineValidator,
}

def get_validator_by_name(validator_name: str):
    """
    Get validator class by name.
    
    Args:
        validator_name (str): Name of the validator
        
    Returns:
        Class: Validator class or None if not found
    """
    return COMPONENT_VALIDATORS.get(validator_name)

def get_available_components():
    """Get list of available component validators."""
    return list(COMPONENT_VALIDATORS.keys())