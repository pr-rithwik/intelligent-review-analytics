"""
Phase Validators Package

File: src/data/validation/phase_validators/__init__.py

Phase-specific validation modules for systematic project progression.
Provides specialized validators for each project phase with clear handoff criteria.
"""

from .components.dataset_validator import DatasetRequirementsValidator

# Public API
__all__ = [
    'DatasetRequirementsValidator'
]

# Version info
__version__ = '1.0.0'

# Module metadata
PHASE_VALIDATORS_INFO = {
    'description': 'Phase-specific validation modules for systematic project progression',
    'components': [
        'DatasetRequirementsValidator - Dataset validation for phase requirements'
    ],
    'supported_phases': ['Phase1', 'Phase2', 'Phase3'],
    'integration': 'Designed for PhaseCheckpointsValidator orchestrator'
}

def get_available_validators():
    """Get list of available phase validators."""
    return {
        'dataset_requirements': DatasetRequirementsValidator
    }