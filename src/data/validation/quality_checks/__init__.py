"""
Quality Checks Package

Modular quality check implementations for data validation including
missing values, text quality, sentiment validation, and data consistency.
"""

from .base_checker import BaseQualityChecker, QualityCheckUtils
from .missing_values import MissingValueChecker
from .text_quality import TextQualityChecker
from .sentiment_validation import SentimentValidator
from .data_consistency import DataConsistencyChecker

# Public API
__all__ = [
    'BaseQualityChecker',
    'QualityCheckUtils',
    'MissingValueChecker',
    'TextQualityChecker', 
    'SentimentValidator',
    'DataConsistencyChecker',
    'run_all_quality_checks',
    'create_quality_check_suite'
]

def run_all_quality_checks(df, config=None):
    """
    Convenience function to run all quality checks on a dataset.
    
    Args:
        df (pd.DataFrame): Dataset to validate
        config (Dict, optional): Validation configuration
        
    Returns:
        Dict[str, Any]: Combined quality check results
    """
    checkers = [
        MissingValueChecker(config),
        TextQualityChecker(config),
        SentimentValidator(config),
        DataConsistencyChecker(config)
    ]
    
    results = {}
    for checker in checkers:
        try:
            results[checker.checker_name] = checker.check(df)
        except Exception as e:
            results[checker.checker_name] = {
                'error': str(e),
                'score': 0,
                'status': 'ERROR',
                'critical_issues': [f"Checker failed: {str(e)}"],
                'warnings': [],
                'recommendations': ['Fix checker implementation or data format'],
                'details': {},
                'statistics': {}
            }
    
    return results

def create_quality_check_suite(checks_to_include=None, config=None):
    """
    Create a custom quality check suite with specific checkers.
    
    Args:
        checks_to_include (List[str], optional): List of checker names to include
        config (Dict, optional): Validation configuration
        
    Returns:
        List[BaseQualityChecker]: List of configured quality checkers
    """
    available_checkers = {
        'missing_values': MissingValueChecker,
        'text_quality': TextQualityChecker,
        'sentiment_validation': SentimentValidator,
        'data_consistency': DataConsistencyChecker
    }
    
    if checks_to_include is None:
        checks_to_include = list(available_checkers.keys())
    
    checkers = []
    for check_name in checks_to_include:
        if check_name in available_checkers:
            checkers.append(available_checkers[check_name](config))
    
    return checkers

def get_available_checkers():
    """Get list of available quality checker names."""
    return ['missing_values', 'text_quality', 'sentiment_validation', 'data_consistency']

def validate_single_aspect(df, aspect_name, config=None, **kwargs):
    """
    Run a single quality check aspect.
    
    Args:
        df (pd.DataFrame): Dataset to validate
        aspect_name (str): Name of quality aspect to check
        config (Dict, optional): Validation configuration
        **kwargs: Additional parameters for specific checkers
        
    Returns:
        Dict[str, Any]: Single aspect validation results
    """
    checker_map = {
        'missing_values': MissingValueChecker,
        'text_quality': TextQualityChecker,
        'sentiment_validation': SentimentValidator,
        'data_consistency': DataConsistencyChecker
    }
    
    if aspect_name not in checker_map:
        return {
            'error': f"Unknown aspect '{aspect_name}'. Available: {list(checker_map.keys())}",
            'score': 0,
            'status': 'ERROR'
        }
    
    try:
        checker = checker_map[aspect_name](config)
        return checker.check(df, **kwargs)
    except Exception as e:
        return {
            'error': str(e),
            'score': 0,
            'status': 'ERROR',
            'critical_issues': [f"Checker failed: {str(e)}"],
            'warnings': [],
            'recommendations': ['Fix checker implementation or data format']
        }