"""
Validation Configuration for Intelligent Review Analytics Platform

Centralized configuration for all validation thresholds, business rules,
and quality assessment parameters used throughout the validation framework.

"""

from typing import Dict, List, Any, Union

# =============================================================================
# QUALITY THRESHOLDS AND SCORING
# =============================================================================

QUALITY_THRESHOLDS = {
    'excellent': 95.0,      # A grade - exceptional quality
    'good': 85.0,           # B grade - good quality, ready for production
    'acceptable': 75.0,     # C grade - acceptable, may need monitoring
    'poor': 60.0,           # D grade - poor quality, needs improvement
    'failing': 0.0          # F grade - critical issues, must be fixed
}

QUALITY_GRADES = {
    'A': {'min_score': 95.0, 'description': 'Excellent - Production Ready'},
    'B': {'min_score': 85.0, 'description': 'Good - Minor Issues'},
    'C': {'min_score': 75.0, 'description': 'Acceptable - Monitor Closely'},
    'D': {'min_score': 60.0, 'description': 'Poor - Needs Improvement'},
    'F': {'min_score': 0.0, 'description': 'Failing - Critical Issues'}
}

# =============================================================================
# MISSING VALUE THRESHOLDS
# =============================================================================

MISSING_VALUE_THRESHOLDS = {
    'critical_columns': {
        'max_percentage': 5.0,      # Max 5% missing for critical columns
        'penalty_multiplier': 2.0,  # Heavy penalty for critical column issues
        'columns': ['sentiment', 'text']
    },
    'important_columns': {
        'max_percentage': 15.0,     # Max 15% missing for important columns
        'penalty_multiplier': 1.0,  # Standard penalty
        'columns': ['productId', 'userId']
    },
    'optional_columns': {
        'max_percentage': 50.0,     # Max 50% missing for optional columns
        'penalty_multiplier': 0.1,  # Light penalty
        'columns': ['summary', 'helpfulnessNumerator', 'helpfulnessDenominator']
    }
}

# =============================================================================
# TEXT QUALITY RULES
# =============================================================================

TEXT_QUALITY_RULES = {
    'word_count': {
        'min_words': 1,
        'max_words': 10000,
        'optimal_min': 10,      # Optimal minimum for meaningful reviews
        'optimal_max': 200      # Optimal maximum for readability
    },
    'character_count': {
        'min_chars': 3,
        'max_chars': 50000
    },
    'empty_text': {
        'max_empty_percentage': 5.0,    # Max 5% empty texts allowed
        'penalty_per_percent': 6.0      # 6 points penalty per percent
    },
    'text_diversity': {
        'min_std_word_count': 5.0,      # Minimum standard deviation
        'duplicate_threshold': 0.95     # Similarity threshold for duplicates
    }
}

# =============================================================================
# SENTIMENT VALIDATION RULES
# =============================================================================

SENTIMENT_RULES = {
    'valid_values': [-1, 0, 1],         # Valid sentiment values
    'max_missing_percentage': 0.0,      # Sentiment is critical - no missing allowed
    'class_balance': {
        'severe_imbalance_threshold': 90.0,     # >90% in one class is severe
        'moderate_imbalance_threshold': 80.0,   # >80% in one class is moderate
        'penalty_severe': 15.0,                 # Penalty for severe imbalance
        'penalty_moderate': 5.0                 # Penalty for moderate imbalance
    },
    'invalid_value_penalty': 50.0      # Heavy penalty for invalid values
}

# =============================================================================
# BUSINESS LOGIC RULES
# =============================================================================

BUSINESS_RULES = {
    'product_coverage': {
        'min_reviews_per_product': 1,
        'warning_threshold': 5,         # Warn if product has <5 reviews
        'ideal_min_reviews': 10         # Ideal minimum for statistical significance
    },
    'user_activity': {
        'min_reviews_per_user': 1,
        'power_user_threshold': 10,     # Users with 10+ reviews
        'casual_user_threshold': 3      # Users with <3 reviews are casual
    },
    'duplicate_detection': {
        'max_duplicate_percentage': 5.0,        # Max 5% duplicates allowed
        'penalty_per_percent': 4.0,             # 4 points penalty per percent
        'text_similarity_threshold': 0.95       # 95% similarity = duplicate
    },
    'helpfulness_validation': {
        'logical_consistency': True,             # numerator <= denominator
        'allow_negative_values': False,         # No negative helpfulness
        'min_denominator_for_analysis': 3       # Min votes for meaningful analysis
    }
}

# =============================================================================
# DATA CONSISTENCY RULES
# =============================================================================

CONSISTENCY_RULES = {
    'data_types': {
        'expected_types': {
            'sentiment': ['int64', 'float64', 'object'],
            'text': ['object'],
            'summary': ['object'],
            'productId': ['object'],
            'userId': ['object'],
            'helpfulnessNumerator': ['int64', 'float64'],
            'helpfulnessDenominator': ['int64', 'float64']
        },
        'type_mismatch_penalty': 25.0
    },
    'id_consistency': {
        'max_missing_product_id': 5.0,      # Max 5% missing product IDs
        'max_missing_user_id': 5.0,         # Max 5% missing user IDs
        'empty_id_penalty': 10.0
    },
    'text_sentiment_consistency': {
        'check_empty_text_with_sentiment': True,
        'penalty_per_inconsistency': 15.0
    }
}

# =============================================================================
# PHASE VALIDATION CONFIGURATION
# =============================================================================

PHASE1_REQUIREMENTS = {
    'critical_datasets': ['train', 'validation', 'test'],
    'minimum_dataset_sizes': {
        'train': 4000,
        'validation': 500,
        'test': 500
    },
    'required_insights': [
        'product_categories', 
        'user_behavior', 
        'review_quality', 
        'roi_analysis'
    ],
    'minimum_insights_count': 5,
    'baseline_performance_thresholds': {
        'min_accuracy': 0.65,          # 65% minimum accuracy
        'min_f1_score': 0.60,          # 60% minimum F1-score
        'max_training_time': 300       # 5 minutes maximum training time
    },
    'preprocessing_requirements': {
        'min_success_rate': 90.0,      # 90% preprocessing success rate
        'required_metrics': ['accuracy', 'f1_score', 'training_time_seconds']
    }
}

PHASE2_READINESS_CRITERIA = {
    'scoring_weights': {
        'data_availability': 20,       # 20 points for data availability
        'data_quality': 25,            # 25 points for data quality
        'preprocessing_pipeline': 20,   # 20 points for preprocessing
        'baseline_performance': 20,     # 20 points for baseline model
        'feature_engineering': 15      # 15 points for feature readiness
    },
    'readiness_levels': {
        'READY': 80,                   # 80+ points = ready
        'MOSTLY_READY': 60,            # 60-79 points = mostly ready
        'NEEDS_IMPROVEMENT': 40,       # 40-59 points = needs work
        'NOT_READY': 0                 # <40 points = not ready
    }
}

# =============================================================================
# REPORT GENERATION CONFIGURATION
# =============================================================================

REPORT_CONFIG = {
    'summary_limits': {
        'max_critical_issues_shown': 5,
        'max_warnings_shown': 3,
        'max_recommendations_shown': 5,
        'max_action_items': 5
    },
    'formatting': {
        'decimal_places': 2,
        'percentage_precision': 1,
        'large_number_threshold': 1000
    },
    'severity_indicators': {
        'critical': '🚨',
        'warning': '⚠️',
        'info': 'ℹ️',
        'success': '✅',
        'failure': '❌',
        'recommendation': '💡'
    }
}

# =============================================================================
# BUSINESS METRICS CONFIGURATION
# =============================================================================

BUSINESS_METRICS_CONFIG = {
    'roi_calculation': {
        'manual_processing': {
            'time_per_review_minutes': 5.0,
            'hourly_labor_cost_usd': 60.0,
            'accuracy_rate': 0.85,
            'consistency_factor': 0.75
        },
        'automated_processing': {
            'time_per_review_seconds': 0.1,
            'infrastructure_cost_monthly': 500.0,
            'maintenance_hours_monthly': 8.0
        }
    },
    'impact_thresholds': {
        'high_impact_savings': 50000,      # $50K+ = high impact
        'medium_impact_savings': 20000,     # $20K+ = medium impact
        'low_impact_savings': 5000          # $5K+ = low impact
    }
}

# =============================================================================
# DEFAULT CONFIGURATIONS
# =============================================================================

def get_default_validation_config() -> Dict[str, Any]:
    """
    Get the complete default validation configuration.
    
    Returns:
        Dict[str, Any]: Complete validation configuration
    """
    return {
        'quality_thresholds': QUALITY_THRESHOLDS,
        'missing_value_thresholds': MISSING_VALUE_THRESHOLDS,
        'text_quality_rules': TEXT_QUALITY_RULES,
        'sentiment_rules': SENTIMENT_RULES,
        'business_rules': BUSINESS_RULES,
        'consistency_rules': CONSISTENCY_RULES,
        'phase1_requirements': PHASE1_REQUIREMENTS,
        'phase2_readiness_criteria': PHASE2_READINESS_CRITERIA,
        'report_config': REPORT_CONFIG,
        'business_metrics_config': BUSINESS_METRICS_CONFIG
    }

def create_custom_validation_config(
    quality_threshold: float = 75.0,
    missing_value_tolerance: float = 5.0,
    min_text_length: int = 1,
    max_text_length: int = 10000,
    baseline_accuracy_threshold: float = 0.65
) -> Dict[str, Any]:
    """
    Create a custom validation configuration with specified parameters.
    
    Args:
        quality_threshold (float): Minimum quality score for acceptance
        missing_value_tolerance (float): Maximum missing value percentage for critical columns
        min_text_length (int): Minimum text length in words
        max_text_length (int): Maximum text length in words
        baseline_accuracy_threshold (float): Minimum baseline model accuracy
        
    Returns:
        Dict[str, Any]: Custom validation configuration
    """
    config = get_default_validation_config()
    
    # Update quality thresholds
    config['quality_thresholds']['acceptable'] = quality_threshold
    config['quality_thresholds']['good'] = min(85.0, quality_threshold + 10)
    config['quality_thresholds']['excellent'] = min(95.0, quality_threshold + 20)
    config['quality_thresholds']['poor'] = max(50.0, quality_threshold - 15)
    
    # Update missing value tolerance
    config['missing_value_thresholds']['critical_columns']['max_percentage'] = missing_value_tolerance
    config['missing_value_thresholds']['important_columns']['max_percentage'] = missing_value_tolerance * 3
    
    # Update text quality rules
    config['text_quality_rules']['word_count']['min_words'] = min_text_length
    config['text_quality_rules']['word_count']['max_words'] = max_text_length
    config['text_quality_rules']['character_count']['min_chars'] = max(1, min_text_length * 3)
    
    # Update baseline performance requirements
    config['phase1_requirements']['baseline_performance_thresholds']['min_accuracy'] = baseline_accuracy_threshold
    config['phase1_requirements']['baseline_performance_thresholds']['min_f1_score'] = max(0.50, baseline_accuracy_threshold - 0.05)
    
    return config

def get_validation_config_for_environment(environment: str = 'development') -> Dict[str, Any]:
    """
    Get validation configuration optimized for specific environment.
    
    Args:
        environment (str): Environment type ('development', 'staging', 'production')
        
    Returns:
        Dict[str, Any]: Environment-specific validation configuration
    """
    base_config = get_default_validation_config()
    
    if environment == 'development':
        # More lenient thresholds for development
        base_config['quality_thresholds']['acceptable'] = 60.0
        base_config['missing_value_thresholds']['critical_columns']['max_percentage'] = 10.0
        base_config['phase1_requirements']['baseline_performance_thresholds']['min_accuracy'] = 0.55
        
    elif environment == 'staging':
        # Standard thresholds for staging
        pass  # Use defaults
        
    elif environment == 'production':
        # Stricter thresholds for production
        base_config['quality_thresholds']['acceptable'] = 85.0
        base_config['quality_thresholds']['good'] = 90.0
        base_config['missing_value_thresholds']['critical_columns']['max_percentage'] = 2.0
        base_config['phase1_requirements']['baseline_performance_thresholds']['min_accuracy'] = 0.70
    
    return base_config

# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that a configuration dictionary has all required fields.
    
    Args:
        config (Dict[str, Any]): Configuration to validate
        
    Returns:
        Dict[str, Any]: Validation result with status and issues
    """
    required_sections = [
        'quality_thresholds',
        'missing_value_thresholds', 
        'text_quality_rules',
        'sentiment_rules',
        'business_rules'
    ]
    
    missing_sections = [section for section in required_sections if section not in config]
    
    issues = []
    
    # Check for missing sections
    if missing_sections:
        issues.append(f"Missing required configuration sections: {missing_sections}")
    
    # Validate quality thresholds
    if 'quality_thresholds' in config:
        thresholds = config['quality_thresholds']
        required_thresholds = ['excellent', 'good', 'acceptable', 'poor']
        missing_thresholds = [t for t in required_thresholds if t not in thresholds]
        if missing_thresholds:
            issues.append(f"Missing quality thresholds: {missing_thresholds}")
        
        # Check threshold ordering
        if all(t in thresholds for t in required_thresholds):
            if not (thresholds['excellent'] >= thresholds['good'] >= 
                   thresholds['acceptable'] >= thresholds['poor']):
                issues.append("Quality thresholds must be in descending order")
    
    # Validate missing value thresholds
    if 'missing_value_thresholds' in config:
        mv_config = config['missing_value_thresholds']
        required_mv_sections = ['critical_columns', 'important_columns', 'optional_columns']
        missing_mv_sections = [s for s in required_mv_sections if s not in mv_config]
        if missing_mv_sections:
            issues.append(f"Missing missing value threshold sections: {missing_mv_sections}")
    
    return {
        'is_valid': len(issues) == 0,
        'issues': issues,
        'missing_sections': missing_sections
    }

# =============================================================================
# CONSTANTS FOR EXTERNAL USE
# =============================================================================

# Column categorization for easy reference
CRITICAL_COLUMNS = MISSING_VALUE_THRESHOLDS['critical_columns']['columns']
IMPORTANT_COLUMNS = MISSING_VALUE_THRESHOLDS['important_columns']['columns']
OPTIONAL_COLUMNS = MISSING_VALUE_THRESHOLDS['optional_columns']['columns']

# Valid sentiment values for easy reference
VALID_SENTIMENT_VALUES = SENTIMENT_RULES['valid_values']

# Phase requirements for easy reference
REQUIRED_DATASETS = PHASE1_REQUIREMENTS['critical_datasets']
MINIMUM_DATASET_SIZES = PHASE1_REQUIREMENTS['minimum_dataset_sizes']

# Default file naming conventions
DEFAULT_REPORT_FILENAME = "validation_report_{timestamp}.json"
DEFAULT_SUMMARY_FILENAME = "validation_summary_{timestamp}.md"