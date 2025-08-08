"""
Model Evaluation Module for Intelligent Review Analytics Platform

This module provides comprehensive model evaluation, comparison, and validation
frameworks with statistical significance testing and business-relevant metrics.

Key Components:
    metrics: Core evaluation metrics calculation (accuracy, F1, AUC-ROC)
    validator: Cross-validation framework with statistical testing
    comparator: Model comparison analysis with significance testing
    performance: Performance tracking and monitoring utilities

Evaluation Framework:
    - Stratified k-fold cross-validation for robust performance estimates
    - Statistical significance testing (t-tests, Mann-Whitney U)
    - Business-relevant metrics beyond accuracy
    - Confidence interval calculation
    - Performance vs speed trade-off analysis

Example:
    from src.evaluation import ModelEvaluator, ModelComparator
    
    # Evaluate single model
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(model, X_test, y_test)
    
    # Compare multiple models
    comparator = ModelComparator()
    comparison = comparator.compare_models(models_dict, X_test, y_test)
"""

import numpy as np
from typing import Dict, List, Tuple, Any

# Evaluation metrics configuration
EVALUATION_METRICS = {
    'primary_metrics': {
        'accuracy': {
            'name': 'Accuracy',
            'description': 'Overall classification accuracy',
            'business_interpretation': 'Percentage of reviews correctly classified',
            'optimization_target': 'maximize',
            'threshold_excellent': 0.85,
            'threshold_good': 0.80,
            'threshold_acceptable': 0.75
        },
        'f1_score': {
            'name': 'F1-Score',
            'description': 'Harmonic mean of precision and recall',
            'business_interpretation': 'Balance between catching positive reviews and avoiding false positives',
            'optimization_target': 'maximize',
            'threshold_excellent': 0.84,
            'threshold_good': 0.79,
            'threshold_acceptable': 0.74
        },
        'roc_auc': {
            'name': 'AUC-ROC',
            'description': 'Area under receiver operating characteristic curve',
            'business_interpretation': 'Model ability to distinguish between positive and negative reviews',
            'optimization_target': 'maximize',
            'threshold_excellent': 0.90,
            'threshold_good': 0.85,
            'threshold_acceptable': 0.80
        }
    },
    'secondary_metrics': {
        'precision': {
            'name': 'Precision',
            'description': 'True positives / (True positives + False positives)',
            'business_interpretation': 'When model predicts positive, how often is it correct?',
            'optimization_target': 'maximize'
        },
        'recall': {
            'name': 'Recall (Sensitivity)',
            'description': 'True positives / (True positives + False negatives)',
            'business_interpretation': 'How many positive reviews does the model catch?',
            'optimization_target': 'maximize'
        },
        'specificity': {
            'name': 'Specificity',
            'description': 'True negatives / (True negatives + False positives)',
            'business_interpretation': 'How many negative reviews does the model correctly identify?',
            'optimization_target': 'maximize'
        }
    },
    'business_metrics': {
        'inference_time': {
            'name': 'Inference Time',
            'description': 'Average prediction time per review',
            'business_interpretation': 'Speed of processing for real-time applications',
            'optimization_target': 'minimize',
            'unit': 'milliseconds',
            'threshold_excellent': 10,
            'threshold_good': 50,
            'threshold_acceptable': 100
        },
        'training_time': {
            'name': 'Training Time',
            'description': 'Time required to train the model',
            'business_interpretation': 'Development and retraining cost considerations',
            'optimization_target': 'minimize',
            'unit': 'minutes',
            'threshold_excellent': 5,
            'threshold_good': 15,
            'threshold_acceptable': 30
        },
        'memory_usage': {
            'name': 'Memory Usage',
            'description': 'Memory required for model inference',
            'business_interpretation': 'Infrastructure cost for deployment',
            'optimization_target': 'minimize',
            'unit': 'MB',
            'threshold_excellent': 50,
            'threshold_good': 200,
            'threshold_acceptable': 500
        }
    }
}

# Cross-validation configuration
CROSS_VALIDATION_CONFIG = {
    'default_cv_folds': 5,
    'stratified': True,
    'random_state': 42,
    'shuffle': True,
    'scoring_metrics': ['accuracy', 'f1', 'precision', 'recall', 'roc_auc'],
    'confidence_level': 0.95,
    'min_samples_per_fold': 100
}

# Statistical testing configuration
STATISTICAL_TESTING_CONFIG = {
    'significance_level': 0.05,
    'multiple_comparison_correction': 'bonferroni',
    'effect_size_thresholds': {
        'small': 0.2,
        'medium': 0.5,
        'large': 0.8
    },
    'minimum_improvement_threshold': 0.01,  # 1% improvement
    'tests_to_perform': [
        'paired_t_test',
        'mann_whitney_u',
        'mcnemar_test'
    ]
}

# Performance benchmarks for model comparison
PERFORMANCE_BENCHMARKS = {
    'baseline_expectations': {
        'logistic_regression': {
            'accuracy_mean': 0.78,
            'accuracy_std': 0.03,
            'f1_mean': 0.77,
            'f1_std': 0.03,
            'inference_time_ms': 1,
            'training_time_min': 0.5
        },
        'naive_bayes': {
            'accuracy_mean': 0.75,
            'accuracy_std': 0.04,
            'f1_mean': 0.74,
            'f1_std': 0.04,
            'inference_time_ms': 0.5,
            'training_time_min': 0.2
        },
        'majority_class': {
            'accuracy_mean': 0.65,  # Assuming 65/35 class split
            'f1_mean': 0.39,  # F1 is poor for majority class baseline
            'inference_time_ms': 0.1,
            'training_time_min': 0.0
        }
    },
    'advanced_expectations': {
        'svm': {
            'accuracy_improvement_min': 0.02,  # At least 2% over baseline
            'f1_improvement_min': 0.02,
            'inference_time_max_ms': 10,
            'training_time_max_min': 5
        },
        'random_forest': {
            'accuracy_improvement_min': 0.03,
            'f1_improvement_min': 0.03,
            'inference_time_max_ms': 15,
            'training_time_max_min': 5
        },
        'xgboost': {
            'accuracy_improvement_min': 0.04,
            'f1_improvement_min': 0.04,
            'inference_time_max_ms': 20,
            'training_time_max_min': 15
        },
        'bert': {
            'accuracy_improvement_min': 0.06,
            'f1_improvement_min': 0.06,
            'inference_time_max_ms': 200,
            'training_time_max_min': 60
        }
    }
}

# Model comparison criteria
MODEL_COMPARISON_CRITERIA = {
    'accuracy_focused': {
        'primary_metric': 'accuracy',
        'secondary_metrics': ['f1_score', 'roc_auc'],
        'performance_weight': 0.8,
        'speed_weight': 0.1,
        'interpretability_weight': 0.1,
        'use_case': 'Maximum accuracy for critical business decisions'
    },
    'speed_focused': {
        'primary_metric': 'inference_time',
        'secondary_metrics': ['accuracy', 'memory_usage'],
        'performance_weight': 0.4,
        'speed_weight': 0.5,
        'interpretability_weight': 0.1,
        'use_case': 'High-volume real-time processing'
    },
    'balanced': {
        'primary_metric': 'f1_score',
        'secondary_metrics': ['accuracy', 'inference_time'],
        'performance_weight': 0.5,
        'speed_weight': 0.3,
        'interpretability_weight': 0.2,
        'use_case': 'General purpose production deployment'
    },
    'interpretable': {
        'primary_metric': 'accuracy',
        'secondary_metrics': ['f1_score', 'feature_importance'],
        'performance_weight': 0.6,
        'speed_weight': 0.1,
        'interpretability_weight': 0.3,
        'use_case': 'Business stakeholder communication and transparency'
    }
}

# Evaluation report templates
REPORT_TEMPLATES = {
    'executive_summary': {
        'sections': [
            'model_recommendation',
            'key_metrics',
            'business_impact',
            'implementation_timeline'
        ],
        'max_length_words': 500,
        'technical_detail_level': 'low'
    },
    'technical_detailed': {
        'sections': [
            'methodology',
            'all_metrics',
            'statistical_significance',
            'performance_analysis',
            'recommendations'
        ],
        'max_length_words': 2000,
        'technical_detail_level': 'high'
    },
    'model_comparison': {
        'sections': [
            'algorithm_comparison',
            'trade_off_analysis',
            'deployment_recommendations',
            'confidence_intervals'
        ],
        'max_length_words': 1000,
        'technical_detail_level': 'medium'
    }
}

# Public API - will be populated as modules are implemented
__all__ = [
    "EVALUATION_METRICS",
    "CROSS_VALIDATION_CONFIG",
    "STATISTICAL_TESTING_CONFIG",
    "PERFORMANCE_BENCHMARKS",
    "MODEL_COMPARISON_CRITERIA",
    "REPORT_TEMPLATES"
]