"""
Machine Learning Models Module for Intelligent Review Analytics Platform

This module implements progressive ML algorithms from simple baselines to
advanced transformer models with systematic evaluation and comparison.

Algorithm Progression:
    1. Logistic Regression (Baseline) - Fast, interpretable
    2. SVM - Better text pattern recognition
    3. Random Forest - Feature interactions, uncertainty estimates
    4. XGBoost - Production-grade performance
    5. BERT - State-of-art transformer accuracy

Key Components:
    baseline: Logistic Regression implementation
    traditional: SVM, Random Forest, XGBoost models
    transformer: BERT implementation with fine-tuning
    ensemble: Model combination strategies
    predictor: Unified prediction interface

Example:
    from src.models import BaselineModel, BERTModel
    from src.models.predictor import UnifiedPredictor
    
    # Train baseline model
    baseline = BaselineModel()
    baseline.train(X_train, y_train)
    
    # Use unified predictor for inference
    predictor = UnifiedPredictor()
    prediction = predictor.predict(text, model_type='baseline')
"""

# Algorithm configurations
ALGORITHM_CONFIGS = {
    'baseline': {
        'name': 'Logistic Regression',
        'description': 'Fast, interpretable baseline with L2 regularization',
        'use_case': 'Quick prototyping, interpretable results',
        'inference_speed': 'Very Fast (<1ms)',
        'training_time': 'Fast (<30s)',
        'interpretability': 'High'
    },
    'svm': {
        'name': 'Support Vector Machine',
        'description': 'Robust text classification with RBF kernel',
        'use_case': 'Better pattern recognition than linear models',
        'inference_speed': 'Fast (<5ms)', 
        'training_time': 'Medium (<5min)',
        'interpretability': 'Medium'
    },
    'random_forest': {
        'name': 'Random Forest',
        'description': 'Ensemble method with feature importance',
        'use_case': 'Feature interactions, uncertainty estimates',
        'inference_speed': 'Medium (<10ms)',
        'training_time': 'Medium (<5min)',
        'interpretability': 'Medium-High'
    },
    'xgboost': {
        'name': 'XGBoost',
        'description': 'Gradient boosting for production performance',
        'use_case': 'Production deployment, high accuracy',
        'inference_speed': 'Medium (<15ms)',
        'training_time': 'Medium-Long (<15min)',
        'interpretability': 'Medium'
    },
    'bert': {
        'name': 'BERT Transformer',
        'description': 'State-of-art pre-trained transformer',
        'use_case': 'Maximum accuracy for critical decisions',
        'inference_speed': 'Slow (>100ms)',
        'training_time': 'Long (>30min)',
        'interpretability': 'Low'
    }
}

# Model performance expectations (placeholders)
PERFORMANCE_EXPECTATIONS = {
    'baseline': {
        'accuracy_range': (0.75, 0.85),
        'f1_range': (0.74, 0.84),
        'training_samples_needed': 1000,
        'memory_usage_mb': 10
    },
    'svm': {
        'accuracy_range': (0.78, 0.88),
        'f1_range': (0.77, 0.87),
        'training_samples_needed': 2000,
        'memory_usage_mb': 25
    },
    'random_forest': {
        'accuracy_range': (0.80, 0.88),
        'f1_range': (0.79, 0.87),
        'training_samples_needed': 1500,
        'memory_usage_mb': 50
    },
    'xgboost': {
        'accuracy_range': (0.82, 0.90),
        'f1_range': (0.81, 0.89),
        'training_samples_needed': 2000,
        'memory_usage_mb': 75
    },
    'bert': {
        'accuracy_range': (0.85, 0.92),
        'f1_range': (0.84, 0.91),
        'training_samples_needed': 3000,
        'memory_usage_mb': 500
    }
}

# Business justifications for each algorithm
BUSINESS_JUSTIFICATIONS = {
    'baseline': {
        'speed': 'Fastest inference for high-volume processing',
        'cost': 'Lowest computational cost for budget-conscious deployment',
        'interpretability': 'Feature weights directly interpretable by business users',
        'deployment': 'Easiest to deploy and maintain in production'
    },
    'svm': {
        'pattern_recognition': 'Superior handling of complex text patterns',
        'robustness': 'Less prone to overfitting on limited training data',
        'versatility': 'Effective across different product categories',
        'proven_performance': 'Well-established in text classification applications'
    },
    'random_forest': {
        'feature_interactions': 'Captures complex relationships between features',
        'uncertainty_quantification': 'Provides confidence estimates for predictions',
        'feature_importance': 'Clear ranking of most important words/features',
        'robustness': 'Handles missing values and noisy data well'
    },
    'xgboost': {
        'production_performance': 'Optimized for production deployment scenarios',
        'accuracy_speed_balance': 'Best balance of accuracy and inference speed',
        'scalability': 'Handles large datasets efficiently',
        'industry_standard': 'Widely adopted in production ML systems'
    },
    'bert': {
        'maximum_accuracy': 'Highest accuracy for critical business decisions',
        'context_understanding': 'Superior understanding of complex review language',
        'transfer_learning': 'Leverages massive pre-training for better performance',
        'future_proof': 'State-of-art approach showing continued improvement'
    }
}

# Public API - will be populated as modules are implemented
__all__ = [
    "ALGORITHM_CONFIGS",
    "PERFORMANCE_EXPECTATIONS", 
    "BUSINESS_JUSTIFICATIONS"
]