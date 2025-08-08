"""
Configuration Management for Intelligent Review Analytics Platform

Centralized configuration settings for all project components including
data paths, model parameters, evaluation metrics, and deployment settings.

"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import json

class Config:
    """
    Centralized configuration management for the entire project.
    
    Handles paths, model parameters, evaluation settings, and deployment
    configuration with environment-specific overrides.
    """
    
    # Project structure
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
    
    # Data configuration
    DATA_PATHS = {
        'raw': DATA_DIR / "raw",
        'processed': DATA_DIR / "processed", 
        'insights': DATA_DIR / "insights"
    }
    
    # Dataset file names
    DATASET_FILES = {
        'train': 'reviews_train.tsv',
        'validation': 'reviews_validation.tsv',
        'test': 'reviews_test.tsv',
        'submit': 'reviews_submit.tsv'
    }
    
    # Text preprocessing configuration
    TEXT_PREPROCESSING = {
        'min_length': 1,
        'max_length': 10000,
        'encoding': 'utf-8',
        'remove_html': True,
        'remove_urls': True,
        'remove_special_chars': True,
        'lowercase': True,
        'remove_stopwords': True,
        'lemmatize': True
    }
    
    # Feature engineering configuration
    FEATURE_ENGINEERING = {
        'tfidf': {
            'max_features': 10000,
            'ngram_range': (1, 3),
            'min_df': 5,
            'max_df': 0.95,
            'stop_words': 'english',
            'sublinear_tf': True
        },
        'text_stats': {
            'include_length': True,
            'include_word_count': True,
            'include_char_count': True,
            'include_sentence_count': True,
            'include_readability': True,
            'include_sentiment_intensity': True
        }
    }
    
    # Model configuration
    MODEL_CONFIG = {
        'baseline': {
            'algorithm': 'LogisticRegression',
            'params': {
                'C': 1.0,
                'max_iter': 1000,
                'random_state': 42
            }
        },
        'svm': {
            'algorithm': 'SVC',
            'params': {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale',
                'random_state': 42
            }
        },
        'random_forest': {
            'algorithm': 'RandomForestClassifier',
            'params': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'random_state': 42
            }
        },
        'xgboost': {
            'algorithm': 'XGBClassifier',
            'params': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42
            }
        },
        'bert': {
            'model_name': 'bert-base-uncased',
            'params': {
                'num_labels': 2,
                'learning_rate': 2e-5,
                'num_train_epochs': 3,
                'per_device_train_batch_size': 16,
                'warmup_steps': 500,
                'weight_decay': 0.01
            }
        }
    }
    
    # Evaluation configuration
    EVALUATION_CONFIG = {
        'cv_folds': 5,
        'test_size': 0.2,
        'random_state': 42,
        'scoring_metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        'confidence_threshold': 0.8,
        'statistical_tests': True,
        'significance_level': 0.05
    }
    
    # Business intelligence configuration
    BUSINESS_CONFIG = {
        'insights': {
            'min_category_size': 50,
            'min_user_reviews': 5,
            'quality_threshold': 0.8,
            'helpfulness_threshold': 0.7
        },
        'roi_analysis': {
            'manual_time_per_review': 5.0,  # minutes
            'automated_time_per_review': 0.1,  # seconds
            'hourly_labor_cost': 60.0,  # USD
            'annual_review_volume': 100000,
            'customer_lifetime_value': 500.0  # USD
        }
    }
    
    # Visualization configuration
    VISUALIZATION_CONFIG = {
        'style': 'whitegrid',
        'palette': 'husl',
        'figure_size': (12, 8),
        'dpi': 300,
        'save_format': 'png',
        'color_scheme': {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#17becf'
        }
    }
    
    # Streamlit app configuration
    STREAMLIT_CONFIG = {
        'page_title': 'Intelligent Review Analytics Platform',
        'page_icon': '🎯',
        'layout': 'wide',
        'initial_sidebar_state': 'expanded',
        'cache_ttl': 3600,  # 1 hour
        'max_upload_size': 100  # MB
    }
    
    # Logging configuration
    LOGGING_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'log_file': 'logs/app.log',
        'max_bytes': 10485760,  # 10MB
        'backup_count': 5
    }
    
    @classmethod
    def get_data_path(cls, dataset_type: str) -> Path:
        """Get path for specific dataset type."""
        return cls.DATA_PATHS.get(dataset_type, cls.DATA_PATHS['raw'])
    
    @classmethod
    def get_model_config(cls, model_name: str) -> Dict[str, Any]:
        """Get configuration for specific model."""
        return cls.MODEL_CONFIG.get(model_name, cls.MODEL_CONFIG['baseline'])
    
    @classmethod
    def get_output_path(cls, output_type: str = 'insights') -> Path:
        """Get output path for specific output type."""
        output_paths = {
            'insights': cls.OUTPUTS_DIR / "insights",
            'figures': cls.OUTPUTS_DIR / "figures",
            'reports': cls.OUTPUTS_DIR / "reports",
            'models': cls.MODELS_DIR
        }
        return output_paths.get(output_type, cls.OUTPUTS_DIR)
    
    @classmethod
    def create_directories(cls) -> None:
        """Create all necessary directories if they don't exist."""
        directories = [
            cls.DATA_DIR / "raw",
            cls.DATA_DIR / "processed", 
            cls.DATA_DIR / "insights",
            cls.MODELS_DIR / "baseline",
            cls.MODELS_DIR / "traditional",
            cls.MODELS_DIR / "advanced",
            cls.OUTPUTS_DIR / "insights",
            cls.OUTPUTS_DIR / "figures",
            cls.OUTPUTS_DIR / "reports",
            Path("logs")
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def load_custom_config(cls, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load custom configuration from JSON file."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
    
    @classmethod
    def save_config(cls, config_path: str) -> None:
        """Save current configuration to JSON file."""
        config_dict = {
            'data_paths': {str(k): str(v) for k, v in cls.DATA_PATHS.items()},
            'dataset_files': cls.DATASET_FILES,
            'text_preprocessing': cls.TEXT_PREPROCESSING,
            'feature_engineering': cls.FEATURE_ENGINEERING,
            'model_config': cls.MODEL_CONFIG,
            'evaluation_config': cls.EVALUATION_CONFIG,
            'business_config': cls.BUSINESS_CONFIG,
            'visualization_config': cls.VISUALIZATION_CONFIG
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)

# Initialize directories when module is imported
Config.create_directories()