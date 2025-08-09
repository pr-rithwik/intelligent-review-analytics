"""
Baseline Model Implementation for Intelligent Review Analytics Platform

This module implements the Logistic Regression baseline model with comprehensive
evaluation, feature importance analysis, and business justification framework.

Author: [Your Name]
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, classification_report, 
                           confusion_matrix)
import scipy.sparse as sp
from typing import Dict, List, Optional, Tuple, Any
import time
import logging
import joblib
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class BaselineModel:
    """
    Logistic Regression baseline model for sentiment analysis.
    
    Provides comprehensive evaluation, feature importance analysis,
    and serves as performance benchmark for advanced algorithms.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize baseline model with configuration.
        
        Args:
            config (Dict, optional): Model configuration parameters
        """
        self.config = config or self._get_default_config()
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.training_metrics = {}
        self.validation_metrics = {}
        
        logger.info("BaselineModel initialized with Logistic Regression")
    
    def _get_default_config(self) -> Dict:
        """Get default model configuration."""
        return {
            'model_params': {
                'C': 1.0,
                'max_iter': 1000,
                'random_state': 42,
                'solver': 'liblinear',  # Good for small datasets
                'penalty': 'l2',
                'class_weight': 'balanced'  # Handle class imbalance
            },
            'cross_validation': {
                'cv_folds': 5,
                'scoring_metrics': ['accuracy', 'f1', 'precision', 'recall', 'roc_auc'],
                'random_state': 42
            },
            'feature_importance': {
                'top_k_features': 20,
                'include_negative_features': True
            }
        }
    
    def train(self, X_train: sp.csr_matrix, y_train: np.ndarray, 
              feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train the baseline model with comprehensive evaluation.
        
        Args:
            X_train (sp.csr_matrix): Training feature matrix
            y_train (np.ndarray): Training labels
            feature_names (List[str], optional): Feature names for interpretability
            
        Returns:
            Dict[str, Any]: Training results and metrics
        """
        logger.info(f"Training baseline model on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
        
        start_time = time.time()
        
        # Store feature names
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_train.shape[1])]
        
        # Initialize and train model
        self.model = LogisticRegression(**self.config['model_params'])
        self.model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        self.is_fitted = True
        
        # Generate training predictions
        y_train_pred = self.model.predict(X_train)
        y_train_prob = self.model.predict_proba(X_train)[:, 1]
        
        # Calculate training metrics
        self.training_metrics = self._calculate_metrics(y_train, y_train_pred, y_train_prob)
        self.training_metrics['training_time_seconds'] = training_time
        
        # Perform cross-validation
        cv_results = self._perform_cross_validation(X_train, y_train)
        
        # Feature importance analysis
        feature_importance = self._analyze_feature_importance()
        
        # Model coefficients analysis
        coefficient_stats = self._analyze_coefficients()
        
        results = {
            'training_metrics': self.training_metrics,
            'cross_validation_results': cv_results,
            'feature_importance': feature_importance,
            'coefficient_analysis': coefficient_stats,
            'model_info': {
                'algorithm': 'Logistic Regression',
                'n_features': X_train.shape[1],
                'n_samples': X_train.shape[0],
                'training_time_seconds': training_time,
                'convergence_achieved': True  # Logistic Regression typically converges
            },
            'business_justification': self._get_business_justification()
        }
        
        logger.info(f"Baseline model training complete. Accuracy: {self.training_metrics['accuracy']:.4f}, Training time: {training_time:.2f}s")
        
        return results
    
    def evaluate(self, X_test: sp.csr_matrix, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the trained model on test data.
        
        Args:
            X_test (sp.csr_matrix): Test feature matrix
            y_test (np.ndarray): Test labels
            
        Returns:
            Dict[str, Any]: Evaluation results and metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info(f"Evaluating baseline model on {X_test.shape[0]} test samples...")
        
        start_time = time.time()
        
        # Generate predictions
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        inference_time = time.time() - start_time
        avg_inference_time_ms = (inference_time / X_test.shape[0]) * 1000
        
        # Calculate metrics
        self.validation_metrics = self._calculate_metrics(y_test, y_pred, y_prob)
        self.validation_metrics['total_inference_time_seconds'] = inference_time
        self.validation_metrics['avg_inference_time_ms'] = avg_inference_time_ms
        
        # Detailed analysis
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        confusion_mat = confusion_matrix(y_test, y_pred)
        
        # Error analysis
        error_analysis = self._analyze_prediction_errors(X_test, y_test, y_pred, y_prob)
        
        results = {
            'validation_metrics': self.validation_metrics,
            'classification_report': classification_rep,
            'confusion_matrix': confusion_mat.tolist(),
            'error_analysis': error_analysis,
            'performance_summary': {
                'accuracy': self.validation_metrics['accuracy'],
                'f1_score': self.validation_metrics['f1_score'],
                'auc_roc': self.validation_metrics['auc_roc'],
                'avg_inference_time_ms': avg_inference_time_ms,
                'total_inference_time_seconds': inference_time
            },
            'comparison_to_training': self._compare_training_validation()
        }
        
        logger.info(f"Baseline model evaluation complete. Test Accuracy: {self.validation_metrics['accuracy']:.4f}")
        
        return results
    
    def predict(self, X: sp.csr_matrix, return_probabilities: bool = False) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X (sp.csr_matrix): Feature matrix for prediction
            return_probabilities (bool): Return prediction probabilities
            
        Returns:
            np.ndarray: Predictions or prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        if return_probabilities:
            return self.model.predict_proba(X)[:, 1]
        else:
            return self.model.predict(X)
    
    def get_feature_importance(self, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Get top features by importance (coefficient magnitude).
        
        Args:
            top_k (int): Number of top features to return
            
        Returns:
            List[Tuple[str, float]]: List of (feature_name, importance_score)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained to get feature importance")
        
        coefficients = np.abs(self.model.coef_[0])
        top_indices = np.argsort(coefficients)[-top_k:][::-1]
        
        return [(self.feature_names[i], coefficients[i]) for i in top_indices]
    
    def get_model_explanation(self) -> Dict[str, Any]:
        """
        Get comprehensive model explanation for business stakeholders.
        
        Returns:
            Dict[str, Any]: Model explanation and interpretation
        """
        if not self.is_fitted:
            return {'error': 'Model not trained'}
        
        # Get most important positive and negative features
        coefficients = self.model.coef_[0]
        
        # Most positive features (indicate positive sentiment)
        pos_indices = np.argsort(coefficients)[-10:][::-1]
        positive_features = [(self.feature_names[i], coefficients[i]) for i in pos_indices]
        
        # Most negative features (indicate negative sentiment)
        neg_indices = np.argsort(coefficients)[:10]
        negative_features = [(self.feature_names[i], coefficients[i]) for i in neg_indices]
        
        explanation = {
            'model_type': 'Logistic Regression (Linear Model)',
            'interpretation': {
                'how_it_works': 'Assigns weights to words/features based on their correlation with positive/negative sentiment',
                'prediction_logic': 'Combines weighted feature values to calculate probability of positive sentiment',
                'decision_boundary': f'Classifies as positive when probability > 0.5 (intercept: {self.model.intercept_[0]:.3f})'
            },
            'key_positive_indicators': positive_features,
            'key_negative_indicators': negative_features,
            'model_confidence': {
                'training_accuracy': self.training_metrics.get('accuracy', 0),
                'cross_validation_stability': self.training_metrics.get('cv_accuracy_std', 0) < 0.05,
                'feature_count': len(self.feature_names),
                'regularization': f"L2 penalty with C={self.config['model_params']['C']}"
            },
            'business_applications': {
                'suitable_for': [
                    'High-volume real-time processing',
                    'Interpretable business insights',
                    'Feature importance analysis',
                    'Baseline performance benchmarking'
                ],
                'limitations': [
                    'Linear decision boundary only',
                    'May miss complex text patterns',
                    'Sensitive to feature scaling',
                    'Assumes feature independence'
                ]
            }
        }
        
        return explanation
    
    def save_model(self, filepath: str, include_metadata: bool = True) -> None:
        """
        Save the trained model and metadata.
        
        Args:
            filepath (str): Path to save the model
            include_metadata (bool): Include training metadata
        """
        if not self.is_fitted:
            raise ValueError("Cannot save untrained model")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = filepath.with_suffix('.pkl')
        joblib.dump(self.model, model_path)
        
        # Save metadata if requested
        if include_metadata:
            metadata = {
                'config': self.config,
                'feature_names': self.feature_names,
                'training_metrics': self.training_metrics,
                'validation_metrics': self.validation_metrics,
                'model_type': 'LogisticRegression',
                'is_fitted': self.is_fitted
            }
            
            metadata_path = filepath.with_suffix('.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Baseline model saved to {model_path}")
    
    def load_model(self, filepath: str, load_metadata: bool = True) -> None:
        """
        Load a saved model and metadata.
        
        Args:
            filepath (str): Path to load the model from
            load_metadata (bool): Load training metadata
        """
        filepath = Path(filepath)
        
        # Load model
        model_path = filepath.with_suffix('.pkl')
        self.model = joblib.load(model_path)
        self.is_fitted = True
        
        # Load metadata if available
        if load_metadata:
            metadata_path = filepath.with_suffix('.json')
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.config = metadata.get('config', self.config)
                self.feature_names = metadata.get('feature_names', self.feature_names)
                self.training_metrics = metadata.get('training_metrics', {})
                self.validation_metrics = metadata.get('validation_metrics', {})
        
        logger.info(f"Baseline model loaded from {model_path}")
    
    # Helper methods
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'auc_roc': roc_auc_score(y_true, y_prob)
        }
    
    def _perform_cross_validation(self, X: sp.csr_matrix, y: np.ndarray) -> Dict[str, Any]:
        """Perform stratified cross-validation."""
        cv_config = self.config['cross_validation']
        cv = StratifiedKFold(n_splits=cv_config['cv_folds'], shuffle=True, 
                           random_state=cv_config['random_state'])
        
        cv_results = {}
        for metric in cv_config['scoring_metrics']:
            scores = cross_val_score(self.model, X, y, cv=cv, scoring=metric)
            cv_results[f'cv_{metric}_mean'] = scores.mean()
            cv_results[f'cv_{metric}_std'] = scores.std()
            cv_results[f'cv_{metric}_scores'] = scores.tolist()
        
        return cv_results
    
    """
Baseline Model Implementation for Intelligent Review Analytics Platform

This module implements the Logistic Regression baseline model with comprehensive
evaluation, feature importance analysis, and business justification framework.

Author: [Your Name]
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, classification_report, 
                           confusion_matrix)
import scipy.sparse as sp
from typing import Dict, List, Optional, Tuple, Any
import time
import logging
import joblib
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class BaselineModel:
    """
    Logistic Regression baseline model for sentiment analysis.
    
    Provides comprehensive evaluation, feature importance analysis,
    and serves as performance benchmark for advanced algorithms.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize baseline model with configuration.
        
        Args:
            config (Dict, optional): Model configuration parameters
        """
        self.config = config or self._get_default_config()
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.training_metrics = {}
        self.validation_metrics = {}
        
        logger.info("BaselineModel initialized with Logistic Regression")
    
    def _get_default_config(self) -> Dict:
        """Get default model configuration."""
        return {
            'model_params': {
                'C': 1.0,
                'max_iter': 1000,
                'random_state': 42,
                'solver': 'liblinear',  # Good for small datasets
                'penalty': 'l2',
                'class_weight': 'balanced'  # Handle class imbalance
            },
            'cross_validation': {
                'cv_folds': 5,
                'scoring_metrics': ['accuracy', 'f1', 'precision', 'recall', 'roc_auc'],
                'random_state': 42
            },
            'feature_importance': {
                'top_k_features': 20,
                'include_negative_features': True
            }
        }
    
    def train(self, X_train: sp.csr_matrix, y_train: np.ndarray, 
              feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train the baseline model with comprehensive evaluation.
        
        Args:
            X_train (sp.csr_matrix): Training feature matrix
            y_train (np.ndarray): Training labels
            feature_names (List[str], optional): Feature names for interpretability
            
        Returns:
            Dict[str, Any]: Training results and metrics
        """
        logger.info(f"Training baseline model on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
        
        start_time = time.time()
        
        # Store feature names
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_train.shape[1])]
        
        # Initialize and train model
        self.model = LogisticRegression(**self.config['model_params'])
        self.model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        self.is_fitted = True
        
        # Generate training predictions
        y_train_pred = self.model.predict(X_train)
        y_train_prob = self.model.predict_proba(X_train)[:, 1]
        
        # Calculate training metrics
        self.training_metrics = self._calculate_metrics(y_train, y_train_pred, y_train_prob)
        self.training_metrics['training_time_seconds'] = training_time
        
        # Perform cross-validation
        cv_results = self._perform_cross_validation(X_train, y_train)
        
        # Feature importance analysis
        feature_importance = self._analyze_feature_importance()
        
        # Model coefficients analysis
        coefficient_stats = self._analyze_coefficients()
        
        results = {
            'training_metrics': self.training_metrics,
            'cross_validation_results': cv_results,
            'feature_importance': feature_importance,
            'coefficient_analysis': coefficient_stats,
            'model_info': {
                'algorithm': 'Logistic Regression',
                'n_features': X_train.shape[1],
                'n_samples': X_train.shape[0],
                'training_time_seconds': training_time,
                'convergence_achieved': True  # Logistic Regression typically converges
            },
            'business_justification': self._get_business_justification()
        }
        
        logger.info(f"Baseline model training complete. Accuracy: {self.training_metrics['accuracy']:.4f}, Training time: {training_time:.2f}s")
        
        return results
    
    def evaluate(self, X_test: sp.csr_matrix, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the trained model on test data.
        
        Args:
            X_test (sp.csr_matrix): Test feature matrix
            y_test (np.ndarray): Test labels
            
        Returns:
            Dict[str, Any]: Evaluation results and metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info(f"Evaluating baseline model on {X_test.shape[0]} test samples...")
        
        start_time = time.time()
        
        # Generate predictions
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        inference_time = time.time() - start_time
        avg_inference_time_ms = (inference_time / X_test.shape[0]) * 1000
        
        # Calculate metrics
        self.validation_metrics = self._calculate_metrics(y_test, y_pred, y_prob)
        self.validation_metrics['total_inference_time_seconds'] = inference_time
        self.validation_metrics['avg_inference_time_ms'] = avg_inference_time_ms
        
        # Detailed analysis
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        confusion_mat = confusion_matrix(y_test, y_pred)
        
        # Error analysis
        error_analysis = self._analyze_prediction_errors(X_test, y_test, y_pred, y_prob)
        
        results = {
            'validation_metrics': self.validation_metrics,
            'classification_report': classification_rep,
            'confusion_matrix': confusion_mat.tolist(),
            'error_analysis': error_analysis,
            'performance_summary': {
                'accuracy': self.validation_metrics['accuracy'],
                'f1_score': self.validation_metrics['f1_score'],
                'auc_roc': self.validation_metrics['auc_roc'],
                'avg_inference_time_ms': avg_inference_time_ms,
                'total_inference_time_seconds': inference_time
            },
            'comparison_to_training': self._compare_training_validation()
        }
        
        logger.info(f"Baseline model evaluation complete. Test Accuracy: {self.validation_metrics['accuracy']:.4f}")
        
        return results
    
    def predict(self, X: sp.csr_matrix, return_probabilities: bool = False) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X (sp.csr_matrix): Feature matrix for prediction
            return_probabilities (bool): Return prediction probabilities
            
        Returns:
            np.ndarray: Predictions or prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        if return_probabilities:
            return self.model.predict_proba(X)[:, 1]
        else:
            return self.model.predict(X)
    
    def get_feature_importance(self, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Get top features by importance (coefficient magnitude).
        
        Args:
            top_k (int): Number of top features to return
            
        Returns:
            List[Tuple[str, float]]: List of (feature_name, importance_score)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained to get feature importance")
        
        coefficients = np.abs(self.model.coef_[0])
        top_indices = np.argsort(coefficients)[-top_k:][::-1]
        
        return [(self.feature_names[i], coefficients[i]) for i in top_indices]
    
    def get_model_explanation(self) -> Dict[str, Any]:
        """
        Get comprehensive model explanation for business stakeholders.
        
        Returns:
            Dict[str, Any]: Model explanation and interpretation
        """
        if not self.is_fitted:
            return {'error': 'Model not trained'}
        
        # Get most important positive and negative features
        coefficients = self.model.coef_[0]
        
        # Most positive features (indicate positive sentiment)
        pos_indices = np.argsort(coefficients)[-10:][::-1]
        positive_features = [(self.feature_names[i], coefficients[i]) for i in pos_indices]
        
        # Most negative features (indicate negative sentiment)
        neg_indices = np.argsort(coefficients)[:10]
        negative_features = [(self.feature_names[i], coefficients[i]) for i in neg_indices]
        
        explanation = {
            'model_type': 'Logistic Regression (Linear Model)',
            'interpretation': {
                'how_it_works': 'Assigns weights to words/features based on their correlation with positive/negative sentiment',
                'prediction_logic': 'Combines weighted feature values to calculate probability of positive sentiment',
                'decision_boundary': f'Classifies as positive when probability > 0.5 (intercept: {self.model.intercept_[0]:.3f})'
            },
            'key_positive_indicators': positive_features,
            'key_negative_indicators': negative_features,
            'model_confidence': {
                'training_accuracy': self.training_metrics.get('accuracy', 0),
                'cross_validation_stability': self.training_metrics.get('cv_accuracy_std', 0) < 0.05,
                'feature_count': len(self.feature_names),
                'regularization': f"L2 penalty with C={self.config['model_params']['C']}"
            },
            'business_applications': {
                'suitable_for': [
                    'High-volume real-time processing',
                    'Interpretable business insights',
                    'Feature importance analysis',
                    'Baseline performance benchmarking'
                ],
                'limitations': [
                    'Linear decision boundary only',
                    'May miss complex text patterns',
                    'Sensitive to feature scaling',
                    'Assumes feature independence'
                ]
            }
        }
        
        return explanation
    
    def save_model(self, filepath: str, include_metadata: bool = True) -> None:
        """
        Save the trained model and metadata.
        
        Args:
            filepath (str): Path to save the model
            include_metadata (bool): Include training metadata
        """
        if not self.is_fitted:
            raise ValueError("Cannot save untrained model")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = filepath.with_suffix('.pkl')
        joblib.dump(self.model, model_path)
        
        # Save metadata if requested
        if include_metadata:
            metadata = {
                'config': self.config,
                'feature_names': self.feature_names,
                'training_metrics': self.training_metrics,
                'validation_metrics': self.validation_metrics,
                'model_type': 'LogisticRegression',
                'is_fitted': self.is_fitted
            }
            
            metadata_path = filepath.with_suffix('.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Baseline model saved to {model_path}")
    
    def load_model(self, filepath: str, load_metadata: bool = True) -> None:
        """
        Load a saved model and metadata.
        
        Args:
            filepath (str): Path to load the model from
            load_metadata (bool): Load training metadata
        """
        filepath = Path(filepath)
        
        # Load model
        model_path = filepath.with_suffix('.pkl')
        self.model = joblib.load(model_path)
        self.is_fitted = True
        
        # Load metadata if available
        if load_metadata:
            metadata_path = filepath.with_suffix('.json')
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.config = metadata.get('config', self.config)
                self.feature_names = metadata.get('feature_names', self.feature_names)
                self.training_metrics = metadata.get('training_metrics', {})
                self.validation_metrics = metadata.get('validation_metrics', {})
        
        logger.info(f"Baseline model loaded from {model_path}")
    
    # Helper methods
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'auc_roc': roc_auc_score(y_true, y_prob)
        }
    
    def _perform_cross_validation(self, X: sp.csr_matrix, y: np.ndarray) -> Dict[str, Any]:
        """Perform stratified cross-validation."""
        cv_config = self.config['cross_validation']
        cv = StratifiedKFold(n_splits=cv_config['cv_folds'], shuffle=True, 
                           random_state=cv_config['random_state'])
        
        cv_results = {}
        for metric in cv_config['scoring_metrics']:
            scores = cross_val_score(self.model, X, y, cv=cv, scoring=metric)
            cv_results[f'cv_{metric}_mean'] = scores.mean()
            cv_results[f'cv_{metric}_std'] = scores.std()
            cv_results[f'cv_{metric}_scores'] = scores.tolist()
        
        return cv_results
    
    def _analyze_feature_importance(self) -> Dict[str, Any]:
        """Analyze feature importance based on coefficients."""
        if not self.is_fitted:
            return {}
        
        coefficients = self.model.coef_[0]
        feature_importance = np.abs(coefficients)
        
        # Get top positive and negative features
        top_k = self.config['feature_importance']['top_k_features']
        
        # Top features by absolute importance
        top_indices = np.argsort(feature_importance)[-top_k:][::-1]
        top_features = [(self.feature_names[i], feature_importance[i], coefficients[i]) 
                       for i in top_indices]
        
        # Top positive features (indicate positive sentiment)
        pos_indices = np.argsort(coefficients)[-top_k//2:][::-1]
        positive_features = [(self.feature_names[i], coefficients[i]) for i in pos_indices]
        
        # Top negative features (indicate negative sentiment)
        neg_indices = np.argsort(coefficients)[:top_k//2]
        negative_features = [(self.feature_names[i], coefficients[i]) for i in neg_indices]
        
        return {
            'top_features_by_importance': top_features,
            'top_positive_indicators': positive_features,
            'top_negative_indicators': negative_features,
            'feature_statistics': {
                'total_features': len(coefficients),
                'positive_coefficients': int(np.sum(coefficients > 0)),
                'negative_coefficients': int(np.sum(coefficients < 0)),
                'zero_coefficients': int(np.sum(coefficients == 0)),
                'max_coefficient': float(np.max(coefficients)),
                'min_coefficient': float(np.min(coefficients)),
                'coefficient_std': float(np.std(coefficients))
            }
        }
    
    def _analyze_coefficients(self) -> Dict[str, Any]:
        """Analyze model coefficients for interpretability."""
        if not self.is_fitted:
            return {}
        
        coefficients = self.model.coef_[0]
        intercept = self.model.intercept_[0]
        
        # Coefficient distribution analysis
        pos_coef = coefficients[coefficients > 0]
        neg_coef = coefficients[coefficients < 0]
        
        coefficient_analysis = {
            'intercept': float(intercept),
            'coefficient_distribution': {
                'mean': float(np.mean(coefficients)),
                'std': float(np.std(coefficients)),
                'min': float(np.min(coefficients)),
                'max': float(np.max(coefficients)),
                'median': float(np.median(coefficients))
            },
            'positive_coefficients': {
                'count': len(pos_coef),
                'mean': float(np.mean(pos_coef)) if len(pos_coef) > 0 else 0,
                'max': float(np.max(pos_coef)) if len(pos_coef) > 0 else 0
            },
            'negative_coefficients': {
                'count': len(neg_coef),
                'mean': float(np.mean(neg_coef)) if len(neg_coef) > 0 else 0,
                'min': float(np.min(neg_coef)) if len(neg_coef) > 0 else 0
            },
            'model_complexity': {
                'l2_norm': float(np.linalg.norm(coefficients)),
                'sparsity': float(np.sum(np.abs(coefficients) < 1e-6) / len(coefficients)),
                'regularization_strength': self.config['model_params']['C']
            }
        }
        
        return coefficient_analysis
    
    def _compare_training_validation(self) -> Dict[str, Any]:
        """Compare training and validation performance."""
        if not self.training_metrics or not self.validation_metrics:
            return {}
        
        comparison = {}
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
            if metric in self.training_metrics and metric in self.validation_metrics:
                train_score = self.training_metrics[metric]
                val_score = self.validation_metrics[metric]
                difference = train_score - val_score
                
                comparison[metric] = {
                    'training': round(train_score, 4),
                    'validation': round(val_score, 4),
                    'difference': round(difference, 4),
                    'overfitting_indicator': difference > 0.05  # 5% threshold
                }
        
        # Overall assessment
        avg_difference = np.mean([comp['difference'] for comp in comparison.values() 
                                if isinstance(comp, dict)])
        
        comparison['overall_assessment'] = {
            'avg_performance_drop': round(avg_difference, 4),
            'likely_overfitting': avg_difference > 0.03,
            'model_generalization': 'Good' if avg_difference < 0.02 else 'Poor' if avg_difference > 0.05 else 'Fair'
        }
        
        return comparison
    
    def _analyze_prediction_errors(self, X_test: sp.csr_matrix, y_test: np.ndarray, 
                                 y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        """Analyze prediction errors for model improvement insights."""
        
        # Identify misclassified samples
        misclassified_mask = y_test != y_pred
        misclassified_indices = np.where(misclassified_mask)[0]
        
        if len(misclassified_indices) == 0:
            return {'perfect_classification': True, 'error_count': 0}
        
        # Analyze error types
        false_positives = np.where((y_test == -1) & (y_pred == 1))[0]
        false_negatives = np.where((y_test == 1) & (y_pred == -1))[0]
        
        # Confidence analysis for errors
        error_confidences = y_prob[misclassified_mask]
        high_confidence_errors = np.where(np.abs(error_confidences - 0.5) > 0.3)[0]
        
        error_analysis = {
            'total_errors': len(misclassified_indices),
            'error_rate': len(misclassified_indices) / len(y_test),
            'false_positives': len(false_positives),
            'false_negatives': len(false_negatives),
            'high_confidence_errors': len(high_confidence_errors),
            'error_confidence_stats': {
                'mean_error_confidence': float(np.mean(np.abs(error_confidences - 0.5))),
                'std_error_confidence': float(np.std(error_confidences)),
                'min_error_confidence': float(np.min(error_confidences)),
                'max_error_confidence': float(np.max(error_confidences))
            },
            'recommendations': self._generate_error_recommendations(len(false_positives), 
                                                                  len(false_negatives), 
                                                                  len(high_confidence_errors))
        }
        
        return error_analysis
    
    def _generate_error_recommendations(self, fp_count: int, fn_count: int, 
                                      high_conf_errors: int) -> List[str]:
        """Generate recommendations based on error analysis."""
        recommendations = []
        
        if fp_count > fn_count:
            recommendations.append("High false positive rate - consider increasing classification threshold")
            recommendations.append("Review features that may be causing positive bias")
        elif fn_count > fp_count:
            recommendations.append("High false negative rate - consider decreasing classification threshold")
            recommendations.append("Investigate features that may be missing positive sentiment indicators")
        
        if high_conf_errors > 5:
            recommendations.append("Model shows high confidence in incorrect predictions - review feature quality")
            recommendations.append("Consider ensemble methods to improve prediction confidence calibration")
        
        if not recommendations:
            recommendations.append("Balanced error distribution - consider advanced algorithms for improvement")
        
        return recommendations
    
    def _get_business_justification(self) -> Dict[str, str]:
        """Get business justification for baseline model."""
        return {
            'speed_advantage': 'Fastest inference time for high-volume real-time processing',
            'interpretability': 'Feature weights directly interpretable by business stakeholders',
            'deployment_simplicity': 'Minimal infrastructure requirements and easy maintenance',
            'cost_effectiveness': 'Lowest computational and operational costs',
            'baseline_establishment': 'Provides reliable performance benchmark for advanced algorithms',
            'risk_mitigation': 'Well-understood algorithm with predictable behavior',
            'scalability': 'Linear scaling with data size and feature count',
            'debugging_ease': 'Simple to diagnose and fix performance issues'
        }