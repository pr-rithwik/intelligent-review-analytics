"""
Baseline Model Module
Train and evaluate Logistic Regression baseline model.
"""

from sklearn.linear_model import LogisticRegression
import numpy as np
import logging

logger = logging.getLogger(__name__)


def create_baseline_model(random_state: int = 42,
                         max_iter: int = 1000,
                         solver: str = 'lbfgs',
                         C: float = 1.0) -> LogisticRegression:
    """
    Initialize Logistic Regression baseline model.
    
    Args:
        random_state: Random seed for reproducibility
        max_iter: Maximum iterations for convergence
        solver: Optimization algorithm ('lbfgs', 'liblinear', etc.)
        C: Inverse regularization strength (smaller = stronger regularization)
        
    Returns:
        LogisticRegression instance
        
    Note:
        - 'lbfgs' solver works well for small to medium datasets
        - max_iter=1000 ensures convergence for most cases
        - C=1.0 is a good default (no strong regularization)
    """
    model = LogisticRegression(
        random_state=random_state,
        max_iter=max_iter,
        solver=solver,
        C=C,
        verbose=0,
        n_jobs=-1  # Use all available CPU cores
    )
    
    logger.info(f"Created Logistic Regression model: solver={solver}, "
                f"C={C}, max_iter={max_iter}")
    
    return model


def train_model(X_train, y_train, model: LogisticRegression) -> LogisticRegression:
    """
    Train the model on training data.
    
    Args:
        X_train: TF-IDF features (sparse matrix)
        y_train: Target labels
        model: LogisticRegression instance
        
    Returns:
        Trained model
        
    Note:
        Handles sparse matrices efficiently.
    """
    logger.info(f"Training model on {X_train.shape[0]} samples with "
                f"{X_train.shape[1]} features")
    
    # Fit model
    model.fit(X_train, y_train)
    
    logger.info("Model training complete")
    
    # Log coefficient statistics
    if hasattr(model, 'coef_'):
        coef_stats = {
            'mean': np.mean(np.abs(model.coef_)),
            'std': np.std(model.coef_),
            'max': np.max(np.abs(model.coef_)),
            'min': np.min(np.abs(model.coef_))
        }
        logger.info(f"Coefficient statistics: {coef_stats}")
    
    return model


def predict(model: LogisticRegression, X) -> tuple:
    """
    Make predictions on new data.
    
    Args:
        model: Trained LogisticRegression model
        X: TF-IDF features (sparse matrix)
        
    Returns:
        Tuple of (predictions, probabilities)
        - predictions: Class predictions (array)
        - probabilities: Probability scores for each class (array)
        
    Note:
        probabilities shape: (n_samples, n_classes)
    """
    logger.info(f"Making predictions on {X.shape[0]} samples")
    
    # Get class predictions
    predictions = model.predict(X)
    
    # Get probability scores
    probabilities = model.predict_proba(X)
    
    logger.info("Predictions complete")
    
    return predictions, probabilities


def predict_single(model: LogisticRegression,
                  X,
                  return_confidence: bool = True) -> dict:
    """
    Make prediction for a single sample with detailed output.
    
    Args:
        model: Trained model
        X: TF-IDF features for single sample (1D or 2D array)
        return_confidence: Whether to return confidence score
        
    Returns:
        Dictionary with prediction details:
        - prediction: Class label
        - confidence: Confidence score (if requested)
        - probabilities: All class probabilities (if requested)
    """
    # Ensure 2D shape
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    prediction = model.predict(X)[0]
    
    result = {'prediction': prediction}
    
    if return_confidence:
        probabilities = model.predict_proba(X)[0]
        confidence = np.max(probabilities)
        
        result['confidence'] = confidence
        result['probabilities'] = probabilities
    
    return result


def get_top_coefficients(model: LogisticRegression,
                        feature_names: list,
                        top_n: int = 20) -> dict:
    """
    Get top positive and negative coefficients (most influential features).
    
    Args:
        model: Trained LogisticRegression model
        feature_names: List of feature names (from vectorizer)
        top_n: Number of top features to return for each class
        
    Returns:
        Dictionary with 'positive' and 'negative' top features
    """
    if not hasattr(model, 'coef_'):
        raise ValueError("Model must be trained before extracting coefficients")
    
    # Get coefficients (for binary classification, shape is (1, n_features))
    coefficients = model.coef_[0]
    
    # Get top positive coefficients
    top_positive_idx = np.argsort(coefficients)[-top_n:][::-1]
    top_positive = [(feature_names[i], coefficients[i]) for i in top_positive_idx]
    
    # Get top negative coefficients
    top_negative_idx = np.argsort(coefficients)[:top_n]
    top_negative = [(feature_names[i], coefficients[i]) for i in top_negative_idx]
    
    return {
        'positive': top_positive,
        'negative': top_negative
    }