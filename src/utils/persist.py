"""
Model Persistence Module
Save and load trained models and vectorizers.
"""

import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def save_model(model, filepath: Path):
    """
    Save trained model to disk using joblib.
    
    Args:
        model: Trained sklearn model or any picklable object
        filepath: Path where model will be saved (.pkl extension)
        
    Note:
        - Creates parent directories if they don't exist
        - Overwrites existing file
        - Uses joblib for efficient sklearn model serialization
    """
    # Ensure parent directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(model, filepath)
    
    # Get file size for logging
    file_size_mb = filepath.stat().st_size / (1024 * 1024)
    
    logger.info(f"Model saved successfully to: {filepath}")
    logger.info(f"File size: {file_size_mb:.2f} MB")


def load_model(filepath: Path):
    """
    Load trained model from disk.
    
    Args:
        filepath: Path to saved model file (.pkl)
        
    Returns:
        Loaded model object
        
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    logger.info(f"Loading model from: {filepath}")
    
    # Load model
    model = joblib.load(filepath)
    
    logger.info("Model loaded successfully")
    
    return model


def save_vectorizer(vectorizer, filepath: Path):
    """
    Save fitted TF-IDF vectorizer to disk.
    
    Args:
        vectorizer: Fitted TfidfVectorizer
        filepath: Path where vectorizer will be saved (.pkl extension)
        
    Note:
        This is critical - the vectorizer must be saved with the model
        to ensure consistent feature extraction at inference time.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(vectorizer, filepath)
    
    file_size_mb = filepath.stat().st_size / (1024 * 1024)
    
    logger.info(f"Vectorizer saved successfully to: {filepath}")
    logger.info(f"File size: {file_size_mb:.2f} MB")
    
    # Log vocabulary size if available
    if hasattr(vectorizer, 'vocabulary_'):
        vocab_size = len(vectorizer.vocabulary_)
        logger.info(f"Vocabulary size: {vocab_size} terms")


def load_vectorizer(filepath: Path):
    """
    Load fitted TF-IDF vectorizer from disk.
    
    Args:
        filepath: Path to saved vectorizer file (.pkl)
        
    Returns:
        Loaded TfidfVectorizer object
        
    Raises:
        FileNotFoundError: If vectorizer file doesn't exist
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Vectorizer file not found: {filepath}")
    
    logger.info(f"Loading vectorizer from: {filepath}")
    
    vectorizer = joblib.load(filepath)
    
    logger.info("Vectorizer loaded successfully")
    
    # Log vocabulary info if available
    if hasattr(vectorizer, 'vocabulary_'):
        vocab_size = len(vectorizer.vocabulary_)
        logger.info(f"Vocabulary size: {vocab_size} terms")
    
    return vectorizer


def save_artifacts(model, vectorizer, model_path: Path, vectorizer_path: Path):
    """
    Save both model and vectorizer together.
    
    Args:
        model: Trained model
        vectorizer: Fitted vectorizer
        model_path: Path for model file
        vectorizer_path: Path for vectorizer file
        
    Note:
        Convenience function to save both artifacts in one call.
    """
    logger.info("Saving model artifacts...")
    
    save_model(model, model_path)
    save_vectorizer(vectorizer, vectorizer_path)
    
    logger.info("All artifacts saved successfully")


def load_artifacts(model_path: Path, vectorizer_path: Path) -> tuple:
    """
    Load both model and vectorizer together.
    
    Args:
        model_path: Path to model file
        vectorizer_path: Path to vectorizer file
        
    Returns:
        Tuple of (model, vectorizer)
        
    Note:
        Convenience function to load both artifacts in one call.
    """
    logger.info("Loading model artifacts...")
    
    model = load_model(model_path)
    vectorizer = load_vectorizer(vectorizer_path)
    
    logger.info("All artifacts loaded successfully")
    
    return model, vectorizer