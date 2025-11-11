"""
Feature Engineering Module
Convert text to numerical features using TF-IDF vectorization.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import logging
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


def create_tfidf_vectorizer(max_features: int = 5000,
                            min_df: int = 2,
                            max_df: float = 0.95,
                            ngram_range: tuple = (1, 1)) -> TfidfVectorizer:
    """
    Initialize TF-IDF vectorizer with specified parameters.
    
    Args:
        max_features: Maximum vocabulary size
        min_df: Minimum document frequency (ignore terms appearing in fewer documents)
        max_df: Maximum document frequency (ignore terms appearing in more than this proportion)
        ngram_range: Range of n-grams to extract (default: unigrams only)
        
    Returns:
        Configured TfidfVectorizer instance
        
    Note:
        - Uses default tokenization (word-level)
        - Removes English stop words by default
        - Lowercase conversion is handled in preprocessing
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        strip_accents='unicode',
        stop_words='english',
        sublinear_tf=True  # Use sublinear TF scaling (logarithmic)
    )
    
    logger.info(f"Created TF-IDF vectorizer with max_features={max_features}, "
                f"min_df={min_df}, max_df={max_df}")
    
    return vectorizer


def fit_transform_tfidf(texts: pd.Series,
                        vectorizer: TfidfVectorizer) -> tuple:
    """
    Fit vectorizer on texts and transform to TF-IDF features.
    
    Args:
        texts: Series of cleaned text documents
        vectorizer: TfidfVectorizer instance
        
    Returns:
        Tuple of (X_features, fitted_vectorizer)
        - X_features: Sparse matrix of TF-IDF features
        - fitted_vectorizer: Vectorizer fitted on the texts
        
    Note:
        This should only be called on training data.
        For test data, use transform_tfidf() with the fitted vectorizer.
    """
    logger.info(f"Fitting TF-IDF vectorizer on {len(texts)} documents")
    
    # Fit and transform
    X_features = vectorizer.fit_transform(texts)
    
    # Log vocabulary statistics
    vocabulary_size = len(vectorizer.vocabulary_)
    logger.info(f"Vocabulary size: {vocabulary_size} terms")
    logger.info(f"Feature matrix shape: {X_features.shape}")
    logger.info(f"Feature matrix sparsity: {(1 - X_features.nnz / (X_features.shape[0] * X_features.shape[1])) * 100:.2f}%")
    
    return X_features, vectorizer


def transform_tfidf(texts: pd.Series,
                   vectorizer: TfidfVectorizer) -> csr_matrix:
    """
    Transform new text using pre-fitted vectorizer.
    
    Args:
        texts: Series of cleaned text documents
        vectorizer: Already fitted TfidfVectorizer
        
    Returns:
        Sparse matrix of TF-IDF features
        
    Note:
        - Only transforms, does not fit
        - Vocabulary is fixed from training
        - Used for test data and new predictions
    """
    if not hasattr(vectorizer, 'vocabulary_'):
        raise ValueError("Vectorizer must be fitted before transforming. "
                        "Use fit_transform_tfidf() on training data first.")
    
    logger.info(f"Transforming {len(texts)} documents using fitted vectorizer")
    
    # Transform only
    X_features = vectorizer.transform(texts)
    
    logger.info(f"Transformed feature matrix shape: {X_features.shape}")
    
    return X_features


def get_feature_names(vectorizer: TfidfVectorizer) -> list:
    """
    Get feature names (vocabulary terms) from fitted vectorizer.
    
    Args:
        vectorizer: Fitted TfidfVectorizer
        
    Returns:
        List of feature names
    """
    if not hasattr(vectorizer, 'vocabulary_'):
        raise ValueError("Vectorizer must be fitted before extracting features")
    
    return vectorizer.get_feature_names_out().tolist()


def get_top_features(vectorizer: TfidfVectorizer,
                    X: csr_matrix,
                    top_n: int = 20) -> list:
    """
    Get top features by average TF-IDF score across documents.
    
    Args:
        vectorizer: Fitted TfidfVectorizer
        X: TF-IDF feature matrix
        top_n: Number of top features to return
        
    Returns:
        List of (feature_name, avg_score) tuples
    """
    feature_names = get_feature_names(vectorizer)
    
    # Calculate mean TF-IDF score for each feature
    mean_scores = X.mean(axis=0).A1
    
    # Get top features
    top_indices = mean_scores.argsort()[-top_n:][::-1]
    top_features = [(feature_names[i], mean_scores[i]) for i in top_indices]
    
    return top_features