"""
Feature Engineering Module for Intelligent Review Analytics Platform

This module provides comprehensive feature extraction capabilities including
TF-IDF vectorization, n-grams, text statistics, and custom sentiment features.

"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy.sparse as sp
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import pickle
import joblib
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class ReviewFeatureEngineer:
    """
    Comprehensive feature engineering for review sentiment analysis.
    
    Combines TF-IDF vectorization, text statistics, and custom features
    to create optimal feature representations for ML algorithms.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize feature engineer with configuration.
        
        Args:
            config (Dict, optional): Feature engineering configuration
        """
        self.config = config or self._get_default_config()
        self.vectorizer = None
        self.scaler = None
        self.feature_names = []
        self.is_fitted = False
        
        logger.info("ReviewFeatureEngineer initialized")
    
    def _get_default_config(self) -> Dict:
        """Get default feature engineering configuration."""
        return {
            'tfidf': {
                'max_features': 10000,
                'ngram_range': (1, 3),
                'min_df': 5,
                'max_df': 0.95,
                'stop_words': 'english',
                'sublinear_tf': True,
                'norm': 'l2'
            },
            'text_features': {
                'include_length_features': True,
                'include_readability_features': True,
                'include_sentiment_features': True,
                'include_lexical_features': True,
                'include_structural_features': True
            },
            'scaling': {
                'scale_tfidf': False,
                'scale_text_features': True,
                'scaler_type': 'standard'  # 'standard' or 'minmax'
            },
            'feature_selection': {
                'enable': False,
                'method': 'chi2',
                'k_best': 5000
            }
        }
    
    def extract_text_statistics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract comprehensive text statistics features.
        
        Args:
            df (pd.DataFrame): DataFrame with text quality metrics
            
        Returns:
            pd.DataFrame: Text statistics features
        """
        features = pd.DataFrame(index=df.index)
        
        if not self.config['text_features']['include_length_features']:
            return features
        
        # Length features
        if 'word_count' in df.columns:
            features['word_count'] = df['word_count'].fillna(0)
            features['word_count_log'] = np.log1p(features['word_count'])
        
        if 'char_count' in df.columns:
            features['char_count'] = df['char_count'].fillna(0)
            features['char_count_log'] = np.log1p(features['char_count'])
        
        if 'sentence_count' in df.columns:
            features['sentence_count'] = df['sentence_count'].fillna(0)
        
        # Derived length features
        if 'avg_word_length' in df.columns:
            features['avg_word_length'] = df['avg_word_length'].fillna(0)
        
        if 'avg_sentence_length' in df.columns:
            features['avg_sentence_length'] = df['avg_sentence_length'].fillna(0)
        
        # Readability features
        if self.config['text_features']['include_readability_features']:
            if 'flesch_reading_ease' in df.columns:
                features['flesch_reading_ease'] = df['flesch_reading_ease'].fillna(50)
                
                # Create readability categories
                features['readability_very_easy'] = (features['flesch_reading_ease'] >= 90).astype(int)
                features['readability_easy'] = ((features['flesch_reading_ease'] >= 80) & 
                                              (features['flesch_reading_ease'] < 90)).astype(int)
                features['readability_standard'] = ((features['flesch_reading_ease'] >= 60) & 
                                                   (features['flesch_reading_ease'] < 80)).astype(int)
                features['readability_difficult'] = (features['flesch_reading_ease'] < 60).astype(int)
        
        # Sentiment features
        if self.config['text_features']['include_sentiment_features']:
            sentiment_cols = ['sentiment_compound', 'sentiment_positive', 
                            'sentiment_negative', 'sentiment_neutral']
            for col in sentiment_cols:
                if col in df.columns:
                    features[col] = df[col].fillna(0)
            
            # Sentiment intensity categories
            if 'sentiment_compound' in df.columns:
                compound = df['sentiment_compound'].fillna(0)
                features['sentiment_very_positive'] = (compound >= 0.5).astype(int)
                features['sentiment_positive'] = ((compound >= 0.1) & (compound < 0.5)).astype(int)
                features['sentiment_neutral'] = ((compound >= -0.1) & (compound < 0.1)).astype(int)
                features['sentiment_negative'] = ((compound >= -0.5) & (compound < -0.1)).astype(int)
                features['sentiment_very_negative'] = (compound < -0.5).astype(int)
        
        # Lexical diversity features
        if self.config['text_features']['include_lexical_features']:
            if 'diversity_ratio' in df.columns:
                features['diversity_ratio'] = df['diversity_ratio'].fillna(0)
            
            if 'unique_word_count' in df.columns:
                features['unique_word_count'] = df['unique_word_count'].fillna(0)
                features['unique_word_count_log'] = np.log1p(features['unique_word_count'])
        
        # Structural features
        if self.config['text_features']['include_structural_features']:
            # Word to sentence ratio
            if 'word_count' in features.columns and 'sentence_count' in features.columns:
                features['words_per_sentence'] = (features['word_count'] / 
                                                 features['sentence_count'].replace(0, 1))
            
            # Character to word ratio
            if 'char_count' in features.columns and 'word_count' in features.columns:
                features['chars_per_word'] = (features['char_count'] / 
                                             features['word_count'].replace(0, 1))
        
        logger.info(f"Extracted {features.shape[1]} text statistics features")
        return features
    
    def create_tfidf_features(self, texts: List[str], fit: bool = True) -> sp.csr_matrix:
        """
        Create TF-IDF features from text data.
        
        Args:
            texts (List[str]): List of cleaned texts
            fit (bool): Whether to fit the vectorizer
            
        Returns:
            scipy.sparse.csr_matrix: TF-IDF feature matrix
        """
        if fit:
            self.vectorizer = TfidfVectorizer(**self.config['tfidf'])
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Store feature names
            self.tfidf_feature_names = [f'tfidf_{name}' for name in self.vectorizer.get_feature_names_out()]
            
            logger.info(f"TF-IDF vectorizer fitted with {tfidf_matrix.shape[1]} features")
        else:
            if self.vectorizer is None:
                raise ValueError("Vectorizer not fitted. Call with fit=True first.")
            tfidf_matrix = self.vectorizer.transform(texts)
        
        return tfidf_matrix
    
    def create_ngram_features(self, texts: List[str], ngram_range: Tuple[int, int] = (2, 3), 
                             max_features: int = 1000, fit: bool = True) -> sp.csr_matrix:
        """
        Create n-gram features (character or word level).
        
        Args:
            texts (List[str]): List of cleaned texts
            ngram_range (Tuple[int, int]): N-gram range
            max_features (int): Maximum number of features
            fit (bool): Whether to fit the vectorizer
            
        Returns:
            scipy.sparse.csr_matrix: N-gram feature matrix
        """
        if not hasattr(self, 'ngram_vectorizer') or fit:
            self.ngram_vectorizer = CountVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                min_df=2,
                stop_words='english',
                binary=True  # Binary features for n-grams
            )
            ngram_matrix = self.ngram_vectorizer.fit_transform(texts)
            
            # Store feature names
            self.ngram_feature_names = [f'ngram_{name}' for name in self.ngram_vectorizer.get_feature_names_out()]
            
            logger.info(f"N-gram vectorizer fitted with {ngram_matrix.shape[1]} features")
        else:
            ngram_matrix = self.ngram_vectorizer.transform(texts)
        
        return ngram_matrix
    
    def create_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create custom domain-specific features for review analysis.
        
        Args:
            df (pd.DataFrame): DataFrame with review data
            
        Returns:
            pd.DataFrame: Custom features
        """
        features = pd.DataFrame(index=df.index)
        
        # Product-related features (if available)
        if 'productId' in df.columns:
            # Product frequency (how often this product is reviewed)
            product_counts = df['productId'].value_counts()
            features['product_review_count'] = df['productId'].map(product_counts).fillna(0)
            features['product_review_count_log'] = np.log1p(features['product_review_count'])
            
            # Product popularity category
            features['product_high_volume'] = (features['product_review_count'] >= 100).astype(int)
            features['product_medium_volume'] = ((features['product_review_count'] >= 20) & 
                                               (features['product_review_count'] < 100)).astype(int)
            features['product_low_volume'] = (features['product_review_count'] < 20).astype(int)
        
        # User-related features (if available)
        if 'userId' in df.columns:
            # User activity level
            user_counts = df['userId'].value_counts()
            features['user_review_count'] = df['userId'].map(user_counts).fillna(0)
            features['user_review_count_log'] = np.log1p(features['user_review_count'])
            
            # User type categories
            features['user_power_reviewer'] = (features['user_review_count'] >= 10).astype(int)
            features['user_regular_reviewer'] = ((features['user_review_count'] >= 3) & 
                                               (features['user_review_count'] < 10)).astype(int)
            features['user_casual_reviewer'] = (features['user_review_count'] < 3).astype(int)
        
        # Helpfulness features (if available)
        if 'helpfulnessNumerator' in df.columns and 'helpfulnessDenominator' in df.columns:
            # Helpfulness ratio
            denominator = df['helpfulnessDenominator'].replace(0, 1)  # Avoid division by zero
            features['helpfulness_ratio'] = df['helpfulnessNumerator'] / denominator
            
            # Helpfulness categories
            features['very_helpful'] = (features['helpfulness_ratio'] >= 0.8).astype(int)
            features['helpful'] = ((features['helpfulness_ratio'] >= 0.5) & 
                                 (features['helpfulness_ratio'] < 0.8)).astype(int)
            features['not_helpful'] = (features['helpfulness_ratio'] < 0.5).astype(int)
            
            # Total helpfulness votes
            features['total_helpfulness_votes'] = df['helpfulnessDenominator'].fillna(0)
            features['total_helpfulness_votes_log'] = np.log1p(features['total_helpfulness_votes'])
        
        # Time-based features (if available)
        if 'timestamp' in df.columns:
            try:
                timestamps = pd.to_datetime(df['timestamp'])
                features['review_year'] = timestamps.dt.year
                features['review_month'] = timestamps.dt.month
                features['review_day_of_week'] = timestamps.dt.dayofweek
                features['review_quarter'] = timestamps.dt.quarter
            except:
                logger.warning("Could not parse timestamp column for time-based features")
        
        logger.info(f"Created {features.shape[1]} custom features")
        return features
    
    def fit_transform(self, df: pd.DataFrame, text_column: str = 'cleaned_text') -> Tuple[sp.csr_matrix, List[str]]:
        """
        Fit feature engineering pipeline and transform data.
        
        Args:
            df (pd.DataFrame): DataFrame with preprocessed text and quality metrics
            text_column (str): Column containing cleaned text
            
        Returns:
            Tuple[sp.csr_matrix, List[str]]: Feature matrix and feature names
        """
        logger.info(f"Fitting feature engineering pipeline on {len(df)} samples...")
        
        # Validate input
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in DataFrame")
        
        # Get text data
        texts = df[text_column].fillna('').astype(str).tolist()
        
        feature_matrices = []
        all_feature_names = []
        
        # 1. Create TF-IDF features
        tfidf_matrix = self.create_tfidf_features(texts, fit=True)
        feature_matrices.append(tfidf_matrix)
        all_feature_names.extend(self.tfidf_feature_names)
        
        # 2. Create text statistics features
        text_stats_features = self.extract_text_statistics_features(df)
        if not text_stats_features.empty:
            text_stats_matrix = sp.csr_matrix(text_stats_features.values)
            feature_matrices.append(text_stats_matrix)
            all_feature_names.extend([f'text_stat_{col}' for col in text_stats_features.columns])
        
        # 3. Create custom domain features
        custom_features = self.create_custom_features(df)
        if not custom_features.empty:
            custom_matrix = sp.csr_matrix(custom_features.values)
            feature_matrices.append(custom_matrix)
            all_feature_names.extend([f'custom_{col}' for col in custom_features.columns])
        
        # 4. Combine all features
        combined_matrix = sp.hstack(feature_matrices)
        
        # 5. Apply scaling if configured
        if self.config['scaling']['scale_text_features'] or self.config['scaling']['scale_tfidf']:
            if self.config['scaling']['scaler_type'] == 'standard':
                self.scaler = StandardScaler(with_mean=False)  # Sparse matrices
            else:
                self.scaler = MinMaxScaler()
            
            combined_matrix = self.scaler.fit_transform(combined_matrix)
        
        # Store feature information
        self.feature_names = all_feature_names
        self.is_fitted = True
        
        logger.info(f"Feature engineering complete: {combined_matrix.shape[1]} features created")
        
        return combined_matrix, self.feature_names
    
    def transform(self, df: pd.DataFrame, text_column: str = 'cleaned_text') -> sp.csr_matrix:
        """
        Transform new data using fitted pipeline.
        
        Args:
            df (pd.DataFrame): DataFrame with preprocessed text and quality metrics
            text_column (str): Column containing cleaned text
            
        Returns:
            scipy.sparse.csr_matrix: Transformed feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Feature engineer not fitted. Call fit_transform first.")
        
        logger.info(f"Transforming {len(df)} samples...")
        
        # Get text data
        texts = df[text_column].fillna('').astype(str).tolist()
        
        feature_matrices = []
        
        # 1. Transform TF-IDF features
        tfidf_matrix = self.create_tfidf_features(texts, fit=False)
        feature_matrices.append(tfidf_matrix)
        
        # 2. Extract text statistics features
        text_stats_features = self.extract_text_statistics_features(df)
        if not text_stats_features.empty:
            text_stats_matrix = sp.csr_matrix(text_stats_features.values)
            feature_matrices.append(text_stats_matrix)
        
        # 3. Create custom domain features
        custom_features = self.create_custom_features(df)
        if not custom_features.empty:
            custom_matrix = sp.csr_matrix(custom_features.values)
            feature_matrices.append(custom_matrix)
        
        # 4. Combine all features
        combined_matrix = sp.hstack(feature_matrices)
        
        # 5. Apply scaling if fitted
        if self.scaler is not None:
            combined_matrix = self.scaler.transform(combined_matrix)
        
        return combined_matrix
    
    def get_feature_importance_names(self, importance_scores: np.ndarray, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Get top feature names with importance scores.
        
        Args:
            importance_scores (np.ndarray): Feature importance scores from model
            top_k (int): Number of top features to return
            
        Returns:
            List[Tuple[str, float]]: List of (feature_name, importance_score)
        """
        if len(importance_scores) != len(self.feature_names):
            raise ValueError("Importance scores length doesn't match number of features")
        
        # Get top features
        top_indices = np.argsort(importance_scores)[-top_k:][::-1]
        
        return [(self.feature_names[i], importance_scores[i]) for i in top_indices]
    
    def save_pipeline(self, filepath: str) -> None:
        """
        Save fitted feature engineering pipeline.
        
        Args:
            filepath (str): Path to save the pipeline
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted pipeline")
        
        pipeline_data = {
            'config': self.config,
            'vectorizer': self.vectorizer,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        # Save additional vectorizers if they exist
        if hasattr(self, 'ngram_vectorizer'):
            pipeline_data['ngram_vectorizer'] = self.ngram_vectorizer
            pipeline_data['ngram_feature_names'] = self.ngram_feature_names
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(pipeline_data, f)
        
        logger.info(f"Feature engineering pipeline saved to {filepath}")
    
    def load_pipeline(self, filepath: str) -> None:
        """
        Load fitted feature engineering pipeline.
        
        Args:
            filepath (str): Path to load the pipeline from
        """
        with open(filepath, 'rb') as f:
            pipeline_data = pickle.load(f)
        
        self.config = pipeline_data['config']
        self.vectorizer = pipeline_data['vectorizer']
        self.scaler = pipeline_data['scaler']
        self.feature_names = pipeline_data['feature_names']
        self.is_fitted = pipeline_data['is_fitted']
        
        # Load additional vectorizers if they exist
        if 'ngram_vectorizer' in pipeline_data:
            self.ngram_vectorizer = pipeline_data['ngram_vectorizer']
            self.ngram_feature_names = pipeline_data['ngram_feature_names']
        
        logger.info(f"Feature engineering pipeline loaded from {filepath}")
    
    def get_feature_summary(self) -> Dict:
        """
        Get summary of created features.
        
        Returns:
            Dict: Feature summary statistics
        """
        if not self.is_fitted:
            return {'error': 'Pipeline not fitted'}
        
        # Count features by type
        feature_types = {}
        for name in self.feature_names:
            if name.startswith('tfidf_'):
                feature_types['tfidf'] = feature_types.get('tfidf', 0) + 1
            elif name.startswith('text_stat_'):
                feature_types['text_statistics'] = feature_types.get('text_statistics', 0) + 1
            elif name.startswith('custom_'):
                feature_types['custom_domain'] = feature_types.get('custom_domain', 0) + 1
            elif name.startswith('ngram_'):
                feature_types['ngrams'] = feature_types.get('ngrams', 0) + 1
        
        return {
            'total_features': len(self.feature_names),
            'feature_types': feature_types,
            'tfidf_config': self.config['tfidf'],
            'scaling_applied': self.scaler is not None,
            'scaler_type': self.config['scaling']['scaler_type'] if self.scaler else None
        }

# Convenience functions
def create_features(df: pd.DataFrame, config: Optional[Dict] = None) -> Tuple[sp.csr_matrix, List[str], ReviewFeatureEngineer]:
    """
    Convenience function for feature creation.
    
    Args:
        df (pd.DataFrame): DataFrame with preprocessed reviews
        config (Dict, optional): Feature engineering configuration
        
    Returns:
        Tuple: Feature matrix, feature names, fitted engineer
    """
    engineer = ReviewFeatureEngineer(config)
    features, names = engineer.fit_transform(df)
    return features, names, engineer

def create_feature_config(
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 3),
    include_text_stats: bool = True,
    include_custom_features: bool = True,
    scale_features: bool = True
) -> Dict:
    """
    Create feature engineering configuration.
    
    Args:
        max_features (int): Maximum TF-IDF features
        ngram_range (Tuple[int, int]): N-gram range for TF-IDF
        include_text_stats (bool): Include text statistics features
        include_custom_features (bool): Include domain-specific features
        scale_features (bool): Apply feature scaling
        
    Returns:
        Dict: Feature engineering configuration
    """
    return {
        'tfidf': {
            'max_features': max_features,
            'ngram_range': ngram_range,
            'min_df': 5,
            'max_df': 0.95,
            'stop_words': 'english',
            'sublinear_tf': True,
            'norm': 'l2'
        },
        'text_features': {
            'include_length_features': include_text_stats,
            'include_readability_features': include_text_stats,
            'include_sentiment_features': include_text_stats,
            'include_lexical_features': include_text_stats,
            'include_structural_features': include_text_stats
        },
        'scaling': {
            'scale_tfidf': False,
            'scale_text_features': scale_features,
            'scaler_type': 'standard'
        }
    }