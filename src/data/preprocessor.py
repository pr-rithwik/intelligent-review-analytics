"""
Text Preprocessing Module for Intelligent Review Analytics Platform

This module provides comprehensive text preprocessing capabilities including
cleaning, normalization, tokenization, and quality assessment for review text data.
"""

import re
import string
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import logging
from typing import List, Dict, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)

class ReviewTextPreprocessor:
    """
    Comprehensive text preprocessing for Amazon review data.
    
    Handles cleaning, normalization, tokenization, and quality assessment
    with configurable options for different processing requirements.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize preprocessor with configuration options.
        
        Args:
            config (Dict, optional): Preprocessing configuration parameters
        """
        self.config = config or self._get_default_config()
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.sia = SentimentIntensityAnalyzer()
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
        
        logger.info("ReviewTextPreprocessor initialized with configuration")
    
    def _get_default_config(self) -> Dict:
        """Get default preprocessing configuration."""
        return {
            'min_length': 1,
            'max_length': 10000,
            'remove_html': True,
            'remove_urls': True,
            'remove_special_chars': True,
            'lowercase': True,
            'remove_stopwords': True,
            'lemmatize': True,
            'remove_extra_whitespace': True,
            'preserve_sentence_structure': False,
            'calculate_quality_metrics': True
        }
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for text cleaning."""
        self.patterns = {
            'html': re.compile(r'<[^>]+>'),
            'url': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'email': re.compile(r'\S+@\S+\.\S+'),
            'special_chars': re.compile(r'[^a-zA-Z0-9\s]'),
            'extra_whitespace': re.compile(r'\s+'),
            'numbers_only': re.compile(r'^\d+$'),
            'repeated_chars': re.compile(r'(.)\1{3,}')  # 4+ repeated characters
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text according to configuration.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        
        cleaned = text
        
        # Remove HTML tags
        if self.config['remove_html']:
            cleaned = self.patterns['html'].sub(' ', cleaned)
        
        # Remove URLs and emails
        if self.config['remove_urls']:
            cleaned = self.patterns['url'].sub(' ', cleaned)
            cleaned = self.patterns['email'].sub(' ', cleaned)
        
        # Fix repeated characters (e.g., "sooooo good" -> "so good")
        cleaned = self.patterns['repeated_chars'].sub(r'\1\1', cleaned)
        
        # Convert to lowercase
        if self.config['lowercase']:
            cleaned = cleaned.lower()
        
        # Remove special characters
        if self.config['remove_special_chars']:
            cleaned = self.patterns['special_chars'].sub(' ', cleaned)
        
        # Remove extra whitespace
        if self.config['remove_extra_whitespace']:
            cleaned = self.patterns['extra_whitespace'].sub(' ', cleaned)
        
        return cleaned.strip()
    
    def tokenize_and_filter(self, text: str) -> List[str]:
        """
        Tokenize text and apply filtering (stopwords, lemmatization).
        
        Args:
            text (str): Cleaned text to tokenize
            
        Returns:
            List[str]: Filtered tokens
        """
        if not text:
            return []
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Filter tokens
        filtered_tokens = []
        for token in tokens:
            # Skip very short tokens or numbers-only
            if len(token) < 2 or self.patterns['numbers_only'].match(token):
                continue
                
            # Remove stopwords if configured
            if self.config['remove_stopwords'] and token.lower() in self.stopwords:
                continue
            
            # Lemmatize if configured
            if self.config['lemmatize']:
                token = self.lemmatizer.lemmatize(token.lower())
            
            filtered_tokens.append(token)
        
        return filtered_tokens
    
    def calculate_text_quality_metrics(self, text: str) -> Dict:
        """
        Calculate quality metrics for text analysis.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict: Quality metrics
        """
        if not text or not text.strip():
            return self._empty_quality_metrics()
        
        # Basic counts
        word_count = len(text.split())
        char_count = len(text)
        sentence_count = len(sent_tokenize(text))
        
        # Advanced metrics
        avg_word_length = np.mean([len(word) for word in text.split()]) if word_count > 0 else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Readability (simplified Flesch reading ease approximation)
        if sentence_count > 0 and word_count > 0:
            avg_sentence_length_score = avg_sentence_length
            avg_syllables = self._estimate_syllables_per_word(text)
            flesch_score = 206.835 - (1.015 * avg_sentence_length_score) - (84.6 * avg_syllables)
            flesch_score = max(0, min(100, flesch_score))  # Clamp between 0-100
        else:
            flesch_score = 0
        
        # Sentiment intensity
        sentiment_scores = self.sia.polarity_scores(text)
        
        # Text diversity (unique words ratio)
        unique_words = len(set(text.lower().split()))
        diversity_ratio = unique_words / word_count if word_count > 0 else 0
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'sentence_count': sentence_count,
            'avg_word_length': round(avg_word_length, 2),
            'avg_sentence_length': round(avg_sentence_length, 2),
            'flesch_reading_ease': round(flesch_score, 2),
            'sentiment_compound': round(sentiment_scores['compound'], 3),
            'sentiment_positive': round(sentiment_scores['pos'], 3),
            'sentiment_negative': round(sentiment_scores['neg'], 3),
            'sentiment_neutral': round(sentiment_scores['neu'], 3),
            'diversity_ratio': round(diversity_ratio, 3),
            'unique_word_count': unique_words
        }
    
    def _empty_quality_metrics(self) -> Dict:
        """Return empty quality metrics for invalid text."""
        return {
            'word_count': 0,
            'char_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0.0,
            'avg_sentence_length': 0.0,
            'flesch_reading_ease': 0.0,
            'sentiment_compound': 0.0,
            'sentiment_positive': 0.0,
            'sentiment_negative': 0.0,
            'sentiment_neutral': 0.0,
            'diversity_ratio': 0.0,
            'unique_word_count': 0
        }
    
    def _estimate_syllables_per_word(self, text: str) -> float:
        """
        Estimate average syllables per word (simplified method).
        
        Args:
            text (str): Text to analyze
            
        Returns:
            float: Average syllables per word
        """
        words = text.split()
        if not words:
            return 0
        
        total_syllables = 0
        for word in words:
            word = word.lower().strip(string.punctuation)
            if word:
                # Simple syllable estimation: count vowel groups
                syllables = len(re.findall(r'[aeiouy]+', word))
                if word.endswith('e'):
                    syllables -= 1
                syllables = max(1, syllables)  # At least 1 syllable
                total_syllables += syllables
        
        return total_syllables / len(words)
    
    def preprocess_single_text(self, text: str) -> Dict:
        """
        Complete preprocessing of a single text with quality metrics.
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            Dict: Preprocessing results including cleaned text and metrics
        """
        # Validate input
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        # Initial quality check
        if len(text.strip()) < self.config['min_length']:
            return {
                'original_text': text,
                'cleaned_text': "",
                'tokens': [],
                'is_valid': False,
                'reason': 'too_short',
                'quality_metrics': self._empty_quality_metrics()
            }
        
        if len(text) > self.config['max_length']:
            return {
                'original_text': text,
                'cleaned_text': "",
                'tokens': [],
                'is_valid': False,
                'reason': 'too_long',
                'quality_metrics': self._empty_quality_metrics()
            }
        
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Check if cleaning removed too much content
        if not cleaned_text or len(cleaned_text.strip()) < self.config['min_length']:
            return {
                'original_text': text,
                'cleaned_text': cleaned_text,
                'tokens': [],
                'is_valid': False,
                'reason': 'insufficient_content_after_cleaning',
                'quality_metrics': self._empty_quality_metrics()
            }
        
        # Tokenize and filter
        tokens = self.tokenize_and_filter(cleaned_text)
        
        # Calculate quality metrics
        quality_metrics = (self.calculate_text_quality_metrics(cleaned_text) 
                         if self.config['calculate_quality_metrics'] else {})
        
        return {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'tokens': tokens,
            'is_valid': True,
            'reason': 'processed_successfully',
            'quality_metrics': quality_metrics
        }
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> Tuple[pd.DataFrame, Dict]:
        """
        Preprocess text data in a pandas DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing text data
            text_column (str): Name of the column containing text
            
        Returns:
            Tuple[pd.DataFrame, Dict]: Processed DataFrame and processing report
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        logger.info(f"Starting preprocessing of {len(df)} texts...")
        
        # Initialize results
        processed_df = df.copy()
        processing_results = []
        
        # Process each text
        for idx, text in enumerate(df[text_column]):
            result = self.preprocess_single_text(text)
            processing_results.append(result)
            
            if idx % 1000 == 0 and idx > 0:
                logger.info(f"Processed {idx}/{len(df)} texts...")
        
        # Add results to DataFrame
        processed_df['cleaned_text'] = [r['cleaned_text'] for r in processing_results]
        processed_df['is_valid'] = [r['is_valid'] for r in processing_results]
        processed_df['processing_reason'] = [r['reason'] for r in processing_results]
        processed_df['token_count'] = [len(r['tokens']) for r in processing_results]
        
        # Add quality metrics if calculated
        if self.config['calculate_quality_metrics']:
            quality_metrics_df = pd.DataFrame([r['quality_metrics'] for r in processing_results])
            processed_df = pd.concat([processed_df, quality_metrics_df], axis=1)
        
        # Generate processing report
        processing_report = self._generate_processing_report(processing_results, df)
        
        logger.info(f"Preprocessing complete. Success rate: {processing_report['success_rate']:.2f}%")
        
        return processed_df, processing_report
    
    def _generate_processing_report(self, processing_results: List[Dict], original_df: pd.DataFrame) -> Dict:
        """Generate comprehensive processing report."""
        total_texts = len(processing_results)
        valid_texts = sum(1 for r in processing_results if r['is_valid'])
        
        # Count reasons for invalid texts
        reasons = {}
        for result in processing_results:
            if not result['is_valid']:
                reason = result['reason']
                reasons[reason] = reasons.get(reason, 0) + 1
        
        # Calculate quality statistics
        valid_results = [r for r in processing_results if r['is_valid']]
        if valid_results and self.config['calculate_quality_metrics']:
            word_counts = [r['quality_metrics']['word_count'] for r in valid_results]
            quality_stats = {
                'avg_word_count': np.mean(word_counts),
                'median_word_count': np.median(word_counts),
                'min_word_count': np.min(word_counts),
                'max_word_count': np.max(word_counts),
                'std_word_count': np.std(word_counts)
            }
        else:
            quality_stats = {}
        
        return {
            'total_texts': total_texts,
            'valid_texts': valid_texts,
            'invalid_texts': total_texts - valid_texts,
            'success_rate': (valid_texts / total_texts * 100) if total_texts > 0 else 0,
            'invalid_reasons': reasons,
            'quality_statistics': quality_stats,
            'processing_config': self.config.copy()
        }
    
    def get_preprocessing_summary(self, processing_report: Dict) -> str:
        """
        Generate human-readable preprocessing summary.
        
        Args:
            processing_report (Dict): Processing report from preprocess_dataframe
            
        Returns:
            str: Formatted summary
        """
        summary = []
        summary.append("=== TEXT PREPROCESSING SUMMARY ===")
        summary.append(f"Total texts processed: {processing_report['total_texts']:,}")
        summary.append(f"Successfully processed: {processing_report['valid_texts']:,}")
        summary.append(f"Failed processing: {processing_report['invalid_texts']:,}")
        summary.append(f"Success rate: {processing_report['success_rate']:.2f}%")
        
        if processing_report['invalid_reasons']:
            summary.append("\nFailure reasons:")
            for reason, count in processing_report['invalid_reasons'].items():
                summary.append(f"  - {reason}: {count:,} texts")
        
        if processing_report['quality_statistics']:
            stats = processing_report['quality_statistics']
            summary.append("\nQuality statistics:")
            summary.append(f"  - Average word count: {stats['avg_word_count']:.1f}")
            summary.append(f"  - Median word count: {stats['median_word_count']:.0f}")
            summary.append(f"  - Word count range: {stats['min_word_count']:.0f} - {stats['max_word_count']:.0f}")
        
        summary.append("=" * 35)
        
        return "\n".join(summary)

# Convenience functions
def preprocess_reviews(df: pd.DataFrame, config: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function for preprocessing review DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with review text
        config (Dict, optional): Preprocessing configuration
        
    Returns:
        Tuple[pd.DataFrame, Dict]: Processed DataFrame and report
    """
    preprocessor = ReviewTextPreprocessor(config)
    return preprocessor.preprocess_dataframe(df, 'text')

def create_preprocessing_config(
    remove_stopwords: bool = True,
    lemmatize: bool = True,
    min_length: int = 1,
    max_length: int = 10000,
    calculate_quality_metrics: bool = True
) -> Dict:
    """
    Create preprocessing configuration with common parameters.
    
    Args:
        remove_stopwords (bool): Remove English stopwords
        lemmatize (bool): Apply lemmatization
        min_length (int): Minimum text length
        max_length (int): Maximum text length
        calculate_quality_metrics (bool): Calculate quality metrics
        
    Returns:
        Dict: Preprocessing configuration
    """
    return {
        'min_length': min_length,
        'max_length': max_length,
        'remove_html': True,
        'remove_urls': True,
        'remove_special_chars': True,
        'lowercase': True,
        'remove_stopwords': remove_stopwords,
        'lemmatize': lemmatize,
        'remove_extra_whitespace': True,
        'preserve_sentence_structure': False,
        'calculate_quality_metrics': calculate_quality_metrics
    }