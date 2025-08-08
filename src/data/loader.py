"""
Data Loading and Validation Module for Intelligent Review Analytics Platform

This module provides robust data loading utilities for Amazon review datasets
with comprehensive validation, quality assessment, and error handling.

"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReviewDataLoader:
    """
    Comprehensive data loading and validation class for Amazon review datasets.
    
    Handles TSV file loading, data quality assessment, and provides detailed
    statistics for business intelligence analysis.
    """
    
    def __init__(self, data_path: str = "data/raw/"):
        """
        Initialize the data loader with path configuration.
        
        Args:
            data_path (str): Path to directory containing TSV files
        """
        self.data_path = data_path
        self.datasets = {}
        self.data_quality_report = {}
        
    def load_tsv_file(self, filename: str) -> pd.DataFrame:
        """
        Load a single TSV file with robust error handling.
        
        Args:
            filename (str): Name of TSV file to load
            
        Returns:
            pd.DataFrame: Loaded dataset with validated structure
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file structure is invalid
        """
        filepath = os.path.join(self.data_path, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        try:
            # Load TSV with appropriate encoding and error handling
            df = pd.read_csv(filepath, sep='\t', encoding='utf-8', 
                           na_values=['', 'NULL', 'null', 'NaN'])
            
            logger.info(f"Successfully loaded {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Basic validation
            if df.empty:
                raise ValueError(f"File {filename} is empty")
                
            return df
            
        except Exception as e:
            logger.error(f"Error loading {filename}: {str(e)}")
            raise
    
    def validate_dataset_structure(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """
        Validate dataset structure and generate quality metrics.
        
        Args:
            df (pd.DataFrame): Dataset to validate
            dataset_name (str): Name of dataset for reporting
            
        Returns:
            Dict: Comprehensive validation report
        """
        validation_report = {
            'dataset_name': dataset_name,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': list(df.columns),
            'missing_values': {},
            'data_types': {},
            'quality_score': 0
        }
        
        # Check for required columns
        required_columns = ['sentiment', 'text']
        missing_required = [col for col in required_columns if col not in df.columns]
        
        if missing_required:
            logger.warning(f"Missing required columns in {dataset_name}: {missing_required}")
            validation_report['missing_required_columns'] = missing_required
        
        # Analyze missing values
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_percentage = (missing_count / len(df)) * 100
            validation_report['missing_values'][column] = {
                'count': int(missing_count),
                'percentage': round(missing_percentage, 2)
            }
            validation_report['data_types'][column] = str(df[column].dtype)
        
        # Calculate overall quality score
        total_missing_percentage = sum([v['percentage'] for v in validation_report['missing_values'].values()])
        avg_missing_percentage = total_missing_percentage / len(df.columns)
        validation_report['quality_score'] = round(max(0, 100 - avg_missing_percentage), 2)
        
        return validation_report
    
    def analyze_text_characteristics(self, df: pd.DataFrame) -> Dict:
        """
        Analyze text characteristics for business intelligence insights.
        
        Args:
            df (pd.DataFrame): Dataset with text column
            
        Returns:
            Dict: Text analysis statistics
        """
        if 'text' not in df.columns:
            return {'error': 'No text column found'}
        
        # Remove null values for analysis
        text_series = df['text'].dropna()
        
        if text_series.empty:
            return {'error': 'No valid text data found'}
        
        # Calculate word counts
        word_counts = text_series.apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
        
        # Calculate character counts
        char_counts = text_series.apply(lambda x: len(str(x)) if pd.notna(x) else 0)
        
        text_analysis = {
            'total_reviews': len(text_series),
            'word_count_stats': {
                'min': int(word_counts.min()),
                'max': int(word_counts.max()),
                'mean': round(word_counts.mean(), 2),
                'median': int(word_counts.median()),
                'std': round(word_counts.std(), 2)
            },
            'char_count_stats': {
                'min': int(char_counts.min()),
                'max': int(char_counts.max()),
                'mean': round(char_counts.mean(), 2),
                'median': int(char_counts.median())
            },
            'empty_reviews': int((text_series == '').sum()),
            'processing_success_rate': round((len(text_series) / len(df)) * 100, 2)
        }
        
        return text_analysis
    
    def analyze_sentiment_distribution(self, df: pd.DataFrame) -> Dict:
        """
        Analyze sentiment distribution for class balance assessment.
        
        Args:
            df (pd.DataFrame): Dataset with sentiment column
            
        Returns:
            Dict: Sentiment distribution statistics
        """
        if 'sentiment' not in df.columns:
            return {'error': 'No sentiment column found'}
        
        sentiment_counts = df['sentiment'].value_counts()
        total_samples = len(df['sentiment'].dropna())
        
        sentiment_analysis = {
            'total_samples': total_samples,
            'unique_sentiments': list(sentiment_counts.index),
            'sentiment_counts': sentiment_counts.to_dict(),
            'sentiment_percentages': {}
        }
        
        # Calculate percentages
        for sentiment, count in sentiment_counts.items():
            percentage = round((count / total_samples) * 100, 2)
            sentiment_analysis['sentiment_percentages'][sentiment] = percentage
        
        # Assess class balance
        percentages = list(sentiment_analysis['sentiment_percentages'].values())
        balance_score = 100 - (max(percentages) - min(percentages)) if len(percentages) > 1 else 100
        sentiment_analysis['class_balance_score'] = round(balance_score, 2)
        
        return sentiment_analysis
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all required datasets with comprehensive validation.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing all loaded datasets
            
        Raises:
            Exception: If critical datasets cannot be loaded
        """
        dataset_files = {
            'train': 'reviews_train.tsv',
            'validation': 'reviews_validation.tsv', 
            'test': 'reviews_test.tsv',
            'submit': 'reviews_submit.tsv'
        }
        
        logger.info("Starting comprehensive dataset loading and validation...")
        
        # Load each dataset
        for dataset_name, filename in dataset_files.items():
            try:
                df = self.load_tsv_file(filename)
                self.datasets[dataset_name] = df
                
                # Generate validation report
                validation_report = self.validate_dataset_structure(df, dataset_name)
                text_analysis = self.analyze_text_characteristics(df)
                sentiment_analysis = self.analyze_sentiment_distribution(df)
                
                # Combine reports
                self.data_quality_report[dataset_name] = {
                    'validation': validation_report,
                    'text_analysis': text_analysis,
                    'sentiment_analysis': sentiment_analysis
                }
                
            except FileNotFoundError:
                logger.warning(f"Optional file {filename} not found, skipping...")
                continue
            except Exception as e:
                if dataset_name in ['train', 'validation', 'test']:
                    logger.error(f"Critical dataset {dataset_name} failed to load: {str(e)}")
                    raise
                else:
                    logger.warning(f"Optional dataset {dataset_name} failed to load: {str(e)}")
                    continue
        
        # Validate critical datasets loaded successfully
        critical_datasets = ['train', 'validation', 'test']
        loaded_critical = [name for name in critical_datasets if name in self.datasets]
        
        if len(loaded_critical) < 3:
            missing = set(critical_datasets) - set(loaded_critical)
            raise Exception(f"Critical datasets missing: {missing}")
        
        logger.info(f"Successfully loaded {len(self.datasets)} datasets")
        return self.datasets
    
    def get_dataset_summary(self) -> Dict:
        """
        Generate comprehensive summary of all loaded datasets.
        
        Returns:
            Dict: Complete summary with key metrics for business reporting
        """
        if not self.datasets:
            return {'error': 'No datasets loaded'}
        
        summary = {
            'total_datasets': len(self.datasets),
            'dataset_overview': {},
            'combined_statistics': {
                'total_reviews': 0,
                'total_unique_words_estimate': 0,
                'overall_quality_score': 0
            }
        }
        
        quality_scores = []
        total_reviews = 0
        
        for dataset_name, df in self.datasets.items():
            dataset_info = {
                'rows': len(df),
                'columns': len(df.columns),
                'quality_score': self.data_quality_report.get(dataset_name, {}).get('validation', {}).get('quality_score', 0),
                'text_word_range': 'N/A',
                'sentiment_distribution': 'N/A'
            }
            
            # Add text statistics if available
            if dataset_name in self.data_quality_report:
                text_stats = self.data_quality_report[dataset_name].get('text_analysis', {})
                if 'word_count_stats' in text_stats:
                    min_words = text_stats['word_count_stats']['min']
                    max_words = text_stats['word_count_stats']['max']
                    dataset_info['text_word_range'] = f"{min_words}-{max_words} words"
                
                # Add sentiment distribution
                sentiment_stats = self.data_quality_report[dataset_name].get('sentiment_analysis', {})
                if 'sentiment_percentages' in sentiment_stats:
                    dataset_info['sentiment_distribution'] = sentiment_stats['sentiment_percentages']
            
            summary['dataset_overview'][dataset_name] = dataset_info
            quality_scores.append(dataset_info['quality_score'])
            total_reviews += dataset_info['rows']
        
        # Calculate combined statistics
        summary['combined_statistics']['total_reviews'] = total_reviews
        summary['combined_statistics']['overall_quality_score'] = round(np.mean(quality_scores), 2) if quality_scores else 0
        
        return summary
    
    def get_validation_checkpoint(self) -> Dict:
        """
        Generate validation checkpoint for Phase 1 completion verification.
        
        Returns:
            Dict: Key metrics for project progression validation
        """
        if not self.datasets:
            return {'status': 'FAILED', 'error': 'No datasets loaded'}
        
        checkpoint = {
            'status': 'SUCCESS',
            'timestamp': pd.Timestamp.now().isoformat(),
            'critical_validations': {
                'train_dataset_loaded': 'train' in self.datasets,
                'validation_dataset_loaded': 'validation' in self.datasets,
                'test_dataset_loaded': 'test' in self.datasets,
                'expected_train_size': len(self.datasets.get('train', [])) >= 4000 if 'train' in self.datasets else False,
                'expected_val_size': len(self.datasets.get('validation', [])) >= 500 if 'validation' in self.datasets else False,
                'expected_test_size': len(self.datasets.get('test', [])) >= 500 if 'test' in self.datasets else False
            },
            'key_metrics': {}
        }
        
        # Extract key metrics for placeholder updates
        if 'train' in self.datasets:
            train_df = self.datasets['train']
            checkpoint['key_metrics']['TOTAL_REVIEWS'] = len(train_df)
            
            # Get text statistics
            if 'train' in self.data_quality_report:
                text_stats = self.data_quality_report['train'].get('text_analysis', {})
                if 'word_count_stats' in text_stats:
                    checkpoint['key_metrics']['MIN_WORDS'] = text_stats['word_count_stats']['min']
                    checkpoint['key_metrics']['MAX_WORDS'] = text_stats['word_count_stats']['max']
                    checkpoint['key_metrics']['AVG_REVIEW_LENGTH'] = text_stats['word_count_stats']['mean']
                
                # Processing success rate
                checkpoint['key_metrics']['PREPROCESSING_SUCCESS_RATE'] = text_stats.get('processing_success_rate', 0)
                
                # Sentiment distribution
                sentiment_stats = self.data_quality_report['train'].get('sentiment_analysis', {})
                if 'sentiment_percentages' in sentiment_stats:
                    sentiments = sentiment_stats['sentiment_percentages']
                    checkpoint['key_metrics']['POSITIVE_PERCENT'] = sentiments.get(1, 0)
                    checkpoint['key_metrics']['NEGATIVE_PERCENT'] = sentiments.get(-1, 0)
        
        # Overall validation status
        all_critical_passed = all(checkpoint['critical_validations'].values())
        checkpoint['status'] = 'SUCCESS' if all_critical_passed else 'VALIDATION_ISSUES'
        
        return checkpoint

def load_and_validate_datasets(data_path: str = "data/raw/") -> Tuple[Dict[str, pd.DataFrame], Dict]:
    """
    Convenience function for loading and validating all datasets.
    
    Args:
        data_path (str): Path to directory containing TSV files
        
    Returns:
        Tuple[Dict[str, pd.DataFrame], Dict]: Datasets and validation checkpoint
    """
    loader = ReviewDataLoader(data_path)
    datasets = loader.load_all_datasets()
    checkpoint = loader.get_validation_checkpoint()
    
    return datasets, checkpoint

# Example usage and testing
if __name__ == "__main__":
    # Test the data loader
    try:
        loader = ReviewDataLoader()
        datasets = loader.load_all_datasets()
        summary = loader.get_dataset_summary()
        checkpoint = loader.get_validation_checkpoint()
        
        print("=== DATA LOADING VALIDATION ===")
        print(f"Status: {checkpoint['status']}")
        print(f"Datasets loaded: {len(datasets)}")
        
        for name, df in datasets.items():
            print(f"  - {name}: {len(df)} rows")
        
        print(f"Overall quality score: {summary['combined_statistics']['overall_quality_score']}%")
        print("=== VALIDATION COMPLETE ===")
        
    except Exception as e:
        print(f"Data loading failed: {str(e)}")
        print("Please ensure TSV files are in the correct location.")