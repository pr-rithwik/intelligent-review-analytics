"""
Base Quality Checker Module

Provides common functionality and utilities for all quality check implementations
in the validation framework.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
from datetime import datetime
import logging

from ..config import get_default_validation_config

logger = logging.getLogger(__name__)

class BaseQualityChecker(ABC):
    """
    Abstract base class for all quality checkers.
    
    Provides common functionality including configuration management,
    result formatting, logging, and utility methods.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize base quality checker with configuration.
        
        Args:
            config (Dict, optional): Validation configuration
        """
        self.config = config or get_default_validation_config()
        self.checker_name = self.__class__.__name__
        
        logger.info(f"{self.checker_name} initialized")
    
    @abstractmethod
    def check(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Abstract method for performing quality check.
        
        Args:
            df (pd.DataFrame): Dataset to check
            **kwargs: Additional parameters specific to checker
            
        Returns:
            Dict[str, Any]: Quality check results
        """
        pass
    
    def create_result_template(self, score: float = 0) -> Dict[str, Any]:
        """
        Create standardized result template for quality checks.
        
        Args:
            score (float): Quality score (0-100)
            
        Returns:
            Dict[str, Any]: Standardized result template
        """
        return {
            'checker_name': self.checker_name,
            'validation_timestamp': datetime.now().isoformat(),
            'score': round(score, 1),
            'status': self._get_status_from_score(score),
            'critical_issues': [],
            'warnings': [],
            'recommendations': [],
            'details': {},
            'statistics': {}
        }
    
    def _get_status_from_score(self, score: float) -> str:
        """Get status label from score."""
        quality_thresholds = self.config['quality_thresholds']
        
        if score >= quality_thresholds['excellent']:
            return 'EXCELLENT'
        elif score >= quality_thresholds['good']:
            return 'GOOD'
        elif score >= quality_thresholds['acceptable']:
            return 'ACCEPTABLE'
        elif score >= quality_thresholds['poor']:
            return 'POOR'
        else:
            return 'FAILING'
    
    def calculate_penalty_score(self, penalty: float, max_penalty: float = 100) -> float:
        """
        Calculate quality score from penalty.
        
        Args:
            penalty (float): Total penalty points
            max_penalty (float): Maximum possible penalty
            
        Returns:
            float: Quality score (0-100)
        """
        return max(0, 100 - min(penalty, max_penalty))
    
    def add_issue(self, result: Dict, message: str, is_critical: bool = True) -> None:
        """
        Add issue to result dictionary.
        
        Args:
            result (Dict): Result dictionary to modify
            message (str): Issue message
            is_critical (bool): Whether issue is critical
        """
        if is_critical:
            result['critical_issues'].append(message)
        else:
            result['warnings'].append(message)
    
    def add_recommendation(self, result: Dict, recommendation: str) -> None:
        """
        Add recommendation to result dictionary.
        
        Args:
            result (Dict): Result dictionary to modify
            recommendation (str): Recommendation message
        """
        result['recommendations'].append(recommendation)
    
    def validate_column_exists(self, df: pd.DataFrame, column: str) -> bool:
        """
        Validate that a column exists in the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame to check
            column (str): Column name to validate
            
        Returns:
            bool: True if column exists
        """
        return column in df.columns
    
    def get_column_info(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        Get basic information about a column.
        
        Args:
            df (pd.DataFrame): DataFrame containing the column
            column (str): Column name
            
        Returns:
            Dict[str, Any]: Column information
        """
        if not self.validate_column_exists(df, column):
            return {'exists': False, 'error': f"Column '{column}' not found"}
        
        series = df[column]
        
        return {
            'exists': True,
            'dtype': str(series.dtype),
            'total_values': len(series),
            'non_null_values': int(series.notna().sum()),
            'null_values': int(series.isnull().sum()),
            'null_percentage': round((series.isnull().sum() / len(series)) * 100, 2) if len(series) > 0 else 0,
            'unique_values': int(series.nunique()),
            'memory_usage_kb': round(series.memory_usage(deep=True) / 1024, 2)
        }
    
    def categorize_column_importance(self, column: str) -> str:
        """
        Categorize column by business importance.
        
        Args:
            column (str): Column name
            
        Returns:
            str: Importance category ('critical', 'important', 'optional', 'unknown')
        """
        from ..config import CRITICAL_COLUMNS, IMPORTANT_COLUMNS, OPTIONAL_COLUMNS
        
        if column in CRITICAL_COLUMNS:
            return 'critical'
        elif column in IMPORTANT_COLUMNS:
            return 'important'
        elif column in OPTIONAL_COLUMNS:
            return 'optional'
        else:
            return 'unknown'
    
    def calculate_percentage(self, part: int, total: int, decimal_places: int = 2) -> float:
        """
        Calculate percentage with proper handling of edge cases.
        
        Args:
            part (int): Part value
            total (int): Total value
            decimal_places (int): Number of decimal places
            
        Returns:
            float: Percentage value
        """
        if total == 0:
            return 0.0
        return round((part / total) * 100, decimal_places)
    
    def format_large_number(self, number: Union[int, float]) -> str:
        """
        Format large numbers with appropriate suffixes.
        
        Args:
            number (Union[int, float]): Number to format
            
        Returns:
            str: Formatted number
        """
        if abs(number) >= 1e6:
            return f"{number/1e6:.1f}M"
        elif abs(number) >= 1e3:
            return f"{number/1e3:.1f}K"
        else:
            return str(int(number)) if isinstance(number, float) and number.is_integer() else f"{number:.1f}"
    
    def create_statistics_summary(self, series: pd.Series, 
                                include_distribution: bool = True) -> Dict[str, Any]:
        """
        Create comprehensive statistics summary for a pandas Series.
        
        Args:
            series (pd.Series): Series to analyze
            include_distribution (bool): Include distribution statistics
            
        Returns:
            Dict[str, Any]: Statistics summary
        """
        if series.empty:
            return {'empty': True, 'count': 0}
        
        summary = {
            'count': len(series),
            'non_null_count': int(series.notna().sum()),
            'null_count': int(series.isnull().sum()),
            'null_percentage': self.calculate_percentage(series.isnull().sum(), len(series)),
            'unique_count': int(series.nunique()),
            'dtype': str(series.dtype)
        }
        
        # Add numeric statistics if applicable
        if pd.api.types.is_numeric_dtype(series):
            numeric_series = pd.to_numeric(series, errors='coerce').dropna()
            if not numeric_series.empty:
                summary.update({
                    'min': float(numeric_series.min()),
                    'max': float(numeric_series.max()),
                    'mean': round(float(numeric_series.mean()), 3),
                    'median': float(numeric_series.median()),
                    'std': round(float(numeric_series.std()), 3)
                })
                
                if include_distribution:
                    summary.update({
                        'q25': float(numeric_series.quantile(0.25)),
                        'q75': float(numeric_series.quantile(0.75)),
                        'iqr': float(numeric_series.quantile(0.75) - numeric_series.quantile(0.25))
                    })
        
        # Add categorical statistics if applicable
        elif pd.api.types.is_object_dtype(series):
            non_null_series = series.dropna()
            if not non_null_series.empty:
                value_counts = non_null_series.value_counts()
                summary.update({
                    'most_common_value': value_counts.index[0] if len(value_counts) > 0 else None,
                    'most_common_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'unique_ratio': round(series.nunique() / len(non_null_series), 3) if len(non_null_series) > 0 else 0
                })
        
        return summary
    
    def log_check_start(self, check_name: str, dataset_info: str = "") -> None:
        """Log the start of a quality check."""
        logger.info(f"Starting {check_name} check {dataset_info}")
    
    def log_check_complete(self, check_name: str, score: float, 
                          issues_count: int, warnings_count: int) -> None:
        """Log the completion of a quality check."""
        logger.info(
            f"Completed {check_name} check - Score: {score:.1f}, "
            f"Issues: {issues_count}, Warnings: {warnings_count}"
        )
    
    def merge_check_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge multiple check results into a consolidated result.
        
        Args:
            results (List[Dict]): List of individual check results
            
        Returns:
            Dict[str, Any]: Merged results
        """
        if not results:
            return self.create_result_template(0)
        
        # Calculate weighted average score
        total_score = sum(result['score'] for result in results)
        avg_score = total_score / len(results)
        
        merged = self.create_result_template(avg_score)
        merged['checker_name'] = 'MergedResults'
        
        # Aggregate all issues, warnings, and recommendations
        for result in results:
            merged['critical_issues'].extend(result.get('critical_issues', []))
            merged['warnings'].extend(result.get('warnings', []))
            merged['recommendations'].extend(result.get('recommendations', []))
        
        # Add individual results to details
        merged['details']['individual_results'] = results
        merged['details']['check_count'] = len(results)
        merged['details']['avg_score'] = round(avg_score, 1)
        
        # Summary statistics
        merged['statistics'] = {
            'total_issues': len(merged['critical_issues']),
            'total_warnings': len(merged['warnings']),
            'total_recommendations': len(merged['recommendations']),
            'checks_passed': sum(1 for r in results if r['score'] >= 75),
            'checks_failed': sum(1 for r in results if r['score'] < 75)
        }
        
        return merged


class QualityCheckUtils:
    """
    Utility functions for quality checking operations.
    
    Static methods that can be used across different quality checkers
    without requiring instantiation.
    """
    
    @staticmethod
    def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> Dict[str, Any]:
        """
        Detect outliers using IQR method.
        
        Args:
            series (pd.Series): Numeric series to analyze
            multiplier (float): IQR multiplier for outlier detection
            
        Returns:
            Dict[str, Any]: Outlier detection results
        """
        if not pd.api.types.is_numeric_dtype(series):
            return {'error': 'Series must be numeric for outlier detection'}
        
        numeric_series = pd.to_numeric(series, errors='coerce').dropna()
        
        if numeric_series.empty:
            return {'outliers_count': 0, 'outlier_indices': [], 'outlier_values': []}
        
        q1 = numeric_series.quantile(0.25)
        q3 = numeric_series.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        outlier_mask = (numeric_series < lower_bound) | (numeric_series > upper_bound)
        outlier_indices = numeric_series[outlier_mask].index.tolist()
        outlier_values = numeric_series[outlier_mask].tolist()
        
        return {
            'outliers_count': len(outlier_indices),
            'outlier_percentage': round((len(outlier_indices) / len(numeric_series)) * 100, 2),
            'outlier_indices': outlier_indices,
            'outlier_values': outlier_values,
            'bounds': {'lower': lower_bound, 'upper': upper_bound},
            'iqr_stats': {'q1': q1, 'q3': q3, 'iqr': iqr}
        }
    
    @staticmethod
    def calculate_text_metrics(text_series: pd.Series) -> Dict[str, Any]:
        """
        Calculate comprehensive text metrics for a text series.
        
        Args:
            text_series (pd.Series): Text series to analyze
            
        Returns:
            Dict[str, Any]: Text metrics
        """
        if text_series.empty:
            return {'empty': True}
        
        # Convert to string and handle nulls
        text_series = text_series.astype(str).replace('nan', '')
        
        # Word counts
        word_counts = text_series.apply(lambda x: len(x.split()) if x.strip() else 0)
        
        # Character counts
        char_counts = text_series.apply(len)
        
        # Empty texts
        empty_count = (text_series.str.strip() == '').sum()
        
        return {
            'total_texts': len(text_series),
            'empty_texts': int(empty_count),
            'empty_percentage': round((empty_count / len(text_series)) * 100, 2),
            'word_stats': {
                'min': int(word_counts.min()),
                'max': int(word_counts.max()),
                'mean': round(word_counts.mean(), 2),
                'median': int(word_counts.median()),
                'std': round(word_counts.std(), 2)
            },
            'char_stats': {
                'min': int(char_counts.min()),
                'max': int(char_counts.max()),
                'mean': round(char_counts.mean(), 2),
                'median': int(char_counts.median())
            },
            'uniqueness': {
                'unique_texts': int(text_series.nunique()),
                'duplicate_count': len(text_series) - text_series.nunique(),
                'uniqueness_ratio': round(text_series.nunique() / len(text_series), 3)
            }
        }
    
    @staticmethod
    def validate_numeric_range(series: pd.Series, min_val: Optional[float] = None,
                             max_val: Optional[float] = None) -> Dict[str, Any]:
        """
        Validate that numeric values fall within specified range.
        
        Args:
            series (pd.Series): Numeric series to validate
            min_val (float, optional): Minimum allowed value
            max_val (float, optional): Maximum allowed value
            
        Returns:
            Dict[str, Any]: Range validation results
        """
        numeric_series = pd.to_numeric(series, errors='coerce').dropna()
        
        if numeric_series.empty:
            return {'error': 'No valid numeric values found'}
        
        results = {
            'total_values': len(numeric_series),
            'within_range': 0,
            'below_min': 0,
            'above_max': 0,
            'out_of_range_indices': []
        }
        
        # Check minimum value
        if min_val is not None:
            below_min = numeric_series < min_val
            results['below_min'] = int(below_min.sum())
            results['out_of_range_indices'].extend(numeric_series[below_min].index.tolist())
        
        # Check maximum value
        if max_val is not None:
            above_max = numeric_series > max_val
            results['above_max'] = int(above_max.sum())
            results['out_of_range_indices'].extend(numeric_series[above_max].index.tolist())
        
        # Calculate within range
        out_of_range_count = results['below_min'] + results['above_max']
        results['within_range'] = len(numeric_series) - out_of_range_count
        results['within_range_percentage'] = round((results['within_range'] / len(numeric_series)) * 100, 2)
        
        return results