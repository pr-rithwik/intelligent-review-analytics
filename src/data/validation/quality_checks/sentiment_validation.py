"""
Sentiment Validation Checker

Validates sentiment column values, distribution, and logical consistency.
Ensures sentiment values are valid and checks class balance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Set
import logging

from .base_checker import BaseQualityChecker
from ..config import SENTIMENT_RULES, VALID_SENTIMENT_VALUES

logger = logging.getLogger(__name__)

class SentimentValidator(BaseQualityChecker):
    """
    Validates sentiment column values, distribution, and logical consistency.
    
    Ensures sentiment values are valid, checks class balance, and validates
    business logic constraints for sentiment classification.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize sentiment validator with configuration."""
        super().__init__(config)
        self.sentiment_rules = self.config['sentiment_rules']
        self.valid_values = set(self.sentiment_rules['valid_values'])
    
    def check(self, df: pd.DataFrame, sentiment_column: str = 'sentiment', **kwargs) -> Dict[str, Any]:
        """
        Comprehensive sentiment column validation.
        
        Args:
            df (pd.DataFrame): Dataset to validate
            sentiment_column (str): Name of sentiment column
            
        Returns:
            Dict[str, Any]: Sentiment validation results
        """
        self.log_check_start("sentiment validation", f"for column '{sentiment_column}' on {len(df)} rows")
        
        result = self.create_result_template()
        
        # Validate column exists
        if not self.validate_column_exists(df, sentiment_column):
            self.add_issue(result, f"Sentiment column '{sentiment_column}' not found in dataset")
            result['score'] = 0
            return result
        
        sentiment_series = df[sentiment_column].dropna()
        
        # Check if column is completely empty
        if sentiment_series.empty:
            self.add_issue(result, f"Sentiment column '{sentiment_column}' is completely empty")
            result['score'] = 0
            return result
        
        # Analyze sentiment distribution and values
        sentiment_analysis = self._analyze_sentiment_distribution(sentiment_series, len(df))
        
        # Validate sentiment values and calculate score
        validation_results = self._validate_sentiment_values(sentiment_series, sentiment_analysis)
        
        # Set score and status
        result['score'] = validation_results['score']
        result['status'] = self._get_status_from_score(validation_results['score'])
        
        # Add issues and warnings
        result['critical_issues'].extend(validation_results['critical_issues'])
        result['warnings'].extend(validation_results['warnings'])
        
        # Add detailed results
        result['details'] = {
            'sentiment_analysis': sentiment_analysis,
            'validation_results': validation_results,
            'column_analyzed': sentiment_column
        }
        
        # Generate recommendations
        result['recommendations'] = self._generate_sentiment_recommendations(
            sentiment_analysis, validation_results
        )
        
        # Create summary statistics
        result['statistics'] = self._create_sentiment_statistics(sentiment_analysis, validation_results)
        
        self.log_check_complete("sentiment validation", result['score'], 
                               len(result['critical_issues']), len(result['warnings']))
        
        return result
    
    def _analyze_sentiment_distribution(self, sentiment_series: pd.Series, 
                                      total_rows: int) -> Dict[str, Any]:
        """Analyze sentiment value distribution and statistics."""
        # Value counts and percentages
        value_counts = sentiment_series.value_counts().sort_index()
        value_percentages = (sentiment_series.value_counts(normalize=True) * 100).round(2).sort_index()
        
        # Missing value analysis
        missing_count = total_rows - len(sentiment_series)
        missing_percentage = self.calculate_percentage(missing_count, total_rows)
        
        # Class balance analysis
        balance_analysis = self._analyze_class_balance(value_percentages)
        
        return {
            'basic_info': {
                'total_values': len(sentiment_series),
                'total_rows': total_rows,
                'missing_count': missing_count,
                'missing_percentage': missing_percentage,
                'coverage_percentage': self.calculate_percentage(len(sentiment_series), total_rows)
            },
            'value_distribution': {
                'unique_values': sorted(sentiment_series.unique()),
                'value_counts': value_counts.to_dict(),
                'value_percentages': value_percentages.to_dict(),
                'most_common_value': value_counts.idxmax() if not value_counts.empty else None,
                'least_common_value': value_counts.idxmin() if not value_counts.empty else None
            },
            'class_balance': balance_analysis
        }
    
    def _analyze_class_balance(self, value_percentages: pd.Series) -> Dict[str, Any]:
        """Analyze class balance and imbalance severity."""
        if len(value_percentages) <= 1:
            return {
                'max_class_percentage': value_percentages.iloc[0] if len(value_percentages) > 0 else 0,
                'min_class_percentage': 0,
                'balance_ratio': float('inf'),
                'is_severely_imbalanced': True,
                'is_moderately_imbalanced': True,
                'imbalance_severity': 'SEVERE'
            }
        
        max_class_percentage = value_percentages.max()
        min_class_percentage = value_percentages.min()
        balance_ratio = max_class_percentage / min_class_percentage if min_class_percentage > 0 else float('inf')
        
        # Determine imbalance severity
        severe_threshold = self.sentiment_rules['class_balance']['severe_imbalance_threshold']
        moderate_threshold = self.sentiment_rules['class_balance']['moderate_imbalance_threshold']
        
        is_severely_imbalanced = max_class_percentage > severe_threshold
        is_moderately_imbalanced = max_class_percentage > moderate_threshold
        
        if is_severely_imbalanced:
            severity = 'SEVERE'
        elif is_moderately_imbalanced:
            severity = 'MODERATE'
        else:
            severity = 'ACCEPTABLE'
        
        return {
            'max_class_percentage': max_class_percentage,
            'min_class_percentage': min_class_percentage,
            'balance_ratio': balance_ratio if balance_ratio != float('inf') else None,
            'is_severely_imbalanced': is_severely_imbalanced,
            'is_moderately_imbalanced': is_moderately_imbalanced,
            'imbalance_severity': severity,
            'class_distribution_entropy': self._calculate_entropy(value_percentages)
        }
    
    def _calculate_entropy(self, percentages: pd.Series) -> float:
        """Calculate entropy of class distribution (higher = more balanced)."""
        if len(percentages) <= 1:
            return 0.0
        
        # Convert percentages to probabilities
        probabilities = percentages / 100.0
        
        # Calculate entropy
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(percentages))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return round(normalized_entropy, 3)
    
    def _validate_sentiment_values(self, sentiment_series: pd.Series, 
                                 sentiment_analysis: Dict) -> Dict[str, Any]:
        """Validate sentiment values against business rules."""
        actual_values = set(sentiment_series.unique())
        invalid_values = actual_values - self.valid_values
        
        critical_issues = []
        warnings = []
        penalty = 0
        
        # Check for invalid values
        if invalid_values:
            penalty += self.sentiment_rules['invalid_value_penalty']
            critical_issues.append(f"Invalid sentiment values found: {sorted(invalid_values)}")
        
        # Check missing values
        missing_percentage = sentiment_analysis['basic_info']['missing_percentage']
        max_missing = self.sentiment_rules['max_missing_percentage']
        
        if missing_percentage > max_missing:
            penalty += 40
            critical_issues.append(
                f"Sentiment column has {missing_percentage:.1f}% missing values "
                f"(max allowed: {max_missing}%)"
            )
        
        # Check class balance
        balance_info = sentiment_analysis['class_balance']
        
        if balance_info['is_severely_imbalanced']:
            penalty += self.sentiment_rules['class_balance']['penalty_severe']
            warnings.append(
                f"Severe class imbalance: {balance_info['max_class_percentage']:.1f}% "
                f"vs {balance_info['min_class_percentage']:.1f}%"
            )
        elif balance_info['is_moderately_imbalanced']:
            penalty += self.sentiment_rules['class_balance']['penalty_moderate']
            warnings.append(
                f"Moderate class imbalance: {balance_info['max_class_percentage']:.1f}% "
                f"vs {balance_info['min_class_percentage']:.1f}%"
            )
        
        # Calculate final score
        score = self.calculate_penalty_score(penalty)
        
        return {
            'score': round(score, 1),
            'penalty_applied': round(penalty, 1),
            'valid_values_check': {
                'expected_values': sorted(self.valid_values),
                'actual_values': sorted(actual_values),
                'invalid_values': sorted(invalid_values),
                'has_invalid_values': len(invalid_values) > 0,
                'valid_value_coverage': len(actual_values & self.valid_values)
            },
            'critical_issues': critical_issues,
            'warnings': warnings,
            'validation_passed': len(critical_issues) == 0
        }
    
    def _generate_sentiment_recommendations(self, sentiment_analysis: Dict,
                                          validation_results: Dict) -> List[str]:
        """Generate recommendations for sentiment validation issues."""
        recommendations = []
        
        # Invalid values recommendations
        if validation_results['valid_values_check']['has_invalid_values']:
            invalid_vals = validation_results['valid_values_check']['invalid_values']
            recommendations.append(
                f"Fix invalid sentiment values: {invalid_vals}. "
                f"Valid values are: {sorted(self.valid_values)}"
            )
        
        # Missing values recommendations
        missing_percentage = sentiment_analysis['basic_info']['missing_percentage']
        if missing_percentage > 0:
            recommendations.append(
                "Address missing sentiment values - sentiment is critical for analysis"
            )
        
        # Class balance recommendations
        balance_info = sentiment_analysis['class_balance']
        if balance_info['is_severely_imbalanced']:
            recommendations.append(
                "Consider data collection strategies to improve class balance or "
                "use appropriate sampling techniques during model training"
            )
        elif balance_info['is_moderately_imbalanced']:
            recommendations.append(
                "Monitor model performance carefully due to class imbalance, "
                "consider using stratified sampling"
            )
        
        # Coverage recommendations
        coverage = sentiment_analysis['basic_info']['coverage_percentage']
        if coverage < 100:
            recommendations.append("Ensure all rows have sentiment labels for complete analysis")
        
        # Data quality recommendations
        if balance_info['class_distribution_entropy'] < 0.5:
            recommendations.append(
                "Very low class diversity detected - review data collection process"
            )
        
        # If no issues, provide positive feedback
        if not recommendations:
            recommendations.append("Sentiment column validation passed - ready for model training")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _create_sentiment_statistics(self, sentiment_analysis: Dict, 
                                   validation_results: Dict) -> Dict[str, Any]:
        """Create summary statistics for sentiment validation."""
        value_dist = sentiment_analysis['value_distribution']
        balance_info = sentiment_analysis['class_balance']
        basic_info = sentiment_analysis['basic_info']
        
        return {
            'distribution_summary': {
                'total_sentiment_values': basic_info['total_values'],
                'unique_sentiment_values': len(value_dist['unique_values']),
                'missing_percentage': basic_info['missing_percentage'],
                'coverage_percentage': basic_info['coverage_percentage']
            },
            'value_analysis': {
                'valid_values_found': validation_results['valid_values_check']['valid_value_coverage'],
                'invalid_values_count': len(validation_results['valid_values_check']['invalid_values']),
                'most_common_sentiment': value_dist['most_common_value'],
                'least_common_sentiment': value_dist['least_common_value']
            },
            'balance_metrics': {
                'imbalance_severity': balance_info['imbalance_severity'],
                'max_class_percentage': balance_info['max_class_percentage'],
                'min_class_percentage': balance_info['min_class_percentage'],
                'balance_ratio': balance_info['balance_ratio'],
                'distribution_entropy': balance_info['class_distribution_entropy']
            },
            'validation_summary': {
                'validation_passed': validation_results['validation_passed'],
                'critical_issues_count': len(validation_results['critical_issues']),
                'warnings_count': len(validation_results['warnings']),
                'overall_score': validation_results['score']
            }
        }