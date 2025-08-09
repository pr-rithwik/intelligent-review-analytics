"""
Text Quality Checker

Validates text column quality including length, content, diversity, and characteristics.
Analyzes text for business-relevant quality metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

from .base_checker import BaseQualityChecker, QualityCheckUtils
from ..config import TEXT_QUALITY_RULES

logger = logging.getLogger(__name__)

class TextQualityChecker(BaseQualityChecker):
    """
    Validates text column quality including length, content, and characteristics.
    
    Analyzes text for business-relevant quality metrics including readability,
    length distribution, empty content, and text diversity.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize text quality checker with configuration."""
        super().__init__(config)
        self.text_rules = self.config['text_quality_rules']
    
    def check(self, df: pd.DataFrame, text_column: str = 'text', **kwargs) -> Dict[str, Any]:
        """
        Comprehensive text quality analysis.
        
        Args:
            df (pd.DataFrame): Dataset to analyze
            text_column (str): Name of text column to analyze
            
        Returns:
            Dict[str, Any]: Text quality analysis results
        """
        self.log_check_start("text quality", f"for column '{text_column}' on {len(df)} rows")
        
        result = self.create_result_template()
        
        # Validate column exists
        if not self.validate_column_exists(df, text_column):
            self.add_issue(result, f"Text column '{text_column}' not found in dataset")
            result['score'] = 0
            return result
        
        text_series = df[text_column].dropna()
        
        # Check if column is completely empty
        if text_series.empty:
            self.add_issue(result, f"Text column '{text_column}' is completely empty")
            result['score'] = 0
            return result
        
        # Calculate comprehensive text statistics
        text_statistics = self._calculate_comprehensive_text_statistics(text_series, len(df))
        
        # Perform quality assessment
        quality_assessment = self._assess_text_quality(text_statistics, text_series)
        
        # Set score and status
        result['score'] = quality_assessment['score']
        result['status'] = self._get_status_from_score(quality_assessment['score'])
        
        # Add issues and warnings
        result['critical_issues'].extend(quality_assessment['critical_issues'])
        result['warnings'].extend(quality_assessment['warnings'])
        
        # Add detailed results
        result['details'] = {
            'text_statistics': text_statistics,
            'quality_assessment': quality_assessment,
            'column_analyzed': text_column
        }
        
        # Generate recommendations
        result['recommendations'] = self._generate_text_quality_recommendations(
            text_statistics, quality_assessment
        )
        
        # Create summary statistics
        result['statistics'] = self._create_text_quality_statistics(text_statistics)
        
        self.log_check_complete("text quality", result['score'], 
                               len(result['critical_issues']), len(result['warnings']))
        
        return result
    
    def _calculate_comprehensive_text_statistics(self, text_series: pd.Series, 
                                               total_rows: int) -> Dict[str, Any]:
        """Calculate comprehensive text statistics using utility functions and custom analysis."""
        # Use utility function for basic metrics
        basic_metrics = QualityCheckUtils.calculate_text_metrics(text_series)
        
        # Add business-specific analysis
        word_counts = text_series.apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
        
        # Content quality analysis
        empty_texts = (text_series.str.strip() == '').sum()
        very_short_texts = (word_counts < self.text_rules['word_count']['min_words']).sum()
        very_long_texts = (word_counts > self.text_rules['word_count']['max_words']).sum()
        
        # Optimal length analysis
        optimal_min = self.text_rules['word_count'].get('optimal_min', 10)
        optimal_max = self.text_rules['word_count'].get('optimal_max', 200)
        optimal_length_texts = ((word_counts >= optimal_min) & (word_counts <= optimal_max)).sum()
        
        # Enhanced statistics
        enhanced_stats = {
            'coverage_analysis': {
                'total_rows_in_dataset': total_rows,
                'texts_analyzed': len(text_series),
                'coverage_percentage': self.calculate_percentage(len(text_series), total_rows)
            },
            'content_quality': {
                'empty_texts': int(empty_texts),
                'empty_percentage': self.calculate_percentage(empty_texts, total_rows),
                'very_short_texts': int(very_short_texts),
                'very_short_percentage': self.calculate_percentage(very_short_texts, len(text_series)),
                'very_long_texts': int(very_long_texts),
                'very_long_percentage': self.calculate_percentage(very_long_texts, len(text_series)),
                'optimal_length_texts': int(optimal_length_texts),
                'optimal_length_percentage': self.calculate_percentage(optimal_length_texts, len(text_series))
            },
            'length_distribution': {
                'word_count_quartiles': {
                    'q1': int(word_counts.quantile(0.25)),
                    'q2': int(word_counts.quantile(0.50)),
                    'q3': int(word_counts.quantile(0.75)),
                    'iqr': int(word_counts.quantile(0.75) - word_counts.quantile(0.25))
                },
                'length_variance': round(word_counts.var(), 2),
                'length_coefficient_variation': round(word_counts.std() / word_counts.mean(), 3) if word_counts.mean() > 0 else 0
            }
        }
        
        # Combine basic metrics with enhanced analysis
        return {**basic_metrics, **enhanced_stats}
    
    def _assess_text_quality(self, text_stats: Dict, text_series: pd.Series) -> Dict[str, Any]:
        """Assess overall text quality and generate issues/warnings."""
        critical_issues = []
        warnings = []
        penalty = 0
        
        # Check empty text percentage
        empty_threshold = self.text_rules['empty_text']['max_empty_percentage']
        empty_percentage = text_stats['content_quality']['empty_percentage']
        
        if empty_percentage > empty_threshold:
            penalty += self.text_rules['empty_text']['penalty_per_percent'] * empty_percentage
            critical_issues.append(
                f"{empty_percentage:.1f}% of texts are empty (threshold: {empty_threshold}%)"
            )
        elif empty_percentage > empty_threshold / 2:
            warnings.append(f"{empty_percentage:.1f}% of texts are empty")
        
        # Check length distribution issues
        penalty += self._assess_length_distribution(text_stats, warnings)
        
        # Check text diversity
        penalty += self._assess_text_diversity(text_stats, warnings)
        
        # Check coverage
        penalty += self._assess_text_coverage(text_stats, warnings)
        
        # Calculate final score
        score = self.calculate_penalty_score(penalty)
        
        return {
            'score': round(score, 1),
            'penalty_applied': round(penalty, 1),
            'critical_issues': critical_issues,
            'warnings': warnings,
            'quality_indicators': self._create_quality_indicators(text_stats, empty_threshold)
        }
    
    def _assess_length_distribution(self, text_stats: Dict, warnings: List[str]) -> float:
        """Assess text length distribution and return penalty."""
        penalty = 0
        
        # Check very short texts
        very_short_percentage = text_stats['content_quality']['very_short_percentage']
        if very_short_percentage > 10:  # More than 10% too short
            penalty += 15
            warnings.append(
                f"{very_short_percentage:.1f}% of texts are very short "
                f"(< {self.text_rules['word_count']['min_words']} words)"
            )
        
        # Check very long texts
        very_long_percentage = text_stats['content_quality']['very_long_percentage']
        if very_long_percentage > 5:  # More than 5% too long
            penalty += 10
            warnings.append(
                f"{very_long_percentage:.1f}% of texts are very long "
                f"(> {self.text_rules['word_count']['max_words']} words)"
            )
        
        # Check length variance
        min_std = self.text_rules['text_diversity']['min_std_word_count']
        word_std = text_stats['word_stats']['std']
        
        if word_std < min_std:
            penalty += 10
            warnings.append(
                f"Low text length variation (std: {word_std:.1f}) may indicate data quality issues"
            )
        
        return penalty
    
    def _assess_text_diversity(self, text_stats: Dict, warnings: List[str]) -> float:
        """Assess text diversity and return penalty."""
        penalty = 0
        
        # Check uniqueness ratio
        uniqueness_ratio = text_stats['uniqueness']['uniqueness_ratio']
        if uniqueness_ratio < 0.8:  # Less than 80% unique
            penalty += 15
            warnings.append(
                f"Low text uniqueness ({uniqueness_ratio:.1%}) - many duplicate texts detected"
            )
        elif uniqueness_ratio < 0.9:  # Less than 90% unique
            penalty += 5
            warnings.append(
                f"Moderate text duplication detected ({uniqueness_ratio:.1%} unique)"
            )
        
        return penalty
    
    def _assess_text_coverage(self, text_stats: Dict, warnings: List[str]) -> float:
        """Assess text coverage and return penalty."""
        penalty = 0
        
        coverage = text_stats['coverage_analysis']['coverage_percentage']
        if coverage < 95:
            penalty += 10
            warnings.append(f"Text coverage is {coverage:.1f}% - many missing texts")
        elif coverage < 98:
            penalty += 3
            warnings.append(f"Text coverage is {coverage:.1f}% - some missing texts")
        
        return penalty
    
    def _create_quality_indicators(self, text_stats: Dict, empty_threshold: float) -> Dict[str, bool]:
        """Create boolean quality indicators for quick assessment."""
        return {
            'empty_text_acceptable': text_stats['content_quality']['empty_percentage'] <= empty_threshold,
            'length_distribution_good': text_stats['word_stats']['std'] >= self.text_rules['text_diversity']['min_std_word_count'],
            'uniqueness_acceptable': text_stats['uniqueness']['uniqueness_ratio'] >= 0.8,
            'coverage_good': text_stats['coverage_analysis']['coverage_percentage'] >= 95,
            'optimal_length_ratio_good': text_stats['content_quality']['optimal_length_percentage'] >= 50
        }
    
    def _generate_text_quality_recommendations(self, text_stats: Dict, 
                                             quality_assessment: Dict) -> List[str]:
        """Generate recommendations for text quality improvement."""
        recommendations = []
        
        # Empty text recommendations
        empty_percentage = text_stats['content_quality']['empty_percentage']
        if empty_percentage > 5:
            recommendations.append("Remove or populate empty text entries before processing")
        elif empty_percentage > 0:
            recommendations.append("Consider reviewing empty text entries for data quality")
        
        # Length distribution recommendations
        very_short_percentage = text_stats['content_quality']['very_short_percentage']
        if very_short_percentage > 10:
            recommendations.append("Consider filtering out very short texts or reviewing data quality")
        
        very_long_percentage = text_stats['content_quality']['very_long_percentage']
        if very_long_percentage > 5:
            recommendations.append("Review very long texts for data quality or consider truncation")
        
        # Diversity recommendations
        uniqueness_ratio = text_stats['uniqueness']['uniqueness_ratio']
        if uniqueness_ratio < 0.8:
            recommendations.append("Investigate and remove duplicate texts if appropriate")
        
        # Coverage recommendations
        coverage = text_stats['coverage_analysis']['coverage_percentage']
        if coverage < 95:
            recommendations.append("Improve text data collection to reduce missing values")
        
        # Optimal length recommendations
        optimal_percentage = text_stats['content_quality']['optimal_length_percentage']
        if optimal_percentage < 50:
            optimal_min = self.text_rules['word_count'].get('optimal_min', 10)
            optimal_max = self.text_rules['word_count'].get('optimal_max', 200)
            recommendations.append(
                f"Encourage reviews between {optimal_min}-{optimal_max} words for optimal quality"
            )
        
        # If no specific issues, provide general guidance
        if not recommendations:
            recommendations.append("Text quality is good - ready for advanced processing")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _create_text_quality_statistics(self, text_stats: Dict) -> Dict[str, Any]:
        """Create summary statistics for text quality assessment."""
        return {
            'summary': {
                'total_texts_analyzed': text_stats['total_texts'],
                'coverage_percentage': text_stats['coverage_analysis']['coverage_percentage'],
                'empty_text_percentage': text_stats['content_quality']['empty_percentage'],
                'duplicate_percentage': round((1 - text_stats['uniqueness']['uniqueness_ratio']) * 100, 2),
                'optimal_length_percentage': text_stats['content_quality']['optimal_length_percentage']
            },
            'length_analysis': {
                'avg_word_count': text_stats['word_stats']['mean'],
                'median_word_count': text_stats['word_stats']['median'],
                'word_count_std': text_stats['word_stats']['std'],
                'length_coefficient_variation': text_stats['length_distribution']['length_coefficient_variation'],
                'very_short_texts': text_stats['content_quality']['very_short_texts'],
                'very_long_texts': text_stats['content_quality']['very_long_texts']
            },
            'quality_distribution': {
                'empty_texts': text_stats['content_quality']['empty_texts'],
                'optimal_length_texts': text_stats['content_quality']['optimal_length_texts'],
                'unique_texts': text_stats['uniqueness']['unique_texts'],
                'duplicate_texts': text_stats['uniqueness']['duplicate_count']
            },
            'business_metrics': {
                'texts_ready_for_processing': text_stats['total_texts'] - text_stats['content_quality']['empty_texts'],
                'processing_success_rate_estimate': round(
                    ((text_stats['total_texts'] - text_stats['content_quality']['empty_texts'] - 
                      text_stats['content_quality']['very_short_texts']) / text_stats['total_texts']) * 100, 2
                ) if text_stats['total_texts'] > 0 else 0
            }
        }