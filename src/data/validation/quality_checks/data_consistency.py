"""
Data Consistency Checker

Validates data consistency, logical relationships, and integrity constraints.
Checks for logical inconsistencies and business rule violations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

from .base_checker import BaseQualityChecker
from ..config import CONSISTENCY_RULES

logger = logging.getLogger(__name__)

class DataConsistencyChecker(BaseQualityChecker):
    """
    Validates data consistency, logical relationships, and integrity constraints.
    
    Checks for logical inconsistencies, data type issues, and business rule
    violations across related columns.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize data consistency checker with configuration."""
        super().__init__(config)
        self.consistency_rules = self.config['consistency_rules']
        self.expected_types = self.consistency_rules['data_types']['expected_types']
    
    def check(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Comprehensive data consistency validation.
        
        Args:
            df (pd.DataFrame): Dataset to validate
            
        Returns:
            Dict[str, Any]: Data consistency validation results
        """
        self.log_check_start("data consistency", f"on {len(df)} rows")
        
        result = self.create_result_template()
        
        # Run individual consistency checks
        consistency_checks = {
            'data_type_consistency': self._check_data_type_consistency(df),
            'id_consistency': self._check_id_consistency(df),
            'helpfulness_logic': self._check_helpfulness_logic(df),
            'text_sentiment_consistency': self._check_text_sentiment_consistency(df),
            'numerical_ranges': self._check_numerical_ranges(df)
        }
        
        # Aggregate results
        total_penalty = sum(check.get('penalty', 0) for check in consistency_checks.values())
        all_issues = []
        all_warnings = []
        
        for check_name, check_result in consistency_checks.items():
            all_issues.extend(check_result.get('issues', []))
            all_warnings.extend(check_result.get('warnings', []))
        
        # Calculate overall score
        overall_score = self.calculate_penalty_score(total_penalty)
        
        # Set result properties
        result['score'] = overall_score
        result['status'] = self._get_status_from_score(overall_score)
        result['critical_issues'] = all_issues
        result['warnings'] = all_warnings
        
        # Add detailed results
        result['details'] = {
            'consistency_checks': consistency_checks,
            'total_penalty': round(total_penalty, 1)
        }
        
        # Generate recommendations
        result['recommendations'] = self._generate_consistency_recommendations(consistency_checks)
        
        # Create summary statistics
        result['statistics'] = self._create_consistency_statistics(consistency_checks)
        
        self.log_check_complete("data consistency", result['score'], 
                               len(result['critical_issues']), len(result['warnings']))
        
        return result
    
    def _check_data_type_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data type consistency for expected columns."""
        type_penalty = self.consistency_rules['data_types']['type_mismatch_penalty']
        
        issues = []
        warnings = []
        penalty = 0
        type_checks = {}
        
        for column in df.columns:
            if column in self.expected_types:
                actual_type = str(df[column].dtype)
                expected = self.expected_types[column]
                is_correct = actual_type in expected
                
                # Additional validation for numeric columns
                if not is_correct and any('int' in t or 'float' in t for t in expected):
                    # Check if values can be converted to numeric
                    non_null_series = df[column].dropna()
                    if not non_null_series.empty:
                        numeric_conversion = pd.to_numeric(non_null_series, errors='coerce')
                        conversion_success_rate = (numeric_conversion.notna().sum() / len(non_null_series))
                        
                        type_checks[column] = {
                            'actual_type': actual_type,
                            'expected_types': expected,
                            'is_correct': is_correct,
                            'numeric_conversion_rate': round(conversion_success_rate, 3),
                            'can_be_converted': conversion_success_rate > 0.95
                        }
                        
                        if conversion_success_rate > 0.95:
                            warnings.append(
                                f"Column '{column}' can be converted to numeric type "
                                f"({conversion_success_rate:.1%} success rate)"
                            )
                        else:
                            penalty += type_penalty
                            issues.append(
                                f"Column '{column}' has incorrect type '{actual_type}' and "
                                f"cannot be reliably converted (only {conversion_success_rate:.1%} convertible)"
                            )
                    else:
                        type_checks[column] = {
                            'actual_type': actual_type,
                            'expected_types': expected,
                            'is_correct': is_correct,
                            'all_null': True
                        }
                else:
                    type_checks[column] = {
                        'actual_type': actual_type,
                        'expected_types': expected,
                        'is_correct': is_correct
                    }
                    
                    if not is_correct:
                        if column in ['sentiment']:  # Critical columns
                            penalty += type_penalty
                            issues.append(
                                f"Critical column '{column}' has incorrect type '{actual_type}', "
                                f"expected one of {expected}"
                            )
                        else:
                            warnings.append(
                                f"Column '{column}' has type '{actual_type}', expected one of {expected}"
                            )
        
        return {
            'type_checks': type_checks,
            'issues': issues,
            'warnings': warnings,
            'penalty': penalty,
            'passed': len(issues) == 0,
            'compliance_rate': self._calculate_type_compliance_rate(type_checks)
        }
    
    def _check_id_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check ID column consistency and completeness."""
        id_rules = self.consistency_rules['id_consistency']
        issues = []
        warnings = []
        penalty = 0
        id_checks = {}
        
        # Check Product ID consistency
        if 'productId' in df.columns:
            product_id_analysis = self._analyze_id_column(df, 'productId')
            id_checks['productId'] = product_id_analysis
            
            if product_id_analysis['empty_percentage'] > id_rules['max_missing_product_id']:
                penalty += id_rules['empty_id_penalty']
                warnings.append(
                    f"Product ID missing for {product_id_analysis['empty_percentage']:.1f}% of reviews "
                    f"(threshold: {id_rules['max_missing_product_id']}%)"
                )
        
        # Check User ID consistency
        if 'userId' in df.columns:
            user_id_analysis = self._analyze_id_column(df, 'userId')
            id_checks['userId'] = user_id_analysis
            
            if user_id_analysis['empty_percentage'] > id_rules['max_missing_user_id']:
                penalty += id_rules['empty_id_penalty']
                warnings.append(
                    f"User ID missing for {user_id_analysis['empty_percentage']:.1f}% of reviews "
                    f"(threshold: {id_rules['max_missing_user_id']}%)"
                )
        
        return {
            'id_checks': id_checks,
            'issues': issues,
            'warnings': warnings,
            'penalty': penalty,
            'passed': len(issues) == 0
        }
    
    def _check_helpfulness_logic(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check helpfulness column logical consistency."""
        issues = []
        warnings = []
        penalty = 0
        helpfulness_checks = {}
        
        if 'helpfulnessNumerator' in df.columns and 'helpfulnessDenominator' in df.columns:
            numerator = df['helpfulnessNumerator']
            denominator = df['helpfulnessDenominator']
            
            # Check for logical inconsistencies (numerator > denominator)
            logical_inconsistencies = (numerator > denominator).sum()
            if logical_inconsistencies > 0:
                penalty += 15
                warnings.append(
                    f"{logical_inconsistencies} reviews have helpfulness numerator > denominator"
                )
            
            # Check for negative values
            negative_numerator = (numerator < 0).sum()
            negative_denominator = (denominator < 0).sum()
            
            if negative_numerator > 0:
                penalty += 10
                warnings.append(f"{negative_numerator} reviews have negative helpfulness numerator")
            
            if negative_denominator > 0:
                penalty += 10
                warnings.append(f"{negative_denominator} reviews have negative helpfulness denominator")
            
            # Check for zero denominators with non-zero numerators
            zero_denom_nonzero_num = ((denominator == 0) & (numerator > 0)).sum()
            if zero_denom_nonzero_num > 0:
                penalty += 5
                warnings.append(
                    f"{zero_denom_nonzero_num} reviews have zero denominator with positive numerator"
                )
            
            helpfulness_checks = {
                'logical_inconsistencies': int(logical_inconsistencies),
                'negative_numerator': int(negative_numerator),
                'negative_denominator': int(negative_denominator),
                'zero_denom_nonzero_num': int(zero_denom_nonzero_num),
                'valid_helpfulness_entries': int(len(df) - logical_inconsistencies - negative_numerator - negative_denominator)
            }
        
        return {
            'helpfulness_checks': helpfulness_checks,
            'issues': issues,
            'warnings': warnings,
            'penalty': penalty,
            'passed': len(issues) == 0
        }
    
    def _check_text_sentiment_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check consistency between text and sentiment columns."""
        issues = []
        warnings = []
        penalty = 0
        consistency_checks = {}
        
        if 'text' in df.columns and 'sentiment' in df.columns:
            # Check for empty text with non-null sentiment
            empty_text_mask = (df['text'].isnull()) | (df['text'].str.strip() == '')
            has_sentiment_mask = df['sentiment'].notnull()
            
            empty_text_with_sentiment = (empty_text_mask & has_sentiment_mask).sum()
            
            if empty_text_with_sentiment > 0:
                penalty += self.consistency_rules['text_sentiment_consistency']['penalty_per_inconsistency']
                warnings.append(
                    f"{empty_text_with_sentiment} reviews have sentiment labels but empty text"
                )
            
            # Check for text without sentiment
            has_text_mask = (~empty_text_mask)
            no_sentiment_mask = df['sentiment'].isnull()
            
            text_without_sentiment = (has_text_mask & no_sentiment_mask).sum()
            
            if text_without_sentiment > 0:
                penalty += 5  # Lighter penalty as this might be intentional for test data
                warnings.append(
                    f"{text_without_sentiment} reviews have text but no sentiment labels"
                )
            
            consistency_checks = {
                'empty_text_with_sentiment': int(empty_text_with_sentiment),
                'text_without_sentiment': int(text_without_sentiment),
                'consistent_entries': int(len(df) - empty_text_with_sentiment - text_without_sentiment),
                'consistency_rate': self.calculate_percentage(
                    len(df) - empty_text_with_sentiment - text_without_sentiment, len(df)
                )
            }
        
        return {
            'consistency_checks': consistency_checks,
            'issues': issues,
            'warnings': warnings,
            'penalty': penalty,
            'passed': len(issues) == 0
        }
    
    def _check_numerical_ranges(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check numerical columns for reasonable ranges."""
        issues = []
        warnings = []
        penalty = 0
        range_checks = {}
        
        # Define expected ranges for known columns
        expected_ranges = {
            'sentiment': {'min': -1, 'max': 1},
            'helpfulnessNumerator': {'min': 0, 'max': None},
            'helpfulnessDenominator': {'min': 0, 'max': None}
        }
        
        for column, range_def in expected_ranges.items():
            if column in df.columns:
                numeric_series = pd.to_numeric(df[column], errors='coerce').dropna()
                
                if not numeric_series.empty:
                    min_val = range_def.get('min')
                    max_val = range_def.get('max')
                    
                    out_of_range_count = 0
                    
                    if min_val is not None:
                        below_min = (numeric_series < min_val).sum()
                        out_of_range_count += below_min
                        
                        if below_min > 0:
                            penalty += 10
                            warnings.append(
                                f"{below_min} values in '{column}' are below minimum ({min_val})"
                            )
                    
                    if max_val is not None:
                        above_max = (numeric_series > max_val).sum()
                        out_of_range_count += above_max
                        
                        if above_max > 0:
                            penalty += 10
                            warnings.append(
                                f"{above_max} values in '{column}' are above maximum ({max_val})"
                            )
                    
                    range_checks[column] = {
                        'total_values': len(numeric_series),
                        'within_range': len(numeric_series) - out_of_range_count,
                        'out_of_range': out_of_range_count,
                        'range_compliance_rate': self.calculate_percentage(
                            len(numeric_series) - out_of_range_count, len(numeric_series)
                        ),
                        'min_value': float(numeric_series.min()),
                        'max_value': float(numeric_series.max())
                    }
        
        return {
            'range_checks': range_checks,
            'issues': issues,
            'warnings': warnings,
            'penalty': penalty,
            'passed': len(issues) == 0
        }
    
    def _analyze_id_column(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Analyze ID column characteristics."""
        series = df[column]
        
        empty_count = series.isnull().sum()
        empty_percentage = self.calculate_percentage(empty_count, len(series))
        
        # Check for empty strings
        if series.dtype == 'object':
            empty_strings = (series.str.strip() == '').sum()
            total_empty = empty_count + empty_strings
            total_empty_percentage = self.calculate_percentage(total_empty, len(series))
        else:
            empty_strings = 0
            total_empty = empty_count
            total_empty_percentage = empty_percentage
        
        return {
            'total_values': len(series),
            'null_values': int(empty_count),
            'empty_strings': int(empty_strings),
            'total_empty': int(total_empty),
            'empty_percentage': total_empty_percentage,
            'unique_values': int(series.nunique()),
            'uniqueness_ratio': round(series.nunique() / len(series), 3) if len(series) > 0 else 0
        }
    
    def _calculate_type_compliance_rate(self, type_checks: Dict) -> float:
        """Calculate overall type compliance rate."""
        if not type_checks:
            return 100.0
        
        compliant_count = sum(1 for check in type_checks.values() if check.get('is_correct', False))
        return self.calculate_percentage(compliant_count, len(type_checks))
    
    def _generate_consistency_recommendations(self, consistency_checks: Dict) -> List[str]:
        """Generate recommendations for consistency improvements."""
        recommendations = []
        
        # Data type recommendations
        type_check = consistency_checks.get('data_type_consistency', {})
        if not type_check.get('passed', True):
            recommendations.append("Fix data type mismatches for proper processing and analysis")
        
        # ID consistency recommendations
        id_check = consistency_checks.get('id_consistency', {})
        if id_check.get('id_checks'):
            for id_type, analysis in id_check['id_checks'].items():
                if analysis['empty_percentage'] > 10:
                    recommendations.append(f"Improve {id_type} data collection to reduce missing values")
        
        # Helpfulness logic recommendations
        helpfulness_check = consistency_checks.get('helpfulness_logic', {})
        if helpfulness_check.get('helpfulness_checks'):
            checks = helpfulness_check['helpfulness_checks']
            if checks.get('logical_inconsistencies', 0) > 0:
                recommendations.append("Fix helpfulness data where numerator exceeds denominator")
            if checks.get('negative_numerator', 0) > 0 or checks.get('negative_denominator', 0) > 0:
                recommendations.append("Remove negative values from helpfulness columns")
        
        # Text-sentiment consistency recommendations
        text_sentiment_check = consistency_checks.get('text_sentiment_consistency', {})
        if text_sentiment_check.get('consistency_checks'):
            checks = text_sentiment_check['consistency_checks']
            if checks.get('empty_text_with_sentiment', 0) > 0:
                recommendations.append("Remove sentiment labels from empty text entries or populate text content")
            if checks.get('text_without_sentiment', 0) > 0:
                recommendations.append("Add sentiment labels to text entries or remove unlabeled text")
        
        # Numerical range recommendations
        range_check = consistency_checks.get('numerical_ranges', {})
        if range_check.get('range_checks'):
            for column, analysis in range_check['range_checks'].items():
                if analysis['range_compliance_rate'] < 95:
                    recommendations.append(f"Review {column} values outside expected range")
        
        # General data quality recommendations
        total_issues = sum(len(check.get('issues', [])) for check in consistency_checks.values())
        total_warnings = sum(len(check.get('warnings', [])) for check in consistency_checks.values())
        
        if total_issues == 0 and total_warnings == 0:
            recommendations.append("Data consistency is excellent - ready for advanced processing")
        elif total_issues == 0 and total_warnings < 5:
            recommendations.append("Data consistency is good with minor warnings to monitor")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _create_consistency_statistics(self, consistency_checks: Dict) -> Dict[str, Any]:
        """Create summary statistics for consistency validation."""
        # Aggregate statistics across all checks
        total_issues = sum(len(check.get('issues', [])) for check in consistency_checks.values())
        total_warnings = sum(len(check.get('warnings', [])) for check in consistency_checks.values())
        total_penalty = sum(check.get('penalty', 0) for check in consistency_checks.values())
        
        checks_passed = sum(1 for check in consistency_checks.values() if check.get('passed', False))
        total_checks = len(consistency_checks)
        
        # Specific check statistics
        check_summary = {}
        for check_name, check_result in consistency_checks.items():
            check_summary[check_name] = {
                'passed': check_result.get('passed', False),
                'penalty': check_result.get('penalty', 0),
                'issues_count': len(check_result.get('issues', [])),
                'warnings_count': len(check_result.get('warnings', []))
            }
        
        # Calculate compliance rates
        type_compliance = 100.0
        if 'data_type_consistency' in consistency_checks:
            type_compliance = consistency_checks['data_type_consistency'].get('compliance_rate', 100.0)
        
        return {
            'overall_summary': {
                'total_consistency_checks': total_checks,
                'checks_passed': checks_passed,
                'checks_failed': total_checks - checks_passed,
                'pass_rate': self.calculate_percentage(checks_passed, total_checks),
                'total_issues': total_issues,
                'total_warnings': total_warnings,
                'total_penalty_applied': round(total_penalty, 1)
            },
            'check_details': check_summary,
            'compliance_metrics': {
                'data_type_compliance_rate': type_compliance,
                'overall_consistency_score': self.calculate_penalty_score(total_penalty)
            },
            'severity_distribution': {
                'critical_checks_failed': sum(1 for check in consistency_checks.values() 
                                            if not check.get('passed', True) and check.get('penalty', 0) > 15),
                'minor_issues_detected': sum(1 for check in consistency_checks.values() 
                                           if check.get('passed', True) and len(check.get('warnings', [])) > 0),
                'clean_checks': sum(1 for check in consistency_checks.values() 
                                  if check.get('passed', True) and len(check.get('warnings', [])) == 0)
            }
        }