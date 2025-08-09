"""
Missing Value Quality Checker

Validates missing values according to column importance and business rules.
Applies different thresholds and penalties based on business criticality.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

from .base_checker import BaseQualityChecker
from ..config import MISSING_VALUE_THRESHOLDS

logger = logging.getLogger(__name__)

class MissingValueChecker(BaseQualityChecker):
    """
    Validates missing values according to column importance and business rules.
    
    Applies different thresholds and penalties based on whether columns
    are critical, important, or optional for business operations.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize missing value checker with configuration."""
        super().__init__(config)
        self.thresholds = self.config['missing_value_thresholds']
    
    def check(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Comprehensive missing value analysis with business impact scoring.
        
        Args:
            df (pd.DataFrame): Dataset to analyze
            
        Returns:
            Dict[str, Any]: Missing value analysis results
        """
        self.log_check_start("missing values", f"on {len(df)} rows")
        
        result = self.create_result_template()
        
        # Analyze each column
        missing_analysis = {}
        total_penalty = 0
        
        for column in df.columns:
            column_analysis = self._analyze_column_missing_values(df, column)
            missing_analysis[column] = column_analysis
            total_penalty += column_analysis['penalty_applied']
            
            # Add issues and warnings
            if column_analysis['status'] == 'CRITICAL':
                self.add_issue(result, column_analysis['issue_message'], is_critical=True)
            elif column_analysis['status'] == 'WARNING':
                self.add_issue(result, column_analysis['warning_message'], is_critical=False)
        
        # Calculate overall score
        score = self.calculate_penalty_score(total_penalty)
        result['score'] = score
        result['status'] = self._get_status_from_score(score)
        
        # Add detailed analysis
        result['details'] = {
            'missing_analysis': missing_analysis,
            'overall_statistics': self._calculate_overall_missing_stats(df, missing_analysis),
            'total_penalty': round(total_penalty, 1)
        }
        
        # Generate recommendations
        result['recommendations'] = self._generate_missing_value_recommendations(missing_analysis)
        
        # Create summary statistics
        result['statistics'] = self._create_missing_value_statistics(missing_analysis)
        
        self.log_check_complete("missing values", score, 
                               len(result['critical_issues']), len(result['warnings']))
        
        return result
    
    def _analyze_column_missing_values(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Analyze missing values for a single column."""
        missing_count = df[column].isnull().sum()
        missing_percentage = self.calculate_percentage(missing_count, len(df))
        column_type = self.categorize_column_importance(column)
        
        column_analysis = {
            'missing_count': int(missing_count),
            'missing_percentage': missing_percentage,
            'total_values': len(df),
            'non_null_values': int(len(df) - missing_count),
            'column_type': column_type,
            'status': 'OK',
            'penalty_applied': 0,
            'issue_message': None,
            'warning_message': None
        }
        
        # Calculate penalty and determine status
        penalty = self._calculate_missing_value_penalty(column, missing_percentage, column_type)
        column_analysis['penalty_applied'] = penalty
        
        # Assess status and generate messages
        status, issue_msg, warning_msg = self._assess_missing_value_status(
            column, missing_percentage, column_type
        )
        
        column_analysis.update({
            'status': status,
            'issue_message': issue_msg,
            'warning_message': warning_msg
        })
        
        return column_analysis
    
    def _calculate_missing_value_penalty(self, column: str, missing_percentage: float, 
                                       column_type: str) -> float:
        """Calculate penalty for missing values based on column importance."""
        if column_type == 'critical':
            threshold = self.thresholds['critical_columns']['max_percentage']
            multiplier = self.thresholds['critical_columns']['penalty_multiplier']
            if missing_percentage > threshold:
                return min(50, (missing_percentage - threshold) * multiplier * 5)
        
        elif column_type == 'important':
            threshold = self.thresholds['important_columns']['max_percentage']
            multiplier = self.thresholds['important_columns']['penalty_multiplier']
            if missing_percentage > threshold:
                return min(20, (missing_percentage - threshold) * multiplier * 2)
        
        elif column_type == 'optional':
            threshold = self.thresholds['optional_columns']['max_percentage']
            multiplier = self.thresholds['optional_columns']['penalty_multiplier']
            if missing_percentage > threshold:
                return min(5, (missing_percentage - threshold) * multiplier)
        
        return 0
    
    def _assess_missing_value_status(self, column: str, missing_percentage: float, 
                                   column_type: str) -> tuple:
        """Assess status and generate messages for missing values."""
        issue_msg = None
        warning_msg = None
        status = 'OK'
        
        if column_type == 'critical':
            threshold = self.thresholds['critical_columns']['max_percentage']
            if missing_percentage > threshold:
                status = 'CRITICAL'
                issue_msg = (f"Critical column '{column}' has {missing_percentage:.1f}% missing values "
                           f"(threshold: {threshold}%)")
            elif missing_percentage > threshold / 2:
                status = 'WARNING'
                warning_msg = f"Critical column '{column}' has {missing_percentage:.1f}% missing values"
        
        elif column_type == 'important':
            threshold = self.thresholds['important_columns']['max_percentage']
            if missing_percentage > threshold:
                status = 'WARNING'
                warning_msg = (f"Important column '{column}' has {missing_percentage:.1f}% missing values "
                             f"(threshold: {threshold}%)")
        
        elif column_type == 'optional':
            threshold = self.thresholds['optional_columns']['max_percentage']
            if missing_percentage > threshold:
                status = 'WARNING'
                warning_msg = (f"Optional column '{column}' has {missing_percentage:.1f}% missing values "
                             f"(threshold: {threshold}%)")
        
        return status, issue_msg, warning_msg
    
    def _calculate_overall_missing_stats(self, df: pd.DataFrame, 
                                       missing_analysis: Dict) -> Dict[str, Any]:
        """Calculate overall missing value statistics."""
        total_cells = df.size
        total_missing = df.isnull().sum().sum()
        
        columns_with_missing = sum(1 for analysis in missing_analysis.values() 
                                 if analysis['missing_count'] > 0)
        
        critical_cols_with_issues = sum(1 for analysis in missing_analysis.values()
                                      if analysis['column_type'] == 'critical' and 
                                      analysis['status'] in ['CRITICAL', 'WARNING'])
        
        completely_empty_columns = [col for col, analysis in missing_analysis.items() 
                                  if analysis['missing_percentage'] == 100]
        
        return {
            'total_missing_values': int(total_missing),
            'total_cells': int(total_cells),
            'overall_missing_percentage': self.calculate_percentage(total_missing, total_cells),
            'columns_with_missing': columns_with_missing,
            'total_columns': len(missing_analysis),
            'critical_columns_with_issues': critical_cols_with_issues,
            'completely_empty_columns': completely_empty_columns,
            'columns_by_type': self._categorize_columns_by_type(missing_analysis)
        }
    
    def _categorize_columns_by_type(self, missing_analysis: Dict) -> Dict[str, List[str]]:
        """Categorize columns by their importance type."""
        by_type = {'critical': [], 'important': [], 'optional': [], 'unknown': []}
        
        for column, analysis in missing_analysis.items():
            column_type = analysis['column_type']
            by_type[column_type].append(column)
        
        return by_type
    
    def _generate_missing_value_recommendations(self, missing_analysis: Dict) -> List[str]:
        """Generate recommendations for missing value issues."""
        recommendations = []
        
        # Critical column recommendations
        critical_issues = [col for col, analysis in missing_analysis.items()
                         if analysis['column_type'] == 'critical' and analysis['status'] == 'CRITICAL']
        if critical_issues:
            recommendations.append(
                f"URGENT: Address missing values in critical columns: {', '.join(critical_issues)}"
            )
        
        # High missing percentage recommendations
        high_missing_cols = [col for col, analysis in missing_analysis.items()
                           if analysis['missing_percentage'] > 25]
        if high_missing_cols:
            recommendations.append(
                f"Review data collection for columns with >25% missing: {', '.join(high_missing_cols)}"
            )
        
        # Completely empty columns
        empty_cols = [col for col, analysis in missing_analysis.items()
                     if analysis['missing_percentage'] == 100]
        if empty_cols:
            recommendations.append(
                f"Remove or populate completely empty columns: {', '.join(empty_cols)}"
            )
        
        # Data quality improvement recommendations
        if not recommendations:
            avg_missing = np.mean([analysis['missing_percentage'] for analysis in missing_analysis.values()])
            if avg_missing > 10:
                recommendations.append("Consider improving data collection processes to reduce missing values")
            else:
                recommendations.append("Missing value levels are acceptable for most analyses")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _create_missing_value_statistics(self, missing_analysis: Dict) -> Dict[str, Any]:
        """Create summary statistics for missing values."""
        missing_percentages = [analysis['missing_percentage'] for analysis in missing_analysis.values()]
        
        # Statistics by column type
        type_stats = {}
        for col_type in ['critical', 'important', 'optional', 'unknown']:
            type_analyses = [analysis for analysis in missing_analysis.values() 
                           if analysis['column_type'] == col_type]
            
            if type_analyses:
                type_missing = [analysis['missing_percentage'] for analysis in type_analyses]
                type_stats[col_type] = {
                    'column_count': len(type_analyses),
                    'avg_missing_percentage': round(np.mean(type_missing), 2),
                    'max_missing_percentage': round(max(type_missing), 2),
                    'columns_with_issues': sum(1 for analysis in type_analyses 
                                             if analysis['status'] in ['CRITICAL', 'WARNING'])
                }
        
        return {
            'overall': {
                'total_columns_analyzed': len(missing_analysis),
                'avg_missing_percentage': round(np.mean(missing_percentages), 2),
                'median_missing_percentage': round(np.median(missing_percentages), 2),
                'max_missing_percentage': round(max(missing_percentages), 2),
                'min_missing_percentage': round(min(missing_percentages), 2),
                'columns_with_no_missing': sum(1 for p in missing_percentages if p == 0)
            },
            'by_column_type': type_stats,
            'severity_distribution': {
                'critical_issues': sum(1 for analysis in missing_analysis.values() 
                                     if analysis['status'] == 'CRITICAL'),
                'warnings': sum(1 for analysis in missing_analysis.values() 
                              if analysis['status'] == 'WARNING'),
                'acceptable': sum(1 for analysis in missing_analysis.values() 
                                if analysis['status'] == 'OK')
            }
        }