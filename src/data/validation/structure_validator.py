"""
Dataset Structure Validation Module

Validates dataset structure, schema compliance, and basic data characteristics
for the Intelligent Review Analytics Platform validation framework.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from .config import (
    CRITICAL_COLUMNS, IMPORTANT_COLUMNS, OPTIONAL_COLUMNS,
    CONSISTENCY_RULES, get_default_validation_config
)

logger = logging.getLogger(__name__)

class DatasetStructureValidator:
    """
    Validates dataset structure, schema compliance, and basic characteristics.
    
    Focuses on structural integrity, column presence, data types, and
    basic dataset properties without diving into content quality.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize structure validator with configuration.
        
        Args:
            config (Dict, optional): Validation configuration
        """
        self.config = config or get_default_validation_config()
        self.expected_types = self.config['consistency_rules']['data_types']['expected_types']
        
    def validate_basic_structure(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Validate basic dataset structure and properties.
        
        Args:
            df (pd.DataFrame): Dataset to validate
            dataset_name (str): Name of dataset for reporting
            
        Returns:
            Dict[str, Any]: Basic structure validation results
        """
        logger.info(f"Validating basic structure for {dataset_name}...")
        
        validation_result = {
            'dataset_name': dataset_name,
            'validation_timestamp': datetime.now().isoformat(),
            'basic_properties': {},
            'issues': [],
            'warnings': [],
            'is_valid': True
        }
        
        # Basic properties
        basic_properties = {
            'is_empty': len(df) == 0,
            'row_count': len(df),
            'column_count': len(df.columns),
            'has_columns': len(df.columns) > 0,
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': round((df.duplicated().sum() / len(df)) * 100, 2) if len(df) > 0 else 0
        }
        
        # Check for critical structural issues
        if basic_properties['is_empty']:
            validation_result['issues'].append("Dataset is completely empty")
            validation_result['is_valid'] = False
        
        if not basic_properties['has_columns']:
            validation_result['issues'].append("Dataset has no columns")
            validation_result['is_valid'] = False
        
        if basic_properties['duplicate_percentage'] > 10:
            validation_result['warnings'].append(
                f"High duplicate row percentage: {basic_properties['duplicate_percentage']:.1f}%"
            )
        
        # Memory usage warning for large datasets
        if basic_properties['memory_usage_mb'] > 1000:  # > 1GB
            validation_result['warnings'].append(
                f"Large memory usage: {basic_properties['memory_usage_mb']:.1f} MB"
            )
        
        validation_result['basic_properties'] = basic_properties
        
        return validation_result
    
    def validate_column_presence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate presence of required and optional columns.
        
        Args:
            df (pd.DataFrame): Dataset to validate
            
        Returns:
            Dict[str, Any]: Column presence validation results
        """
        available_columns = set(df.columns)
        
        # Check required columns
        missing_critical = [col for col in CRITICAL_COLUMNS if col not in available_columns]
        missing_important = [col for col in IMPORTANT_COLUMNS if col not in available_columns]
        present_optional = [col for col in OPTIONAL_COLUMNS if col in available_columns]
        
        # Check for unexpected columns
        all_expected = set(CRITICAL_COLUMNS + IMPORTANT_COLUMNS + OPTIONAL_COLUMNS)
        unexpected_columns = available_columns - all_expected
        
        column_coverage = {
            'critical_columns': {
                'required': CRITICAL_COLUMNS,
                'present': [col for col in CRITICAL_COLUMNS if col in available_columns],
                'missing': missing_critical,
                'coverage_percentage': round(
                    (len(CRITICAL_COLUMNS) - len(missing_critical)) / len(CRITICAL_COLUMNS) * 100, 1
                ) if CRITICAL_COLUMNS else 100
            },
            'important_columns': {
                'expected': IMPORTANT_COLUMNS,
                'present': [col for col in IMPORTANT_COLUMNS if col in available_columns],
                'missing': missing_important,
                'coverage_percentage': round(
                    (len(IMPORTANT_COLUMNS) - len(missing_important)) / len(IMPORTANT_COLUMNS) * 100, 1
                ) if IMPORTANT_COLUMNS else 100
            },
            'optional_columns': {
                'possible': OPTIONAL_COLUMNS,
                'present': present_optional,
                'coverage_percentage': round(
                    len(present_optional) / len(OPTIONAL_COLUMNS) * 100, 1
                ) if OPTIONAL_COLUMNS else 100
            },
            'unexpected_columns': list(unexpected_columns)
        }
        
        # Determine validation status
        is_valid = len(missing_critical) == 0
        issues = []
        warnings = []
        
        if missing_critical:
            issues.append(f"Missing critical columns: {missing_critical}")
        
        if missing_important:
            warnings.append(f"Missing important columns: {missing_important}")
        
        if unexpected_columns:
            warnings.append(f"Unexpected columns found: {list(unexpected_columns)}")
        
        return {
            'column_coverage': column_coverage,
            'is_valid': is_valid,
            'issues': issues,
            'warnings': warnings
        }
    
    def validate_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data types against expected schema.
        
        Args:
            df (pd.DataFrame): Dataset to validate
            
        Returns:
            Dict[str, Any]: Data type validation results
        """
        type_compliance = {}
        issues = []
        warnings = []
        
        for column in df.columns:
            actual_type = str(df[column].dtype)
            
            if column in self.expected_types:
                expected_types = self.expected_types[column]
                is_compliant = actual_type in expected_types
                
                type_compliance[column] = {
                    'actual_type': actual_type,
                    'expected_types': expected_types,
                    'is_compliant': is_compliant,
                    'recommendation': self._get_type_recommendation(column, actual_type, expected_types)
                }
                
                if not is_compliant:
                    if column in CRITICAL_COLUMNS:
                        issues.append(
                            f"Critical column '{column}' has incorrect type '{actual_type}', "
                            f"expected one of {expected_types}"
                        )
                    else:
                        warnings.append(
                            f"Column '{column}' has type '{actual_type}', "
                            f"expected one of {expected_types}"
                        )
            else:
                # Unknown column - document but don't penalize
                type_compliance[column] = {
                    'actual_type': actual_type,
                    'expected_types': ['unknown'],
                    'is_compliant': True,
                    'recommendation': 'Column not in expected schema'
                }
        
        # Calculate overall compliance rate
        compliant_columns = sum(1 for col_info in type_compliance.values() if col_info['is_compliant'])
        compliance_rate = round(compliant_columns / len(type_compliance) * 100, 1) if type_compliance else 100
        
        return {
            'type_compliance': type_compliance,
            'compliance_rate': compliance_rate,
            'compliant_columns': compliant_columns,
            'total_columns': len(type_compliance),
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }
    
    def validate_dataset_size_requirements(self, df: pd.DataFrame, 
                                         dataset_name: str) -> Dict[str, Any]:
        """
        Validate dataset meets minimum size requirements.
        
        Args:
            df (pd.DataFrame): Dataset to validate
            dataset_name (str): Name of dataset (train/validation/test)
            
        Returns:
            Dict[str, Any]: Size requirement validation results
        """
        from .config import MINIMUM_DATASET_SIZES
        
        actual_size = len(df)
        required_size = MINIMUM_DATASET_SIZES.get(dataset_name, 0)
        
        meets_requirement = actual_size >= required_size
        size_ratio = actual_size / required_size if required_size > 0 else float('inf')
        
        issues = []
        warnings = []
        
        if not meets_requirement:
            if required_size > 0:
                issues.append(
                    f"Dataset '{dataset_name}' has {actual_size} rows, "
                    f"minimum required: {required_size}"
                )
        elif size_ratio < 1.2:  # Less than 20% above minimum
            warnings.append(
                f"Dataset '{dataset_name}' is close to minimum size requirement "
                f"({actual_size} vs {required_size} required)"
            )
        
        return {
            'dataset_name': dataset_name,
            'actual_size': actual_size,
            'required_size': required_size,
            'meets_requirement': meets_requirement,
            'size_ratio': round(size_ratio, 2) if size_ratio != float('inf') else None,
            'is_valid': meets_requirement,
            'issues': issues,
            'warnings': warnings
        }
    
    def analyze_data_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze basic data distribution characteristics.
        
        Args:
            df (pd.DataFrame): Dataset to analyze
            
        Returns:
            Dict[str, Any]: Data distribution analysis
        """
        distribution_analysis = {
            'null_value_summary': {},
            'numeric_columns_summary': {},
            'categorical_columns_summary': {},
            'memory_efficiency': {}
        }
        
        # Null value analysis
        total_cells = df.size
        total_nulls = df.isnull().sum().sum()
        
        distribution_analysis['null_value_summary'] = {
            'total_null_values': int(total_nulls),
            'total_cells': int(total_cells),
            'null_percentage': round((total_nulls / total_cells) * 100, 2) if total_cells > 0 else 0,
            'columns_with_nulls': int((df.isnull().sum() > 0).sum()),
            'completely_null_columns': list(df.columns[df.isnull().all()])
        }
        
        # Numeric columns analysis
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            distribution_analysis['numeric_columns_summary'] = {
                'count': len(numeric_columns),
                'columns': list(numeric_columns),
                'basic_stats': df[numeric_columns].describe().round(2).to_dict() if len(numeric_columns) > 0 else {}
            }
        
        # Categorical columns analysis
        categorical_columns = df.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            cat_summary = {}
            for col in categorical_columns:
                cat_summary[col] = {
                    'unique_values': int(df[col].nunique()),
                    'most_common': df[col].mode().iloc[0] if not df[col].empty and not df[col].mode().empty else None,
                    'null_count': int(df[col].isnull().sum())
                }
            
            distribution_analysis['categorical_columns_summary'] = {
                'count': len(categorical_columns),
                'columns': list(categorical_columns),
                'details': cat_summary
            }
        
        # Memory efficiency analysis
        memory_usage = df.memory_usage(deep=True)
        distribution_analysis['memory_efficiency'] = {
            'total_memory_mb': round(memory_usage.sum() / 1024**2, 2),
            'avg_memory_per_row_bytes': round(memory_usage.sum() / len(df), 2) if len(df) > 0 else 0,
            'largest_column': memory_usage.idxmax(),
            'largest_column_mb': round(memory_usage.max() / 1024**2, 2)
        }
        
        return distribution_analysis
    
    def generate_structure_summary(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Generate comprehensive structure validation summary.
        
        Args:
            df (pd.DataFrame): Dataset to validate
            dataset_name (str): Name of dataset
            
        Returns:
            Dict[str, Any]: Complete structure validation summary
        """
        logger.info(f"Generating structure summary for {dataset_name}...")
        
        # Run all structure validations
        basic_structure = self.validate_basic_structure(df, dataset_name)
        column_presence = self.validate_column_presence(df)
        data_types = self.validate_data_types(df)
        size_requirements = self.validate_dataset_size_requirements(df, dataset_name)
        distribution_analysis = self.analyze_data_distribution(df)
        
        # Aggregate issues and warnings
        all_issues = []
        all_warnings = []
        
        for validation in [basic_structure, column_presence, data_types, size_requirements]:
            all_issues.extend(validation.get('issues', []))
            all_warnings.extend(validation.get('warnings', []))
        
        # Determine overall validation status
        overall_valid = all(
            validation.get('is_valid', True) 
            for validation in [basic_structure, column_presence, data_types, size_requirements]
        )
        
        # Calculate structure score
        structure_score = self._calculate_structure_score(
            basic_structure, column_presence, data_types, size_requirements
        )
        
        summary = {
            'dataset_name': dataset_name,
            'validation_timestamp': datetime.now().isoformat(),
            'overall_valid': overall_valid,
            'structure_score': structure_score,
            'validations': {
                'basic_structure': basic_structure,
                'column_presence': column_presence,
                'data_types': data_types,
                'size_requirements': size_requirements
            },
            'distribution_analysis': distribution_analysis,
            'summary_statistics': {
                'total_issues': len(all_issues),
                'total_warnings': len(all_warnings),
                'critical_issues': len([issue for issue in all_issues if 'critical' in issue.lower()]),
                'type_compliance_rate': data_types.get('compliance_rate', 0)
            },
            'all_issues': all_issues,
            'all_warnings': all_warnings,
            'recommendations': self._generate_structure_recommendations(
                basic_structure, column_presence, data_types, size_requirements
            )
        }
        
        return summary
    
    def _get_type_recommendation(self, column: str, actual_type: str, 
                               expected_types: List[str]) -> str:
        """Generate recommendation for type mismatch."""
        if actual_type in expected_types:
            return "Type is correct"
        
        # Common type conversion recommendations
        if column in ['sentiment', 'helpfulnessNumerator', 'helpfulnessDenominator']:
            return f"Convert to numeric type using pd.to_numeric(df['{column}'], errors='coerce')"
        elif column in ['text', 'summary', 'productId', 'userId']:
            return f"Convert to string type using df['{column}'].astype(str)"
        else:
            return f"Review data type - expected one of {expected_types}"
    
    def _calculate_structure_score(self, basic_structure: Dict, column_presence: Dict,
                                 data_types: Dict, size_requirements: Dict) -> float:
        """Calculate overall structure score (0-100)."""
        score = 100.0
        
        # Deduct for basic structure issues
        if not basic_structure.get('is_valid', True):
            score -= 40  # Major penalty for structural issues
        
        # Deduct for missing critical columns
        missing_critical = len(column_presence.get('column_coverage', {}).get('critical_columns', {}).get('missing', []))
        score -= missing_critical * 20  # 20 points per missing critical column
        
        # Deduct for missing important columns
        missing_important = len(column_presence.get('column_coverage', {}).get('important_columns', {}).get('missing', []))
        score -= missing_important * 5  # 5 points per missing important column
        
        # Deduct for type compliance issues
        type_compliance_rate = data_types.get('compliance_rate', 100)
        score -= (100 - type_compliance_rate) * 0.3  # Moderate penalty for type issues
        
        # Deduct for size requirement issues
        if not size_requirements.get('meets_requirement', True):
            score -= 25  # Penalty for not meeting size requirements
        
        return max(0.0, round(score, 1))
    
    def _generate_structure_recommendations(self, basic_structure: Dict, column_presence: Dict,
                                          data_types: Dict, size_requirements: Dict) -> List[str]:
        """Generate recommendations for structure improvements."""
        recommendations = []
        
        # Basic structure recommendations
        if not basic_structure.get('is_valid', True):
            if basic_structure['basic_properties'].get('is_empty', False):
                recommendations.append("Load data into the dataset - it is currently empty")
            if not basic_structure['basic_properties'].get('has_columns', True):
                recommendations.append("Ensure dataset has proper column structure")
        
        # Column presence recommendations
        missing_critical = column_presence.get('column_coverage', {}).get('critical_columns', {}).get('missing', [])
        if missing_critical:
            recommendations.append(f"Add missing critical columns: {', '.join(missing_critical)}")
        
        missing_important = column_presence.get('column_coverage', {}).get('important_columns', {}).get('missing', [])
        if missing_important:
            recommendations.append(f"Consider adding important columns: {', '.join(missing_important)}")
        
        # Data type recommendations
        type_issues = [issue for issue in data_types.get('issues', []) if 'type' in issue.lower()]
        if type_issues:
            recommendations.append("Fix data type mismatches for proper processing")
        
        # Size requirement recommendations
        if not size_requirements.get('meets_requirement', True):
            dataset_name = size_requirements.get('dataset_name', 'dataset')
            required_size = size_requirements.get('required_size', 0)
            recommendations.append(f"Increase {dataset_name} dataset size to at least {required_size} rows")
        
        # Efficiency recommendations
        if basic_structure['basic_properties'].get('memory_usage_mb', 0) > 500:
            recommendations.append("Consider optimizing memory usage for large dataset")
        
        if basic_structure['basic_properties'].get('duplicate_percentage', 0) > 5:
            recommendations.append("Review and remove duplicate rows if appropriate")
        
        return recommendations[:5]  # Limit to top 5 recommendations