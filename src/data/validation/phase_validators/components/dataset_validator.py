"""
Dataset Requirements Validator - Modular Components Structure

File: src/data/validation/phase_validators/components/dataset_validator.py

Validates dataset requirements for phase completion including size requirements,
data quality standards, and business readiness criteria.

This component validator is part of the modular phase validation architecture.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging

from ...config import PHASE1_REQUIREMENTS, MINIMUM_DATASET_SIZES

logger = logging.getLogger(__name__)

class DatasetRequirementsValidator:
    """
    Comprehensive dataset requirements validator for phase completion.
    
    Validates dataset presence, size requirements, data quality standards,
    and business readiness criteria for systematic phase progression.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize dataset requirements validator.
        
        Args:
            config (Dict, optional): Validation configuration
        """
        self.config = config or self._get_default_config()
        self.phase1_requirements = PHASE1_REQUIREMENTS
        self.minimum_sizes = MINIMUM_DATASET_SIZES
        
        logger.info("DatasetRequirementsValidator initialized")
    
    def _get_default_config(self) -> Dict:
        """Get default validation configuration."""
        return {
            'size_tolerance_percentage': 10,  # Allow 10% below minimum as warning
            'quality_threshold': 85.0,  # Minimum data quality score
            'critical_columns': ['sentiment', 'text'],
            'important_columns': ['productId', 'userId'],
            'optional_columns': ['summary', 'helpfulnessNumerator', 'helpfulnessDenominator'],
            'class_balance_thresholds': {
                'severe_imbalance': 90.0,  # >90% one class
                'moderate_imbalance': 80.0  # >80% one class
            }
        }
    
    def validate(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Comprehensive dataset validation for phase requirements.
        
        Args:
            datasets (Dict[str, pd.DataFrame]): Dictionary of loaded datasets
            
        Returns:
            Dict[str, Any]: Complete dataset validation results
        """
        logger.info(f"Validating dataset requirements for {len(datasets)} datasets...")
        
        validation_result = {
            'validator': 'DatasetRequirementsValidator',
            'validation_timestamp': datetime.now().isoformat(),
            'score': 0.0,
            'critical_issues': [],
            'warnings': [],
            'dataset_assessments': {},
            'overall_assessment': {},
            'business_readiness': {},
            'recommendations': []
        }
        
        try:
            # Validate individual datasets
            dataset_assessments = {}
            total_penalty = 0
            
            for dataset_name, df in datasets.items():
                assessment = self._validate_individual_dataset(dataset_name, df)
                dataset_assessments[dataset_name] = assessment
                total_penalty += assessment['penalty']
            
            validation_result['dataset_assessments'] = dataset_assessments
            
            # Validate critical dataset presence
            critical_presence = self._validate_critical_dataset_presence(datasets)
            total_penalty += critical_presence['penalty']
            
            # Validate size requirements
            size_validation = self._validate_size_requirements(datasets)
            total_penalty += size_validation['penalty']
            
            # Cross-dataset consistency validation
            consistency_validation = self._validate_cross_dataset_consistency(datasets)
            total_penalty += consistency_validation['penalty']
            
            # Calculate overall score
            overall_score = max(0, 100 - total_penalty)
            validation_result['score'] = round(overall_score, 1)
            
            # Aggregate issues and warnings
            for assessment in dataset_assessments.values():
                validation_result['critical_issues'].extend(assessment.get('critical_issues', []))
                validation_result['warnings'].extend(assessment.get('warnings', []))
            
            validation_result['critical_issues'].extend(critical_presence.get('critical_issues', []))
            validation_result['critical_issues'].extend(size_validation.get('critical_issues', []))
            validation_result['critical_issues'].extend(consistency_validation.get('critical_issues', []))
            
            validation_result['warnings'].extend(critical_presence.get('warnings', []))
            validation_result['warnings'].extend(size_validation.get('warnings', []))
            validation_result['warnings'].extend(consistency_validation.get('warnings', []))
            
            # Overall assessment
            validation_result['overall_assessment'] = {
                'total_datasets': len(datasets),
                'datasets_meeting_requirements': len([a for a in dataset_assessments.values() if a['meets_requirements']]),
                'critical_datasets_present': critical_presence['all_critical_present'],
                'size_requirements_met': size_validation['all_sizes_adequate'],
                'cross_dataset_consistency': consistency_validation['consistency_score'],
                'overall_penalty': total_penalty,
                'all_requirements_met': overall_score >= self.config['quality_threshold']
            }
            
            # Business readiness assessment
            validation_result['business_readiness'] = self._assess_business_readiness(
                datasets, dataset_assessments, overall_score
            )
            
            # Generate recommendations
            validation_result['recommendations'] = self._generate_dataset_recommendations(
                dataset_assessments, critical_presence, size_validation, consistency_validation
            )
            
            logger.info(f"Dataset validation complete. Score: {overall_score:.1f}")
            
        except Exception as e:
            logger.error(f"Dataset validation failed: {str(e)}")
            validation_result['score'] = 0
            validation_result['critical_issues'].append(f"Validation process failed: {str(e)}")
        
        return validation_result
    
    def _validate_individual_dataset(self, dataset_name: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate individual dataset requirements."""
        assessment = {
            'dataset_name': dataset_name,
            'penalty': 0,
            'critical_issues': [],
            'warnings': [],
            'metrics': {},
            'meets_requirements': True
        }
        
        # Basic structure validation
        if df.empty:
            assessment['penalty'] += 40
            assessment['critical_issues'].append(f"Dataset '{dataset_name}' is empty")
            assessment['meets_requirements'] = False
            return assessment
        
        # Column presence validation
        column_assessment = self._validate_column_presence(df, dataset_name)
        assessment['penalty'] += column_assessment['penalty']
        assessment['critical_issues'].extend(column_assessment['critical_issues'])
        assessment['warnings'].extend(column_assessment['warnings'])
        
        # Data quality validation
        quality_assessment = self._validate_data_quality(df, dataset_name)
        assessment['penalty'] += quality_assessment['penalty']
        assessment['critical_issues'].extend(quality_assessment['critical_issues'])
        assessment['warnings'].extend(quality_assessment['warnings'])
        
        # Class balance validation (for datasets with sentiment)
        if 'sentiment' in df.columns:
            balance_assessment = self._validate_class_balance(df, dataset_name)
            assessment['penalty'] += balance_assessment['penalty']
            assessment['warnings'].extend(balance_assessment['warnings'])
            assessment['metrics']['class_balance'] = balance_assessment['balance_metrics']
        
        # Text content validation
        if 'text' in df.columns:
            text_assessment = self._validate_text_content(df, dataset_name)
            assessment['penalty'] += text_assessment['penalty']
            assessment['warnings'].extend(text_assessment['warnings'])
            assessment['metrics']['text_quality'] = text_assessment['text_metrics']
        
        # Update overall assessment
        assessment['meets_requirements'] = assessment['penalty'] < 25
        assessment['metrics'].update({
            'row_count': len(df),
            'column_count': len(df.columns),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            'missing_data_percentage': round((df.isnull().sum().sum() / df.size) * 100, 2)
        })
        
        return assessment
    
    def _validate_column_presence(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Validate required column presence."""
        result = {'penalty': 0, 'critical_issues': [], 'warnings': []}
        
        # Check critical columns
        missing_critical = [col for col in self.config['critical_columns'] if col not in df.columns]
        if missing_critical:
            result['penalty'] += len(missing_critical) * 20
            result['critical_issues'].append(
                f"Dataset '{dataset_name}' missing critical columns: {missing_critical}"
            )
        
        # Check important columns
        missing_important = [col for col in self.config['important_columns'] if col not in df.columns]
        if missing_important:
            result['penalty'] += len(missing_important) * 5
            result['warnings'].append(
                f"Dataset '{dataset_name}' missing important columns: {missing_important}"
            )
        
        return result
    
    def _validate_data_quality(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Validate basic data quality metrics."""
        result = {'penalty': 0, 'critical_issues': [], 'warnings': []}
        
        # Check for excessive missing values
        missing_percentage = (df.isnull().sum().sum() / df.size) * 100
        if missing_percentage > 30:
            result['penalty'] += 15
            result['critical_issues'].append(
                f"Dataset '{dataset_name}' has {missing_percentage:.1f}% missing values"
            )
        elif missing_percentage > 15:
            result['penalty'] += 5
            result['warnings'].append(
                f"Dataset '{dataset_name}' has {missing_percentage:.1f}% missing values"
            )
        
        # Check for duplicate rows
        duplicate_percentage = (df.duplicated().sum() / len(df)) * 100
        if duplicate_percentage > 20:
            result['penalty'] += 10
            result['warnings'].append(
                f"Dataset '{dataset_name}' has {duplicate_percentage:.1f}% duplicate rows"
            )
        
        # Check critical column completeness
        for col in self.config['critical_columns']:
            if col in df.columns:
                col_missing_pct = (df[col].isnull().sum() / len(df)) * 100
                if col_missing_pct > 5:
                    result['penalty'] += 10
                    result['critical_issues'].append(
                        f"Critical column '{col}' in '{dataset_name}' has {col_missing_pct:.1f}% missing values"
                    )
        
        return result
    
    def _validate_class_balance(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Validate sentiment class balance."""
        result = {'penalty': 0, 'warnings': [], 'balance_metrics': {}}
        
        # Calculate class distribution
        sentiment_counts = df['sentiment'].value_counts()
        total_samples = len(df['sentiment'].dropna())
        
        if total_samples == 0:
            result['warnings'].append(f"No valid sentiment values in '{dataset_name}'")
            return result
        
        # Calculate class percentages
        class_percentages = (sentiment_counts / total_samples * 100).to_dict()
        max_class_percentage = max(class_percentages.values()) if class_percentages else 0
        
        result['balance_metrics'] = {
            'class_distribution': class_percentages,
            'max_class_percentage': max_class_percentage,
            'total_samples': total_samples
        }
        
        # Check for severe imbalance
        if max_class_percentage > self.config['class_balance_thresholds']['severe_imbalance']:
            result['penalty'] += 8
            result['warnings'].append(
                f"Severe class imbalance in '{dataset_name}': {max_class_percentage:.1f}% in dominant class"
            )
        elif max_class_percentage > self.config['class_balance_thresholds']['moderate_imbalance']:
            result['penalty'] += 3
            result['warnings'].append(
                f"Moderate class imbalance in '{dataset_name}': {max_class_percentage:.1f}% in dominant class"
            )
        
        return result
    
    def _validate_text_content(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Validate text content quality."""
        result = {'penalty': 0, 'warnings': [], 'text_metrics': {}}
        
        text_series = df['text'].dropna()
        if text_series.empty:
            result['warnings'].append(f"No valid text content in '{dataset_name}'")
            return result
        
        # Calculate text statistics
        word_counts = text_series.apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
        empty_texts = (text_series.str.strip() == '').sum()
        very_short_texts = (word_counts < 3).sum()
        
        empty_percentage = (empty_texts / len(text_series)) * 100
        very_short_percentage = (very_short_texts / len(text_series)) * 100
        
        result['text_metrics'] = {
            'avg_word_count': round(word_counts.mean(), 2),
            'median_word_count': int(word_counts.median()),
            'empty_text_percentage': round(empty_percentage, 2),
            'very_short_text_percentage': round(very_short_percentage, 2),
            'text_length_std': round(word_counts.std(), 2),
            'uniqueness_ratio': round(text_series.nunique() / len(text_series), 3)
        }
        
        # Validate text quality
        if empty_percentage > 10:
            result['penalty'] += 10
            result['warnings'].append(
                f"High empty text percentage in '{dataset_name}': {empty_percentage:.1f}%"
            )
        
        if very_short_percentage > 25:
            result['penalty'] += 5
            result['warnings'].append(
                f"High very short text percentage in '{dataset_name}': {very_short_percentage:.1f}%"
            )
        
        return result
    
    def _validate_critical_dataset_presence(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Validate presence of critical datasets."""
        result = {'penalty': 0, 'critical_issues': [], 'warnings': []}
        
        critical_datasets = self.phase1_requirements['critical_datasets']
        missing_critical = set(critical_datasets) - set(datasets.keys())
        
        if missing_critical:
            result['penalty'] += len(missing_critical) * 25
            result['critical_issues'].append(f"Missing critical datasets: {list(missing_critical)}")
        
        result['all_critical_present'] = len(missing_critical) == 0
        return result
    
    def _validate_size_requirements(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Validate dataset size requirements."""
        result = {'penalty': 0, 'critical_issues': [], 'warnings': []}
        
        sizes_adequate = []
        for dataset_name, min_size in self.minimum_sizes.items():
            if dataset_name in datasets:
                actual_size = len(datasets[dataset_name])
                tolerance = min_size * (self.config['size_tolerance_percentage'] / 100)
                
                if actual_size < min_size:
                    result['penalty'] += 20
                    result['critical_issues'].append(
                        f"Dataset '{dataset_name}' too small: {actual_size} < {min_size} required"
                    )
                    sizes_adequate.append(False)
                elif actual_size < min_size + tolerance:
                    result['penalty'] += 5
                    result['warnings'].append(
                        f"Dataset '{dataset_name}' close to minimum size: {actual_size} vs {min_size} required"
                    )
                    sizes_adequate.append(True)
                else:
                    sizes_adequate.append(True)
        
        result['all_sizes_adequate'] = all(sizes_adequate) if sizes_adequate else False
        return result
    
    def _validate_cross_dataset_consistency(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Validate consistency across datasets."""
        result = {'penalty': 0, 'critical_issues': [], 'warnings': [], 'consistency_score': 100}
        
        if len(datasets) < 2:
            return result
        
        # Check schema consistency
        schemas = {name: set(df.columns) for name, df in datasets.items()}
        
        if len(set(frozenset(schema) for schema in schemas.values())) > 1:
            all_columns = set.union(*schemas.values())
            common_columns = set.intersection(*schemas.values())
            
            consistency_percentage = (len(common_columns) / len(all_columns)) * 100
            result['consistency_score'] = round(consistency_percentage, 1)
            
            if consistency_percentage < 80:
                result['penalty'] += 10
                result['warnings'].append(
                    f"Schema inconsistency across datasets: only {consistency_percentage:.1f}% columns common"
                )
        
        return result
    
    def _assess_business_readiness(self, datasets: Dict[str, pd.DataFrame], 
                                 dataset_assessments: Dict[str, Any], 
                                 overall_score: float) -> Dict[str, Any]:
        """Assess business readiness based on dataset validation."""
        total_reviews = sum(len(df) for df in datasets.values())
        datasets_meeting_requirements = sum(1 for a in dataset_assessments.values() if a['meets_requirements'])
        
        # Calculate data coverage
        train_coverage = len(datasets.get('train', pd.DataFrame())) / self.minimum_sizes.get('train', 1)
        
        # Assess sentiment coverage
        sentiment_coverage = 0
        if 'train' in datasets and 'sentiment' in datasets['train'].columns:
            sentiment_coverage = (datasets['train']['sentiment'].notna().sum() / len(datasets['train'])) * 100
        
        return {
            'total_review_volume': total_reviews,
            'datasets_meeting_requirements': datasets_meeting_requirements,
            'total_datasets': len(datasets),
            'data_coverage_score': round(min(100, train_coverage * 100), 1),
            'sentiment_coverage_percentage': round(sentiment_coverage, 1),
            'ready_for_ml_development': (
                overall_score >= self.config['quality_threshold'] and
                datasets_meeting_requirements >= len(datasets) * 0.8
            ),
            'business_value_indicators': {
                'sufficient_volume': total_reviews >= 5000,
                'diverse_coverage': len(datasets) >= 3,
                'quality_adequate': overall_score >= self.config['quality_threshold']
            }
        }
    
    def _generate_dataset_recommendations(self, dataset_assessments: Dict[str, Any],
                                        critical_presence: Dict[str, Any],
                                        size_validation: Dict[str, Any],
                                        consistency_validation: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations for dataset improvements."""
        recommendations = []
        
        # Critical dataset issues
        if not critical_presence['all_critical_present']:
            recommendations.append("URGENT: Obtain missing critical datasets before proceeding to Phase 2")
        
        # Size requirement issues
        if not size_validation['all_sizes_adequate']:
            recommendations.append("Collect additional data to meet minimum size requirements")
        
        # Individual dataset issues
        failing_datasets = [name for name, assessment in dataset_assessments.items() 
                           if not assessment['meets_requirements']]
        if failing_datasets:
            recommendations.append(f"Address data quality issues in: {', '.join(failing_datasets)}")
        
        # Schema consistency issues
        if consistency_validation['consistency_score'] < 90:
            recommendations.append("Standardize column schemas across all datasets")
        
        # Success case
        if not recommendations:
            recommendations.append("Dataset requirements met - ready for Phase 2 ML development")
        
        return recommendations[:5]