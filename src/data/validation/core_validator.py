"""
Core Data Quality Validator

Main orchestrator for comprehensive data quality validation combining
structure validation, quality checks, and business rules validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging

from .config import get_default_validation_config, QUALITY_THRESHOLDS
from .structure_validator import DatasetStructureValidator
from .quality_checks import (
    run_all_quality_checks, MissingValueChecker, TextQualityChecker,
    SentimentValidator, DataConsistencyChecker
)

logger = logging.getLogger(__name__)

class DataQualityValidator:
    """
    Comprehensive data quality validator orchestrating all validation components.
    
    Combines structure validation, quality checks, and business rules validation
    into a unified framework with detailed reporting and scoring.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize data quality validator with configuration.
        
        Args:
            config (Dict, optional): Validation configuration
        """
        self.config = config or get_default_validation_config()
        self.validation_results = {}
        self.overall_score = 0.0
        
        # Initialize component validators
        self.structure_validator = DatasetStructureValidator(self.config)
        
        logger.info("DataQualityValidator initialized with integrated framework")
    
    def validate_dataset_quality(self, df: pd.DataFrame, dataset_name: str,
                                validation_scope: str = 'comprehensive') -> Dict[str, Any]:
        """
        Perform comprehensive dataset quality validation.
        
        Args:
            df (pd.DataFrame): Dataset to validate
            dataset_name (str): Name of dataset for reporting
            validation_scope (str): Scope of validation ('basic', 'standard', 'comprehensive')
            
        Returns:
            Dict[str, Any]: Complete validation results
        """
        logger.info(f"Starting comprehensive validation for {dataset_name} ({len(df)} rows)")
        
        validation_result = {
            'dataset_name': dataset_name,
            'validation_timestamp': datetime.now().isoformat(),
            'validation_scope': validation_scope,
            'dataset_info': {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2)
            },
            'validations': {},
            'overall_assessment': {},
            'recommendations': [],
            'validation_summary': {}
        }
        
        try:
            # Phase 1: Structure Validation (always performed)
            logger.info("Phase 1: Structure validation...")
            structure_results = self.structure_validator.generate_structure_summary(df, dataset_name)
            validation_result['validations']['structure'] = structure_results
            
            # Phase 2: Quality Checks (based on scope)
            if validation_scope in ['standard', 'comprehensive']:
                logger.info("Phase 2: Quality checks...")
                quality_results = self._run_quality_checks(df, validation_scope)
                validation_result['validations']['quality_checks'] = quality_results
            
            # Phase 3: Business Rules (comprehensive only)
            if validation_scope == 'comprehensive':
                logger.info("Phase 3: Business rules validation...")
                business_results = self._run_business_validation(df)
                validation_result['validations']['business_rules'] = business_results
            
            # Calculate overall assessment
            overall_assessment = self._calculate_overall_assessment(validation_result['validations'])
            validation_result['overall_assessment'] = overall_assessment
            
            # Generate consolidated recommendations
            recommendations = self._generate_consolidated_recommendations(validation_result['validations'])
            validation_result['recommendations'] = recommendations
            
            # Create validation summary
            validation_summary = self._create_validation_summary(validation_result)
            validation_result['validation_summary'] = validation_summary
            
            # Store results
            self.validation_results[dataset_name] = validation_result
            self.overall_score = overall_assessment['overall_score']
            
            logger.info(f"Validation complete for {dataset_name}. Overall score: {self.overall_score:.1f}")
            
        except Exception as e:
            logger.error(f"Validation failed for {dataset_name}: {str(e)}")
            validation_result['error'] = str(e)
            validation_result['overall_assessment'] = {
                'overall_score': 0.0,
                'status': 'FAILED',
                'critical_issues': [f"Validation process failed: {str(e)}"]
            }
        
        return validation_result
    
    def validate_multiple_datasets(self, datasets: Dict[str, pd.DataFrame],
                                 validation_scope: str = 'comprehensive') -> Dict[str, Any]:
        """
        Validate multiple datasets and provide cross-dataset analysis.
        
        Args:
            datasets (Dict[str, pd.DataFrame]): Dictionary of datasets to validate
            validation_scope (str): Scope of validation
            
        Returns:
            Dict[str, Any]: Multi-dataset validation results
        """
        logger.info(f"Starting multi-dataset validation for {len(datasets)} datasets")
        
        multi_results = {
            'validation_timestamp': datetime.now().isoformat(),
            'validation_scope': validation_scope,
            'datasets_validated': len(datasets),
            'individual_results': {},
            'cross_dataset_analysis': {},
            'consolidated_assessment': {}
        }
        
        # Validate each dataset individually
        for dataset_name, df in datasets.items():
            try:
                individual_result = self.validate_dataset_quality(df, dataset_name, validation_scope)
                multi_results['individual_results'][dataset_name] = individual_result
            except Exception as e:
                logger.error(f"Failed to validate {dataset_name}: {str(e)}")
                multi_results['individual_results'][dataset_name] = {
                    'error': str(e),
                    'overall_assessment': {'overall_score': 0.0, 'status': 'FAILED'}
                }
        
        # Perform cross-dataset analysis
        if len(multi_results['individual_results']) > 1:
            cross_analysis = self._perform_cross_dataset_analysis(datasets, multi_results['individual_results'])
            multi_results['cross_dataset_analysis'] = cross_analysis
        
        # Create consolidated assessment
        consolidated = self._create_consolidated_assessment(multi_results['individual_results'])
        multi_results['consolidated_assessment'] = consolidated
        
        return multi_results
    
    def get_validation_status(self, dataset_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current validation status for a dataset or overall.
        
        Args:
            dataset_name (str, optional): Specific dataset name
            
        Returns:
            Dict[str, Any]: Validation status information
        """
        if dataset_name and dataset_name in self.validation_results:
            result = self.validation_results[dataset_name]
            return {
                'dataset_name': dataset_name,
                'overall_score': result['overall_assessment']['overall_score'],
                'status': result['overall_assessment']['status'],
                'last_validated': result['validation_timestamp'],
                'critical_issues_count': len(result['overall_assessment'].get('critical_issues', [])),
                'warnings_count': len(result['overall_assessment'].get('warnings', []))
            }
        else:
            # Return overall status across all validated datasets
            if not self.validation_results:
                return {'status': 'NO_VALIDATIONS_RUN', 'datasets_validated': 0}
            
            scores = [r['overall_assessment']['overall_score'] for r in self.validation_results.values()]
            avg_score = np.mean(scores)
            
            return {
                'datasets_validated': len(self.validation_results),
                'average_score': round(avg_score, 1),
                'status': self._get_status_from_score(avg_score),
                'score_range': {'min': round(min(scores), 1), 'max': round(max(scores), 1)},
                'last_validation': max(r['validation_timestamp'] for r in self.validation_results.values())
            }
    
    def _run_quality_checks(self, df: pd.DataFrame, validation_scope: str) -> Dict[str, Any]:
        """Run quality checks based on validation scope."""
        if validation_scope == 'standard':
            # Run essential quality checks only
            checkers_to_run = ['missing_values', 'sentiment_validation']
        else:  # comprehensive
            # Run all quality checks
            checkers_to_run = None  # None means all checkers
        
        # Run quality checks
        quality_results = run_all_quality_checks(df, self.config)
        
        # Filter results if needed
        if checkers_to_run:
            filtered_results = {}
            checker_name_map = {
                'missing_values': 'MissingValueChecker',
                'text_quality': 'TextQualityChecker',
                'sentiment_validation': 'SentimentValidator',
                'data_consistency': 'DataConsistencyChecker'
            }
            
            for check_type in checkers_to_run:
                checker_name = checker_name_map.get(check_type, check_type)
                if checker_name in quality_results:
                    filtered_results[checker_name] = quality_results[checker_name]
            
            quality_results = filtered_results
        
        # Calculate aggregate quality score
        if quality_results:
            scores = [result.get('score', 0) for result in quality_results.values() if 'score' in result]
            aggregate_score = np.mean(scores) if scores else 0
        else:
            aggregate_score = 0
        
        return {
            'individual_checks': quality_results,
            'aggregate_score': round(aggregate_score, 1),
            'checks_run': len(quality_results),
            'checks_passed': sum(1 for r in quality_results.values() if r.get('score', 0) >= 75)
        }
    
    def _run_business_validation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run business-specific validation rules."""
        # Placeholder for business validation - would integrate business rules checker
        business_results = {
            'data_completeness': self._check_data_completeness(df),
            'business_logic': self._check_basic_business_logic(df),
            'aggregate_score': 85.0  # Placeholder
        }
        
        return business_results
    
    def _check_data_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data completeness for business requirements."""
        required_columns = ['sentiment', 'text']
        missing_required = [col for col in required_columns if col not in df.columns]
        
        completeness_score = 100.0
        if missing_required:
            completeness_score -= len(missing_required) * 30
        
        # Check data coverage
        if 'text' in df.columns:
            non_empty_text = df['text'].notna().sum()
            text_coverage = (non_empty_text / len(df)) * 100 if len(df) > 0 else 0
            if text_coverage < 95:
                completeness_score -= (95 - text_coverage)
        
        return {
            'score': max(0, completeness_score),
            'missing_required_columns': missing_required,
            'text_coverage_percentage': text_coverage if 'text' in df.columns else 0,
            'passed': len(missing_required) == 0 and completeness_score >= 75
        }
    
    def _check_basic_business_logic(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check basic business logic constraints."""
        issues = []
        score = 100.0
        
        # Check sentiment-text consistency
        if 'sentiment' in df.columns and 'text' in df.columns:
            empty_text_with_sentiment = ((df['text'].isnull() | (df['text'] == '')) & 
                                        df['sentiment'].notna()).sum()
            if empty_text_with_sentiment > 0:
                score -= 15
                issues.append(f"{empty_text_with_sentiment} rows have sentiment without text")
        
        # Check helpfulness logic
        if 'helpfulnessNumerator' in df.columns and 'helpfulnessDenominator' in df.columns:
            invalid_helpfulness = (df['helpfulnessNumerator'] > df['helpfulnessDenominator']).sum()
            if invalid_helpfulness > 0:
                score -= 10
                issues.append(f"{invalid_helpfulness} rows have invalid helpfulness ratios")
        
        return {
            'score': max(0, score),
            'issues': issues,
            'passed': len(issues) == 0
        }
    
    def _calculate_overall_assessment(self, validations: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall assessment from all validation results."""
        scores = []
        all_issues = []
        all_warnings = []
        
        # Collect scores and issues from structure validation
        if 'structure' in validations:
            structure = validations['structure']
            if 'structure_score' in structure:
                scores.append(structure['structure_score'])
            all_issues.extend(structure.get('all_issues', []))
            all_warnings.extend(structure.get('all_warnings', []))
        
        # Collect scores and issues from quality checks
        if 'quality_checks' in validations:
            quality = validations['quality_checks']
            if 'aggregate_score' in quality:
                scores.append(quality['aggregate_score'])
            
            for check_result in quality.get('individual_checks', {}).values():
                all_issues.extend(check_result.get('critical_issues', []))
                all_warnings.extend(check_result.get('warnings', []))
        
        # Collect scores from business validation
        if 'business_rules' in validations:
            business = validations['business_rules']
            if 'aggregate_score' in business:
                scores.append(business['aggregate_score'])
        
        # Calculate overall score
        overall_score = np.mean(scores) if scores else 0
        status = self._get_status_from_score(overall_score)
        
        return {
            'overall_score': round(overall_score, 1),
            'status': status,
            'component_scores': {
                'structure': validations.get('structure', {}).get('structure_score', 0),
                'quality_checks': validations.get('quality_checks', {}).get('aggregate_score', 0),
                'business_rules': validations.get('business_rules', {}).get('aggregate_score', 0)
            },
            'critical_issues': all_issues[:10],  # Limit to top 10
            'warnings': all_warnings[:10],  # Limit to top 10
            'total_issues_count': len(all_issues),
            'total_warnings_count': len(all_warnings)
        }
    
    def _generate_consolidated_recommendations(self, validations: Dict[str, Any]) -> List[str]:
        """Generate consolidated recommendations from all validation components."""
        all_recommendations = []
        
        # Collect recommendations from structure validation
        if 'structure' in validations:
            all_recommendations.extend(validations['structure'].get('recommendations', []))
        
        # Collect recommendations from quality checks
        if 'quality_checks' in validations:
            for check_result in validations['quality_checks'].get('individual_checks', {}).values():
                all_recommendations.extend(check_result.get('recommendations', []))
        
        # Collect recommendations from business validation
        if 'business_rules' in validations:
            # Add business-specific recommendations
            completeness = validations['business_rules'].get('data_completeness', {})
            if not completeness.get('passed', True):
                all_recommendations.append("Address data completeness issues for business requirements")
            
            business_logic = validations['business_rules'].get('business_logic', {})
            if not business_logic.get('passed', True):
                all_recommendations.append("Fix business logic violations")
        
        # Remove duplicates while preserving order
        unique_recommendations = []
        seen = set()
        for rec in all_recommendations:
            if rec not in seen:
                unique_recommendations.append(rec)
                seen.add(rec)
        
        return unique_recommendations[:8]  # Limit to top 8 recommendations
    
    def _create_validation_summary(self, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create concise validation summary."""
        overall = validation_result['overall_assessment']
        
        return {
            'dataset_name': validation_result['dataset_name'],
            'overall_score': overall['overall_score'],
            'status': overall['status'],
            'grade': self._get_grade_from_score(overall['overall_score']),
            'validation_scope': validation_result['validation_scope'],
            'key_metrics': {
                'total_issues': overall['total_issues_count'],
                'total_warnings': overall['total_warnings_count'],
                'validations_run': len(validation_result['validations']),
                'data_size': f"{validation_result['dataset_info']['rows']:,} rows"
            },
            'top_recommendations': validation_result['recommendations'][:3],
            'validation_timestamp': validation_result['validation_timestamp']
        }
    
    def _perform_cross_dataset_analysis(self, datasets: Dict[str, pd.DataFrame],
                                      individual_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform analysis across multiple datasets."""
        cross_analysis = {
            'consistency_checks': {},
            'comparative_metrics': {},
            'recommendations': []
        }
        
        # Check schema consistency across datasets
        schemas = {}
        for name, df in datasets.items():
            schemas[name] = set(df.columns)
        
        if len(schemas) > 1:
            common_columns = set.intersection(*schemas.values())
            all_columns = set.union(*schemas.values())
            
            cross_analysis['consistency_checks']['schema'] = {
                'common_columns': list(common_columns),
                'total_unique_columns': len(all_columns),
                'schema_consistency_score': round((len(common_columns) / len(all_columns)) * 100, 1)
            }
            
            if len(common_columns) / len(all_columns) < 0.8:
                cross_analysis['recommendations'].append(
                    "Schema inconsistency detected across datasets - review column naming"
                )
        
        # Compare quality scores
        scores = {}
        for name, result in individual_results.items():
            if 'overall_assessment' in result:
                scores[name] = result['overall_assessment']['overall_score']
        
        if scores:
            cross_analysis['comparative_metrics'] = {
                'score_range': {'min': min(scores.values()), 'max': max(scores.values())},
                'score_variance': round(np.var(list(scores.values())), 2),
                'datasets_by_score': dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
            }
        
        return cross_analysis
    
    def _create_consolidated_assessment(self, individual_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create consolidated assessment across all datasets."""
        valid_results = [r for r in individual_results.values() if 'overall_assessment' in r]
        
        if not valid_results:
            return {'status': 'NO_VALID_RESULTS', 'overall_score': 0}
        
        scores = [r['overall_assessment']['overall_score'] for r in valid_results]
        avg_score = np.mean(scores)
        
        total_issues = sum(r['overall_assessment'].get('total_issues_count', 0) for r in valid_results)
        total_warnings = sum(r['overall_assessment'].get('total_warnings_count', 0) for r in valid_results)
        
        return {
            'overall_score': round(avg_score, 1),
            'status': self._get_status_from_score(avg_score),
            'datasets_analyzed': len(valid_results),
            'score_statistics': {
                'min': round(min(scores), 1),
                'max': round(max(scores), 1),
                'std': round(np.std(scores), 2)
            },
            'total_issues': total_issues,
            'total_warnings': total_warnings,
            'datasets_passing': sum(1 for score in scores if score >= 75),
            'datasets_failing': sum(1 for score in scores if score < 60)
        }
    
    def _get_status_from_score(self, score: float) -> str:
        """Get status label from score."""
        thresholds = QUALITY_THRESHOLDS
        
        if score >= thresholds['excellent']:
            return 'EXCELLENT'
        elif score >= thresholds['good']:
            return 'GOOD'
        elif score >= thresholds['acceptable']:
            return 'ACCEPTABLE'
        elif score >= thresholds['poor']:
            return 'POOR'
        else:
            return 'FAILING'
    
    def _get_grade_from_score(self, score: float) -> str:
        """Get letter grade from score."""
        if score >= 95:
            return 'A+'
        elif score >= 90:
            return 'A'
        elif score >= 85:
            return 'B+'
        elif score >= 80:
            return 'B'
        elif score >= 75:
            return 'C+'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'


# Convenience functions for easy usage
def validate_dataset_quality(df: pd.DataFrame, dataset_name: str = "dataset",
                           config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Convenience function for validating a single dataset.
    
    Args:
        df (pd.DataFrame): Dataset to validate
        dataset_name (str): Name of dataset
        config (Dict, optional): Validation configuration
        
    Returns:
        Dict[str, Any]: Validation results
    """
    validator = DataQualityValidator(config)
    return validator.validate_dataset_quality(df, dataset_name)

def validate_multiple_datasets(datasets: Dict[str, pd.DataFrame],
                             config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Convenience function for validating multiple datasets.
    
    Args:
        datasets (Dict[str, pd.DataFrame]): Datasets to validate
        config (Dict, optional): Validation configuration
        
    Returns:
        Dict[str, Any]: Multi-dataset validation results
    """
    validator = DataQualityValidator(config)
    return validator.validate_multiple_datasets(datasets)