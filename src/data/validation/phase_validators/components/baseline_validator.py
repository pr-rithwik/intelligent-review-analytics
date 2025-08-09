"""
Baseline Performance Validator - Modular Components Structure

File: src/data/validation/phase_validators/components/baseline_validator.py

Validates baseline model performance requirements for phase completion including
accuracy thresholds, training metrics, and model readiness for comparison benchmarks.

This component validator ensures baseline model establishment meets Phase 1
completion criteria and provides adequate benchmarks for Phase 2 progression.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging

from ...config import PHASE1_REQUIREMENTS

logger = logging.getLogger(__name__)

class BaselinePerformanceValidator:
    """
    Comprehensive baseline model performance validator for phase completion.
    
    Validates baseline model performance against phase requirements, ensures
    adequate benchmarks for advanced algorithm comparison, and assesses
    model readiness for production considerations.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize baseline performance validator.
        
        Args:
            config (Dict, optional): Validation configuration
        """
        self.config = config or self._get_default_config()
        self.performance_thresholds = PHASE1_REQUIREMENTS['baseline_performance_thresholds']
        
        logger.info("BaselinePerformanceValidator initialized")
    
    def _get_default_config(self) -> Dict:
        """Get default validation configuration."""
        return {
            'required_metrics': ['accuracy', 'f1_score', 'precision', 'recall'],
            'optional_metrics': ['auc_roc', 'training_time_seconds', 'inference_time_ms'],
            'performance_grades': {
                'excellent': 90.0,
                'good': 80.0,
                'acceptable': 70.0,
                'poor': 60.0
            },
            'cross_validation_requirements': {
                'min_cv_folds': 3,
                'max_std_threshold': 0.10,  # Max 10% std deviation across folds
                'min_consistency_score': 0.85
            },
            'training_efficiency_thresholds': {
                'max_training_time_seconds': 300,  # 5 minutes
                'max_inference_time_ms': 100,
                'ideal_training_time_seconds': 60,
                'ideal_inference_time_ms': 10
            },
            'model_reliability_checks': {
                'overfitting_threshold': 0.15,  # 15% performance drop train->val
                'min_sample_coverage': 0.95,  # 95% of samples should be processed
                'convergence_required': True
            }
        }
    
    def validate(self, baseline_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive baseline model performance validation.
        
        Args:
            baseline_metrics (Dict): Baseline model metrics and results
            
        Returns:
            Dict[str, Any]: Complete baseline performance validation results
        """
        logger.info("Validating baseline model performance requirements...")
        
        validation_result = {
            'validator': 'BaselinePerformanceValidator',
            'validation_timestamp': datetime.now().isoformat(),
            'score': 0.0,
            'critical_issues': [],
            'warnings': [],
            'performance_assessment': {},
            'benchmark_readiness': {},
            'model_reliability': {},
            'efficiency_analysis': {},
            'recommendations': []
        }
        
        try:
            if not baseline_metrics:
                validation_result['score'] = 0
                validation_result['critical_issues'].append("No baseline metrics provided")
                return validation_result
            
            # Core performance validation
            performance_assessment = self._validate_core_performance(baseline_metrics)
            total_penalty = performance_assessment['penalty']
            
            # Cross-validation analysis
            cv_assessment = self._validate_cross_validation_results(baseline_metrics)
            total_penalty += cv_assessment['penalty']
            
            # Training efficiency validation
            efficiency_assessment = self._validate_training_efficiency(baseline_metrics)
            total_penalty += efficiency_assessment['penalty']
            
            # Model reliability checks
            reliability_assessment = self._validate_model_reliability(baseline_metrics)
            total_penalty += reliability_assessment['penalty']
            
            # Benchmark readiness evaluation
            benchmark_assessment = self._assess_benchmark_readiness(baseline_metrics, total_penalty)
            
            # Calculate overall score
            overall_score = max(0, 100 - total_penalty)
            validation_result['score'] = round(overall_score, 1)
            
            # Aggregate all assessments
            validation_result['performance_assessment'] = performance_assessment
            validation_result['benchmark_readiness'] = benchmark_assessment
            validation_result['model_reliability'] = reliability_assessment
            validation_result['efficiency_analysis'] = efficiency_assessment
            
            # Aggregate issues and warnings
            for assessment in [performance_assessment, cv_assessment, efficiency_assessment, reliability_assessment]:
                validation_result['critical_issues'].extend(assessment.get('critical_issues', []))
                validation_result['warnings'].extend(assessment.get('warnings', []))
            
            # Add cross-validation specific results
            if cv_assessment.get('cv_results'):
                validation_result['cross_validation_analysis'] = cv_assessment['cv_results']
            
            # Generate recommendations
            validation_result['recommendations'] = self._generate_baseline_recommendations(
                performance_assessment, cv_assessment, efficiency_assessment, 
                reliability_assessment, overall_score
            )
            
            # Final assessment flags
            validation_result['performance_adequate'] = overall_score >= self.config['performance_grades']['acceptable']
            validation_result['ready_for_phase2'] = (
                overall_score >= self.config['performance_grades']['good'] and
                benchmark_assessment['benchmark_quality_score'] >= 80
            )
            
            logger.info(f"Baseline validation complete. Score: {overall_score:.1f}")
            
        except Exception as e:
            logger.error(f"Baseline validation failed: {str(e)}")
            validation_result['score'] = 0
            validation_result['critical_issues'].append(f"Validation process failed: {str(e)}")
        
        return validation_result
    
    def _validate_core_performance(self, baseline_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Validate core performance metrics against thresholds."""
        assessment = {
            'penalty': 0,
            'critical_issues': [],
            'warnings': [],
            'metrics_analysis': {},
            'grade': 'F'
        }
        
        # Extract core metrics from various possible locations
        core_metrics = self._extract_core_metrics(baseline_metrics)
        assessment['metrics_analysis'] = core_metrics
        
        # Validate accuracy
        accuracy = core_metrics.get('accuracy', 0)
        min_accuracy = self.performance_thresholds['min_accuracy']
        
        if accuracy < min_accuracy:
            assessment['penalty'] += 30
            assessment['critical_issues'].append(
                f"Accuracy {accuracy:.3f} below minimum threshold {min_accuracy:.3f}"
            )
        elif accuracy < min_accuracy + 0.05:  # Within 5% of threshold
            assessment['penalty'] += 10
            assessment['warnings'].append(
                f"Accuracy {accuracy:.3f} close to minimum threshold {min_accuracy:.3f}"
            )
        
        # Validate F1-score
        f1_score = core_metrics.get('f1_score', 0)
        min_f1 = self.performance_thresholds['min_f1_score']
        
        if f1_score < min_f1:
            assessment['penalty'] += 25
            assessment['critical_issues'].append(
                f"F1-score {f1_score:.3f} below minimum threshold {min_f1:.3f}"
            )
        elif f1_score < min_f1 + 0.05:
            assessment['penalty'] += 8
            assessment['warnings'].append(
                f"F1-score {f1_score:.3f} close to minimum threshold {min_f1:.3f}"
            )
        
        # Validate required metrics presence
        missing_required = [metric for metric in self.config['required_metrics'] 
                           if metric not in core_metrics or core_metrics[metric] is None]
        
        if missing_required:
            assessment['penalty'] += len(missing_required) * 10
            assessment['critical_issues'].append(f"Missing required metrics: {missing_required}")
        
        # Calculate performance grade
        avg_performance = np.mean([core_metrics.get(metric, 0) for metric in self.config['required_metrics'] if core_metrics.get(metric) is not None])
        assessment['grade'] = self._calculate_performance_grade(avg_performance * 100)
        
        # Additional metric validations
        precision = core_metrics.get('precision', 0)
        recall = core_metrics.get('recall', 0)
        
        # Check for extremely poor precision/recall
        if precision < 0.5:
            assessment['penalty'] += 15
            assessment['warnings'].append(f"Low precision: {precision:.3f}")
        
        if recall < 0.5:
            assessment['penalty'] += 15
            assessment['warnings'].append(f"Low recall: {recall:.3f}")
        
        # Check for imbalanced precision/recall
        if abs(precision - recall) > 0.3:
            assessment['penalty'] += 5
            assessment['warnings'].append(
                f"Significant precision-recall imbalance: P={precision:.3f}, R={recall:.3f}"
            )
        
        return assessment
    
    def _validate_cross_validation_results(self, baseline_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Validate cross-validation consistency and reliability."""
        assessment = {
            'penalty': 0,
            'critical_issues': [],
            'warnings': [],
            'cv_results': {}
        }
        
        # Look for cross-validation results
        cv_data = self._extract_cv_metrics(baseline_metrics)
        
        if not cv_data:
            assessment['penalty'] += 15
            assessment['warnings'].append("No cross-validation results found")
            return assessment
        
        assessment['cv_results'] = cv_data
        
        # Validate CV consistency
        for metric_name, cv_scores in cv_data.items():
            if isinstance(cv_scores, list) and len(cv_scores) >= self.config['cross_validation_requirements']['min_cv_folds']:
                cv_mean = np.mean(cv_scores)
                cv_std = np.std(cv_scores)
                cv_coefficient = cv_std / cv_mean if cv_mean > 0 else float('inf')
                
                # Check for high variance across folds
                if cv_coefficient > self.config['cross_validation_requirements']['max_std_threshold']:
                    assessment['penalty'] += 10
                    assessment['warnings'].append(
                        f"High variance in {metric_name}: std={cv_std:.3f}, mean={cv_mean:.3f}"
                    )
                
                # Check for poor consistency
                min_score = min(cv_scores)
                max_score = max(cv_scores)
                if (max_score - min_score) > 0.2:  # 20% difference between best and worst fold
                    assessment['penalty'] += 8
                    assessment['warnings'].append(
                        f"Poor consistency in {metric_name}: range [{min_score:.3f}, {max_score:.3f}]"
                    )
        
        # Check CV fold count
        fold_counts = [len(scores) for scores in cv_data.values() if isinstance(scores, list)]
        if fold_counts and min(fold_counts) < self.config['cross_validation_requirements']['min_cv_folds']:
            assessment['penalty'] += 12
            assessment['warnings'].append(
                f"Insufficient CV folds: {min(fold_counts)} < {self.config['cross_validation_requirements']['min_cv_folds']} required"
            )
        
        return assessment
    
    def _validate_training_efficiency(self, baseline_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Validate training and inference efficiency."""
        assessment = {
            'penalty': 0,
            'critical_issues': [],
            'warnings': [],
            'efficiency_metrics': {}
        }
        
        # Extract timing metrics
        training_time = self._extract_metric_value(baseline_metrics, 'training_time_seconds')
        inference_time = self._extract_metric_value(baseline_metrics, 'inference_time_ms')
        
        thresholds = self.config['training_efficiency_thresholds']
        
        # Validate training time
        if training_time:
            assessment['efficiency_metrics']['training_time_seconds'] = training_time
            
            if training_time > thresholds['max_training_time_seconds']:
                assessment['penalty'] += 15
                assessment['warnings'].append(
                    f"Training time {training_time:.1f}s exceeds maximum {thresholds['max_training_time_seconds']}s"
                )
            elif training_time > thresholds['ideal_training_time_seconds']:
                assessment['penalty'] += 5
                assessment['warnings'].append(
                    f"Training time {training_time:.1f}s above ideal {thresholds['ideal_training_time_seconds']}s"
                )
        
        # Validate inference time
        if inference_time:
            assessment['efficiency_metrics']['inference_time_ms'] = inference_time
            
            if inference_time > thresholds['max_inference_time_ms']:
                assessment['penalty'] += 10
                assessment['warnings'].append(
                    f"Inference time {inference_time:.1f}ms exceeds maximum {thresholds['max_inference_time_ms']}ms"
                )
            elif inference_time > thresholds['ideal_inference_time_ms']:
                assessment['penalty'] += 3
                assessment['warnings'].append(
                    f"Inference time {inference_time:.1f}ms above ideal {thresholds['ideal_inference_time_ms']}ms"
                )
        
        # Calculate efficiency score
        efficiency_score = 100
        if training_time:
            training_efficiency = min(100, (thresholds['ideal_training_time_seconds'] / training_time) * 100)
            efficiency_score = min(efficiency_score, training_efficiency)
        
        if inference_time:
            inference_efficiency = min(100, (thresholds['ideal_inference_time_ms'] / inference_time) * 100)
            efficiency_score = min(efficiency_score, inference_efficiency)
        
        assessment['efficiency_metrics']['overall_efficiency_score'] = round(efficiency_score, 1)
        
        return assessment
    
    def _validate_model_reliability(self, baseline_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model reliability and robustness."""
        assessment = {
            'penalty': 0,
            'critical_issues': [],
            'warnings': [],
            'reliability_metrics': {}
        }
        
        reliability_checks = self.config['model_reliability_checks']
        
        # Check for overfitting
        train_accuracy = self._extract_metric_value(baseline_metrics, 'training_accuracy', 
                                                   location_keys=['training_metrics', 'train_metrics'])
        val_accuracy = self._extract_metric_value(baseline_metrics, 'accuracy', 
                                                 location_keys=['validation_metrics', 'val_metrics', 'test_metrics'])
        
        if train_accuracy and val_accuracy:
            performance_drop = train_accuracy - val_accuracy
            assessment['reliability_metrics']['performance_drop'] = round(performance_drop, 3)
            
            if performance_drop > reliability_checks['overfitting_threshold']:
                assessment['penalty'] += 20
                assessment['critical_issues'].append(
                    f"Potential overfitting detected: {performance_drop:.3f} performance drop (train->val)"
                )
            elif performance_drop > reliability_checks['overfitting_threshold'] * 0.5:
                assessment['penalty'] += 8
                assessment['warnings'].append(
                    f"Moderate overfitting concern: {performance_drop:.3f} performance drop"
                )
        
        # Check model convergence
        convergence_info = self._extract_convergence_info(baseline_metrics)
        if convergence_info:
            assessment['reliability_metrics'].update(convergence_info)
            
            if not convergence_info.get('converged', True):
                assessment['penalty'] += 15
                assessment['warnings'].append("Model training did not converge properly")
        
        # Check sample coverage
        total_samples = self._extract_metric_value(baseline_metrics, 'total_samples')
        processed_samples = self._extract_metric_value(baseline_metrics, 'processed_samples')
        
        if total_samples and processed_samples:
            coverage_rate = processed_samples / total_samples
            assessment['reliability_metrics']['sample_coverage_rate'] = round(coverage_rate, 3)
            
            if coverage_rate < reliability_checks['min_sample_coverage']:
                assessment['penalty'] += 12
                assessment['warnings'].append(
                    f"Low sample coverage: {coverage_rate:.3f} < {reliability_checks['min_sample_coverage']} required"
                )
        
        return assessment
    
    def _assess_benchmark_readiness(self, baseline_metrics: Dict[str, Any], total_penalty: float) -> Dict[str, Any]:
        """Assess readiness to serve as benchmark for Phase 2 algorithms."""
        benchmark_score = max(0, 100 - total_penalty)
        
        # Extract key metrics for benchmark comparison
        core_metrics = self._extract_core_metrics(baseline_metrics)
        
        benchmark_assessment = {
            'benchmark_quality_score': round(benchmark_score, 1),
            'suitable_for_comparison': benchmark_score >= 70,
            'baseline_metrics_summary': {
                'accuracy': core_metrics.get('accuracy', 0),
                'f1_score': core_metrics.get('f1_score', 0),
                'training_time': self._extract_metric_value(baseline_metrics, 'training_time_seconds'),
                'inference_time': self._extract_metric_value(baseline_metrics, 'inference_time_ms')
            },
            'comparison_readiness_factors': {
                'metrics_completeness': len([m for m in self.config['required_metrics'] 
                                           if core_metrics.get(m) is not None]) / len(self.config['required_metrics']),
                'performance_stability': self._assess_performance_stability(baseline_metrics),
                'documentation_quality': self._assess_documentation_completeness(baseline_metrics)
            }
        }
        
        # Determine benchmark categories
        if benchmark_score >= 85:
            benchmark_assessment['benchmark_category'] = 'Excellent'
            benchmark_assessment['phase2_recommendation'] = 'Strong baseline - proceed with confidence'
        elif benchmark_score >= 70:
            benchmark_assessment['benchmark_category'] = 'Good'
            benchmark_assessment['phase2_recommendation'] = 'Adequate baseline - proceed with monitoring'
        elif benchmark_score >= 50:
            benchmark_assessment['benchmark_category'] = 'Marginal'
            benchmark_assessment['phase2_recommendation'] = 'Weak baseline - consider improvements'
        else:
            benchmark_assessment['benchmark_category'] = 'Inadequate'
            benchmark_assessment['phase2_recommendation'] = 'Inadequate baseline - must improve before Phase 2'
        
        return benchmark_assessment
    
    def _extract_core_metrics(self, baseline_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Extract core performance metrics from baseline results."""
        core_metrics = {}
        
        for metric in self.config['required_metrics'] + self.config['optional_metrics']:
            value = self._extract_metric_value(baseline_metrics, metric)
            if value is not None:
                core_metrics[metric] = float(value)
        
        return core_metrics
    
    def _extract_cv_metrics(self, baseline_metrics: Dict[str, Any]) -> Dict[str, List]:
        """Extract cross-validation metrics from baseline results."""
        cv_metrics = {}
        
        # Look for CV results in various locations
        possible_cv_locations = [
            baseline_metrics.get('cross_validation_results', {}),
            baseline_metrics.get('cv_results', {}),
            baseline_metrics.get('validation_results', {})
        ]
        
        for cv_data in possible_cv_locations:
            if isinstance(cv_data, dict):
                for key, value in cv_data.items():
                    if key.startswith('cv_') and key.endswith('_scores'):
                        metric_name = key[3:-7]  # Remove 'cv_' prefix and '_scores' suffix
                        if isinstance(value, list):
                            cv_metrics[metric_name] = value
        
        return cv_metrics
    
    def _extract_metric_value(self, baseline_metrics: Dict[str, Any], metric_name: str, 
                            location_keys: Optional[List[str]] = None) -> Optional[float]:
        """Extract metric value from nested baseline metrics structure."""
        # Default locations to search
        if location_keys is None:
            location_keys = [
                'validation_metrics', 'test_metrics', 'performance_summary',
                'training_metrics', 'metrics', ''
            ]
        
        # Add root level search
        locations_to_search = [baseline_metrics]
        
        # Add nested locations
        for location_key in location_keys:
            if location_key and location_key in baseline_metrics:
                nested_location = baseline_metrics[location_key]
                if isinstance(nested_location, dict):
                    locations_to_search.append(nested_location)
        
        # Search all locations
        for location in locations_to_search:
            if isinstance(location, dict) and metric_name in location:
                value = location[metric_name]
                if value is not None:
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        continue
        
        return None
    
    def _extract_convergence_info(self, baseline_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Extract model convergence information."""
        convergence_info = {}
        
        # Look for convergence indicators
        model_info = baseline_metrics.get('model_info', {})
        training_info = baseline_metrics.get('training_info', {})
        
        # Check for explicit convergence flag
        converged = (
            model_info.get('convergence_achieved') or
            training_info.get('converged') or
            baseline_metrics.get('converged')
        )
        
        if converged is not None:
            convergence_info['converged'] = bool(converged)
        
        # Look for iteration info
        iterations = (
            model_info.get('n_iter_') or
            training_info.get('iterations') or
            baseline_metrics.get('iterations')
        )
        
        if iterations is not None:
            convergence_info['iterations'] = int(iterations)
        
        return convergence_info
    
    def _assess_performance_stability(self, baseline_metrics: Dict[str, Any]) -> float:
        """Assess performance stability across validation folds."""
        cv_data = self._extract_cv_metrics(baseline_metrics)
        
        if not cv_data:
            return 0.5  # Neutral score if no CV data
        
        stability_scores = []
        for metric_name, scores in cv_data.items():
            if len(scores) > 1:
                cv_std = np.std(scores)
                cv_mean = np.mean(scores)
                coefficient_of_variation = cv_std / cv_mean if cv_mean > 0 else 1
                stability_score = max(0, 1 - coefficient_of_variation)
                stability_scores.append(stability_score)
        
        return np.mean(stability_scores) if stability_scores else 0.5
    
    def _assess_documentation_completeness(self, baseline_metrics: Dict[str, Any]) -> float:
        """Assess completeness of model documentation."""
        documentation_elements = [
            'model_info',
            'training_metrics',
            'validation_metrics',
            'hyperparameters',
            'feature_importance'
        ]
        
        present_elements = sum(1 for element in documentation_elements 
                             if element in baseline_metrics and baseline_metrics[element])
        
        return present_elements / len(documentation_elements)
    
    def _calculate_performance_grade(self, performance_percentage: float) -> str:
        """Calculate letter grade based on performance percentage."""
        grades = self.config['performance_grades']
        
        if performance_percentage >= grades['excellent']:
            return 'A'
        elif performance_percentage >= grades['good']:
            return 'B'
        elif performance_percentage >= grades['acceptable']:
            return 'C'
        elif performance_percentage >= grades['poor']:
            return 'D'
        else:
            return 'F'
    
    def _generate_baseline_recommendations(self, performance_assessment: Dict[str, Any],
                                         cv_assessment: Dict[str, Any],
                                         efficiency_assessment: Dict[str, Any],
                                         reliability_assessment: Dict[str, Any],
                                         overall_score: float) -> List[str]:
        """Generate actionable recommendations for baseline improvement."""
        recommendations = []
        
        # Performance-based recommendations
        if performance_assessment['penalty'] > 20:
            if any('accuracy' in issue.lower() for issue in performance_assessment['critical_issues']):
                recommendations.append("Improve model accuracy through better feature engineering or hyperparameter tuning")
            if any('f1' in issue.lower() for issue in performance_assessment['critical_issues']):
                recommendations.append("Address F1-score by balancing precision and recall optimization")
        
        # Cross-validation recommendations
        if cv_assessment['penalty'] > 10:
            if any('variance' in warning.lower() for warning in cv_assessment['warnings']):
                recommendations.append("Reduce model variance through regularization or ensemble methods")
            if any('consistency' in warning.lower() for warning in cv_assessment['warnings']):
                recommendations.append("Improve model consistency through better data stratification")
        
        # Efficiency recommendations
        if efficiency_assessment['penalty'] > 10:
            if any('training time' in warning.lower() for warning in efficiency_assessment['warnings']):
                recommendations.append("Optimize training efficiency through algorithm selection or data sampling")
            if any('inference time' in warning.lower() for warning in efficiency_assessment['warnings']):
                recommendations.append("Optimize inference speed for production deployment requirements")
        
        # Reliability recommendations
        if reliability_assessment['penalty'] > 15:
            if any('overfitting' in issue.lower() for issue in reliability_assessment['critical_issues']):
                recommendations.append("Address overfitting through regularization, cross-validation, or more data")
            if any('convergence' in warning.lower() for warning in reliability_assessment['warnings']):
                recommendations.append("Ensure model convergence through learning rate adjustment or more iterations")
        
        # Overall recommendations
        if overall_score >= 85:
            recommendations.append("Excellent baseline established - ready for Phase 2 ML development")
        elif overall_score >= 70:
            recommendations.append("Good baseline performance - proceed to Phase 2 with monitoring")
        elif overall_score >= 50:
            recommendations.append("Baseline performance marginal - consider improvements before Phase 2")
        else:
            recommendations.append("CRITICAL: Baseline performance inadequate - must improve before Phase 2")
        
        return recommendations[:6]  # Limit to top 6 recommendations