"""
Phase Completion Validation Module - Core Orchestrator

Validates completion criteria for each project phase and readiness for progression.
Ensures systematic handoff between Phase 1 → Phase 2 → Phase 3 with clear criteria.

This is the main orchestrator that coordinates component validators for comprehensive
phase validation and readiness assessment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
from pathlib import Path

from .config import (
    PHASE1_REQUIREMENTS, PHASE2_READINESS_CRITERIA, 
    get_default_validation_config
)
from .phase_validators.components import get_validator_by_name

logger = logging.getLogger(__name__)

class PhaseCheckpointsValidator:
    """
    Main orchestrator for phase completion validation and readiness assessment.
    
    Coordinates component validators to provide comprehensive phase validation
    with clear gates between phases and systematic progression criteria.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize phase checkpoints validator with configuration."""
        self.config = config or get_default_validation_config()
        self.phase1_requirements = self.config['phase1_requirements']
        self.phase2_criteria = self.config['phase2_readiness_criteria']
        
        # Initialize component validators (will be injected)
        self.dataset_validator = None
        self.baseline_validator = None
        self.insights_validator = None
        self.preprocessing_validator = None
        
        logger.info("PhaseCheckpointsValidator initialized")
    
    def set_component_validators(self, dataset_validator=None, baseline_validator=None,
                               insights_validator=None, preprocessing_validator=None):
        """
        Inject component validators for modular architecture.
        Can accept validator instances or load by name from component registry.
        
        Args:
            dataset_validator: Dataset requirements validator (instance or string name)
            baseline_validator: Baseline model performance validator  
            insights_validator: Business insights validator
            preprocessing_validator: Preprocessing pipeline validator
        """
        # Handle string names for dynamic loading
        if isinstance(dataset_validator, str):
            validator_class = get_validator_by_name(dataset_validator)
            dataset_validator = validator_class() if validator_class else None
        
        self.dataset_validator = dataset_validator
        self.baseline_validator = baseline_validator
        self.insights_validator = insights_validator
        self.preprocessing_validator = preprocessing_validator
        
        logger.info("Component validators configured")
    
    def validate_phase1_completion(self, datasets: Dict[str, pd.DataFrame],
                                 baseline_metrics: Dict[str, Any],
                                 business_insights: Dict[str, Any],
                                 preprocessing_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive Phase 1 completion validation.
        
        Args:
            datasets (Dict[str, pd.DataFrame]): Loaded datasets
            baseline_metrics (Dict): Baseline model performance metrics
            business_insights (Dict): Business intelligence analysis results
            preprocessing_results (Dict): Text preprocessing pipeline results
            
        Returns:
            Dict[str, Any]: Phase 1 completion validation results
        """
        logger.info("Validating Phase 1 completion criteria...")
        
        validation_result = {
            'phase': 'Phase 1: Foundation & Intelligence',
            'validation_timestamp': datetime.now().isoformat(),
            'overall_status': 'PENDING',
            'completion_score': 0.0,
            'validation_components': {},
            'critical_issues': [],
            'warnings': [],
            'recommendations': [],
            'readiness_assessment': {},
            'next_phase_requirements': self._get_phase2_requirements()
        }
        
        # Run component validations
        try:
            # Dataset validation
            if self.dataset_validator:
                dataset_validation = self.dataset_validator.validate(datasets)
            else:
                dataset_validation = self._fallback_dataset_validation(datasets)
            
            # Baseline model validation
            if self.baseline_validator:
                baseline_validation = self.baseline_validator.validate(baseline_metrics)
            else:
                baseline_validation = self._fallback_baseline_validation(baseline_metrics)
            
            # Business insights validation
            if self.insights_validator:
                insights_validation = self.insights_validator.validate(business_insights)
            else:
                insights_validation = self._fallback_insights_validation(business_insights)
            
            # Preprocessing validation
            if self.preprocessing_validator:
                preprocessing_validation = self.preprocessing_validator.validate(preprocessing_results)
            else:
                preprocessing_validation = self._fallback_preprocessing_validation(preprocessing_results)
            
            # Store component results
            validation_result['validation_components'] = {
                'dataset_requirements': dataset_validation,
                'baseline_performance': baseline_validation,
                'business_insights': insights_validation,
                'preprocessing_pipeline': preprocessing_validation
            }
            
            # Calculate overall completion score
            component_scores = [
                dataset_validation['score'],
                baseline_validation['score'],
                insights_validation['score'],
                preprocessing_validation['score']
            ]
            
            overall_score = np.mean(component_scores)
            validation_result['completion_score'] = round(overall_score, 1)
            
            # Aggregate issues and warnings
            for component in validation_result['validation_components'].values():
                validation_result['critical_issues'].extend(component.get('critical_issues', []))
                validation_result['warnings'].extend(component.get('warnings', []))
            
            # Determine overall status
            validation_result['overall_status'] = self._determine_phase_status(overall_score)
            
            # Generate readiness assessment for Phase 2
            readiness_assessment = self._assess_phase2_readiness(validation_result['validation_components'])
            validation_result['readiness_assessment'] = readiness_assessment
            
            # Generate recommendations
            validation_result['recommendations'] = self._generate_phase1_recommendations(
                validation_result['validation_components'], overall_score
            )
            
            logger.info(f"Phase 1 validation complete. Status: {validation_result['overall_status']}, Score: {overall_score:.1f}")
            
        except Exception as e:
            logger.error(f"Phase 1 validation failed: {str(e)}")
            validation_result['overall_status'] = 'VALIDATION_ERROR'
            validation_result['critical_issues'].append(f"Validation process failed: {str(e)}")
        
        return validation_result
    
    def validate_phase2_readiness(self, phase1_results: Dict[str, Any],
                                feature_engineering_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate readiness to begin Phase 2 ML development.
        
        Args:
            phase1_results (Dict): Complete Phase 1 validation results
            feature_engineering_results (Dict, optional): Feature engineering pipeline results
            
        Returns:
            Dict[str, Any]: Phase 2 readiness validation
        """
        logger.info("Validating Phase 2 readiness...")
        
        readiness_result = {
            'phase_transition': 'Phase 1 → Phase 2',
            'validation_timestamp': datetime.now().isoformat(),
            'readiness_status': 'PENDING',
            'readiness_score': 0.0,
            'readiness_components': {},
            'blocking_issues': [],
            'optimization_suggestions': [],
            'ml_development_plan': {}
        }
        
        try:
            # Validate Phase 1 completion
            phase1_readiness = self._validate_phase1_handoff(phase1_results)
            
            # Validate feature engineering readiness
            feature_readiness = self._validate_feature_engineering_readiness(
                feature_engineering_results or {}
            )
            
            # Validate data quality for ML
            data_ml_readiness = self._validate_data_ml_readiness(phase1_results)
            
            # Validate baseline establishment
            baseline_readiness = self._validate_baseline_establishment(phase1_results)
            
            # Store component results
            readiness_result['readiness_components'] = {
                'phase1_completion': phase1_readiness,
                'feature_engineering': feature_readiness,
                'data_ml_readiness': data_ml_readiness,
                'baseline_establishment': baseline_readiness
            }
            
            # Calculate readiness score using weighted criteria
            weights = self.phase2_criteria['scoring_weights']
            weighted_score = (
                phase1_readiness['score'] * weights['data_availability'] / 100 +
                feature_readiness['score'] * weights['feature_engineering'] / 100 +
                data_ml_readiness['score'] * weights['data_quality'] / 100 +
                baseline_readiness['score'] * weights['baseline_performance'] / 100
            )
            
            readiness_result['readiness_score'] = round(weighted_score, 1)
            
            # Determine readiness status
            readiness_result['readiness_status'] = self._determine_readiness_status(weighted_score)
            
            # Generate blocking issues and suggestions
            readiness_result['blocking_issues'] = self._identify_blocking_issues(
                readiness_result['readiness_components']
            )
            
            readiness_result['optimization_suggestions'] = self._generate_optimization_suggestions(
                readiness_result['readiness_components']
            )
            
            # Create ML development plan
            readiness_result['ml_development_plan'] = self._create_ml_development_plan(
                readiness_result['readiness_status'], readiness_result['readiness_components']
            )
            
            logger.info(f"Phase 2 readiness validation complete. Status: {readiness_result['readiness_status']}")
            
        except Exception as e:
            logger.error(f"Phase 2 readiness validation failed: {str(e)}")
            readiness_result['readiness_status'] = 'VALIDATION_ERROR'
            readiness_result['blocking_issues'].append(f"Validation process failed: {str(e)}")
        
        return readiness_result
    
    def get_phase_status_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get concise phase status summary for quick assessment.
        
        Args:
            validation_results (Dict): Phase validation results
            
        Returns:
            Dict[str, Any]: Concise status summary
        """
        return {
            'phase': validation_results.get('phase', 'Unknown'),
            'status': validation_results.get('overall_status', 'UNKNOWN'),
            'score': validation_results.get('completion_score', 0),
            'critical_issues_count': len(validation_results.get('critical_issues', [])),
            'warnings_count': len(validation_results.get('warnings', [])),
            'ready_for_next_phase': validation_results.get('overall_status') in [
                'PHASE_1_COMPLETE', 'MOSTLY_COMPLETE'
            ],
            'validation_timestamp': validation_results.get('validation_timestamp'),
            'top_recommendations': validation_results.get('recommendations', [])[:3]
        }
    
    # Status determination methods
    def _determine_phase_status(self, overall_score: float) -> str:
        """Determine overall phase status from score."""
        if overall_score >= 85:
            return 'PHASE_1_COMPLETE'
        elif overall_score >= 70:
            return 'MOSTLY_COMPLETE'
        elif overall_score >= 50:
            return 'NEEDS_IMPROVEMENT'
        else:
            return 'NOT_READY'
    
    def _determine_readiness_status(self, weighted_score: float) -> str:
        """Determine Phase 2 readiness status from weighted score."""
        readiness_levels = self.phase2_criteria['readiness_levels']
        
        if weighted_score >= readiness_levels['READY']:
            return 'READY'
        elif weighted_score >= readiness_levels['MOSTLY_READY']:
            return 'MOSTLY_READY'
        elif weighted_score >= readiness_levels['NEEDS_IMPROVEMENT']:
            return 'NEEDS_IMPROVEMENT'
        else:
            return 'NOT_READY'
    
    # Phase 2 readiness validation methods
    def _validate_phase1_handoff(self, phase1_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Phase 1 completion for handoff to Phase 2."""
        completion_score = phase1_results.get('completion_score', 0)
        overall_status = phase1_results.get('overall_status', 'NOT_READY')
        
        score = completion_score
        handoff_ready = overall_status in ['PHASE_1_COMPLETE', 'MOSTLY_COMPLETE']
        
        return {
            'score': score,
            'handoff_ready': handoff_ready,
            'phase1_status': overall_status,
            'completion_score': completion_score,
            'issues': [] if handoff_ready else ['Phase 1 not sufficiently complete']
        }
    
    def _validate_feature_engineering_readiness(self, feature_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate feature engineering pipeline readiness."""
        score = 100.0
        issues = []
        
        if not feature_results:
            return {
                'score': 70.0,  # Assume basic features available from preprocessing
                'ready': True,
                'issues': ['Feature engineering results not provided - assuming basic pipeline'],
                'feature_count': 'Unknown',
                'efficiency': 'Unknown'
            }
        
        # Check feature count
        feature_count = feature_results.get('total_features', 0)
        if feature_count < 1000:
            score -= 20
            issues.append(f"Low feature count: {feature_count}")
        elif feature_count > 50000:
            score -= 10
            issues.append(f"Very high feature count may cause memory issues: {feature_count}")
        
        # Check pipeline efficiency
        processing_time = feature_results.get('processing_time_seconds', 0)
        if processing_time > 600:  # More than 10 minutes
            score -= 15
            issues.append("Feature engineering pipeline may be too slow for iterative development")
        
        return {
            'score': max(0, score),
            'ready': score >= 70,
            'issues': issues,
            'feature_count': feature_count,
            'efficiency': 'Good' if processing_time < 60 else 'Acceptable' if processing_time < 600 else 'Slow'
        }
    
    def _validate_data_ml_readiness(self, phase1_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data readiness for ML development."""
        score = 100.0
        
        # Extract data quality metrics from Phase 1 results
        validation_components = phase1_results.get('validation_components', {})
        dataset_validation = validation_components.get('dataset_requirements', {})
        
        if not dataset_validation.get('all_requirements_met', False):
            score -= 30
        
        # Check preprocessing pipeline
        preprocessing_validation = validation_components.get('preprocessing_pipeline', {})
        if not preprocessing_validation.get('pipeline_ready', False):
            score -= 25
        
        return {
            'score': max(0, score),
            'ml_ready': score >= 80,
            'data_quality_adequate': dataset_validation.get('all_requirements_met', False),
            'preprocessing_ready': preprocessing_validation.get('pipeline_ready', False)
        }
    
    def _validate_baseline_establishment(self, phase1_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate baseline model establishment for comparison."""
        validation_components = phase1_results.get('validation_components', {})
        baseline_validation = validation_components.get('baseline_performance', {})
        
        score = baseline_validation.get('score', 0)
        performance_adequate = baseline_validation.get('performance_adequate', False)
        
        return {
            'score': score,
            'baseline_established': performance_adequate,
            'baseline_metrics': baseline_validation.get('performance_metrics', {}),
            'ready_for_comparison': performance_adequate
        }
    
    # Assessment and recommendation methods
    def _assess_phase2_readiness(self, validation_components: Dict[str, Any]) -> Dict[str, Any]:
        """Assess readiness to begin Phase 2."""
        dataset_ready = validation_components.get('dataset_requirements', {}).get('all_requirements_met', False)
        baseline_ready = validation_components.get('baseline_performance', {}).get('performance_adequate', False)
        insights_ready = validation_components.get('business_insights', {}).get('insights_adequate', False)
        preprocessing_ready = validation_components.get('preprocessing_pipeline', {}).get('pipeline_ready', False)
        
        readiness_score = sum([dataset_ready, baseline_ready, insights_ready, preprocessing_ready]) * 25
        
        return {
            'overall_readiness_score': readiness_score,
            'components_ready': {
                'datasets': dataset_ready,
                'baseline_model': baseline_ready,
                'business_insights': insights_ready,
                'preprocessing_pipeline': preprocessing_ready
            },
            'ready_for_phase2': readiness_score >= 75,
            'recommendations': self._get_phase2_preparation_recommendations(
                dataset_ready, baseline_ready, insights_ready, preprocessing_ready
            )
        }
    
    def _generate_phase1_recommendations(self, validation_components: Dict[str, Any], 
                                       overall_score: float) -> List[str]:
        """Generate recommendations based on Phase 1 validation results."""
        recommendations = []
        
        if overall_score < 50:
            recommendations.append("CRITICAL: Address fundamental issues before proceeding to Phase 2")
        elif overall_score < 70:
            recommendations.append("Complete remaining Phase 1 requirements before Phase 2 transition")
        elif overall_score < 85:
            recommendations.append("Address minor issues to ensure smooth Phase 2 transition")
        else:
            recommendations.append("Phase 1 completed successfully - ready for Phase 2 ML development")
        
        # Component-specific recommendations
        for component_name, component_data in validation_components.items():
            if component_data.get('score', 0) < 70:
                recommendations.append(f"Improve {component_name.replace('_', ' ')} quality before progression")
        
        return recommendations[:5]  # Limit to top 5
    
    def _get_phase2_preparation_recommendations(self, dataset_ready: bool, baseline_ready: bool,
                                             insights_ready: bool, preprocessing_ready: bool) -> List[str]:
        """Generate recommendations for Phase 2 preparation."""
        recommendations = []
        
        if not dataset_ready:
            recommendations.append("Complete dataset preparation and validation before Phase 2")
        if not baseline_ready:
            recommendations.append("Establish robust baseline performance metrics")
        if not insights_ready:
            recommendations.append("Generate comprehensive business insights for stakeholder communication")
        if not preprocessing_ready:
            recommendations.append("Optimize preprocessing pipeline for efficient model training")
        
        if all([dataset_ready, baseline_ready, insights_ready, preprocessing_ready]):
            recommendations.append("All Phase 1 requirements met - ready to begin Phase 2 ML development")
        
        return recommendations
    
    def _identify_blocking_issues(self, readiness_components: Dict[str, Any]) -> List[str]:
        """Identify critical blocking issues for Phase 2."""
        blocking_issues = []
        
        for component_name, component_data in readiness_components.items():
            if not component_data.get('ready', True) and component_data.get('score', 0) < 50:
                blocking_issues.append(f"Critical issue in {component_name.replace('_', ' ')}")
        
        return blocking_issues
    
    def _generate_optimization_suggestions(self, readiness_components: Dict[str, Any]) -> List[str]:
        """Generate optimization suggestions for Phase 2 preparation."""
        suggestions = []
        
        for component_name, component_data in readiness_components.items():
            score = component_data.get('score', 0)
            if 50 <= score < 80:
                suggestions.append(f"Optimize {component_name.replace('_', ' ')} for better Phase 2 performance")
        
        return suggestions
    
    def _create_ml_development_plan(self, readiness_status: str, 
                                  readiness_components: Dict[str, Any]) -> Dict[str, Any]:
        """Create ML development plan based on readiness assessment."""
        plan = {
            'recommended_start_date': 'Immediate' if readiness_status == 'READY' else 'After addressing issues',
            'estimated_duration_weeks': 6 if readiness_status == 'READY' else 8,
            'algorithm_progression': [
                'SVM (Week 1-2)',
                'Random Forest (Week 2-3)', 
                'XGBoost (Week 3-4)',
                'BERT Fine-tuning (Week 4-6)'
            ],
            'success_criteria': [
                'Each algorithm shows statistical improvement over baseline',
                'Complete cross-validation evaluation framework',
                'Business justification documented for each algorithm',
                'Production deployment recommendations established'
            ]
        }
        
        if readiness_status != 'READY':
            plan['prerequisite_actions'] = self._identify_blocking_issues(readiness_components)
        
        return plan
    
    def _get_phase2_requirements(self) -> List[str]:
        """Get Phase 2 requirements and success criteria."""
        return [
            "Implement progressive ML algorithms: SVM → Random Forest → XGBoost → BERT",
            "Establish comprehensive model evaluation framework with cross-validation",
            "Conduct statistical significance testing for model comparisons",
            "Document business justification for each algorithm",
            "Optimize hyperparameters for production deployment",
            "Create model comparison analysis with speed vs accuracy trade-offs",
            "Generate technical documentation for model selection rationale"
        ]
    
    # Fallback validation methods (when component validators not available)
    def _fallback_dataset_validation(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Fallback dataset validation when component validator not available."""
        critical_datasets = self.phase1_requirements['critical_datasets']
        min_sizes = self.phase1_requirements['minimum_dataset_sizes']
        
        score = 100.0
        critical_issues = []
        warnings = []
        
        # Check critical datasets presence
        missing_datasets = set(critical_datasets) - set(datasets.keys())
        if missing_datasets:
            score -= 40
            critical_issues.append(f"Missing critical datasets: {missing_datasets}")
        
        # Check dataset sizes
        for dataset_name, min_size in min_sizes.items():
            if dataset_name in datasets:
                actual_size = len(datasets[dataset_name])
                if actual_size < min_size:
                    score -= 20
                    critical_issues.append(
                        f"Dataset '{dataset_name}' too small: {actual_size} < {min_size} required"
                    )
        
        return {
            'score': max(0, score),
            'critical_issues': critical_issues,
            'warnings': warnings,
            'all_requirements_met': score >= 80
        }
    
    def _fallback_baseline_validation(self, baseline_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback baseline validation when component validator not available."""
        if not baseline_metrics:
            return {
                'score': 0,
                'critical_issues': ['No baseline metrics provided'],
                'warnings': [],
                'performance_adequate': False
            }
        
        # Simple validation - check if key metrics exist
        has_accuracy = 'accuracy' in baseline_metrics
        has_f1 = 'f1_score' in baseline_metrics
        
        score = 80.0 if has_accuracy and has_f1 else 40.0
        
        return {
            'score': score,
            'critical_issues': [] if score >= 70 else ['Baseline metrics incomplete'],
            'warnings': [],
            'performance_adequate': score >= 70
        }
    
    def _fallback_insights_validation(self, business_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback insights validation when component validator not available."""
        if not business_insights:
            return {
                'score': 0,
                'critical_issues': ['No business insights provided'],
                'warnings': [],
                'insights_adequate': False
            }
        
        # Simple validation - check if insights exist
        insight_count = len(business_insights)
        score = min(100, insight_count * 20)  # 20 points per insight category
        
        return {
            'score': score,
            'critical_issues': [] if score >= 60 else ['Insufficient business insights'],
            'warnings': [],
            'insights_adequate': score >= 60
        }
    
    def _fallback_preprocessing_validation(self, preprocessing_results: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback preprocessing validation when component validator not available."""
        if not preprocessing_results:
            return {
                'score': 50,  # Assume basic preprocessing exists
                'critical_issues': [],
                'warnings': ['Preprocessing results not provided - assuming basic pipeline'],
                'pipeline_ready': True
            }
        
        # Simple validation - check success rate if available
        success_rate = preprocessing_results.get('success_rate', 90)
        score = min(100, success_rate)
        
        return {
            'score': score,
            'critical_issues': [] if score >= 80 else ['Preprocessing success rate too low'],
            'warnings': [],
            'pipeline_ready': score >= 80
        }