"""
Business Intelligence Module for Intelligent Review Analytics Platform

This module transforms raw sentiment analysis into actionable business insights
including product category risk assessment, user behavior analysis, and ROI
calculation for stakeholder communication.

Key Components:
    analyzer: Core business intelligence analysis engine
    insights: Insight generation and formatting for stakeholders
    roi_calculator: Cost-benefit analysis and savings calculations
    recommender: Business recommendation engine

Business Intelligence Areas:
    - Product category risk assessment and quality monitoring
    - User behavior segmentation and engagement analysis  
    - Review quality optimization and helpfulness prediction
    - Cost savings quantification and ROI analysis
    - Automated recommendation generation for business actions

Example:
    from src.business import BusinessAnalyzer, ROICalculator
    
    # Generate business insights
    analyzer = BusinessAnalyzer()
    insights = analyzer.analyze_product_categories(reviews_df)
    
    # Calculate ROI
    roi_calc = ROICalculator()
    savings = roi_calc.calculate_annual_savings(processing_metrics)
"""

# Business analysis constants
ANALYSIS_CATEGORIES = {
    'product_risk': {
        'name': 'Product Category Risk Assessment',
        'description': 'Identify high-risk product categories requiring attention',
        'metrics': ['negative_sentiment_rate', 'sentiment_variance', 'review_volume'],
        'output_format': 'risk_scores_by_category'
    },
    'user_behavior': {
        'name': 'User Behavior Segmentation',
        'description': 'Segment users by review patterns and engagement',
        'metrics': ['review_count', 'helpfulness_ratio', 'sentiment_consistency'],
        'output_format': 'user_segments_with_characteristics'
    },
    'review_quality': {
        'name': 'Review Quality Analysis',
        'description': 'Identify factors that make reviews helpful and valuable',
        'metrics': ['length_optimization', 'readability_score', 'sentiment_intensity'],
        'output_format': 'quality_guidelines_and_patterns'
    },
    'roi_analysis': {
        'name': 'Return on Investment Analysis', 
        'description': 'Quantify cost savings and business value creation',
        'metrics': ['time_savings', 'labor_cost_reduction', 'efficiency_gains'],
        'output_format': 'financial_impact_summary'
    }
}

# ROI calculation parameters
ROI_PARAMETERS = {
    'manual_processing': {
        'time_per_review_minutes': 5.0,
        'hourly_labor_cost_usd': 60.0,
        'accuracy_rate': 0.85,
        'consistency_factor': 0.75  # Human variability
    },
    'automated_processing': {
        'time_per_review_seconds': 0.1,
        'infrastructure_cost_monthly': 500.0,
        'maintenance_hours_monthly': 8.0,
        'accuracy_rate_placeholder': '[BERT_ACCURACY]'  # Will be replaced with actual
    },
    'business_impact': {
        'customer_lifetime_value_usd': 500.0,
        'churn_prevention_rate': 0.02,  # 2% churn prevented through faster response
        'response_time_improvement_hours': 4.0,
        'analyst_time_reallocation_percent': 70.0
    }
}

# Insight generation templates
INSIGHT_TEMPLATES = {
    'product_risk_high': {
        'template': 'Product category "{category}" shows {risk_score:.1f}% higher negative sentiment than average, requiring immediate quality review.',
        'action_items': [
            'Review recent product quality feedback',
            'Coordinate with product team for quality assessment',
            'Implement enhanced quality monitoring',
            'Consider customer communication strategy'
        ],
        'priority': 'HIGH'
    },
    'user_segment_power': {
        'template': 'Power users ({count} users, {percentage:.1f}%) contribute {contribution:.1f}% of helpful reviews, providing critical product feedback.',
        'action_items': [
            'Develop power user engagement program',
            'Prioritize power user feedback in product decisions',
            'Create incentive program for detailed reviews',
            'Establish direct communication channel'
        ],
        'priority': 'MEDIUM'
    },
    'review_quality_optimal': {
        'template': 'Reviews with {min_words}-{max_words} words show {helpfulness_rate:.1f}% higher helpfulness rates.',
        'action_items': [
            'Implement review length guidance for users',
            'A/B test review prompts for optimal length',
            'Update review submission interface',
            'Train customer service on optimal review characteristics'
        ],
        'priority': 'MEDIUM'
    },
    'roi_significant': {
        'template': 'Automation reduces analysis time by {time_reduction:.1f}%, saving ${annual_savings:,.0f} annually.',
        'action_items': [
            'Present cost savings to executive leadership',
            'Reallocate analyst time to strategic analysis',
            'Expand automation to additional review categories',
            'Measure ongoing ROI and system improvements'
        ],
        'priority': 'HIGH'
    }
}

# Business metrics thresholds
BUSINESS_THRESHOLDS = {
    'risk_assessment': {
        'high_risk_threshold': 0.4,  # 40% negative sentiment rate
        'medium_risk_threshold': 0.25, # 25% negative sentiment rate
        'minimum_reviews_for_analysis': 50
    },
    'user_segmentation': {
        'power_user_review_threshold': 10,
        'casual_user_review_threshold': 2,
        'helpfulness_threshold': 0.7
    },
    'quality_analysis': {
        'optimal_length_min': 20,   # words
        'optimal_length_max': 150,  # words
        'readability_threshold': 60, # Flesch reading ease
        'minimum_helpfulness_votes': 3
    },
    'roi_analysis': {
        'minimum_roi_threshold': 1.5,  # 150% return
        'payback_period_months': 6,
        'confidence_level': 0.95
    }
}

# Stakeholder communication formats
STAKEHOLDER_FORMATS = {
    'executive_summary': {
        'max_insights': 5,
        'focus_areas': ['roi_analysis', 'product_risk'],
        'include_financials': True,
        'technical_detail_level': 'low'
    },
    'product_team': {
        'max_insights': 8,
        'focus_areas': ['product_risk', 'review_quality'],
        'include_financials': False,
        'technical_detail_level': 'medium'
    },
    'customer_service': {
        'max_insights': 6,
        'focus_areas': ['user_behavior', 'review_quality'],
        'include_financials': False,
        'technical_detail_level': 'low'
    },
    'technical_team': {
        'max_insights': 10,
        'focus_areas': ['user_behavior', 'roi_analysis'],
        'include_financials': True,
        'technical_detail_level': 'high'
    }
}

# Public API - will be populated as modules are implemented
__all__ = [
    "ANALYSIS_CATEGORIES",
    "ROI_PARAMETERS",
    "INSIGHT_TEMPLATES",
    "BUSINESS_THRESHOLDS",
    "STAKEHOLDER_FORMATS"
]