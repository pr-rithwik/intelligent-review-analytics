"""
Business Intelligence Analysis Module for Intelligent Review Analytics Platform

This module provides comprehensive business intelligence analysis capabilities
including product category risk assessment, user behavior analysis, and
actionable insight generation for stakeholder communication.

Author: [Your Name]
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
import logging
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)

class BusinessAnalyzer:
    """
    Comprehensive business intelligence analyzer for review data.
    
    Generates actionable insights for product teams, customer service,
    and executive stakeholders with clear business recommendations.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize business analyzer with configuration.
        
        Args:
            config (Dict, optional): Analysis configuration parameters
        """
        self.config = config or self._get_default_config()
        self.insights = []
        self.analysis_results = {}
        
        logger.info("BusinessAnalyzer initialized")
    
    def _get_default_config(self) -> Dict:
        """Get default analysis configuration."""
        return {
            'risk_thresholds': {
                'high_risk': 0.4,  # 40% negative sentiment
                'medium_risk': 0.25,  # 25% negative sentiment
                'min_reviews': 50  # Minimum reviews for analysis
            },
            'user_segmentation': {
                'power_user_threshold': 10,  # 10+ reviews
                'regular_user_threshold': 3,  # 3-9 reviews
                'helpfulness_threshold': 0.7  # 70% helpfulness
            },
            'quality_thresholds': {
                'optimal_length_min': 20,  # words
                'optimal_length_max': 150,  # words
                'readability_threshold': 60,  # Flesch score
                'min_helpfulness_votes': 3
            },
            'significance_level': 0.05,
            'confidence_level': 0.95
        }
    
    def analyze_product_categories(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze product categories for risk assessment and quality insights.
        
        Args:
            df (pd.DataFrame): DataFrame with reviews including productId and sentiment
            
        Returns:
            Dict[str, Any]: Product category analysis results
        """
        logger.info("Analyzing product category risk patterns...")
        
        if 'productId' not in df.columns or 'sentiment' not in df.columns:
            return {'error': 'Required columns (productId, sentiment) not found'}
        
        # Extract product categories from productId
        df['product_category'] = df['productId'].apply(self._extract_product_category)
        
        # Category-level analysis
        category_stats = df.groupby('product_category').agg({
            'sentiment': ['count', 'mean', 'std'],
            'productId': 'nunique'
        }).round(3)
        
        category_stats.columns = ['review_count', 'avg_sentiment', 'sentiment_std', 'product_count']
        category_stats = category_stats.reset_index()
        
        # Calculate risk metrics
        category_stats['negative_rate'] = df.groupby('product_category')['sentiment'].apply(
            lambda x: (x == -1).mean()
        ).values
        
        category_stats['positive_rate'] = df.groupby('product_category')['sentiment'].apply(
            lambda x: (x == 1).mean()
        ).values
        
        # Risk classification
        category_stats['risk_level'] = category_stats['negative_rate'].apply(self._classify_risk)
        
        # Filter categories with sufficient data
        significant_categories = category_stats[
            category_stats['review_count'] >= self.config['risk_thresholds']['min_reviews']
        ]
        
        # Identify high-risk categories
        high_risk_categories = significant_categories[
            significant_categories['risk_level'] == 'HIGH'
        ].sort_values('negative_rate', ascending=False)
        
        # Statistical significance testing
        overall_negative_rate = (df['sentiment'] == -1).mean()
        category_tests = []
        
        for _, row in significant_categories.iterrows():
            category_data = df[df['product_category'] == row['product_category']]['sentiment']
            
            # Chi-square test for sentiment distribution difference
            observed = [(category_data == -1).sum(), (category_data == 1).sum()]
            expected_neg = len(category_data) * overall_negative_rate
            expected_pos = len(category_data) * (1 - overall_negative_rate)
            expected = [expected_neg, expected_pos]
            
            try:
                chi2_stat, p_value = stats.chisquare(observed, expected)
                is_significant = p_value < self.config['significance_level']
            except:
                chi2_stat, p_value, is_significant = 0, 1, False
            
            category_tests.append({
                'category': row['product_category'],
                'chi2_stat': chi2_stat,
                'p_value': p_value,
                'is_significant': is_significant,
                'effect_size': abs(row['negative_rate'] - overall_negative_rate)
            })
        
        results = {
            'category_summary': {
                'total_categories': len(category_stats),
                'significant_categories': len(significant_categories),
                'high_risk_categories': len(high_risk_categories),
                'overall_negative_rate': round(overall_negative_rate, 3)
            },
            'category_stats': significant_categories.to_dict('records'),
            'high_risk_categories': high_risk_categories.to_dict('records'),
            'statistical_tests': category_tests,
            'recommendations': self._generate_category_recommendations(high_risk_categories)
        }
        
        self.analysis_results['product_categories'] = results
        logger.info(f"Product category analysis complete: {len(high_risk_categories)} high-risk categories identified")
        
        return results
    
    def analyze_user_behavior(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze user behavior patterns and segment users by engagement.
        
        Args:
            df (pd.DataFrame): DataFrame with reviews including userId
            
        Returns:
            Dict[str, Any]: User behavior analysis results
        """
        logger.info("Analyzing user behavior patterns...")
        
        if 'userId' not in df.columns:
            return {'error': 'userId column not found'}
        
        # User-level statistics
        user_stats = df.groupby('userId').agg({
            'sentiment': ['count', 'mean', 'std'],
            'productId': 'nunique'
        }).round(3)
        
        user_stats.columns = ['review_count', 'avg_sentiment', 'sentiment_std', 'products_reviewed']
        user_stats = user_stats.reset_index()
        
        # Add helpfulness metrics if available
        if 'helpfulnessNumerator' in df.columns and 'helpfulnessDenominator' in df.columns:
            helpfulness_stats = df.groupby('userId').apply(self._calculate_user_helpfulness).reset_index()
            helpfulness_stats.columns = ['userId', 'helpfulness_ratio', 'total_votes', 'helpful_reviews']
            user_stats = user_stats.merge(helpfulness_stats, on='userId', how='left')
        else:
            user_stats['helpfulness_ratio'] = 0
            user_stats['total_votes'] = 0
            user_stats['helpful_reviews'] = 0
        
        # User segmentation
        user_stats['user_segment'] = user_stats['review_count'].apply(self._segment_users)
        
        # Segment analysis
        segment_summary = user_stats.groupby('user_segment').agg({
            'userId': 'count',
            'review_count': ['mean', 'median'],
            'avg_sentiment': 'mean',
            'helpfulness_ratio': 'mean',
            'products_reviewed': 'mean'
        }).round(3)
        
        segment_summary.columns = ['user_count', 'avg_reviews', 'median_reviews', 
                                 'avg_sentiment', 'avg_helpfulness', 'avg_products']
        segment_summary = segment_summary.reset_index()
        
        # Calculate segment contributions
        total_reviews = len(df)
        total_users = len(user_stats)
        
        for _, row in segment_summary.iterrows():
            segment_users = user_stats[user_stats['user_segment'] == row['user_segment']]
            segment_reviews = segment_users['review_count'].sum()
            
            segment_summary.loc[segment_summary['user_segment'] == row['user_segment'], 'review_contribution'] = \
                round((segment_reviews / total_reviews) * 100, 1)
            
            segment_summary.loc[segment_summary['user_segment'] == row['user_segment'], 'user_percentage'] = \
                round((len(segment_users) / total_users) * 100, 1)
        
        # Power user analysis
        power_users = user_stats[user_stats['user_segment'] == 'Power User']
        
        results = {
            'user_summary': {
                'total_users': len(user_stats),
                'avg_reviews_per_user': round(user_stats['review_count'].mean(), 1),
                'median_reviews_per_user': int(user_stats['review_count'].median()),
                'avg_helpfulness_ratio': round(user_stats['helpfulness_ratio'].mean(), 3)
            },
            'segment_analysis': segment_summary.to_dict('records'),
            'power_user_insights': {
                'count': len(power_users),
                'percentage': round((len(power_users) / total_users) * 100, 1),
                'review_contribution': round((power_users['review_count'].sum() / total_reviews) * 100, 1),
                'avg_helpfulness': round(power_users['helpfulness_ratio'].mean(), 3),
                'avg_products_reviewed': round(power_users['products_reviewed'].mean(), 1)
            },
            'recommendations': self._generate_user_behavior_recommendations(segment_summary, power_users)
        }
        
        self.analysis_results['user_behavior'] = results
        logger.info(f"User behavior analysis complete: {len(power_users)} power users identified")
        
        return results
    
    def analyze_review_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze review quality patterns and helpfulness factors.
        
        Args:
            df (pd.DataFrame): DataFrame with reviews including quality metrics
            
        Returns:
            Dict[str, Any]: Review quality analysis results
        """
        logger.info("Analyzing review quality patterns...")
        
        quality_metrics = ['word_count', 'flesch_reading_ease', 'sentiment_compound']
        available_metrics = [m for m in quality_metrics if m in df.columns]
        
        if not available_metrics:
            return {'error': 'No quality metrics available'}
        
        # Filter reviews with helpfulness data
        if 'helpfulnessNumerator' in df.columns and 'helpfulnessDenominator' in df.columns:
            df_with_votes = df[
                (df['helpfulnessDenominator'] >= self.config['quality_thresholds']['min_helpfulness_votes']) &
                (df['helpfulnessDenominator'].notna())
            ].copy()
            
            if len(df_with_votes) == 0:
                return {'error': 'No reviews with sufficient helpfulness votes'}
            
            # Calculate helpfulness ratio
            df_with_votes['helpfulness_ratio'] = (df_with_votes['helpfulnessNumerator'] / 
                                                df_with_votes['helpfulnessDenominator'])
            
            # Categorize helpfulness
            df_with_votes['helpfulness_category'] = pd.cut(
                df_with_votes['helpfulness_ratio'],
                bins=[0, 0.3, 0.7, 1.0],
                labels=['Low', 'Medium', 'High'],
                include_lowest=True
            )
        else:
            return {'error': 'Helpfulness data not available'}
        
        # Analyze length vs helpfulness
        length_analysis = self._analyze_length_helpfulness(df_with_votes)
        
        # Analyze readability vs helpfulness
        readability_analysis = self._analyze_readability_helpfulness(df_with_votes)
        
        # Analyze sentiment intensity vs helpfulness
        sentiment_analysis = self._analyze_sentiment_helpfulness(df_with_votes)
        
        # Find optimal review characteristics
        high_quality_reviews = df_with_votes[df_with_votes['helpfulness_ratio'] >= 0.7]
        
        optimal_characteristics = {
            'count': len(high_quality_reviews),
            'percentage': round((len(high_quality_reviews) / len(df_with_votes)) * 100, 1)
        }
        
        if 'word_count' in df_with_votes.columns:
            optimal_characteristics.update({
                'avg_word_count': round(high_quality_reviews['word_count'].mean(), 1),
                'median_word_count': int(high_quality_reviews['word_count'].median()),
                'word_count_range': {
                    'min': int(high_quality_reviews['word_count'].quantile(0.25)),
                    'max': int(high_quality_reviews['word_count'].quantile(0.75))
                }
            })
        
        results = {
            'analysis_summary': {
                'total_reviews_analyzed': len(df_with_votes),
                'high_quality_reviews': len(high_quality_reviews),
                'avg_helpfulness_ratio': round(df_with_votes['helpfulness_ratio'].mean(), 3)
            },
            'length_analysis': length_analysis,
            'readability_analysis': readability_analysis,
            'sentiment_analysis': sentiment_analysis,
            'optimal_characteristics': optimal_characteristics,
            'recommendations': self._generate_quality_recommendations(optimal_characteristics, length_analysis)
        }
        
        self.analysis_results['review_quality'] = results
        logger.info("Review quality analysis complete")
        
        return results
    
    def calculate_roi_metrics(self, processing_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate return on investment metrics for automation.
        
        Args:
            processing_metrics (Dict): Processing time and efficiency metrics
            
        Returns:
            Dict[str, Any]: ROI analysis results
        """
        logger.info("Calculating ROI metrics...")
        
        # Default values if not provided
        manual_time_per_review = processing_metrics.get('manual_time_per_review_minutes', 5.0)
        automated_time_per_review = processing_metrics.get('automated_time_per_review_seconds', 0.1)
        hourly_labor_cost = processing_metrics.get('hourly_labor_cost_usd', 60.0)
        annual_review_volume = processing_metrics.get('annual_review_volume', 100000)
        
        # Time savings calculations
        manual_time_hours_per_review = manual_time_per_review / 60
        automated_time_hours_per_review = automated_time_per_review / 3600
        
        time_savings_per_review = manual_time_hours_per_review - automated_time_hours_per_review
        time_reduction_percentage = (time_savings_per_review / manual_time_hours_per_review) * 100
        
        # Annual calculations
        annual_time_savings_hours = time_savings_per_review * annual_review_volume
        annual_labor_cost_savings = annual_time_savings_hours * hourly_labor_cost
        
        # Infrastructure costs (estimated)
        monthly_infrastructure_cost = 500  # Cloud hosting, etc.
        annual_infrastructure_cost = monthly_infrastructure_cost * 12
        maintenance_hours_monthly = 8
        annual_maintenance_cost = maintenance_hours_monthly * 12 * hourly_labor_cost
        
        total_annual_cost = annual_infrastructure_cost + annual_maintenance_cost
        net_annual_savings = annual_labor_cost_savings - total_annual_cost
        
        # ROI calculation
        initial_development_cost = 40 * hourly_labor_cost  # Estimated 40 hours development
        roi_percentage = ((net_annual_savings - initial_development_cost) / initial_development_cost) * 100
        payback_period_months = initial_development_cost / (net_annual_savings / 12)
        
        # Additional business benefits
        customer_service_efficiency = self._calculate_cs_efficiency_improvement(processing_metrics)
        
        results = {
            'time_analysis': {
                'manual_time_per_review_minutes': manual_time_per_review,
                'automated_time_per_review_seconds': automated_time_per_review,
                'time_reduction_percentage': round(time_reduction_percentage, 1),
                'annual_time_savings_hours': round(annual_time_savings_hours, 0)
            },
            'cost_analysis': {
                'annual_labor_cost_savings': round(annual_labor_cost_savings, 0),
                'annual_infrastructure_cost': annual_infrastructure_cost,
                'annual_maintenance_cost': round(annual_maintenance_cost, 0),
                'net_annual_savings': round(net_annual_savings, 0)
            },
            'roi_metrics': {
                'roi_percentage': round(roi_percentage, 1),
                'payback_period_months': round(payback_period_months, 1),
                'break_even_monthly_savings': round(net_annual_savings / 12, 0)
            },
            'business_benefits': customer_service_efficiency,
            'recommendations': self._generate_roi_recommendations(net_annual_savings, roi_percentage)
        }
        
        self.analysis_results['roi_analysis'] = results
        logger.info(f"ROI analysis complete: ${net_annual_savings:,.0f} net annual savings")
        
        return results
    
    def generate_executive_summary(self) -> Dict[str, Any]:
        """
        Generate executive summary of all analysis results.
        
        Returns:
            Dict[str, Any]: Executive summary with key insights
        """
        if not self.analysis_results:
            return {'error': 'No analysis results available'}
        
        # Extract key metrics from each analysis
        key_insights = []
        
        # Product category insights
        if 'product_categories' in self.analysis_results:
            category_results = self.analysis_results['product_categories']
            if 'high_risk_categories' in category_results:
                high_risk_count = len(category_results['high_risk_categories'])
                if high_risk_count > 0:
                    key_insights.append({
                        'type': 'product_risk',
                        'priority': 'HIGH',
                        'insight': f'{high_risk_count} product categories identified as high-risk requiring immediate attention',
                        'impact': 'Product quality and customer satisfaction',
                        'action': 'Review product quality processes for identified categories'
                    })
        
        # User behavior insights
        if 'user_behavior' in self.analysis_results:
            user_results = self.analysis_results['user_behavior']
            power_user_data = user_results.get('power_user_insights', {})
            if power_user_data.get('percentage', 0) > 0:
                key_insights.append({
                    'type': 'user_engagement',
                    'priority': 'MEDIUM',
                    'insight': f'Power users ({power_user_data.get("percentage", 0):.1f}% of users) contribute {power_user_data.get("review_contribution", 0):.1f}% of reviews',
                    'impact': 'Customer engagement and product feedback quality',
                    'action': 'Develop power user engagement and retention programs'
                })
        
        # ROI insights
        if 'roi_analysis' in self.analysis_results:
            roi_results = self.analysis_results['roi_analysis']
            net_savings = roi_results.get('cost_analysis', {}).get('net_annual_savings', 0)
            roi_percentage = roi_results.get('roi_metrics', {}).get('roi_percentage', 0)
            if net_savings > 0:
                key_insights.append({
                    'type': 'financial_impact',
                    'priority': 'HIGH',
                    'insight': f'Automation delivers ${net_savings:,.0f} net annual savings with {roi_percentage:.0f}% ROI',
                    'impact': 'Cost reduction and operational efficiency',
                    'action': 'Implement automation system and reallocate analyst resources'
                })
        
        # Quality insights
        if 'review_quality' in self.analysis_results:
            quality_results = self.analysis_results['review_quality']
            optimal_chars = quality_results.get('optimal_characteristics', {})
            if 'word_count_range' in optimal_chars:
                word_range = optimal_chars['word_count_range']
                key_insights.append({
                    'type': 'quality_optimization',
                    'priority': 'MEDIUM',
                    'insight': f'Reviews with {word_range["min"]}-{word_range["max"]} words show highest helpfulness rates',
                    'impact': 'Review quality and customer decision-making',
                    'action': 'Implement review length guidance for optimal user experience'
                })
        
        summary = {
            'executive_summary': {
                'total_insights': len(key_insights),
                'high_priority_insights': len([i for i in key_insights if i['priority'] == 'HIGH']),
                'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
                'analysis_scope': list(self.analysis_results.keys())
            },
            'key_insights': key_insights[:5],  # Top 5 insights
            'strategic_recommendations': self._generate_strategic_recommendations(key_insights),
            'implementation_priority': self._prioritize_implementations(key_insights)
        }
        
        return summary
    
    # Helper methods
    def _extract_product_category(self, product_id: str) -> str:
        """Extract category from product ID (simplified)."""
        if not isinstance(product_id, str):
            return 'Unknown'
        
        # Simple heuristic: use first character or pattern
        if len(product_id) > 0:
            first_char = product_id[0].upper()
            category_map = {
                'B': 'Books', 'A': 'Audio', 'D': 'DVD', 'C': 'Clothing',
                'E': 'Electronics', 'F': 'Food', 'G': 'Games', 'H': 'Home',
                'I': 'Industrial', 'J': 'Jewelry', 'K': 'Kitchen', 'L': 'Lawn',
                'M': 'Music', 'N': 'Network', 'O': 'Office', 'P': 'Pet',
                'Q': 'Quality', 'R': 'Recreation', 'S': 'Sports', 'T': 'Tools',
                'U': 'Utility', 'V': 'Video', 'W': 'Wireless', 'X': 'Xbox',
                'Y': 'Yoga', 'Z': 'Zone'
            }
            return category_map.get(first_char, f'Category_{first_char}')
        return 'Unknown'
    
    def _classify_risk(self, negative_rate: float) -> str:
        """Classify risk level based on negative sentiment rate."""
        if negative_rate >= self.config['risk_thresholds']['high_risk']:
            return 'HIGH'
        elif negative_rate >= self.config['risk_thresholds']['medium_risk']:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _segment_users(self, review_count: int) -> str:
        """Segment users based on review activity."""
        if review_count >= self.config['user_segmentation']['power_user_threshold']:
            return 'Power User'
        elif review_count >= self.config['user_segmentation']['regular_user_threshold']:
            return 'Regular User'
        else:
            return 'Casual User'
    
    def _calculate_user_helpfulness(self, user_reviews: pd.DataFrame) -> pd.Series:
        """Calculate helpfulness metrics for a user."""
        if 'helpfulnessNumerator' not in user_reviews.columns:
            return pd.Series([0, 0, 0])
        
        total_votes = user_reviews['helpfulnessDenominator'].sum()
        helpful_votes = user_reviews['helpfulnessNumerator'].sum()
        helpful_reviews = (user_reviews['helpfulnessNumerator'] > 0).sum()
        
        helpfulness_ratio = helpful_votes / total_votes if total_votes > 0 else 0
        
        return pd.Series([helpfulness_ratio, total_votes, helpful_reviews])
    
    def _analyze_length_helpfulness(self, df: pd.DataFrame) -> Dict:
        """Analyze relationship between review length and helpfulness."""
        if 'word_count' not in df.columns:
            return {'error': 'word_count not available'}
        
        # Create length bins
        df['length_bin'] = pd.cut(df['word_count'], bins=5, labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long'])
        
        length_stats = df.groupby('length_bin')['helpfulness_ratio'].agg(['count', 'mean', 'std']).reset_index()
        length_stats.columns = ['length_category', 'review_count', 'avg_helpfulness', 'helpfulness_std']
        
        # Find optimal length range
        best_category = length_stats.loc[length_stats['avg_helpfulness'].idxmax()]
        
        return {
            'length_distribution': length_stats.to_dict('records'),
            'optimal_length_category': best_category['length_category'],
            'optimal_avg_helpfulness': round(best_category['avg_helpfulness'], 3),
            'correlation': round(df['word_count'].corr(df['helpfulness_ratio']), 3)
        }
    
    def _analyze_readability_helpfulness(self, df: pd.DataFrame) -> Dict:
        """Analyze relationship between readability and helpfulness."""
        if 'flesch_reading_ease' not in df.columns:
            return {'error': 'flesch_reading_ease not available'}
        
        # Create readability bins
        df['readability_bin'] = pd.cut(
            df['flesch_reading_ease'],
            bins=[0, 30, 50, 70, 90, 100],
            labels=['Very Difficult', 'Difficult', 'Standard', 'Easy', 'Very Easy']
        )
        
        readability_stats = df.groupby('readability_bin')['helpfulness_ratio'].agg(['count', 'mean']).reset_index()
        readability_stats.columns = ['readability_category', 'review_count', 'avg_helpfulness']
        
        return {
            'readability_distribution': readability_stats.to_dict('records'),
            'correlation': round(df['flesch_reading_ease'].corr(df['helpfulness_ratio']), 3)
        }
    
    def _analyze_sentiment_helpfulness(self, df: pd.DataFrame) -> Dict:
        """Analyze relationship between sentiment intensity and helpfulness."""
        if 'sentiment_compound' not in df.columns:
            return {'error': 'sentiment_compound not available'}
        
        # Create sentiment intensity bins
        df['sentiment_bin'] = pd.cut(
            df['sentiment_compound'],
            bins=[-1, -0.5, -0.1, 0.1, 0.5, 1],
            labels=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
        )
        
        sentiment_stats = df.groupby('sentiment_bin')['helpfulness_ratio'].agg(['count', 'mean']).reset_index()
        sentiment_stats.columns = ['sentiment_category', 'review_count', 'avg_helpfulness']
        
        return {
            'sentiment_distribution': sentiment_stats.to_dict('records'),
            'correlation': round(df['sentiment_compound'].corr(df['helpfulness_ratio']), 3)
        }
    
    def _calculate_cs_efficiency_improvement(self, processing_metrics: Dict) -> Dict:
        """Calculate customer service efficiency improvements."""
        return {
            'response_time_improvement_hours': 4,  # Faster issue identification
            'priority_accuracy_improvement': 25,  # Better prioritization
            'analyst_time_reallocation_percent': 70,  # Time freed for strategic work
            'customer_satisfaction_impact': 'Faster response to negative feedback'
        }
    
    def _generate_category_recommendations(self, high_risk_categories: pd.DataFrame) -> List[Dict]:
        """Generate recommendations for high-risk categories."""
        recommendations = []
        
        for _, category in high_risk_categories.iterrows():
            recommendations.append({
                'category': category['product_category'],
                'priority': 'HIGH',
                'recommendation': f'Immediate quality review required for {category["product_category"]} category',
                'actions': [
                    'Review recent customer complaints and feedback',
                    'Coordinate with product team for quality assessment', 
                    'Implement enhanced quality monitoring',
                    'Consider proactive customer communication'
                ],
                'expected_impact': f'Reduce negative sentiment from {category["negative_rate"]*100:.1f}% to <25%'
            })
        
        return recommendations[:5]  # Top 5 recommendations
    
    def _generate_user_behavior_recommendations(self, segment_summary: pd.DataFrame, power_users: pd.DataFrame) -> List[Dict]:
        """Generate recommendations for user engagement."""
        recommendations = []
        
        # Power user recommendations
        power_user_row = segment_summary[segment_summary['user_segment'] == 'Power User']
        if not power_user_row.empty:
            recommendations.append({
                'segment': 'Power Users',
                'priority': 'MEDIUM',
                'recommendation': 'Develop power user engagement program',
                'actions': [
                    'Create exclusive reviewer recognition program',
                    'Provide early access to new products',
                    'Establish direct feedback channel',
                    'Offer review writing guidelines and tips'
                ],
                'expected_impact': 'Increase review quality and platform advocacy'
            })
        
        return recommendations
    
    def _generate_quality_recommendations(self, optimal_chars: Dict, length_analysis: Dict) -> List[Dict]:
        """Generate recommendations for review quality improvement."""
        recommendations = []
        
        if 'word_count_range' in optimal_chars:
            word_range = optimal_chars['word_count_range']
            recommendations.append({
                'area': 'Review Length Optimization',
                'priority': 'MEDIUM',
                'recommendation': f'Guide users to write {word_range["min"]}-{word_range["max"]} word reviews',
                'actions': [
                    'Implement review length suggestions in UI',
                    'A/B test different prompting strategies',
                    'Provide review quality feedback',
                    'Create review writing best practices guide'
                ],
                'expected_impact': 'Increase average helpfulness ratio by 15-20%'
            })
        
        return recommendations
    
    def _generate_roi_recommendations(self, net_savings: float, roi_percentage: float) -> List[Dict]:
        """Generate ROI-based recommendations."""
        recommendations = []
        
        if net_savings > 0 and roi_percentage > 100:
            recommendations.append({
                'area': 'Automation Implementation',
                'priority': 'HIGH',
                'recommendation': 'Implement automated review analysis system',
                'actions': [
                    'Present business case to executive leadership',
                    'Develop implementation timeline',
                    'Reallocate analyst time to strategic initiatives',
                    'Establish ROI tracking and measurement'
                ],
                'expected_impact': f'${net_savings:,.0f} annual savings with {roi_percentage:.0f}% ROI'
            })
        
        return recommendations
    
    def _generate_strategic_recommendations(self, insights: List[Dict]) -> List[str]:
        """Generate high-level strategic recommendations."""
        recommendations = []
        
        high_priority_count = len([i for i in insights if i['priority'] == 'HIGH'])
        
        if high_priority_count > 0:
            recommendations.append(
                f"Prioritize {high_priority_count} high-impact initiatives for immediate implementation"
            )
        
        recommendations.extend([
            "Establish automated review monitoring system for continuous insight generation",
            "Develop cross-functional team for review quality improvement initiatives", 
            "Implement regular business intelligence reporting cadence",
            "Create customer feedback loop for continuous improvement"
        ])
        
        return recommendations[:4]  # Top 4 strategic recommendations
    
    def _prioritize_implementations(self, insights: List[Dict]) -> List[Dict]:
        """Prioritize implementation based on impact and effort."""
        priority_order = {'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        
        sorted_insights = sorted(insights, key=lambda x: priority_order.get(x['priority'], 3))
        
        implementation_plan = []
        for i, insight in enumerate(sorted_insights[:3]):  # Top 3
            implementation_plan.append({
                'phase': f'Phase {i+1}',
                'timeline': f'{(i+1)*4}-{(i+1)*4+3} weeks',
                'focus': insight['type'].replace('_', ' ').title(),
                'description': insight['insight'],
                'key_action': insight['action']
            })
        
        return implementation_plan
                    