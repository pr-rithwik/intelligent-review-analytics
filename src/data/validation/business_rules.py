"""
Business Rules Validation Module

Validates business logic constraints, domain-specific rules, and
cross-column relationships for review analytics platform.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime

from .config import BUSINESS_RULES, get_default_validation_config

logger = logging.getLogger(__name__)

class BusinessRulesValidator:
    """
    Validates business logic constraints and domain-specific rules.
    
    Checks product coverage, user activity patterns, review quality relationships,
    and cross-column business logic constraints.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize business rules validator with configuration."""
        self.config = config or get_default_validation_config()
        self.business_rules = self.config['business_rules']
        
        logger.info("BusinessRulesValidator initialized")
    
    def validate_business_rules(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive business rules validation.
        
        Args:
            df (pd.DataFrame): Dataset to validate
            
        Returns:
            Dict[str, Any]: Business rules validation results
        """
        logger.info(f"Validating business rules on {len(df)} rows...")
        
        validation_result = {
            'validation_timestamp': datetime.now().isoformat(),
            'total_rows_analyzed': len(df),
            'business_checks': {},
            'overall_score': 0.0,
            'critical_issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Run individual business rule checks
        business_checks = {
            'product_coverage': self._validate_product_coverage(df),
            'user_activity': self._validate_user_activity(df),
            'review_relationships': self._validate_review_relationships(df),
            'data_integrity': self._validate_data_integrity(df),
            'business_logic': self._validate_business_logic(df)
        }
        
        # Aggregate results
        total_penalty = 0
        all_issues = []
        all_warnings = []
        
        for check_name, check_result in business_checks.items():
            total_penalty += check_result.get('penalty', 0)
            all_issues.extend(check_result.get('issues', []))
            all_warnings.extend(check_result.get('warnings', []))
        
        # Calculate overall score
        overall_score = max(0, 100 - total_penalty)
        
        # Update validation result
        validation_result.update({
            'business_checks': business_checks,
            'overall_score': round(overall_score, 1),
            'critical_issues': all_issues,
            'warnings': all_warnings,
            'recommendations': self._generate_business_recommendations(business_checks),
            'summary_statistics': self._create_business_statistics(business_checks, df)
        })
        
        logger.info(f"Business rules validation complete. Score: {overall_score:.1f}")
        
        return validation_result
    
    def _validate_product_coverage(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate product coverage and review distribution."""
        issues = []
        warnings = []
        penalty = 0
        
        if 'productId' not in df.columns:
            return {
                'skipped': True,
                'reason': 'productId column not found',
                'penalty': 0,
                'issues': [],
                'warnings': []
            }
        
        # Analyze product review distribution
        product_review_counts = df['productId'].value_counts()
        min_reviews_threshold = self.business_rules['product_coverage']['min_reviews_per_product']
        warning_threshold = self.business_rules['product_coverage']['warning_threshold']
        ideal_threshold = self.business_rules['product_coverage']['ideal_min_reviews']
        
        # Products below minimum threshold
        products_below_min = (product_review_counts < min_reviews_threshold).sum()
        if products_below_min > 0:
            penalty += 15
            issues.append(f"{products_below_min} products have fewer than {min_reviews_threshold} reviews")
        
        # Products below warning threshold
        products_below_warning = (product_review_counts < warning_threshold).sum()
        if products_below_warning > len(product_review_counts) * 0.3:  # More than 30%
            penalty += 10
            warnings.append(f"{products_below_warning} products have fewer than {warning_threshold} reviews")
        
        # Products below ideal threshold
        products_below_ideal = (product_review_counts < ideal_threshold).sum()
        
        # Calculate coverage statistics
        total_products = len(product_review_counts)
        avg_reviews_per_product = product_review_counts.mean()
        median_reviews_per_product = product_review_counts.median()
        
        # Check for extreme imbalance
        max_reviews = product_review_counts.max()
        min_reviews = product_review_counts.min()
        review_ratio = max_reviews / min_reviews if min_reviews > 0 else float('inf')
        
        if review_ratio > 100:  # One product has 100x more reviews than another
            penalty += 5
            warnings.append(f"Extreme review imbalance detected (ratio: {review_ratio:.1f})")
        
        return {
            'penalty': penalty,
            'issues': issues,
            'warnings': warnings,
            'statistics': {
                'total_products': total_products,
                'avg_reviews_per_product': round(avg_reviews_per_product, 2),
                'median_reviews_per_product': int(median_reviews_per_product),
                'products_below_minimum': products_below_min,
                'products_below_warning': products_below_warning,
                'products_below_ideal': products_below_ideal,
                'review_distribution_ratio': round(review_ratio, 2) if review_ratio != float('inf') else None,
                'coverage_score': round((1 - products_below_min / total_products) * 100, 1)
            }
        }
    
    def _validate_user_activity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate user activity patterns and engagement."""
        issues = []
        warnings = []
        penalty = 0
        
        if 'userId' not in df.columns:
            return {
                'skipped': True,
                'reason': 'userId column not found',
                'penalty': 0,
                'issues': [],
                'warnings': []
            }
        
        # Analyze user activity distribution
        user_review_counts = df['userId'].value_counts()
        min_reviews_threshold = self.business_rules['user_activity']['min_reviews_per_user']
        power_user_threshold = self.business_rules['user_activity']['power_user_threshold']
        casual_user_threshold = self.business_rules['user_activity']['casual_user_threshold']
        
        # User segmentation
        power_users = (user_review_counts >= power_user_threshold).sum()
        regular_users = ((user_review_counts >= casual_user_threshold) & 
                        (user_review_counts < power_user_threshold)).sum()
        casual_users = (user_review_counts < casual_user_threshold).sum()
        
        total_users = len(user_review_counts)
        
        # Check for concerning patterns
        if casual_users > total_users * 0.8:  # More than 80% casual users
            penalty += 10
            warnings.append(f"{casual_users/total_users*100:.1f}% of users are casual (< {casual_user_threshold} reviews)")
        
        if power_users < total_users * 0.05:  # Less than 5% power users
            penalty += 5
            warnings.append(f"Only {power_users/total_users*100:.1f}% of users are power users")
        
        # Calculate engagement metrics
        avg_reviews_per_user = user_review_counts.mean()
        median_reviews_per_user = user_review_counts.median()
        
        # Check for bot-like behavior (users with excessive reviews)
        suspicious_threshold = user_review_counts.quantile(0.99)  # Top 1%
        suspicious_users = (user_review_counts > suspicious_threshold * 2).sum()
        
        if suspicious_users > 0:
            penalty += 8
            warnings.append(f"{suspicious_users} users have suspiciously high review counts")
        
        return {
            'penalty': penalty,
            'issues': issues,
            'warnings': warnings,
            'statistics': {
                'total_users': total_users,
                'power_users': power_users,
                'regular_users': regular_users,
                'casual_users': casual_users,
                'user_distribution': {
                    'power_user_percentage': round(power_users / total_users * 100, 1),
                    'regular_user_percentage': round(regular_users / total_users * 100, 1),
                    'casual_user_percentage': round(casual_users / total_users * 100, 1)
                },
                'engagement_metrics': {
                    'avg_reviews_per_user': round(avg_reviews_per_user, 2),
                    'median_reviews_per_user': int(median_reviews_per_user),
                    'suspicious_users': suspicious_users
                }
            }
        }
    
    def _validate_review_relationships(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate relationships between review components."""
        issues = []
        warnings = []
        penalty = 0
        
        relationship_checks = {}
        
        # Text-summary relationship
        if 'text' in df.columns and 'summary' in df.columns:
            text_summary_analysis = self._analyze_text_summary_relationship(df)
            relationship_checks['text_summary'] = text_summary_analysis
            penalty += text_summary_analysis.get('penalty', 0)
            issues.extend(text_summary_analysis.get('issues', []))
            warnings.extend(text_summary_analysis.get('warnings', []))
        
        # Helpfulness-length relationship
        if all(col in df.columns for col in ['text', 'helpfulnessNumerator', 'helpfulnessDenominator']):
            helpfulness_analysis = self._analyze_helpfulness_patterns(df)
            relationship_checks['helpfulness_patterns'] = helpfulness_analysis
            penalty += helpfulness_analysis.get('penalty', 0)
            warnings.extend(helpfulness_analysis.get('warnings', []))
        
        # Sentiment-helpfulness relationship
        if all(col in df.columns for col in ['sentiment', 'helpfulnessNumerator', 'helpfulnessDenominator']):
            sentiment_helpfulness = self._analyze_sentiment_helpfulness_relationship(df)
            relationship_checks['sentiment_helpfulness'] = sentiment_helpfulness
            warnings.extend(sentiment_helpfulness.get('warnings', []))
        
        return {
            'penalty': penalty,
            'issues': issues,
            'warnings': warnings,
            'relationship_checks': relationship_checks
        }
    
    def _validate_data_integrity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data integrity and cross-column consistency."""
        issues = []
        warnings = []
        penalty = 0
        
        # Check for duplicate reviews (same user, same product)
        if 'userId' in df.columns and 'productId' in df.columns:
            duplicate_pairs = df.duplicated(subset=['userId', 'productId']).sum()
            duplicate_percentage = (duplicate_pairs / len(df)) * 100
            
            if duplicate_percentage > self.business_rules['duplicate_detection']['max_duplicate_percentage']:
                penalty += self.business_rules['duplicate_detection']['penalty_per_percent'] * duplicate_percentage
                issues.append(f"{duplicate_percentage:.1f}% duplicate user-product pairs found")
            elif duplicate_percentage > 1:
                warnings.append(f"{duplicate_percentage:.1f}% duplicate user-product pairs detected")
        
        # Check for temporal consistency (if timestamp available)
        temporal_issues = self._check_temporal_consistency(df)
        penalty += temporal_issues.get('penalty', 0)
        warnings.extend(temporal_issues.get('warnings', []))
        
        # Check ID format consistency
        id_format_issues = self._check_id_format_consistency(df)
        penalty += id_format_issues.get('penalty', 0)
        warnings.extend(id_format_issues.get('warnings', []))
        
        return {
            'penalty': penalty,
            'issues': issues,
            'warnings': warnings,
            'integrity_checks': {
                'duplicate_analysis': {'duplicate_percentage': duplicate_percentage} if 'userId' in df.columns and 'productId' in df.columns else {},
                'temporal_analysis': temporal_issues,
                'id_format_analysis': id_format_issues
            }
        }
    
    def _validate_business_logic(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate specific business logic rules."""
        issues = []
        warnings = []
        penalty = 0
        
        # Helpfulness logic validation
        if 'helpfulnessNumerator' in df.columns and 'helpfulnessDenominator' in df.columns:
            # Check logical consistency
            invalid_helpfulness = (df['helpfulnessNumerator'] > df['helpfulnessDenominator']).sum()
            if invalid_helpfulness > 0:
                penalty += 15
                issues.append(f"{invalid_helpfulness} reviews have helpfulness numerator > denominator")
            
            # Check for negative values
            negative_values = ((df['helpfulnessNumerator'] < 0) | (df['helpfulnessDenominator'] < 0)).sum()
            if negative_values > 0:
                penalty += 10
                warnings.append(f"{negative_values} reviews have negative helpfulness values")
        
        # Sentiment value validation
        if 'sentiment' in df.columns:
            valid_sentiments = {-1, 0, 1}
            invalid_sentiments = df[~df['sentiment'].isin(valid_sentiments) & df['sentiment'].notna()]
            if len(invalid_sentiments) > 0:
                penalty += 20
                issues.append(f"{len(invalid_sentiments)} reviews have invalid sentiment values")
        
        # Review quality thresholds
        if 'text' in df.columns:
            empty_reviews = (df['text'].isnull() | (df['text'].str.strip() == '')).sum()
            if empty_reviews > len(df) * 0.1:  # More than 10% empty
                penalty += 12
                warnings.append(f"{empty_reviews/len(df)*100:.1f}% of reviews have empty text")
        
        return {
            'penalty': penalty,
            'issues': issues,
            'warnings': warnings,
            'logic_checks': {
                'helpfulness_violations': invalid_helpfulness if 'helpfulnessNumerator' in df.columns else 0,
                'sentiment_violations': len(invalid_sentiments) if 'sentiment' in df.columns else 0,
                'empty_text_percentage': (empty_reviews / len(df)) * 100 if 'text' in df.columns else 0
            }
        }
    
    def _analyze_text_summary_relationship(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze relationship between text and summary fields."""
        issues = []
        warnings = []
        penalty = 0
        
        # Check for mismatched text/summary presence
        has_text = df['text'].notna() & (df['text'].str.strip() != '')
        has_summary = df['summary'].notna() & (df['summary'].str.strip() != '')
        
        text_no_summary = (has_text & ~has_summary).sum()
        summary_no_text = (~has_text & has_summary).sum()
        
        if summary_no_text > 0:
            penalty += 8
            warnings.append(f"{summary_no_text} reviews have summary but no text")
        
        # Analyze length relationships where both exist
        both_exist = has_text & has_summary
        if both_exist.sum() > 10:  # Need reasonable sample size
            text_lengths = df.loc[both_exist, 'text'].str.len()
            summary_lengths = df.loc[both_exist, 'summary'].str.len()
            
            # Summary should generally be shorter than text
            summary_longer = (summary_lengths > text_lengths).sum()
            if summary_longer > both_exist.sum() * 0.3:  # More than 30%
                penalty += 5
                warnings.append(f"{summary_longer} summaries are longer than their corresponding text")
        
        return {
            'penalty': penalty,
            'issues': issues,
            'warnings': warnings,
            'statistics': {
                'text_no_summary': text_no_summary,
                'summary_no_text': summary_no_text,
                'both_present': both_exist.sum(),
                'neither_present': (~has_text & ~has_summary).sum()
            }
        }
    
    def _analyze_helpfulness_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze helpfulness voting patterns."""
        warnings = []
        penalty = 0
        
        # Calculate helpfulness ratios
        valid_helpfulness = (df['helpfulnessDenominator'] > 0)
        if valid_helpfulness.sum() == 0:
            return {'penalty': 0, 'warnings': [], 'statistics': {}}
        
        helpfulness_ratios = (df.loc[valid_helpfulness, 'helpfulnessNumerator'] / 
                            df.loc[valid_helpfulness, 'helpfulnessDenominator'])
        
        # Check for unusual patterns
        always_helpful = (helpfulness_ratios == 1.0).sum()
        never_helpful = (helpfulness_ratios == 0.0).sum()
        
        total_with_votes = valid_helpfulness.sum()
        
        if always_helpful > total_with_votes * 0.8:  # More than 80% perfect scores
            penalty += 5
            warnings.append("Unusually high proportion of perfectly helpful reviews")
        
        if never_helpful > total_with_votes * 0.5:  # More than 50% zero scores
            penalty += 3
            warnings.append("High proportion of reviews marked as not helpful")
        
        return {
            'penalty': penalty,
            'warnings': warnings,
            'statistics': {
                'avg_helpfulness_ratio': round(helpfulness_ratios.mean(), 3),
                'median_helpfulness_ratio': round(helpfulness_ratios.median(), 3),
                'perfect_helpfulness_percentage': round(always_helpful / total_with_votes * 100, 1),
                'zero_helpfulness_percentage': round(never_helpful / total_with_votes * 100, 1)
            }
        }
    
    def _analyze_sentiment_helpfulness_relationship(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze relationship between sentiment and helpfulness."""
        warnings = []
        
        # Calculate helpfulness by sentiment
        valid_data = (df['helpfulnessDenominator'] > 0) & df['sentiment'].notna()
        
        if valid_data.sum() < 50:  # Need reasonable sample size
            return {'warnings': [], 'statistics': {}}
        
        sentiment_helpfulness = df.loc[valid_data].groupby('sentiment').apply(
            lambda x: (x['helpfulnessNumerator'] / x['helpfulnessDenominator']).mean()
        )
        
        # Check for unexpected patterns
        if len(sentiment_helpfulness) >= 2:
            helpfulness_variance = sentiment_helpfulness.var()
            if helpfulness_variance < 0.01:  # Very low variance
                warnings.append("Helpfulness ratings show little variation across sentiment levels")
        
        return {
            'warnings': warnings,
            'statistics': {
                'sentiment_helpfulness_correlation': sentiment_helpfulness.to_dict(),
                'helpfulness_variance_by_sentiment': round(helpfulness_variance, 4) if len(sentiment_helpfulness) >= 2 else 0
            }
        }
    
    def _check_temporal_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check temporal data consistency if timestamp data available."""
        # Placeholder for temporal consistency checks
        # Would be implemented if timestamp data is available
        return {'penalty': 0, 'warnings': []}
    
    def _check_id_format_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check ID format consistency across the dataset."""
        warnings = []
        penalty = 0
        
        # Check productId format consistency
        if 'productId' in df.columns:
            product_ids = df['productId'].dropna()
            if len(product_ids) > 0:
                # Check for consistent format patterns
                id_lengths = product_ids.str.len()
                length_variance = id_lengths.var()
                
                if length_variance > 10:  # High variance in ID lengths
                    penalty += 3
                    warnings.append("Inconsistent productId format detected")
        
        # Check userId format consistency
        if 'userId' in df.columns:
            user_ids = df['userId'].dropna()
            if len(user_ids) > 0:
                id_lengths = user_ids.str.len()
                length_variance = id_lengths.var()
                
                if length_variance > 10:
                    penalty += 3
                    warnings.append("Inconsistent userId format detected")
        
        return {'penalty': penalty, 'warnings': warnings}
    
    def _generate_business_recommendations(self, business_checks: Dict[str, Any]) -> List[str]:
        """Generate business-specific recommendations."""
        recommendations = []
        
        # Product coverage recommendations
        if 'product_coverage' in business_checks and not business_checks['product_coverage'].get('skipped', False):
            stats = business_checks['product_coverage']['statistics']
            if stats['products_below_minimum'] > 0:
                recommendations.append(
                    f"Collect more reviews for {stats['products_below_minimum']} products "
                    f"below minimum threshold"
                )
        
        # User activity recommendations
        if 'user_activity' in business_checks and not business_checks['user_activity'].get('skipped', False):
            stats = business_checks['user_activity']['statistics']
            if stats['user_distribution']['casual_user_percentage'] > 80:
                recommendations.append("Develop user engagement strategies to increase review activity")
        
        # Data integrity recommendations
        if 'data_integrity' in business_checks:
            integrity = business_checks['data_integrity'].get('integrity_checks', {})
            if integrity.get('duplicate_analysis', {}).get('duplicate_percentage', 0) > 5:
                recommendations.append("Review and remove duplicate user-product review pairs")
        
        # Business logic recommendations
        if 'business_logic' in business_checks:
            logic = business_checks['business_logic'].get('logic_checks', {})
            if logic.get('helpfulness_violations', 0) > 0:
                recommendations.append("Fix helpfulness data where numerator exceeds denominator")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Business rules validation passed - data meets quality standards")
        
        return recommendations[:5]
    
    def _create_business_statistics(self, business_checks: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Create summary statistics for business validation."""
        return {
            'validation_scope': {
                'checks_performed': len(business_checks),
                'checks_skipped': sum(1 for check in business_checks.values() if check.get('skipped', False)),
                'total_penalties': sum(check.get('penalty', 0) for check in business_checks.values())
            },
            'data_coverage': {
                'has_product_data': 'productId' in df.columns,
                'has_user_data': 'userId' in df.columns,
                'has_helpfulness_data': all(col in df.columns for col in ['helpfulnessNumerator', 'helpfulnessDenominator']),
                'has_text_summary': all(col in df.columns for col in ['text', 'summary'])
            },
            'business_metrics_summary': {
                'products_analyzed': df['productId'].nunique() if 'productId' in df.columns else 0,
                'users_analyzed': df['userId'].nunique() if 'userId' in df.columns else 0,
                'reviews_with_helpfulness': (df['helpfulnessDenominator'] > 0).sum() if 'helpfulnessDenominator' in df.columns else 0
            }
        }