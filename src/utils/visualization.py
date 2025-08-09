"""
Visualization Module for Intelligent Review Analytics Platform

This module provides comprehensive visualization capabilities for business intelligence,
model performance analysis, and stakeholder communication with interactive charts.

Author: [Your Name]
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Set style defaults
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ReviewAnalyticsVisualizer:
    """
    Comprehensive visualization toolkit for review analytics platform.
    
    Creates publication-ready charts for business intelligence reports,
    model performance analysis, and stakeholder presentations.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize visualizer with configuration.
        
        Args:
            config (Dict, optional): Visualization configuration
        """
        self.config = config or self._get_default_config()
        self.color_palette = self.config['colors']
        
        logger.info("ReviewAnalyticsVisualizer initialized")
    
    def _get_default_config(self) -> Dict:
        """Get default visualization configuration."""
        return {
            'figure_size': (12, 8),
            'dpi': 300,
            'style': 'whitegrid',
            'colors': {
                'primary': '#1f77b4',
                'secondary': '#ff7f0e',
                'success': '#2ca02c',
                'warning': '#d62728',
                'info': '#17becf',
                'neutral': '#7f7f7f',
                'accent': '#bcbd22',
                'dark': '#8c564b'
            },
            'font_sizes': {
                'title': 16,
                'subtitle': 14,
                'axis_label': 12,
                'tick_label': 10,
                'legend': 11
            }
        }
    
    def create_sentiment_distribution_chart(self, df: pd.DataFrame, 
                                          interactive: bool = True) -> Union[go.Figure, plt.Figure]:
        """
        Create sentiment distribution visualization.
        
        Args:
            df (pd.DataFrame): DataFrame with sentiment column
            interactive (bool): Create interactive Plotly chart
            
        Returns:
            Union[go.Figure, plt.Figure]: Chart figure
        """
        sentiment_counts = df['sentiment'].value_counts()
        sentiment_labels = {1: 'Positive', -1: 'Negative', 0: 'Neutral'}
        
        if interactive:
            # Interactive Plotly chart
            labels = [sentiment_labels.get(k, f'Sentiment {k}') for k in sentiment_counts.index]
            values = sentiment_counts.values
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                marker_colors=[self.color_palette['success'] if 'Positive' in label 
                             else self.color_palette['warning'] if 'Negative' in label 
                             else self.color_palette['neutral'] for label in labels]
            )])
            
            fig.update_layout(
                title={
                    'text': 'Review Sentiment Distribution',
                    'x': 0.5,
                    'font': {'size': self.config['font_sizes']['title']}
                },
                annotations=[dict(text=f'Total<br>{len(df):,}', x=0.5, y=0.5, 
                                font_size=14, showarrow=False)],
                showlegend=True,
                width=600,
                height=500
            )
            
            return fig
        
        else:
            # Static matplotlib chart
            fig, ax = plt.subplots(figsize=(8, 6))
            
            labels = [sentiment_labels.get(k, f'Sentiment {k}') for k in sentiment_counts.index]
            colors = [self.color_palette['success'] if 'Positive' in label 
                     else self.color_palette['warning'] if 'Negative' in label 
                     else self.color_palette['neutral'] for label in labels]
            
            wedges, texts, autotexts = ax.pie(sentiment_counts.values, labels=labels, 
                                            colors=colors, autopct='%1.1f%%', startangle=90)
            
            ax.set_title('Review Sentiment Distribution', 
                        fontsize=self.config['font_sizes']['title'], pad=20)
            
            return fig
    
    def create_category_risk_heatmap(self, category_analysis: Dict[str, Any]) -> go.Figure:
        """
        Create product category risk assessment heatmap.
        
        Args:
            category_analysis (Dict): Results from BusinessAnalyzer.analyze_product_categories
            
        Returns:
            go.Figure: Interactive heatmap
        """
        if 'category_stats' not in category_analysis:
            return self._create_error_figure("No category statistics available")
        
        df_stats = pd.DataFrame(category_analysis['category_stats'])
        
        # Prepare data for heatmap
        risk_data = df_stats.pivot_table(
            index='product_category',
            values=['negative_rate', 'review_count', 'sentiment_std'],
            aggfunc='first'
        ).fillna(0)
        
        # Normalize data for better visualization
        risk_data_norm = (risk_data - risk_data.min()) / (risk_data.max() - risk_data.min())
        
        fig = go.Figure(data=go.Heatmap(
            z=risk_data_norm.values,
            x=['Negative Rate', 'Review Volume', 'Sentiment Variance'],
            y=risk_data.index,
            colorscale='RdYlBu_r',
            colorbar=dict(title="Risk Level"),
            hoverongaps=False,
            text=risk_data.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title='Product Category Risk Assessment Matrix',
            xaxis_title='Risk Metrics',
            yaxis_title='Product Categories',
            height=max(400, len(risk_data) * 30),
            width=800
        )
        
        return fig
    
    def create_user_segment_analysis(self, user_analysis: Dict[str, Any]) -> go.Figure:
        """
        Create user segmentation analysis visualization.
        
        Args:
            user_analysis (Dict): Results from BusinessAnalyzer.analyze_user_behavior
            
        Returns:
            go.Figure: User segment analysis chart
        """
        if 'segment_analysis' not in user_analysis:
            return self._create_error_figure("No user segment data available")
        
        segment_data = pd.DataFrame(user_analysis['segment_analysis'])
        
        # Create subplot with multiple charts
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('User Distribution', 'Review Contribution', 
                          'Average Reviews per User', 'Helpfulness by Segment'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # User distribution
        fig.add_trace(go.Bar(
            x=segment_data['user_segment'],
            y=segment_data['user_percentage'],
            name='User %',
            marker_color=self.color_palette['primary']
        ), row=1, col=1)
        
        # Review contribution
        fig.add_trace(go.Bar(
            x=segment_data['user_segment'],
            y=segment_data['review_contribution'],
            name='Review %',
            marker_color=self.color_palette['secondary']
        ), row=1, col=2)
        
        # Average reviews per user
        fig.add_trace(go.Bar(
            x=segment_data['user_segment'],
            y=segment_data['avg_reviews'],
            name='Avg Reviews',
            marker_color=self.color_palette['success']
        ), row=2, col=1)
        
        # Helpfulness by segment
        fig.add_trace(go.Bar(
            x=segment_data['user_segment'],
            y=segment_data['avg_helpfulness'],
            name='Helpfulness',
            marker_color=self.color_palette['info']
        ), row=2, col=2)
        
        fig.update_layout(
            title_text='User Behavior Segmentation Analysis',
            showlegend=False,
            height=800,
            width=1000
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Percentage (%)", row=1, col=1)
        fig.update_yaxes(title_text="Percentage (%)", row=1, col=2)
        fig.update_yaxes(title_text="Average Count", row=2, col=1)
        fig.update_yaxes(title_text="Helpfulness Ratio", row=2, col=2)
        
        return fig
    
    def create_model_performance_comparison(self, model_results: Dict[str, Dict]) -> go.Figure:
        """
        Create comprehensive model performance comparison chart.
        
        Args:
            model_results (Dict): Results from multiple models
            
        Returns:
            go.Figure: Model comparison chart
        """
        # Extract metrics from model results
        models = list(model_results.keys())
        metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'auc_roc']
        
        # Create data matrix
        performance_data = []
        for model in models:
            model_metrics = []
            for metric in metrics:
                # Try to extract metric from various possible locations
                value = self._extract_metric_value(model_results[model], metric)
                model_metrics.append(value)
            performance_data.append(model_metrics)
        
        # Create radar chart
        fig = go.Figure()
        
        colors = [self.color_palette['primary'], self.color_palette['secondary'], 
                 self.color_palette['success'], self.color_palette['warning'], 
                 self.color_palette['info']]
        
        for i, model in enumerate(models):
            fig.add_trace(go.Scatterpolar(
                r=performance_data[i] + [performance_data[i][0]],  # Close the polygon
                theta=metrics + [metrics[0]],
                fill='toself',
                name=model.replace('_', ' ').title(),
                line_color=colors[i % len(colors)],
                fillcolor=colors[i % len(colors)],
                opacity=0.3
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickformat='.2f'
                )),
            showlegend=True,
            title="Model Performance Comparison (All Metrics)",
            width=700,
            height=600
        )
        
        return fig
    
    def create_feature_importance_chart(self, feature_importance: List[Tuple[str, float]], 
                                      top_k: int = 15) -> go.Figure:
        """
        Create feature importance visualization.
        
        Args:
            feature_importance (List[Tuple]): List of (feature_name, importance_score)
            top_k (int): Number of top features to display
            
        Returns:
            go.Figure: Feature importance chart
        """
        # Take top K features
        top_features = feature_importance[:top_k]
        feature_names, importance_scores = zip(*top_features)
        
        # Clean feature names for better display
        clean_names = [self._clean_feature_name(name) for name in feature_names]
        
        fig = go.Figure(data=[go.Bar(
            x=importance_scores,
            y=clean_names,
            orientation='h',
            marker_color=self.color_palette['primary'],
            text=[f'{score:.3f}' for score in importance_scores],
            textposition='auto'
        )])
        
        fig.update_layout(
            title=f'Top {top_k} Most Important Features',
            xaxis_title='Feature Importance Score',
            yaxis_title='Features',
            height=max(400, top_k * 25),
            width=800,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    def create_roi_analysis_dashboard(self, roi_analysis: Dict[str, Any]) -> go.Figure:
        """
        Create ROI analysis dashboard with multiple metrics.
        
        Args:
            roi_analysis (Dict): Results from BusinessAnalyzer.calculate_roi_metrics
            
        Returns:
            go.Figure: ROI dashboard
        """
        if 'cost_analysis' not in roi_analysis or 'time_analysis' not in roi_analysis:
            return self._create_error_figure("ROI analysis data not available")
        
        cost_data = roi_analysis['cost_analysis']
        time_data = roi_analysis['time_analysis']
        roi_data = roi_analysis.get('roi_metrics', {})
        
        # Create subplot dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Annual Cost Savings', 'Time Reduction Impact', 
                          'ROI Metrics', 'Payback Analysis'),
            specs=[[{'type': 'bar'}, {'type': 'indicator'}],
                   [{'type': 'indicator'}, {'type': 'bar'}]]
        )
        
        # Annual cost savings breakdown
        cost_categories = ['Labor Savings', 'Infrastructure Cost', 'Maintenance Cost', 'Net Savings']
        cost_values = [
            cost_data.get('annual_labor_cost_savings', 0),
            -cost_data.get('annual_infrastructure_cost', 0),
            -cost_data.get('annual_maintenance_cost', 0),
            cost_data.get('net_annual_savings', 0)
        ]
        
        colors = [self.color_palette['success'], self.color_palette['warning'], 
                 self.color_palette['warning'], self.color_palette['primary']]
        
        fig.add_trace(go.Bar(
            x=cost_categories,
            y=cost_values,
            marker_color=colors,
            name='Cost Analysis'
        ), row=1, col=1)
        
        # Time reduction indicator
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=time_data.get('time_reduction_percentage', 0),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Time Reduction %"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': self.color_palette['success']},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ), row=1, col=2)
        
        # ROI indicator
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=roi_data.get('roi_percentage', 0),
            title={'text': "ROI %"},
            number={'suffix': "%"},
            delta={'reference': 100, 'relative': True}
        ), row=2, col=1)
        
        # Payback timeline
        payback_months = roi_data.get('payback_period_months', 12)
        timeline_data = [payback_months, 12, 24, 36]
        timeline_labels = ['Payback Period', '1 Year', '2 Years', '3 Years']
        
        fig.add_trace(go.Bar(
            x=timeline_labels,
            y=timeline_data,
            marker_color=[self.color_palette['info'], self.color_palette['neutral'], 
                         self.color_palette['neutral'], self.color_palette['neutral']],
            name='Timeline'
        ), row=2, col=2)
        
        fig.update_layout(
            title_text='ROI Analysis Dashboard',
            showlegend=False,
            height=800,
            width=1000
        )
        
        return fig
    
    def create_quality_optimization_chart(self, quality_analysis: Dict[str, Any]) -> go.Figure:
        """
        Create review quality optimization visualization.
        
        Args:
            quality_analysis (Dict): Results from BusinessAnalyzer.analyze_review_quality
            
        Returns:
            go.Figure: Quality optimization chart
        """
        if 'length_analysis' not in quality_analysis:
            return self._create_error_figure("Quality analysis data not available")
        
        length_data = quality_analysis['length_analysis']
        
        if 'length_distribution' not in length_data:
            return self._create_error_figure("Length distribution data not available")
        
        length_df = pd.DataFrame(length_data['length_distribution'])
        
        # Create scatter plot with trend line
        fig = go.Figure()
        
        # Add scatter points
        fig.add_trace(go.Scatter(
            x=length_df['length_category'],
            y=length_df['avg_helpfulness'],
            mode='markers+lines',
            marker=dict(
                size=length_df['review_count']/10,  # Scale marker size by review count
                color=self.color_palette['primary'],
                opacity=0.7,
                line=dict(width=2, color='DarkSlateGrey')
            ),
            text=[f"Reviews: {count}" for count in length_df['review_count']],
            name='Helpfulness by Length'
        ))
        
        # Add optimal range indicator
        optimal_length = quality_analysis.get('optimal_characteristics', {})
        if 'word_count_range' in optimal_length:
            word_range = optimal_length['word_count_range']
            fig.add_shape(
                type="rect",
                x0=1.5, x1=2.5,  # Approximate position for "Medium" category
                y0=0, y1=1,
                fillcolor=self.color_palette['success'],
                opacity=0.2,
                line_width=0,
            )
            
            fig.add_annotation(
                x=2, y=0.9,
                text=f"Optimal Range<br>{word_range['min']}-{word_range['max']} words",
                showarrow=True,
                arrowhead=1,
                bgcolor="white",
                bordercolor=self.color_palette['success']
            )
        
        fig.update_layout(
            title='Review Length vs Helpfulness Optimization',
            xaxis_title='Review Length Category',
            yaxis_title='Average Helpfulness Ratio',
            height=500,
            width=800,
            showlegend=True
        )
        
        return fig
    
    def create_executive_summary_dashboard(self, executive_summary: Dict[str, Any]) -> go.Figure:
        """
        Create executive summary dashboard for stakeholders.
        
        Args:
            executive_summary (Dict): Executive summary from BusinessAnalyzer
            
        Returns:
            go.Figure: Executive dashboard
        """
        insights = executive_summary.get('key_insights', [])
        
        # Priority distribution
        priority_counts = {}
        for insight in insights:
            priority = insight.get('priority', 'UNKNOWN')
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        # Create dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Priority Distribution', 'Implementation Timeline', 
                          'Key Metrics Summary', 'Impact Areas'),
            specs=[[{'type': 'pie'}, {'type': 'bar'}],
                   [{'type': 'indicator'}, {'type': 'bar'}]]
        )
        
        # Priority distribution pie chart
        if priority_counts:
            fig.add_trace(go.Pie(
                labels=list(priority_counts.keys()),
                values=list(priority_counts.values()),
                hole=0.3,
                marker_colors=[self.color_palette['warning'] if p == 'HIGH' 
                             else self.color_palette['secondary'] if p == 'MEDIUM'
                             else self.color_palette['neutral'] for p in priority_counts.keys()]
            ), row=1, col=1)
        
        # Implementation timeline
        impl_plan = executive_summary.get('implementation_priority', [])
        if impl_plan:
            phases = [phase['phase'] for phase in impl_plan]
            timelines = [int(phase['timeline'].split('-')[0]) for phase in impl_plan]
            
            fig.add_trace(go.Bar(
                x=phases,
                y=timelines,
                marker_color=self.color_palette['primary'],
                name='Timeline (weeks)'
            ), row=1, col=2)
        
        # Key metrics indicator
        total_insights = executive_summary.get('executive_summary', {}).get('total_insights', 0)
        fig.add_trace(go.Indicator(
            mode="number",
            value=total_insights,
            title={'text': "Total Insights Generated"},
            number={'font': {'size': 40}}
        ), row=2, col=1)
        
        # Impact areas
        impact_areas = ['Product Quality', 'User Engagement', 'Cost Savings', 'Process Efficiency']
        impact_values = [len([i for i in insights if 'product' in i.get('impact', '').lower()]),
                        len([i for i in insights if 'user' in i.get('impact', '').lower() or 'customer' in i.get('impact', '').lower()]),
                        len([i for i in insights if 'cost' in i.get('impact', '').lower() or 'savings' in i.get('impact', '').lower()]),
                        len([i for i in insights if 'efficiency' in i.get('impact', '').lower() or 'process' in i.get('impact', '').lower()])]
        
        fig.add_trace(go.Bar(
            x=impact_areas,
            y=impact_values,
            marker_color=self.color_palette['success'],
            name='Impact Count'
        ), row=2, col=2)
        
        fig.update_layout(
            title_text='Executive Summary Dashboard',
            showlegend=False,
            height=800,
            width=1000
        )
        
        return fig
    
    def save_figure(self, fig: Union[go.Figure, plt.Figure], 
                   filepath: str, format: str = 'png') -> None:
        """
        Save figure to file with proper formatting.
        
        Args:
            fig (Union[go.Figure, plt.Figure]): Figure to save
            filepath (str): Output file path
            format (str): Output format (png, pdf, html, svg)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(fig, go.Figure):
            # Plotly figure
            if format.lower() == 'html':
                fig.write_html(str(filepath.with_suffix('.html')))
            elif format.lower() == 'pdf':
                fig.write_image(str(filepath.with_suffix('.pdf')))
            elif format.lower() == 'svg':
                fig.write_image(str(filepath.with_suffix('.svg')))
            else:
                fig.write_image(str(filepath.with_suffix('.png')), 
                               width=1200, height=800, scale=2)
        else:
            # Matplotlib figure
            fig.savefig(str(filepath.with_suffix(f'.{format}')), 
                       dpi=self.config['dpi'], bbox_inches='tight')
        
        logger.info(f"Figure saved to {filepath}")
    
    # Helper methods
    def _extract_metric_value(self, model_result: Dict, metric: str) -> float:
        """Extract metric value from model results."""
        # Try different possible locations for the metric
        locations = [
            model_result.get('validation_metrics', {}),
            model_result.get('training_metrics', {}),
            model_result.get('performance_summary', {}),
            model_result
        ]
        
        for location in locations:
            if metric in location:
                return location[metric]
        
        return 0.0  # Default value if not found
    
    def _clean_feature_name(self, feature_name: str) -> str:
        """Clean feature name for better display."""
        # Remove prefixes like 'tfidf_', 'text_stat_', etc.
        prefixes = ['tfidf_', 'text_stat_', 'custom_', 'ngram_']
        for prefix in prefixes:
            if feature_name.startswith(prefix):
                feature_name = feature_name[len(prefix):]
        
        # Limit length and add ellipsis if needed
        if len(feature_name) > 20:
            feature_name = feature_name[:17] + '...'
        
        return feature_name
    
    def _create_error_figure(self, error_message: str) -> go.Figure:
        """Create error figure when data is not available."""
        fig = go.Figure()
        fig.add_annotation(
            text=f"Data not available: {error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            title="Visualization Error",
            height=400,
            width=600,
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        return fig

# Convenience functions
def create_performance_comparison(model_results: Dict[str, Dict]) -> go.Figure:
    """Create model performance comparison chart."""
    visualizer = ReviewAnalyticsVisualizer()
    return visualizer.create_model_performance_comparison(model_results)

def create_business_dashboard(business_results: Dict[str, Any]) -> go.Figure:
    """Create comprehensive business intelligence dashboard."""
    visualizer = ReviewAnalyticsVisualizer()
    return visualizer.create_executive_summary_dashboard(business_results)