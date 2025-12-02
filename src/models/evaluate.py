"""
Model Evaluation Module
Calculate and visualize performance metrics.
"""

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(y_true, y_pred, y_proba=None) -> dict:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for AUC)
        
    Returns:
        Dictionary containing:
        - accuracy: Overall accuracy
        - precision: Precision score
        - recall: Recall score
        - f1_score: F1 score
        - roc_auc: ROC AUC score (if y_proba provided)
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    # Calculate ROC AUC if probabilities are provided
    if y_proba is not None:
        try:
            # For binary classification, use probabilities of positive class
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba, 
                                                   multi_class='ovr', 
                                                   average='weighted')
        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC: {e}")
            metrics['roc_auc'] = None
    
    logger.info("Calculated evaluation metrics:")
    for metric, value in metrics.items():
        if value is not None:
            logger.info(f"  {metric}: {value:.4f}")
    
    return metrics


def print_classification_report(y_true, y_pred, target_names=None):
    """
    Print detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Optional list of class names
    """
    report = classification_report(y_true, y_pred, 
                                   target_names=target_names,
                                   zero_division=0)
    
    logger.info("Classification Report:")
    logger.info("\n" + report)
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(report)
    print("="*60 + "\n")


def plot_confusion_matrix(y_true, y_pred, 
                         class_names=None,
                         figsize=(8, 6),
                         save_path=None):
    """
    Visualize confusion matrix as heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional list of class names for labels
        figsize: Figure size (width, height)
        save_path: Optional path to save figure
        
    Returns:
        matplotlib figure object
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to: {save_path}")
    
    return fig


def plot_roc_curve(y_true, y_proba, 
                   figsize=(8, 6),
                   save_path=None):
    """
    Plot ROC curve for binary classification.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        matplotlib figure object
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot ROC curve
    ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved ROC curve to: {save_path}")
    
    return fig


def print_metrics_summary(metrics: dict):
    """
    Print formatted metrics summary.
    
    Args:
        metrics: Dictionary of metric name -> value
    """
    print("\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*60)
    
    for metric_name, value in metrics.items():
        if value is not None:
            if isinstance(value, float):
                print(f"{metric_name.upper():.<40} {value:.4f}")
            else:
                print(f"{metric_name.upper():.<40} {value}")
    
    print("="*60 + "\n")


def evaluate_model(model, X_test, y_test, 
                  class_names=None,
                  plot_cm=True,
                  plot_roc=False,
                  save_dir=None) -> dict:
    """
    Comprehensive model evaluation with metrics and visualizations.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        class_names: Optional class names
        plot_cm: Whether to plot confusion matrix
        plot_roc: Whether to plot ROC curve (binary only)
        save_dir: Optional directory to save plots
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Starting comprehensive model evaluation")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_proba)
    
    # Print summary
    print_metrics_summary(metrics)
    
    # Print classification report
    print_classification_report(y_test, y_pred, target_names=class_names)
    
    # Plot confusion matrix
    if plot_cm:
        cm_path = None
        if save_dir:
            from pathlib import Path
            cm_path = Path(save_dir) / 'confusion_matrix.png'
        plot_confusion_matrix(y_test, y_pred, class_names=class_names, 
                            save_path=cm_path)
    
    # Plot ROC curve for binary classification
    if plot_roc and len(np.unique(y_test)) == 2 and y_proba.shape[1] == 2:
        roc_path = None
        if save_dir:
            from pathlib import Path
            roc_path = Path(save_dir) / 'roc_curve.png'
        plot_roc_curve(y_test, y_proba[:, 1], save_path=roc_path)
    
    logger.info("Evaluation complete")
    
    return metrics