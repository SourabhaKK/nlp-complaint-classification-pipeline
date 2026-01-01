"""
Evaluation metrics for binary classification models.

This module provides standard classification metrics to evaluate model
performance on held-out test sets.
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Optional


def evaluate_model(y_true, y_pred, y_proba: Optional[object] = None) -> dict:
    """
    Evaluate binary classification model performance.
    
    Computes standard classification metrics on predictions. Evaluation is
    separated from prediction to allow flexible model assessment without
    retraining or re-predicting.
    
    Metrics computed:
    - accuracy: Proportion of correct predictions (TP + TN) / (TP + TN + FP + FN)
    - precision: Proportion of positive predictions that are correct TP / (TP + FP)
    - recall: Proportion of actual positives correctly identified TP / (TP + FN)
    - f1: Harmonic mean of precision and recall 2 * (precision * recall) / (precision + recall)
    - roc_auc: Area under ROC curve (only if y_proba provided)
    
    Args:
        y_true: True labels (numpy array), shape [n_samples]
        y_pred: Predicted labels (numpy array), shape [n_samples]
        y_proba: Predicted probabilities (numpy array or None), shape [n_samples, n_classes]
                 If provided, ROC-AUC will be computed. If None, ROC-AUC is omitted.
        
    Returns:
        Dictionary containing metric names and values (all floats between 0 and 1)
        
    Raises:
        TypeError: If y_true or y_pred is None
        ValueError: If y_true or y_pred is empty, or if lengths don't match
        
    Note:
        ROC-AUC is only computed when y_proba is provided. This allows the
        function to work with both probabilistic models (LogisticRegression)
        and non-probabilistic models (SVC with default kernel).
    """
    # Validate inputs
    if y_true is None:
        raise TypeError("y_true cannot be None")
    
    if y_pred is None:
        raise TypeError("y_pred cannot be None")
    
    # Convert to numpy arrays for validation
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Check for empty inputs
    if len(y_true) == 0:
        raise ValueError("y_true cannot be empty")
    
    if len(y_pred) == 0:
        raise ValueError("y_pred cannot be empty")
    
    # Check length match
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    # Compute metrics
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0))
    }
    
    # Add ROC-AUC if probabilities are provided
    if y_proba is not None:
        # Extract probabilities for positive class (column 1)
        if hasattr(y_proba, 'shape') and len(y_proba.shape) == 2:
            y_proba_positive = y_proba[:, 1]
        else:
            y_proba_positive = y_proba
        
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba_positive))
    
    return metrics
