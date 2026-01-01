"""
Baseline text classifier training for complaint classification pipeline.

This module provides a simple, interpretable baseline classifier for binary
text classification. The focus is on simplicity and reproducibility rather
than maximum performance.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression


def train_model(X_train, y_train, random_state: int = 42):
    """
    Train a baseline binary text classifier.
    
    This function trains a simple LogisticRegression model as a baseline.
    The choice of LogisticRegression provides:
    - Interpretability (feature weights can be inspected)
    - Fast training
    - Deterministic results with random_state
    - Good baseline performance for text classification
    
    This is intentionally a simple baseline - not final model selection.
    More sophisticated models can be explored after establishing this baseline.
    
    Args:
        X_train: Feature matrix (numpy array or sparse matrix), shape [n_samples, n_features]
        y_train: Target labels (numpy array), shape [n_samples]
        random_state: Random seed for reproducibility (default: 42)
        
    Returns:
        Fitted LogisticRegression model
        
    Raises:
        TypeError: If X_train or y_train is None
        ValueError: If X_train or y_train is empty, lengths mismatch,
                   insufficient samples, or not binary classification
                   
    Note:
        Only binary classification (2 classes) is supported.
        The model is fitted before being returned.
    """
    # Validate inputs
    if X_train is None:
        raise TypeError("X_train cannot be None")
    
    if y_train is None:
        raise TypeError("y_train cannot be None")
    
    # Convert to numpy arrays if needed for validation
    if hasattr(X_train, 'shape'):
        X_shape = X_train.shape
    else:
        raise ValueError("X_train must be array-like")
    
    if hasattr(y_train, '__len__'):
        y_len = len(y_train)
    else:
        raise ValueError("y_train must be array-like")
    
    # Check for empty inputs
    if X_shape[0] == 0 or (hasattr(X_train, 'size') and X_train.size == 0):
        raise ValueError("X_train cannot be empty")
    
    if y_len == 0:
        raise ValueError("y_train cannot be empty")
    
    # Check length match
    if X_shape[0] != y_len:
        raise ValueError("X_train and y_train must have the same length")
    
    # Check minimum samples
    if X_shape[0] < 2:
        raise ValueError("Need at least 2 samples")
    
    # Check binary classification
    unique_labels = np.unique(y_train)
    n_classes = len(unique_labels)
    
    if n_classes < 2:
        raise ValueError("Need at least 2 classes")
    
    if n_classes > 2:
        raise ValueError("Only binary classification is supported")
    
    # Train baseline model
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(X_train, y_train)
    
    return model
