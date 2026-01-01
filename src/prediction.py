"""
Prediction interface for trained models.

This module provides a simple wrapper around sklearn-like models to generate
predictions with a consistent output format.
"""
import numpy as np


def predict(model, X) -> dict:
    """
    Generate predictions using a trained model.
    
    This function wraps model.predict() and optionally model.predict_proba()
    to provide a consistent dictionary output format. It handles models both
    with and without probability support.
    
    Args:
        model: Trained sklearn-like model with .predict() method
        X: Feature matrix (numpy array or sparse matrix), shape [n_samples, n_features]
        
    Returns:
        Dictionary containing:
        - "predictions": Array of predicted class labels
        - "probabilities": Array of class probabilities (only if model supports predict_proba)
        
    Raises:
        ValueError: If model is None or X is empty
        TypeError: If X is None or model lacks predict method
        
    Note:
        Probabilities are optional - they are only included if the model
        exposes a predict_proba() method. This allows the interface to work
        with both probabilistic models (LogisticRegression, RandomForest) and
        non-probabilistic models (SVC with default kernel).
    """
    # Validate model
    if model is None:
        raise ValueError("Model cannot be None")
    
    if not hasattr(model, 'predict'):
        raise TypeError("Model must have a predict method")
    
    # Validate X
    if X is None:
        raise TypeError("Input X cannot be None")
    
    # Check if X is empty
    if hasattr(X, 'shape'):
        if X.shape[0] == 0 or (hasattr(X, 'size') and X.size == 0):
            raise ValueError("Input X cannot be empty")
    elif hasattr(X, '__len__'):
        if len(X) == 0:
            raise ValueError("Input X cannot be empty")
    
    # Generate predictions
    predictions = model.predict(X)
    
    # Build result dictionary
    result = {
        "predictions": predictions
    }
    
    # Add probabilities if model supports it
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)
        result["probabilities"] = probabilities
    
    return result
