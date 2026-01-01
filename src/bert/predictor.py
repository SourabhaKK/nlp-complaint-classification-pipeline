"""
BERT predictor module for complaint classification.

This module provides prediction interface for trained BERT models.
"""
import numpy as np


class BertPredictor:
    """
    BERT predictor for generating predictions and probabilities.
    
    This predictor provides a simple interface for making predictions
    with trained BERT models.
    """
    
    def __init__(self, model):
        """
        Initialize predictor with trained model.
        
        Args:
            model: Trained BERT model instance
        """
        self.model = model
    
    def predict(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> dict:
        """
        Generate predictions and probabilities.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Dictionary with 'labels' and 'probabilities'
            
        Raises:
            ValueError: If inputs are empty or shapes don't match
        """
        # Validate inputs
        if len(input_ids) == 0 or len(attention_mask) == 0:
            raise ValueError("Input cannot be empty")
        
        if input_ids.shape != attention_mask.shape:
            raise ValueError("Input shapes must match")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Forward pass
        logits = self.model.forward(input_ids, attention_mask)
        
        # Convert logits to probabilities using softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Get predicted labels (argmax)
        labels = np.argmax(probabilities, axis=1)
        
        return {
            "labels": labels,
            "probabilities": probabilities
        }
