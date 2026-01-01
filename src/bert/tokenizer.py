"""
BERT tokenizer module for complaint classification.

This module provides a lightweight tokenizer wrapper for BERT-based models.
"""
import numpy as np


class BertTokenizer:
    """
    Lightweight BERT tokenizer wrapper.
    
    This tokenizer provides a simple interface for converting text to
    input_ids and attention_mask tensors with fixed max_length.
    """
    
    def __init__(self, max_length: int = 512):
        """
        Initialize tokenizer with max sequence length.
        
        Args:
            max_length: Maximum sequence length for tokenization
        """
        self.max_length = max_length
    
    def tokenize(self, text: str) -> dict:
        """
        Tokenize a single text into input_ids and attention_mask.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask' arrays
            
        Raises:
            ValueError: If text is None or empty
        """
        if text is None:
            raise ValueError("Input text cannot be None")
        
        if text == "":
            raise ValueError("Input text cannot be empty")
        
        # Simple tokenization: convert to character-level IDs
        # In real implementation, this would use a proper tokenizer
        char_ids = [ord(c) % 1000 for c in text[:self.max_length]]
        
        # Pad to max_length
        input_ids = char_ids + [0] * (self.max_length - len(char_ids))
        
        # Attention mask: 1 for real tokens, 0 for padding
        attention_mask = [1] * len(char_ids) + [0] * (self.max_length - len(char_ids))
        
        return {
            "input_ids": np.array(input_ids),
            "attention_mask": np.array(attention_mask)
        }
    
    def tokenize_batch(self, texts: list) -> dict:
        """
        Tokenize a batch of texts.
        
        Args:
            texts: List of input text strings
            
        Returns:
            Dictionary with batched 'input_ids' and 'attention_mask' arrays
            
        Raises:
            ValueError: If batch is empty
        """
        if len(texts) == 0:
            raise ValueError("Batch cannot be empty")
        
        # Tokenize each text
        batch_input_ids = []
        batch_attention_mask = []
        
        for text in texts:
            result = self.tokenize(text)
            batch_input_ids.append(result["input_ids"])
            batch_attention_mask.append(result["attention_mask"])
        
        return {
            "input_ids": np.array(batch_input_ids),
            "attention_mask": np.array(batch_attention_mask)
        }
