"""
Deterministic text preprocessing for complaint classification pipeline.

This module provides simple character-level text cleaning without
linguistic transformations (no stemming, lemmatization, or tokenization).
"""
import string
import re


def preprocess_text(text: str) -> str:
    """
    Preprocess text deterministically using character-level operations.
    
    Performs the following operations:
    1. Validates input type
    2. Converts to lowercase
    3. Removes punctuation
    4. Removes digits
    5. Normalizes whitespace (tabs, newlines, multiple spaces)
    
    Args:
        text: Input text to preprocess
        
    Returns:
        Preprocessed text with only lowercase alphabetic characters and single spaces
        
    Raises:
        ValueError: If text is None
        TypeError: If text is not a string
        
    Note:
        This function performs no linguistic transformations such as stemming,
        lemmatization, or tokenization. It is purely character-based and deterministic.
    """
    # Validate input
    if text is None:
        raise ValueError("Input text cannot be None")
    
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove digits
    text = text.translate(str.maketrans('', '', string.digits))
    
    # Normalize whitespace (replace tabs, newlines, multiple spaces with single space)
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading and trailing whitespace
    text = text.strip()
    
    return text
