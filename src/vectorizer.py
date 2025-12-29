"""
TF-IDF vectorization with leakage prevention for complaint classification pipeline.

This module provides a wrapper around sklearn's TfidfVectorizer to ensure
proper fit/transform separation and prevent data leakage.
"""
from sklearn.feature_extraction.text import TfidfVectorizer


def fit_vectorizer(texts: list[str]) -> TfidfVectorizer:
    """
    Fit a TF-IDF vectorizer on training texts.
    
    This function should ONLY be called on training data. The fitted vectorizer
    learns the vocabulary and IDF weights from the training set, which must not
    include test data to prevent leakage.
    
    Args:
        texts: List of text strings to fit the vectorizer on
        
    Returns:
        Fitted TfidfVectorizer object
        
    Raises:
        ValueError: If texts is None or empty
        TypeError: If texts is not a list or contains non-string elements
        
    Note:
        The vectorizer uses deterministic defaults to ensure reproducibility.
        Vocabulary is frozen after fitting and will not change during transform.
    """
    # Validate input
    if texts is None:
        raise ValueError("Input texts cannot be None")
    
    if not isinstance(texts, list):
        raise TypeError("Input must be a list")
    
    if len(texts) == 0:
        raise ValueError("Input texts cannot be empty")
    
    # Check all elements are strings
    for item in texts:
        if not isinstance(item, str):
            raise TypeError("All elements must be strings")
    
    # Create and fit vectorizer with deterministic defaults
    vectorizer = TfidfVectorizer()
    vectorizer.fit(texts)
    
    return vectorizer


def transform_texts(vectorizer: TfidfVectorizer, texts: list[str]):
    """
    Transform texts using a fitted vectorizer.
    
    This function applies the vocabulary and IDF weights learned during fitting
    to transform new texts into TF-IDF feature vectors. It does NOT refit the
    vectorizer, ensuring that test data does not leak into the model.
    
    Args:
        vectorizer: Fitted TfidfVectorizer object from fit_vectorizer()
        texts: List of text strings to transform
        
    Returns:
        Sparse matrix of TF-IDF features (shape: [n_samples, n_features])
        
    Raises:
        ValueError: If vectorizer is None or texts is empty
        
    Note:
        Unseen words in the input texts will be ignored (not added to vocabulary).
        This is correct behavior to prevent leakage - the vocabulary is frozen
        at fit time and represents only the training data.
    """
    # Validate input
    if vectorizer is None:
        raise ValueError("Vectorizer cannot be None")
    
    if not isinstance(texts, list) or len(texts) == 0:
        raise ValueError("Input texts cannot be empty")
    
    # Transform using the fitted vectorizer (does NOT refit)
    result = vectorizer.transform(texts)
    
    return result
