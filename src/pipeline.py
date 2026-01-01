"""
End-to-end pipeline orchestration for complaint classification.

This module coordinates all components of the ML pipeline in the correct order
to prevent data leakage and ensure reproducible results.
"""
from src.data_loader import load_complaint_dataset
from src.data_validation import validate_complaint_data
from src.data_splits import validate_labels, split_dataset
from src.text_preprocessing import preprocess_text
from src.vectorizer import fit_vectorizer, transform_texts
from src.model_training import train_model
from src.prediction import predict
from src.evaluation import evaluate_model


def run_pipeline(
    data_source: str = "huggingface",
    test_size: float = 0.2,
    random_state: int = 42
) -> dict:
    """
    Run end-to-end ML pipeline for complaint classification.
    
    This function orchestrates the complete ML workflow from data loading
    through model evaluation. Orchestration is isolated in this module to:
    - Ensure correct execution order
    - Prevent data leakage (fit on train, transform on test)
    - Maintain reproducibility via random_state
    - Provide a single entry point for the pipeline
    
    Pipeline Steps:
    1. Load raw data from specified source
    2. Validate DataFrame schema
    3. Validate labels (binary classification)
    4. Split into train/test sets (stratified)
    5. Preprocess text (lowercase, remove punctuation/digits)
    6. Fit TF-IDF vectorizer on training data ONLY
    7. Transform both train and test using fitted vectorizer
    8. Train baseline classifier on training data
    9. Generate predictions on test data
    10. Evaluate model performance
    
    Leakage Prevention:
    - Vectorizer is fit ONLY on training text
    - Test data never influences model training
    - Preprocessing is deterministic (no data-dependent operations)
    
    Args:
        data_source: Data source identifier (default: "huggingface")
        test_size: Proportion of data for test set (0 < test_size < 1)
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing:
        - "model": Trained model object
        - "metrics": Dictionary of evaluation metrics (accuracy, precision, recall, f1)
        
    Raises:
        ValueError: If data_source is unsupported or test_size is invalid
    """
    # Validate inputs
    if data_source not in ["huggingface"]:
        raise ValueError(f"Unsupported data_source: {data_source}")
    
    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1")
    
    # Step 1: Load raw data
    df = load_complaint_dataset(split='train')
    
    # Step 2: Validate schema
    validate_complaint_data(df)
    
    # Step 3: Validate labels
    validate_labels(df)
    
    # Step 4: Split into train/test
    train_df, test_df = split_dataset(df, test_size=test_size, random_state=random_state)
    
    # Step 5: Preprocess text
    train_texts = [preprocess_text(text) for text in train_df['complaint_text']]
    test_texts = [preprocess_text(text) for text in test_df['complaint_text']]
    
    # Step 6: Fit TF-IDF on train only
    vectorizer = fit_vectorizer(train_texts)
    
    # Step 7: Transform both train and test
    X_train = transform_texts(vectorizer, train_texts)
    X_test = transform_texts(vectorizer, test_texts)
    
    # Extract labels
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    # Step 8: Train baseline model
    model = train_model(X_train, y_train, random_state=random_state)
    
    # Step 9: Generate predictions
    predictions = predict(model, X_test)
    y_pred = predictions["predictions"]
    y_proba = predictions.get("probabilities", None)
    
    # Step 10: Evaluate metrics
    metrics = evaluate_model(y_test, y_pred, y_proba)
    
    # Return model and metrics
    return {
        "model": model,
        "metrics": metrics
    }
