"""
Data validation module for complaint classification pipeline.
"""
import pandas as pd


def validate_complaint_data(df: pd.DataFrame) -> bool:
    """
    Validate complaint DataFrame for required structure and data quality.
    
    Args:
        df: DataFrame to validate with 'complaint_text' and 'label' columns
        
    Returns:
        True if validation passes
        
    Raises:
        ValueError: If validation fails with descriptive message
    """
    # Check if DataFrame is None
    if df is None:
        raise ValueError("DataFrame cannot be None")
    
    # Check if DataFrame is empty
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    # Check for required columns
    if "complaint_text" not in df.columns:
        raise ValueError("Missing required column: 'complaint_text'")
    
    if "label" not in df.columns:
        raise ValueError("Missing required column: 'label'")
    
    # Check for null values in complaint_text
    if df["complaint_text"].isnull().any():
        raise ValueError("'complaint_text' contains null values")
    
    # Check for empty or whitespace-only strings in complaint_text
    if (df["complaint_text"].str.strip() == "").any():
        raise ValueError("'complaint_text' contains empty strings")
    
    # Check for null values in label
    if df["label"].isnull().any():
        raise ValueError("'label' contains null values")
    
    return True
