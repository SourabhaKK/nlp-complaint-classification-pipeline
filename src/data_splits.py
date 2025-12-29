"""
Label validation and train/test splitting for complaint classification pipeline.
"""
import pandas as pd
from sklearn.model_selection import train_test_split


def validate_labels(df: pd.DataFrame, label_col: str = "label") -> None:
    """
    Validate that labels are integers, contiguous, and start from 0.
    
    Args:
        df: DataFrame to validate
        label_col: Name of label column (default: "label")
        
    Raises:
        ValueError: If label column is missing, contains nulls, is not integers,
                   or labels are not contiguous starting from 0
    """
    # Check if label column exists
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found")
    
    # Check for null values
    if df[label_col].isnull().any():
        raise ValueError("Label column contains null values")
    
    # Check if labels are integers
    if not pd.api.types.is_integer_dtype(df[label_col]):
        raise ValueError("Labels must be integers")
    
    # Check if labels are contiguous starting from 0
    unique_labels = sorted(df[label_col].unique())
    expected_labels = list(range(len(unique_labels)))
    
    if unique_labels != expected_labels:
        raise ValueError("Labels must be contiguous starting from 0")


def split_dataset(
    df: pd.DataFrame,
    label_col: str = "label",
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into stratified train and test sets.
    
    Args:
        df: DataFrame to split
        label_col: Column name for stratification (default: "label")
        test_size: Proportion for test set, 0.0 to 1.0 (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        
    Returns:
        Tuple of (train_df, test_df)
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_col]
    )
    
    return train_df, test_df
