"""
Hugging Face dataset loader for complaint classification pipeline.

This loader is intended for development and reproducibility only.
In production, datasets should be frozen to local storage.
"""
import pandas as pd
import datasets


def load_complaint_dataset(split: str = "train") -> pd.DataFrame:
    """
    Load complaint dataset from Hugging Face.
    
    Args:
        split: Dataset split to load ('train', 'test', 'validation')
        
    Returns:
        pd.DataFrame with 'text' and 'label' columns
    """
    dataset = datasets.load_dataset("complaint-dataset", split=split)
    return dataset.to_pandas()
