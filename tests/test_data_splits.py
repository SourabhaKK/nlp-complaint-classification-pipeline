"""
Tests for label validation and train/test splitting.
"""
import pytest
import pandas as pd
import numpy as np
from src.data_splits import validate_labels, split_dataset


class TestLabelValidation:
    """Test suite for label validation."""

    def test_validate_labels_missing_column(self):
        """Test that validation fails when label column is missing."""
        df = pd.DataFrame({
            "text": ["Sample 1", "Sample 2"],
            "other_col": [1, 2]
        })
        with pytest.raises(ValueError, match="Label column 'label' not found"):
            validate_labels(df, label_col="label")

    def test_validate_labels_custom_column_missing(self):
        """Test that validation fails when custom label column is missing."""
        df = pd.DataFrame({
            "text": ["Sample 1", "Sample 2"],
            "label": [0, 1]
        })
        with pytest.raises(ValueError, match="Label column 'category' not found"):
            validate_labels(df, label_col="category")

    def test_validate_labels_contains_nulls(self):
        """Test that validation fails when labels contain null values."""
        df = pd.DataFrame({
            "text": ["Sample 1", "Sample 2", "Sample 3"],
            "label": [0, None, 1]
        })
        with pytest.raises(ValueError, match="Label column contains null values"):
            validate_labels(df)

    def test_validate_labels_contains_nan(self):
        """Test that validation fails when labels contain NaN values."""
        df = pd.DataFrame({
            "text": ["Sample 1", "Sample 2", "Sample 3"],
            "label": [0, np.nan, 1]
        })
        with pytest.raises(ValueError, match="Label column contains null values"):
            validate_labels(df)

    def test_validate_labels_not_integers(self):
        """Test that validation fails when labels are not integers."""
        df = pd.DataFrame({
            "text": ["Sample 1", "Sample 2"],
            "label": ["class_a", "class_b"]
        })
        with pytest.raises(ValueError, match="Labels must be integers"):
            validate_labels(df)

    def test_validate_labels_float_values(self):
        """Test that validation fails when labels are floats."""
        df = pd.DataFrame({
            "text": ["Sample 1", "Sample 2"],
            "label": [0.5, 1.5]
        })
        with pytest.raises(ValueError, match="Labels must be integers"):
            validate_labels(df)

    def test_validate_labels_not_contiguous(self):
        """Test that validation fails when labels are not contiguous."""
        df = pd.DataFrame({
            "text": ["Sample 1", "Sample 2", "Sample 3"],
            "label": [0, 2, 0]  # Missing label 1
        })
        with pytest.raises(ValueError, match="Labels must be contiguous starting from 0"):
            validate_labels(df)

    def test_validate_labels_not_starting_from_zero(self):
        """Test that validation fails when labels don't start from 0."""
        df = pd.DataFrame({
            "text": ["Sample 1", "Sample 2", "Sample 3"],
            "label": [1, 2, 3]  # Should start from 0
        })
        with pytest.raises(ValueError, match="Labels must be contiguous starting from 0"):
            validate_labels(df)

    def test_validate_labels_valid_binary(self):
        """Test that validation passes for valid binary labels."""
        df = pd.DataFrame({
            "text": ["Sample 1", "Sample 2", "Sample 3"],
            "label": [0, 1, 0]
        })
        # Should not raise any exception
        validate_labels(df)

    def test_validate_labels_valid_multiclass(self):
        """Test that validation passes for valid multiclass labels."""
        df = pd.DataFrame({
            "text": ["S1", "S2", "S3", "S4", "S5"],
            "label": [0, 1, 2, 0, 1]
        })
        # Should not raise any exception
        validate_labels(df)

    def test_validate_labels_custom_column_valid(self):
        """Test that validation works with custom label column name."""
        df = pd.DataFrame({
            "text": ["Sample 1", "Sample 2"],
            "category": [0, 1]
        })
        # Should not raise any exception
        validate_labels(df, label_col="category")


class TestDatasetSplit:
    """Test suite for train/test splitting."""

    def test_split_returns_two_dataframes(self):
        """Test that split_dataset returns two DataFrames."""
        df = pd.DataFrame({
            "text": ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"],
            "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })
        train, test = split_dataset(df)
        
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)

    def test_split_sizes_sum_to_original(self):
        """Test that train + test sizes equal original dataset size."""
        df = pd.DataFrame({
            "text": [f"Sample {i}" for i in range(100)],
            "label": [i % 3 for i in range(100)]
        })
        train, test = split_dataset(df, test_size=0.2)
        
        assert len(train) + len(test) == len(df)

    def test_split_default_test_size(self):
        """Test that default test_size is 0.2 (20%)."""
        df = pd.DataFrame({
            "text": [f"Sample {i}" for i in range(100)],
            "label": [i % 2 for i in range(100)]
        })
        train, test = split_dataset(df)
        
        assert len(test) == 20
        assert len(train) == 80

    def test_split_custom_test_size(self):
        """Test that custom test_size is respected."""
        df = pd.DataFrame({
            "text": [f"Sample {i}" for i in range(100)],
            "label": [i % 2 for i in range(100)]
        })
        train, test = split_dataset(df, test_size=0.3)
        
        assert len(test) == 30
        assert len(train) == 70

    def test_split_preserves_class_distribution(self):
        """Test that split uses stratified sampling to preserve class distribution."""
        df = pd.DataFrame({
            "text": [f"Sample {i}" for i in range(100)],
            "label": [0] * 70 + [1] * 30  # 70% class 0, 30% class 1
        })
        train, test = split_dataset(df, test_size=0.2)
        
        # Check train distribution (approximately 70/30)
        train_class_0 = (train["label"] == 0).sum()
        train_class_1 = (train["label"] == 1).sum()
        train_ratio = train_class_0 / len(train)
        assert 0.65 <= train_ratio <= 0.75  # Allow some variance
        
        # Check test distribution (approximately 70/30)
        test_class_0 = (test["label"] == 0).sum()
        test_class_1 = (test["label"] == 1).sum()
        test_ratio = test_class_0 / len(test)
        assert 0.65 <= test_ratio <= 0.75  # Allow some variance

    def test_split_reproducible_with_same_random_state(self):
        """Test that split is reproducible with same random_state."""
        df = pd.DataFrame({
            "text": [f"Sample {i}" for i in range(50)],
            "label": [i % 3 for i in range(50)]
        })
        
        train1, test1 = split_dataset(df, random_state=42)
        train2, test2 = split_dataset(df, random_state=42)
        
        # Should produce identical splits
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(test1, test2)

    def test_split_different_with_different_random_state(self):
        """Test that split produces different results with different random_state."""
        df = pd.DataFrame({
            "text": [f"Sample {i}" for i in range(50)],
            "label": [i % 3 for i in range(50)]
        })
        
        train1, test1 = split_dataset(df, random_state=42)
        train2, test2 = split_dataset(df, random_state=99)
        
        # Should produce different splits
        assert not train1.equals(train2)

    def test_split_no_data_leakage(self):
        """Test that train and test sets have no overlapping indices."""
        df = pd.DataFrame({
            "text": [f"Sample {i}" for i in range(100)],
            "label": [i % 2 for i in range(100)]
        })
        train, test = split_dataset(df)
        
        # Check no index overlap
        train_indices = set(train.index)
        test_indices = set(test.index)
        assert len(train_indices.intersection(test_indices)) == 0

    def test_split_custom_label_column(self):
        """Test that split works with custom label column name."""
        df = pd.DataFrame({
            "text": [f"Sample {i}" for i in range(50)],
            "category": [i % 2 for i in range(50)]
        })
        train, test = split_dataset(df, label_col="category", test_size=0.2)
        
        assert len(train) + len(test) == 50
        assert "category" in train.columns
        assert "category" in test.columns

    def test_split_preserves_all_columns(self):
        """Test that split preserves all columns from original DataFrame."""
        df = pd.DataFrame({
            "text": [f"Sample {i}" for i in range(50)],
            "label": [i % 2 for i in range(50)],
            "extra_col": [i * 2 for i in range(50)]
        })
        train, test = split_dataset(df)
        
        assert set(train.columns) == set(df.columns)
        assert set(test.columns) == set(df.columns)

    def test_split_multiclass_stratification(self):
        """Test that stratification works for multiclass labels."""
        df = pd.DataFrame({
            "text": [f"Sample {i}" for i in range(150)],
            "label": [0] * 50 + [1] * 50 + [2] * 50  # Equal distribution
        })
        train, test = split_dataset(df, test_size=0.2)
        
        # Each class should have approximately equal representation
        for class_label in [0, 1, 2]:
            train_count = (train["label"] == class_label).sum()
            test_count = (test["label"] == class_label).sum()
            # Each class should have ~40 in train and ~10 in test
            assert 35 <= train_count <= 45
            assert 8 <= test_count <= 12
