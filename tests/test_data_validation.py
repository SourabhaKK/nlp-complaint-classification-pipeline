"""
Tests for data validation module.
"""
import pytest
import pandas as pd
from src.data_validation import validate_complaint_data


class TestDataValidation:
    """Test suite for complaint data validation."""

    def test_validate_none_dataframe(self):
        """Test that validation fails when DataFrame is None."""
        with pytest.raises(ValueError, match="DataFrame cannot be None"):
            validate_complaint_data(None)

    def test_validate_missing_complaint_text_column(self):
        """Test that validation fails when 'complaint_text' column is missing."""
        df = pd.DataFrame({
            "label": ["product", "service"]
        })
        with pytest.raises(ValueError, match="Missing required column: 'complaint_text'"):
            validate_complaint_data(df)

    def test_validate_missing_label_column(self):
        """Test that validation fails when 'label' column is missing."""
        df = pd.DataFrame({
            "complaint_text": ["This is a complaint", "Another complaint"]
        })
        with pytest.raises(ValueError, match="Missing required column: 'label'"):
            validate_complaint_data(df)

    def test_validate_missing_both_columns(self):
        """Test that validation fails when both required columns are missing."""
        df = pd.DataFrame({
            "other_column": [1, 2, 3]
        })
        with pytest.raises(ValueError, match="Missing required column"):
            validate_complaint_data(df)

    def test_validate_null_complaint_text(self):
        """Test that validation fails when complaint_text contains null values."""
        df = pd.DataFrame({
            "complaint_text": ["Valid complaint", None, "Another valid"],
            "label": ["product", "service", "billing"]
        })
        with pytest.raises(ValueError, match="'complaint_text' contains null values"):
            validate_complaint_data(df)

    def test_validate_empty_string_complaint_text(self):
        """Test that validation fails when complaint_text contains empty strings."""
        df = pd.DataFrame({
            "complaint_text": ["Valid complaint", "", "Another valid"],
            "label": ["product", "service", "billing"]
        })
        with pytest.raises(ValueError, match="'complaint_text' contains empty strings"):
            validate_complaint_data(df)

    def test_validate_whitespace_only_complaint_text(self):
        """Test that validation fails when complaint_text contains whitespace-only strings."""
        df = pd.DataFrame({
            "complaint_text": ["Valid complaint", "   ", "Another valid"],
            "label": ["product", "service", "billing"]
        })
        with pytest.raises(ValueError, match="'complaint_text' contains empty strings"):
            validate_complaint_data(df)

    def test_validate_null_labels(self):
        """Test that validation fails when label column contains null values."""
        df = pd.DataFrame({
            "complaint_text": ["Complaint 1", "Complaint 2", "Complaint 3"],
            "label": ["product", None, "billing"]
        })
        with pytest.raises(ValueError, match="'label' contains null values"):
            validate_complaint_data(df)

    def test_validate_empty_dataframe(self):
        """Test that validation fails for empty DataFrame with correct columns."""
        df = pd.DataFrame({
            "complaint_text": [],
            "label": []
        })
        with pytest.raises(ValueError, match="DataFrame is empty"):
            validate_complaint_data(df)

    def test_validate_minimal_valid_dataframe(self):
        """Test that validation passes for minimal valid DataFrame."""
        df = pd.DataFrame({
            "complaint_text": ["This is a valid complaint"],
            "label": ["product"]
        })
        # Should not raise any exception
        result = validate_complaint_data(df)
        assert result is True

    def test_validate_valid_dataframe_multiple_rows(self):
        """Test that validation passes for valid DataFrame with multiple rows."""
        df = pd.DataFrame({
            "complaint_text": [
                "Product defect complaint",
                "Service quality issue",
                "Billing error report"
            ],
            "label": ["product", "service", "billing"]
        })
        # Should not raise any exception
        result = validate_complaint_data(df)
        assert result is True

    def test_validate_dataframe_with_extra_columns(self):
        """Test that validation passes when DataFrame has extra columns beyond required ones."""
        df = pd.DataFrame({
            "complaint_text": ["Valid complaint"],
            "label": ["product"],
            "timestamp": ["2024-01-01"],
            "customer_id": [12345]
        })
        # Should not raise any exception - extra columns are allowed
        result = validate_complaint_data(df)
        assert result is True
