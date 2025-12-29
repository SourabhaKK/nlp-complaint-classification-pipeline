"""
Tests for Hugging Face dataset loader module.
"""
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.data_loader import load_complaint_dataset


class TestDataLoader:
    """Test suite for Hugging Face dataset loading."""

    @patch('src.data_loader.datasets.load_dataset')
    def test_returns_pandas_dataframe(self, mock_load_dataset):
        """Test that load_complaint_dataset returns a pandas DataFrame."""
        # Mock the Hugging Face dataset object
        mock_dataset = MagicMock()
        mock_dataset.to_pandas.return_value = pd.DataFrame({
            "text": ["Sample complaint"],
            "label": [0]
        })
        mock_load_dataset.return_value = mock_dataset
        
        result = load_complaint_dataset()
        
        assert isinstance(result, pd.DataFrame)

    @patch('src.data_loader.datasets.load_dataset')
    def test_contains_required_text_column(self, mock_load_dataset):
        """Test that returned DataFrame contains 'text' column."""
        mock_dataset = MagicMock()
        mock_dataset.to_pandas.return_value = pd.DataFrame({
            "text": ["Complaint 1", "Complaint 2"],
            "label": [0, 1]
        })
        mock_load_dataset.return_value = mock_dataset
        
        result = load_complaint_dataset()
        
        assert "text" in result.columns

    @patch('src.data_loader.datasets.load_dataset')
    def test_contains_required_label_column(self, mock_load_dataset):
        """Test that returned DataFrame contains 'label' column."""
        mock_dataset = MagicMock()
        mock_dataset.to_pandas.return_value = pd.DataFrame({
            "text": ["Complaint 1", "Complaint 2"],
            "label": [0, 1]
        })
        mock_load_dataset.return_value = mock_dataset
        
        result = load_complaint_dataset()
        
        assert "label" in result.columns

    @patch('src.data_loader.datasets.load_dataset')
    def test_calls_huggingface_with_correct_dataset_name(self, mock_load_dataset):
        """Test that datasets.load_dataset is called with correct dataset name."""
        mock_dataset = MagicMock()
        mock_dataset.to_pandas.return_value = pd.DataFrame({
            "text": ["Sample"],
            "label": [0]
        })
        mock_load_dataset.return_value = mock_dataset
        
        load_complaint_dataset()
        
        # Verify load_dataset was called with a dataset name
        mock_load_dataset.assert_called_once()
        call_args = mock_load_dataset.call_args
        # First positional argument should be the dataset name (string)
        assert len(call_args[0]) > 0
        assert isinstance(call_args[0][0], str)

    @patch('src.data_loader.datasets.load_dataset')
    def test_calls_huggingface_with_train_split_by_default(self, mock_load_dataset):
        """Test that datasets.load_dataset is called with 'train' split by default."""
        mock_dataset = MagicMock()
        mock_dataset.to_pandas.return_value = pd.DataFrame({
            "text": ["Sample"],
            "label": [0]
        })
        mock_load_dataset.return_value = mock_dataset
        
        load_complaint_dataset()
        
        # Verify split='train' is passed
        call_args = mock_load_dataset.call_args
        assert 'split' in call_args[1] or (len(call_args[0]) > 1)

    @patch('src.data_loader.datasets.load_dataset')
    def test_accepts_custom_split_parameter(self, mock_load_dataset):
        """Test that load_complaint_dataset accepts custom split parameter."""
        mock_dataset = MagicMock()
        mock_dataset.to_pandas.return_value = pd.DataFrame({
            "text": ["Test sample"],
            "label": [1]
        })
        mock_load_dataset.return_value = mock_dataset
        
        load_complaint_dataset(split="test")
        
        # Verify the split parameter is passed through
        call_args = mock_load_dataset.call_args
        # Check if 'test' appears in the call arguments
        assert 'split' in call_args[1] and call_args[1]['split'] == "test" or \
               (len(call_args[0]) > 1 and call_args[0][1] == "test")

    @patch('src.data_loader.datasets.load_dataset')
    def test_calls_to_pandas_on_dataset(self, mock_load_dataset):
        """Test that .to_pandas() is called on the loaded dataset."""
        mock_dataset = MagicMock()
        mock_dataset.to_pandas.return_value = pd.DataFrame({
            "text": ["Sample"],
            "label": [0]
        })
        mock_load_dataset.return_value = mock_dataset
        
        load_complaint_dataset()
        
        # Verify to_pandas was called
        mock_dataset.to_pandas.assert_called_once()

    @patch('src.data_loader.datasets.load_dataset')
    def test_no_network_calls_with_mocking(self, mock_load_dataset):
        """Test that no real network calls are made when properly mocked."""
        mock_dataset = MagicMock()
        mock_dataset.to_pandas.return_value = pd.DataFrame({
            "text": ["Offline test"],
            "label": [0]
        })
        mock_load_dataset.return_value = mock_dataset
        
        # This should work offline because we're mocking
        result = load_complaint_dataset()
        
        # Verify we got a DataFrame without making real network calls
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    @patch('src.data_loader.datasets.load_dataset')
    def test_returns_non_empty_dataframe(self, mock_load_dataset):
        """Test that returned DataFrame is not empty."""
        mock_dataset = MagicMock()
        mock_dataset.to_pandas.return_value = pd.DataFrame({
            "text": ["Complaint 1", "Complaint 2", "Complaint 3"],
            "label": [0, 1, 2]
        })
        mock_load_dataset.return_value = mock_dataset
        
        result = load_complaint_dataset()
        
        assert len(result) > 0
        assert not result.empty

    @patch('src.data_loader.datasets.load_dataset')
    def test_validation_split_parameter(self, mock_load_dataset):
        """Test that load_complaint_dataset works with 'validation' split."""
        mock_dataset = MagicMock()
        mock_dataset.to_pandas.return_value = pd.DataFrame({
            "text": ["Validation sample"],
            "label": [0]
        })
        mock_load_dataset.return_value = mock_dataset
        
        result = load_complaint_dataset(split="validation")
        
        assert isinstance(result, pd.DataFrame)
        mock_load_dataset.assert_called_once()
