"""
Tests for BERT pipeline compatibility.
"""
import pytest
import pandas as pd
from unittest.mock import patch
from src.pipeline import run_pipeline


class TestBertPipelineCompatibility:
    """Test suite for BERT pipeline compatibility."""

    @patch('src.pipeline.load_complaint_dataset')
    @patch('src.pipeline.validate_complaint_data')
    def test_pipeline_supports_bert_model_type(self, mock_validate, mock_load):
        """Test that pipeline supports model_type='bert' parameter."""
        mock_df = pd.DataFrame({
            'complaint_text': ['text ' + str(i) for i in range(10)],
            'label': [i % 2 for i in range(10)]
        })
        mock_load.return_value = mock_df
        mock_validate.return_value = None
        
        result = run_pipeline(model_type="bert")
        
        assert result is not None

    @patch('src.pipeline.load_complaint_dataset')
    @patch('src.pipeline.validate_complaint_data')
    def test_bert_pipeline_returns_model_and_metrics(self, mock_validate, mock_load):
        """Test that BERT pipeline returns model and metrics."""
        mock_df = pd.DataFrame({
            'complaint_text': ['text ' + str(i) for i in range(10)],
            'label': [i % 2 for i in range(10)]
        })
        mock_load.return_value = mock_df
        mock_validate.return_value = None
        
        result = run_pipeline(model_type="bert")
        
        assert "model" in result
        assert "metrics" in result

    @patch('src.pipeline.load_complaint_dataset')
    @patch('src.pipeline.validate_complaint_data')
    def test_bert_metrics_in_valid_range(self, mock_validate, mock_load):
        """Test that BERT pipeline metrics are in valid range."""
        mock_df = pd.DataFrame({
            'complaint_text': ['text ' + str(i) for i in range(10)],
            'label': [i % 2 for i in range(10)]
        })
        mock_load.return_value = mock_df
        mock_validate.return_value = None
        
        result = run_pipeline(model_type="bert")
        
        for metric_value in result["metrics"].values():
            assert 0.0 <= metric_value <= 1.0

    @patch('src.pipeline.load_complaint_dataset')
    @patch('src.pipeline.validate_complaint_data')
    def test_default_tfidf_behavior_unchanged(self, mock_validate, mock_load):
        """Test that default TF-IDF behavior is not affected by BERT addition."""
        mock_df = pd.DataFrame({
            'complaint_text': ['text ' + str(i) for i in range(10)],
            'label': [i % 2 for i in range(10)]
        })
        mock_load.return_value = mock_df
        mock_validate.return_value = None
        
        # Default should still use TF-IDF
        result = run_pipeline()
        
        assert result is not None
        assert "model" in result
        assert "metrics" in result

    @patch('src.pipeline.load_complaint_dataset')
    @patch('src.pipeline.validate_complaint_data')
    def test_bert_and_tfidf_produce_different_models(self, mock_validate, mock_load):
        """Test that BERT and TF-IDF produce different model types."""
        mock_df = pd.DataFrame({
            'complaint_text': ['text ' + str(i) for i in range(10)],
            'label': [i % 2 for i in range(10)]
        })
        mock_load.return_value = mock_df
        mock_validate.return_value = None
        
        result_tfidf = run_pipeline(model_type="tfidf")
        result_bert = run_pipeline(model_type="bert")
        
        # Models should be different types
        assert type(result_tfidf["model"]) != type(result_bert["model"])
