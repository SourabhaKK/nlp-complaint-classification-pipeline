"""
End-to-end integration tests for the complaint classification pipeline.

These tests execute the real pipeline without mocks to verify that all
components work together correctly in practice.

Note: These tests use a mock data loader to avoid external dependencies
while still testing the real pipeline logic.
"""
import pytest
import pandas as pd
from unittest.mock import patch
from src.pipeline import run_pipeline


@pytest.fixture
def mock_complaint_data():
    """Fixture providing realistic complaint data for integration tests."""
    return pd.DataFrame({
        'complaint_text': [
            'My credit card was charged twice for the same purchase',
            'I cannot access my online banking account',
            'The ATM did not dispense cash but debited my account',
            'Unauthorized transaction on my debit card',
            'Customer service was very helpful and resolved my issue',
            'Great experience with the mobile app',
            'Interest rate is too high on my loan',
            'I received excellent service at the branch',
            'My check deposit has not been processed',
            'The bank fees are unreasonable',
            'Very satisfied with the mortgage process',
            'Account statement has errors',
            'Friendly staff at the branch',
            'My card was declined for no reason',
            'Quick response to my inquiry',
            'Overdraft fees are excessive',
            'Smooth loan application process',
            'Cannot transfer money between accounts',
            'Appreciate the low interest rates',
            'Long wait times on customer service phone',
        ],
        'label': [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    })


class TestEndToEndIntegration:
    """Integration tests for full pipeline execution."""

    @patch('src.pipeline.load_complaint_dataset')
    def test_pipeline_executes_successfully(self, mock_load, mock_complaint_data):
        """Test that pipeline completes end-to-end without errors."""
        mock_load.return_value = mock_complaint_data
        
        # Execute real pipeline with minimal configuration
        result = run_pipeline(
            data_source="huggingface",
            test_size=0.3,
            random_state=42
        )
        
        # Verify pipeline completed
        assert result is not None

    @patch('src.pipeline.load_complaint_dataset')
    def test_pipeline_returns_model(self, mock_load, mock_complaint_data):
        """Test that pipeline returns a trained model object."""
        mock_load.return_value = mock_complaint_data
        
        result = run_pipeline(
            data_source="huggingface",
            test_size=0.3,
            random_state=42
        )
        
        # Verify model is returned
        assert "model" in result
        assert result["model"] is not None

    @patch('src.pipeline.load_complaint_dataset')
    def test_pipeline_returns_metrics(self, mock_load, mock_complaint_data):
        """Test that pipeline returns metrics dictionary."""
        mock_load.return_value = mock_complaint_data
        
        result = run_pipeline(
            data_source="huggingface",
            test_size=0.3,
            random_state=42
        )
        
        # Verify metrics dictionary is returned
        assert "metrics" in result
        assert isinstance(result["metrics"], dict)

    @patch('src.pipeline.load_complaint_dataset')
    def test_metrics_have_expected_keys(self, mock_load, mock_complaint_data):
        """Test that metrics contain expected evaluation metrics."""
        mock_load.return_value = mock_complaint_data
        
        result = run_pipeline(
            data_source="huggingface",
            test_size=0.3,
            random_state=42
        )
        
        metrics = result["metrics"]
        
        # Verify all expected metrics are present
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

    @patch('src.pipeline.load_complaint_dataset')
    def test_metrics_in_valid_range(self, mock_load, mock_complaint_data):
        """Test that all metric values are in [0.0, 1.0] range."""
        mock_load.return_value = mock_complaint_data
        
        result = run_pipeline(
            data_source="huggingface",
            test_size=0.3,
            random_state=42
        )
        
        metrics = result["metrics"]
        
        # Verify all metrics are in valid range
        for metric_name, metric_value in metrics.items():
            assert 0.0 <= metric_value <= 1.0, \
                f"{metric_name} = {metric_value} is not in [0.0, 1.0]"

    @patch('src.pipeline.load_complaint_dataset')
    def test_pipeline_is_deterministic(self, mock_load, mock_complaint_data):
        """Test that pipeline produces consistent results with same seed."""
        mock_load.return_value = mock_complaint_data
        
        # Run pipeline twice with same seed
        result1 = run_pipeline(
            data_source="huggingface",
            test_size=0.3,
            random_state=42
        )
        
        # Reset mock for second call
        mock_load.return_value = mock_complaint_data
        
        result2 = run_pipeline(
            data_source="huggingface",
            test_size=0.3,
            random_state=42
        )
        
        # Verify metrics are identical
        assert result1["metrics"] == result2["metrics"]

    @patch('src.pipeline.load_complaint_dataset')
    def test_bert_pipeline_executes_successfully(self, mock_load, mock_complaint_data):
        """Test that BERT pipeline completes end-to-end without errors."""
        mock_load.return_value = mock_complaint_data
        
        # Execute real BERT pipeline with minimal configuration
        result = run_pipeline(
            data_source="huggingface",
            test_size=0.3,
            random_state=42,
            model_type="bert"
        )
        
        # Verify pipeline completed
        assert result is not None
        assert "model" in result
        assert "metrics" in result

    @patch('src.pipeline.load_complaint_dataset')
    def test_bert_metrics_in_valid_range(self, mock_load, mock_complaint_data):
        """Test that BERT pipeline metrics are in [0.0, 1.0] range."""
        mock_load.return_value = mock_complaint_data
        
        result = run_pipeline(
            data_source="huggingface",
            test_size=0.3,
            random_state=42,
            model_type="bert"
        )
        
        metrics = result["metrics"]
        
        # Verify all metrics are in valid range
        for metric_name, metric_value in metrics.items():
            assert 0.0 <= metric_value <= 1.0, \
                f"BERT {metric_name} = {metric_value} is not in [0.0, 1.0]"

    @patch('src.pipeline.load_complaint_dataset')
    def test_tfidf_and_bert_produce_different_models(self, mock_load, mock_complaint_data):
        """Test that TF-IDF and BERT pipelines produce different model types."""
        mock_load.return_value = mock_complaint_data
        
        result_tfidf = run_pipeline(
            data_source="huggingface",
            test_size=0.3,
            random_state=42,
            model_type="tfidf"
        )
        
        # Reset mock for second call
        mock_load.return_value = mock_complaint_data
        
        result_bert = run_pipeline(
            data_source="huggingface",
            test_size=0.3,
            random_state=42,
            model_type="bert"
        )
        
        # Verify different model types
        assert type(result_tfidf["model"]) != type(result_bert["model"])
