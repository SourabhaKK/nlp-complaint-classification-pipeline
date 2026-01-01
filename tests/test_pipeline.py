"""
Tests for end-to-end pipeline orchestration.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from src.pipeline import run_pipeline


class TestEndToEndSuccess:
    """Test suite for end-to-end pipeline execution."""

    @patch('src.pipeline.load_complaint_dataset')
    @patch('src.pipeline.validate_complaint_data')
    def test_pipeline_runs_without_errors(self, mock_validate, mock_load):
        """Test that pipeline runs without errors."""
        # Mock dataset
        mock_df = pd.DataFrame({
            'complaint_text': ['text ' + str(i) for i in range(10)],
            'label': [i % 2 for i in range(10)]
        })
        mock_load.return_value = mock_df
        mock_validate.return_value = None
        
        result = run_pipeline()
        
        assert result is not None

    @patch('src.pipeline.load_complaint_dataset')
    @patch('src.pipeline.validate_complaint_data')
    def test_pipeline_returns_dictionary(self, mock_validate, mock_load):
        """Test that pipeline returns a dictionary."""
        mock_df = pd.DataFrame({
            'complaint_text': ['text ' + str(i) for i in range(10)],
            'label': [i % 2 for i in range(10)]
        })
        mock_load.return_value = mock_df
        mock_validate.return_value = None
        
        result = run_pipeline()
        
        assert isinstance(result, dict)

    @patch('src.pipeline.load_complaint_dataset')
    @patch('src.pipeline.validate_complaint_data')
    def test_pipeline_contains_model_key(self, mock_validate, mock_load):
        """Test that output contains 'model' key."""
        mock_df = pd.DataFrame({
            'complaint_text': ['text ' + str(i) for i in range(10)],
            'label': [i % 2 for i in range(10)]
        })
        mock_load.return_value = mock_df
        mock_validate.return_value = None
        
        result = run_pipeline()
        
        assert "model" in result

    @patch('src.pipeline.load_complaint_dataset')
    @patch('src.pipeline.validate_complaint_data')
    def test_pipeline_contains_metrics_key(self, mock_validate, mock_load):
        """Test that output contains 'metrics' key."""
        mock_df = pd.DataFrame({
            'complaint_text': ['text ' + str(i) for i in range(10)],
            'label': [i % 2 for i in range(10)]
        })
        mock_load.return_value = mock_df
        mock_validate.return_value = None
        
        result = run_pipeline()
        
        assert "metrics" in result


class TestMetricsIntegrity:
    """Test suite for metrics integrity."""

    @patch('src.pipeline.load_complaint_dataset')
    @patch('src.pipeline.validate_complaint_data')
    def test_metrics_contains_accuracy(self, mock_validate, mock_load):
        """Test that metrics dict contains 'accuracy'."""
        mock_df = pd.DataFrame({
            'complaint_text': ['text ' + str(i) for i in range(10)],
            'label': [i % 2 for i in range(10)]
        })
        mock_load.return_value = mock_df
        mock_validate.return_value = None
        
        result = run_pipeline()
        
        assert "accuracy" in result["metrics"]

    @patch('src.pipeline.load_complaint_dataset')
    @patch('src.pipeline.validate_complaint_data')
    def test_metrics_contains_precision(self, mock_validate, mock_load):
        """Test that metrics dict contains 'precision'."""
        mock_df = pd.DataFrame({
            'complaint_text': ['text ' + str(i) for i in range(10)],
            'label': [i % 2 for i in range(10)]
        })
        mock_load.return_value = mock_df
        mock_validate.return_value = None
        
        result = run_pipeline()
        
        assert "precision" in result["metrics"]

    @patch('src.pipeline.load_complaint_dataset')
    @patch('src.pipeline.validate_complaint_data')
    def test_metrics_contains_recall(self, mock_validate, mock_load):
        """Test that metrics dict contains 'recall'."""
        mock_df = pd.DataFrame({
            'complaint_text': ['text ' + str(i) for i in range(10)],
            'label': [i % 2 for i in range(10)]
        })
        mock_load.return_value = mock_df
        mock_validate.return_value = None
        
        result = run_pipeline()
        
        assert "recall" in result["metrics"]

    @patch('src.pipeline.load_complaint_dataset')
    @patch('src.pipeline.validate_complaint_data')
    def test_metrics_contains_f1(self, mock_validate, mock_load):
        """Test that metrics dict contains 'f1'."""
        mock_df = pd.DataFrame({
            'complaint_text': ['text ' + str(i) for i in range(10)],
            'label': [i % 2 for i in range(10)]
        })
        mock_load.return_value = mock_df
        mock_validate.return_value = None
        
        result = run_pipeline()
        
        assert "f1" in result["metrics"]

    @patch('src.pipeline.load_complaint_dataset')
    @patch('src.pipeline.validate_complaint_data')
    def test_metric_values_are_floats(self, mock_validate, mock_load):
        """Test that all metric values are floats."""
        mock_df = pd.DataFrame({
            'complaint_text': ['text ' + str(i) for i in range(10)],
            'label': [i % 2 for i in range(10)]
        })
        mock_load.return_value = mock_df
        mock_validate.return_value = None
        
        result = run_pipeline()
        
        for metric_value in result["metrics"].values():
            assert isinstance(metric_value, (float, np.floating))

    @patch('src.pipeline.load_complaint_dataset')
    @patch('src.pipeline.validate_complaint_data')
    def test_metric_values_in_valid_range(self, mock_validate, mock_load):
        """Test that all metric values are in [0, 1]."""
        mock_df = pd.DataFrame({
            'complaint_text': ['text ' + str(i) for i in range(10)],
            'label': [i % 2 for i in range(10)]
        })
        mock_load.return_value = mock_df
        mock_validate.return_value = None
        
        result = run_pipeline()
        
        for metric_value in result["metrics"].values():
            assert 0.0 <= metric_value <= 1.0


class TestDeterminism:
    """Test suite for deterministic behavior."""

    @patch('src.pipeline.load_complaint_dataset')
    @patch('src.pipeline.validate_complaint_data')
    def test_same_inputs_same_metrics(self, mock_validate, mock_load):
        """Test that running pipeline twice with same inputs yields identical metrics."""
        mock_df = pd.DataFrame({
            'complaint_text': ['text ' + str(i) for i in range(10)],
            'label': [i % 2 for i in range(10)]
        })
        mock_load.return_value = mock_df
        mock_validate.return_value = None
        
        result1 = run_pipeline(random_state=42)
        result2 = run_pipeline(random_state=42)
        
        assert result1["metrics"] == result2["metrics"]

    @patch('src.pipeline.load_complaint_dataset')
    @patch('src.pipeline.validate_complaint_data')
    def test_deterministic_with_same_random_state(self, mock_validate, mock_load):
        """Test that same random_state produces deterministic results."""
        mock_df = pd.DataFrame({
            'complaint_text': ['text ' + str(i) for i in range(10)],
            'label': [i % 2 for i in range(10)]
        })
        mock_load.return_value = mock_df
        mock_validate.return_value = None
        
        result1 = run_pipeline(random_state=99)
        result2 = run_pipeline(random_state=99)
        
        # Metrics should be identical
        for key in result1["metrics"]:
            assert result1["metrics"][key] == result2["metrics"][key]


class TestDependencyWiring:
    """Test suite for dependency wiring."""

    @patch('src.pipeline.load_complaint_dataset')
    @patch('src.pipeline.validate_complaint_data')
    def test_dataset_loader_is_called(self, mock_validate, mock_load):
        """Test that dataset loader is called."""
        mock_df = pd.DataFrame({
            'complaint_text': ['text ' + str(i) for i in range(10)],
            'label': [i % 2 for i in range(10)]
        })
        mock_load.return_value = mock_df
        mock_validate.return_value = None
        
        run_pipeline()
        
        mock_load.assert_called_once()

    @patch('src.pipeline.load_complaint_dataset')
    @patch('src.pipeline.validate_complaint_data')
    def test_validation_is_called(self, mock_validate, mock_load):
        """Test that validation is called."""
        mock_df = pd.DataFrame({
            'complaint_text': ['text ' + str(i) for i in range(10)],
            'label': [i % 2 for i in range(10)]
        })
        mock_load.return_value = mock_df
        mock_validate.return_value = None
        
        run_pipeline()
        
        mock_validate.assert_called_once()

    @patch('src.pipeline.load_complaint_dataset')
    @patch('src.pipeline.validate_complaint_data')
    @patch('src.pipeline.split_dataset')
    def test_train_test_split_is_called(self, mock_split, mock_validate, mock_load):
        """Test that train/test split is called."""
        mock_df = pd.DataFrame({
            'complaint_text': ['text ' + str(i) for i in range(10)],
            'label': [i % 2 for i in range(10)]
        })
        mock_load.return_value = mock_df
        mock_validate.return_value = None
        mock_split.return_value = (
            pd.DataFrame({'complaint_text': ['text ' + str(i) for i in range(10)],
            'label': [i % 2 for i in range(10)]}),
            pd.DataFrame({'complaint_text': ['text ' + str(i) for i in range(10)],
            'label': [i % 2 for i in range(10)]})
        )
        
        run_pipeline()
        
        mock_split.assert_called_once()

    @patch('src.pipeline.load_complaint_dataset')
    @patch('src.pipeline.validate_complaint_data')
    @patch('src.pipeline.fit_vectorizer')
    def test_vectorizer_fit_called_once(self, mock_fit_vec, mock_validate, mock_load):
        """Test that vectorizer fit is called once (on train data only)."""
        mock_df = pd.DataFrame({
            'complaint_text': ['text ' + str(i) for i in range(10)],
            'label': [i % 2 for i in range(10)]
        })
        mock_load.return_value = mock_df
        mock_validate.return_value = None
        mock_vectorizer = Mock()
        mock_fit_vec.return_value = mock_vectorizer
        
        run_pipeline()
        
        # Fit should be called exactly once
        mock_fit_vec.assert_called_once()

    @patch('src.pipeline.load_complaint_dataset')
    @patch('src.pipeline.validate_complaint_data')
    @patch('src.pipeline.train_model')
    def test_model_training_invoked_once(self, mock_train, mock_validate, mock_load):
        """Test that model training is invoked once."""
        mock_df = pd.DataFrame({
            'complaint_text': ['text ' + str(i) for i in range(10)],
            'label': [i % 2 for i in range(10)]
        })
        mock_load.return_value = mock_df
        mock_validate.return_value = None
        mock_model = Mock()
        mock_train.return_value = mock_model
        
        run_pipeline()
        
        mock_train.assert_called_once()

    @patch('src.pipeline.load_complaint_dataset')
    @patch('src.pipeline.validate_complaint_data')
    @patch('src.pipeline.evaluate_model')
    def test_evaluation_invoked_once(self, mock_eval, mock_validate, mock_load):
        """Test that evaluation is invoked once."""
        mock_df = pd.DataFrame({
            'complaint_text': ['text ' + str(i) for i in range(10)],
            'label': [i % 2 for i in range(10)]
        })
        mock_load.return_value = mock_df
        mock_validate.return_value = None
        mock_eval.return_value = {'accuracy': 0.75, 'precision': 0.75, 'recall': 0.75, 'f1': 0.75}
        
        run_pipeline()
        
        mock_eval.assert_called_once()


class TestInputValidation:
    """Test suite for input validation."""

    def test_raises_on_unsupported_data_source(self):
        """Test that pipeline raises ValueError for unsupported data_source."""
        with pytest.raises(ValueError, match="Unsupported data_source"):
            run_pipeline(data_source="invalid_source")

    def test_raises_on_invalid_test_size_too_small(self):
        """Test that pipeline raises ValueError for test_size too small."""
        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            run_pipeline(test_size=-0.1)

    def test_raises_on_invalid_test_size_too_large(self):
        """Test that pipeline raises ValueError for test_size too large."""
        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            run_pipeline(test_size=1.5)

    @patch('src.pipeline.load_complaint_dataset')
    @patch('src.pipeline.validate_complaint_data')
    def test_accepts_valid_test_size(self, mock_validate, mock_load):
        """Test that pipeline accepts valid test_size values."""
        mock_df = pd.DataFrame({
            'complaint_text': ['text ' + str(i) for i in range(10)],
            'label': [i % 2 for i in range(10)]
        })
        mock_load.return_value = mock_df
        mock_validate.return_value = None
        
        result = run_pipeline(test_size=0.3)
        
        assert result is not None
