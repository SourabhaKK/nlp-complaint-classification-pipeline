"""
Tests for prediction interface.
"""
import pytest
import numpy as np
from unittest.mock import Mock
from src.prediction import predict


class TestSuccessfulPrediction:
    """Test suite for successful predictions."""

    def test_predict_returns_dictionary(self):
        """Test that predict returns a dictionary."""
        # Create mock model
        model = Mock()
        model.predict.return_value = np.array([0, 1, 0])
        
        X = np.array([[1, 0], [0, 1], [1, 1]])
        result = predict(model, X)
        
        assert isinstance(result, dict)

    def test_predict_contains_predictions_key(self):
        """Test that output dictionary contains 'predictions' key."""
        model = Mock()
        model.predict.return_value = np.array([0, 1])
        
        X = np.array([[1, 0], [0, 1]])
        result = predict(model, X)
        
        assert "predictions" in result

    def test_predictions_length_matches_samples(self):
        """Test that length of predictions equals number of samples."""
        model = Mock()
        model.predict.return_value = np.array([0, 1, 1, 0])
        
        X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        result = predict(model, X)
        
        assert len(result["predictions"]) == len(X)

    def test_predict_with_single_sample(self):
        """Test prediction with single sample."""
        model = Mock()
        model.predict.return_value = np.array([1])
        
        X = np.array([[1, 0]])
        result = predict(model, X)
        
        assert len(result["predictions"]) == 1

    def test_predict_calls_model_predict(self):
        """Test that predict calls model.predict() method."""
        model = Mock()
        model.predict.return_value = np.array([0, 1])
        
        X = np.array([[1, 0], [0, 1]])
        result = predict(model, X)
        
        model.predict.assert_called_once()


class TestProbabilityOutput:
    """Test suite for probability output."""

    def test_includes_probabilities_if_model_supports(self):
        """Test that output includes probabilities if model has predict_proba."""
        model = Mock()
        model.predict.return_value = np.array([0, 1])
        model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
        
        X = np.array([[1, 0], [0, 1]])
        result = predict(model, X)
        
        assert "probabilities" in result

    def test_probabilities_shape_matches_samples(self):
        """Test that probabilities shape matches number of samples."""
        model = Mock()
        model.predict.return_value = np.array([0, 1, 0])
        model.predict_proba.return_value = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
        
        X = np.array([[1, 0], [0, 1], [1, 1]])
        result = predict(model, X)
        
        assert result["probabilities"].shape[0] == len(X)

    def test_no_error_if_model_lacks_predict_proba(self):
        """Test that predict does not error if model lacks predict_proba."""
        model = Mock()
        model.predict.return_value = np.array([0, 1])
        # Don't set predict_proba - model doesn't support it
        del model.predict_proba
        
        X = np.array([[1, 0], [0, 1]])
        result = predict(model, X)
        
        # Should not raise error
        assert "predictions" in result

    def test_probabilities_omitted_if_not_supported(self):
        """Test that probabilities key may be omitted if not supported."""
        model = Mock()
        model.predict.return_value = np.array([0, 1])
        del model.predict_proba
        
        X = np.array([[1, 0], [0, 1]])
        result = predict(model, X)
        
        # Probabilities may or may not be present, but should not error
        assert isinstance(result, dict)


class TestInputValidation:
    """Test suite for input validation."""

    def test_raises_on_none_model(self):
        """Test that predict raises ValueError if model is None."""
        X = np.array([[1, 0], [0, 1]])
        
        with pytest.raises(ValueError, match="Model cannot be None"):
            predict(None, X)

    def test_raises_on_empty_X(self):
        """Test that predict raises ValueError if X is empty."""
        model = Mock()
        model.predict.return_value = np.array([])
        
        X = np.array([])
        
        with pytest.raises(ValueError, match="Input X cannot be empty"):
            predict(model, X)

    def test_raises_on_none_X(self):
        """Test that predict raises TypeError if X is None."""
        model = Mock()
        
        with pytest.raises(TypeError, match="Input X cannot be None"):
            predict(model, None)

    def test_raises_on_invalid_model_type(self):
        """Test that predict raises TypeError if model is not model-like."""
        X = np.array([[1, 0], [0, 1]])
        
        with pytest.raises(TypeError, match="Model must have a predict method"):
            predict("not a model", X)


class TestDeterminism:
    """Test suite for deterministic behavior."""

    def test_same_input_same_predictions(self):
        """Test that same model and input produce same predictions."""
        model = Mock()
        model.predict.return_value = np.array([0, 1, 0])
        
        X = np.array([[1, 0], [0, 1], [1, 1]])
        
        result1 = predict(model, X)
        result2 = predict(model, X)
        
        np.testing.assert_array_equal(result1["predictions"], result2["predictions"])

    def test_no_randomness_introduced(self):
        """Test that predict does not introduce randomness."""
        model = Mock()
        model.predict.return_value = np.array([0, 1, 1, 0])
        
        X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        
        results = [predict(model, X)["predictions"] for _ in range(5)]
        
        # All results should be identical
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])

    def test_deterministic_with_probabilities(self):
        """Test that probabilities are deterministic if supported."""
        model = Mock()
        model.predict.return_value = np.array([0, 1])
        model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
        
        X = np.array([[1, 0], [0, 1]])
        
        result1 = predict(model, X)
        result2 = predict(model, X)
        
        if "probabilities" in result1 and "probabilities" in result2:
            np.testing.assert_array_equal(result1["probabilities"], result2["probabilities"])


class TestEdgeCases:
    """Test suite for edge cases."""

    def test_predict_with_sparse_input(self):
        """Test prediction with sparse matrix input."""
        from scipy.sparse import csr_matrix
        
        model = Mock()
        model.predict.return_value = np.array([0, 1])
        
        X = csr_matrix([[1, 0], [0, 1]])
        result = predict(model, X)
        
        assert "predictions" in result

    def test_predict_large_batch(self):
        """Test prediction with large batch of samples."""
        model = Mock()
        model.predict.return_value = np.array([0] * 1000)
        
        X = np.random.rand(1000, 10)
        result = predict(model, X)
        
        assert len(result["predictions"]) == 1000

    def test_predictions_are_array_like(self):
        """Test that predictions are array-like (can be indexed)."""
        model = Mock()
        model.predict.return_value = np.array([0, 1, 0])
        
        X = np.array([[1, 0], [0, 1], [1, 1]])
        result = predict(model, X)
        
        # Should be able to index predictions
        assert result["predictions"][0] in [0, 1]
