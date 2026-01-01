"""
Tests for evaluation metrics.
"""
import pytest
import numpy as np
from src.evaluation import evaluate_model


class TestMetricsPresence:
    """Test suite for required metrics presence."""

    def test_evaluate_returns_dictionary(self):
        """Test that evaluate_model returns a dictionary."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        
        result = evaluate_model(y_true, y_pred)
        
        assert isinstance(result, dict)

    def test_contains_accuracy(self):
        """Test that output contains 'accuracy' metric."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        
        result = evaluate_model(y_true, y_pred)
        
        assert "accuracy" in result

    def test_contains_precision(self):
        """Test that output contains 'precision' metric."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        
        result = evaluate_model(y_true, y_pred)
        
        assert "precision" in result

    def test_contains_recall(self):
        """Test that output contains 'recall' metric."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        
        result = evaluate_model(y_true, y_pred)
        
        assert "recall" in result

    def test_contains_f1(self):
        """Test that output contains 'f1' metric."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        
        result = evaluate_model(y_true, y_pred)
        
        assert "f1" in result


class TestProbabilityBasedMetrics:
    """Test suite for probability-based metrics."""

    def test_contains_roc_auc_when_proba_provided(self):
        """Test that output contains 'roc_auc' when y_proba is provided."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
        
        result = evaluate_model(y_true, y_pred, y_proba)
        
        assert "roc_auc" in result

    def test_no_roc_auc_when_proba_not_provided(self):
        """Test that 'roc_auc' is absent when y_proba is None."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        
        result = evaluate_model(y_true, y_pred, y_proba=None)
        
        assert "roc_auc" not in result

    def test_no_error_when_proba_is_none(self):
        """Test that evaluate_model does not error when y_proba is None."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        
        result = evaluate_model(y_true, y_pred, y_proba=None)
        
        # Should not raise error
        assert isinstance(result, dict)

    def test_roc_auc_with_different_probabilities(self):
        """Test ROC AUC calculation with different probability distributions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([[0.8, 0.2], [0.9, 0.1], [0.3, 0.7], [0.2, 0.8]])
        
        result = evaluate_model(y_true, y_pred, y_proba)
        
        assert "roc_auc" in result


class TestMetricValidity:
    """Test suite for metric validity."""

    def test_all_metrics_are_floats(self):
        """Test that all metric values are floats."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        
        result = evaluate_model(y_true, y_pred)
        
        for metric_name, metric_value in result.items():
            assert isinstance(metric_value, (float, np.floating))

    def test_all_metrics_between_0_and_1(self):
        """Test that all metric values are between 0 and 1."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1])
        
        result = evaluate_model(y_true, y_pred)
        
        for metric_name, metric_value in result.items():
            assert 0.0 <= metric_value <= 1.0

    def test_perfect_predictions_give_1_0(self):
        """Test that perfect predictions give metric values of 1.0."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])
        
        result = evaluate_model(y_true, y_pred)
        
        assert result["accuracy"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_metrics_with_imbalanced_data(self):
        """Test metrics with imbalanced class distribution."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 1, 1])
        
        result = evaluate_model(y_true, y_pred)
        
        # All metrics should still be valid
        for metric_value in result.values():
            assert 0.0 <= metric_value <= 1.0


class TestInputValidation:
    """Test suite for input validation."""

    def test_raises_on_empty_y_true(self):
        """Test that evaluate_model raises ValueError on empty y_true."""
        y_true = np.array([])
        y_pred = np.array([0, 1])
        
        with pytest.raises(ValueError, match="y_true cannot be empty"):
            evaluate_model(y_true, y_pred)

    def test_raises_on_empty_y_pred(self):
        """Test that evaluate_model raises ValueError on empty y_pred."""
        y_true = np.array([0, 1])
        y_pred = np.array([])
        
        with pytest.raises(ValueError, match="y_pred cannot be empty"):
            evaluate_model(y_true, y_pred)

    def test_raises_on_length_mismatch(self):
        """Test that evaluate_model raises ValueError when lengths don't match."""
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 1])
        
        with pytest.raises(ValueError, match="y_true and y_pred must have the same length"):
            evaluate_model(y_true, y_pred)

    def test_raises_on_none_y_true(self):
        """Test that evaluate_model raises TypeError on None y_true."""
        y_pred = np.array([0, 1])
        
        with pytest.raises(TypeError, match="y_true cannot be None"):
            evaluate_model(None, y_pred)

    def test_raises_on_none_y_pred(self):
        """Test that evaluate_model raises TypeError on None y_pred."""
        y_true = np.array([0, 1])
        
        with pytest.raises(TypeError, match="y_pred cannot be None"):
            evaluate_model(y_true, None)


class TestDeterminism:
    """Test suite for deterministic behavior."""

    def test_same_inputs_same_outputs(self):
        """Test that same inputs produce identical metric outputs."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1])
        
        result1 = evaluate_model(y_true, y_pred)
        result2 = evaluate_model(y_true, y_pred)
        
        assert result1 == result2

    def test_no_randomness_introduced(self):
        """Test that evaluate_model does not introduce randomness."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1, 1, 0])
        
        results = [evaluate_model(y_true, y_pred) for _ in range(5)]
        
        # All results should be identical
        for i in range(1, len(results)):
            assert results[0] == results[i]

    def test_deterministic_with_probabilities(self):
        """Test that metrics are deterministic when probabilities are provided."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
        
        result1 = evaluate_model(y_true, y_pred, y_proba)
        result2 = evaluate_model(y_true, y_pred, y_proba)
        
        assert result1 == result2


class TestEdgeCases:
    """Test suite for edge cases."""

    def test_all_predictions_correct(self):
        """Test evaluation when all predictions are correct."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        
        result = evaluate_model(y_true, y_pred)
        
        assert result["accuracy"] == 1.0

    def test_all_predictions_incorrect(self):
        """Test evaluation when all predictions are incorrect."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        
        result = evaluate_model(y_true, y_pred)
        
        assert result["accuracy"] == 0.0

    def test_balanced_classes(self):
        """Test evaluation with balanced classes."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        
        result = evaluate_model(y_true, y_pred)
        
        assert isinstance(result, dict)
        assert all(0.0 <= v <= 1.0 for v in result.values())

    def test_highly_imbalanced_classes(self):
        """Test evaluation with highly imbalanced classes."""
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        
        result = evaluate_model(y_true, y_pred)
        
        assert isinstance(result, dict)
        assert all(0.0 <= v <= 1.0 for v in result.values())
