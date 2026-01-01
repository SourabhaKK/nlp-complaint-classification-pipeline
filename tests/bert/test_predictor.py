"""
Tests for BERT predictor.
"""
import pytest
import numpy as np
from unittest.mock import Mock
from src.bert.predictor import BertPredictor


class TestPredictorInitialization:
    """Test suite for predictor initialization."""

    def test_predictor_initializes_with_model(self):
        """Test that predictor initializes with model."""
        mock_model = Mock()
        predictor = BertPredictor(model=mock_model)
        
        assert predictor is not None
        assert predictor.model == mock_model


class TestClassPrediction:
    """Test suite for class prediction."""

    def test_predict_returns_class_labels(self):
        """Test that predict returns class labels."""
        mock_model = Mock()
        mock_model.forward.return_value = np.array([[0.1, 0.9]])
        mock_model.eval.return_value = None
        
        predictor = BertPredictor(model=mock_model)
        
        input_ids = np.random.randint(0, 1000, (1, 128))
        attention_mask = np.ones((1, 128))
        
        predictions = predictor.predict(input_ids, attention_mask)
        
        assert "labels" in predictions

    def test_predict_labels_are_integers(self):
        """Test that predicted labels are integers."""
        mock_model = Mock()
        mock_model.forward.return_value = np.array([[0.1, 0.9], [0.8, 0.2]])
        mock_model.eval.return_value = None
        
        predictor = BertPredictor(model=mock_model)
        
        input_ids = np.random.randint(0, 1000, (2, 128))
        attention_mask = np.ones((2, 128))
        
        predictions = predictor.predict(input_ids, attention_mask)
        
        assert all(isinstance(label, (int, np.integer)) for label in predictions["labels"])

    def test_predict_single_sample(self):
        """Test prediction with single sample."""
        mock_model = Mock()
        mock_model.forward.return_value = np.array([[0.1, 0.9]])
        mock_model.eval.return_value = None
        
        predictor = BertPredictor(model=mock_model)
        
        input_ids = np.random.randint(0, 1000, (1, 128))
        attention_mask = np.ones((1, 128))
        
        predictions = predictor.predict(input_ids, attention_mask)
        
        assert len(predictions["labels"]) == 1


class TestProbabilityScores:
    """Test suite for probability scores."""

    def test_predict_returns_probabilities(self):
        """Test that predict returns probability scores."""
        mock_model = Mock()
        mock_model.forward.return_value = np.array([[0.1, 0.9]])
        mock_model.eval.return_value = None
        
        predictor = BertPredictor(model=mock_model)
        
        input_ids = np.random.randint(0, 1000, (1, 128))
        attention_mask = np.ones((1, 128))
        
        predictions = predictor.predict(input_ids, attention_mask)
        
        assert "probabilities" in predictions

    def test_probabilities_sum_to_one(self):
        """Test that probabilities sum to 1 for each sample."""
        mock_model = Mock()
        mock_model.forward.return_value = np.array([[1.0, 2.0], [3.0, 1.0]])
        mock_model.eval.return_value = None
        
        predictor = BertPredictor(model=mock_model)
        
        input_ids = np.random.randint(0, 1000, (2, 128))
        attention_mask = np.ones((2, 128))
        
        predictions = predictor.predict(input_ids, attention_mask)
        
        for probs in predictions["probabilities"]:
            assert np.isclose(np.sum(probs), 1.0)

    def test_probabilities_in_valid_range(self):
        """Test that probabilities are in [0, 1] range."""
        mock_model = Mock()
        mock_model.forward.return_value = np.array([[0.1, 0.9], [0.8, 0.2]])
        mock_model.eval.return_value = None
        
        predictor = BertPredictor(model=mock_model)
        
        input_ids = np.random.randint(0, 1000, (2, 128))
        attention_mask = np.ones((2, 128))
        
        predictions = predictor.predict(input_ids, attention_mask)
        
        for probs in predictions["probabilities"]:
            assert all(0.0 <= p <= 1.0 for p in probs)


class TestBatchPrediction:
    """Test suite for batch prediction."""

    def test_predict_batch_returns_correct_length(self):
        """Test that batch prediction returns correct number of predictions."""
        mock_model = Mock()
        mock_model.forward.return_value = np.array([[0.1, 0.9], [0.8, 0.2], [0.5, 0.5]])
        mock_model.eval.return_value = None
        
        predictor = BertPredictor(model=mock_model)
        
        batch_size = 3
        input_ids = np.random.randint(0, 1000, (batch_size, 128))
        attention_mask = np.ones((batch_size, 128))
        
        predictions = predictor.predict(input_ids, attention_mask)
        
        assert len(predictions["labels"]) == batch_size
        assert len(predictions["probabilities"]) == batch_size


class TestInputValidation:
    """Test suite for input validation."""

    def test_raises_on_empty_input(self):
        """Test that predictor raises ValueError on empty input."""
        mock_model = Mock()
        predictor = BertPredictor(model=mock_model)
        
        with pytest.raises(ValueError, match="Input cannot be empty"):
            predictor.predict(np.array([]), np.array([]))

    def test_raises_on_mismatched_shapes(self):
        """Test that predictor raises ValueError on mismatched input shapes."""
        mock_model = Mock()
        predictor = BertPredictor(model=mock_model)
        
        input_ids = np.random.randint(0, 1000, (2, 128))
        attention_mask = np.ones((3, 128))  # Wrong shape
        
        with pytest.raises(ValueError, match="Input shapes must match"):
            predictor.predict(input_ids, attention_mask)


class TestModelEvalMode:
    """Test suite for model evaluation mode."""

    def test_predict_sets_model_to_eval_mode(self):
        """Test that predict sets model to evaluation mode."""
        mock_model = Mock()
        mock_model.forward.return_value = np.array([[0.1, 0.9]])
        mock_model.eval.return_value = None
        
        predictor = BertPredictor(model=mock_model)
        
        input_ids = np.random.randint(0, 1000, (1, 128))
        attention_mask = np.ones((1, 128))
        
        predictor.predict(input_ids, attention_mask)
        
        mock_model.eval.assert_called()
