"""
Tests for BERT model.
"""
import pytest
import numpy as np
from unittest.mock import Mock
from src.bert.model import BertClassifier


class TestModelInitialization:
    """Test suite for model initialization."""

    def test_model_initializes_with_num_labels(self):
        """Test that model initializes with number of labels."""
        model = BertClassifier(num_labels=2)
        
        assert model is not None

    def test_model_has_num_labels_attribute(self):
        """Test that model has num_labels attribute."""
        model = BertClassifier(num_labels=2)
        
        assert hasattr(model, 'num_labels')
        assert model.num_labels == 2

    def test_model_not_trained_on_init(self):
        """Test that model is not trained during initialization."""
        model = BertClassifier(num_labels=2)
        
        # Model should not have training history or be in trained state
        assert not hasattr(model, 'training_history') or model.training_history is None


class TestModelForward:
    """Test suite for model forward pass."""

    def test_forward_returns_logits(self):
        """Test that forward pass returns logits."""
        model = BertClassifier(num_labels=2)
        
        # Mock input tensors
        input_ids = np.random.randint(0, 1000, (4, 128))
        attention_mask = np.ones((4, 128))
        
        logits = model.forward(input_ids, attention_mask)
        
        assert logits is not None

    def test_logits_shape_matches_batch_and_labels(self):
        """Test that logits shape is (batch_size, num_labels)."""
        model = BertClassifier(num_labels=2)
        
        batch_size = 8
        input_ids = np.random.randint(0, 1000, (batch_size, 128))
        attention_mask = np.ones((batch_size, 128))
        
        logits = model.forward(input_ids, attention_mask)
        
        assert logits.shape == (batch_size, 2)

    def test_forward_with_single_sample(self):
        """Test forward pass with single sample."""
        model = BertClassifier(num_labels=2)
        
        input_ids = np.random.randint(0, 1000, (1, 128))
        attention_mask = np.ones((1, 128))
        
        logits = model.forward(input_ids, attention_mask)
        
        assert logits.shape == (1, 2)


class TestModelConfiguration:
    """Test suite for model configuration."""

    def test_model_supports_binary_classification(self):
        """Test that model supports binary classification."""
        model = BertClassifier(num_labels=2)
        
        assert model.num_labels == 2

    def test_model_raises_on_invalid_num_labels(self):
        """Test that model raises ValueError for invalid num_labels."""
        with pytest.raises(ValueError, match="num_labels must be at least 2"):
            BertClassifier(num_labels=1)


class TestModelMethods:
    """Test suite for model methods."""

    def test_model_has_eval_method(self):
        """Test that model has eval method."""
        model = BertClassifier(num_labels=2)
        
        assert hasattr(model, 'eval')
        assert callable(model.eval)

    def test_model_has_train_method(self):
        """Test that model has train method."""
        model = BertClassifier(num_labels=2)
        
        assert hasattr(model, 'train')
        assert callable(model.train)
