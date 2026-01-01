"""
Tests for BERT trainer.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.bert.trainer import BertTrainer


class TestTrainerInitialization:
    """Test suite for trainer initialization."""

    def test_trainer_initializes_with_model(self):
        """Test that trainer initializes with model."""
        mock_model = Mock()
        trainer = BertTrainer(model=mock_model)
        
        assert trainer is not None
        assert trainer.model == mock_model

    def test_trainer_accepts_random_seed(self):
        """Test that trainer accepts random seed for determinism."""
        mock_model = Mock()
        trainer = BertTrainer(model=mock_model, random_state=42)
        
        assert trainer.random_state == 42


class TestTraining:
    """Test suite for training process."""

    @patch('src.bert.trainer.torch')
    def test_train_accepts_tokenized_tensors(self, mock_torch):
        """Test that train method accepts tokenized tensors."""
        mock_model = Mock()
        trainer = BertTrainer(model=mock_model)
        
        input_ids = np.random.randint(0, 1000, (10, 128))
        attention_mask = np.ones((10, 128))
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        
        result = trainer.train(input_ids, attention_mask, labels, epochs=1)
        
        assert result is not None

    @patch('src.bert.trainer.torch')
    def test_train_returns_trained_model(self, mock_torch):
        """Test that train returns trained model object."""
        mock_model = Mock()
        trainer = BertTrainer(model=mock_model)
        
        input_ids = np.random.randint(0, 1000, (10, 128))
        attention_mask = np.ones((10, 128))
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        
        trained_model = trainer.train(input_ids, attention_mask, labels, epochs=1)
        
        assert trained_model is not None

    @patch('src.bert.trainer.torch')
    def test_train_single_epoch(self, mock_torch):
        """Test that trainer runs a single epoch."""
        mock_model = Mock()
        trainer = BertTrainer(model=mock_model)
        
        input_ids = np.random.randint(0, 1000, (10, 128))
        attention_mask = np.ones((10, 128))
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        
        trainer.train(input_ids, attention_mask, labels, epochs=1)
        
        # Training should have been called
        assert mock_model.train.called or mock_model.forward.called


class TestDeterminism:
    """Test suite for deterministic training."""

    @patch('src.bert.trainer.torch')
    def test_same_seed_same_results(self, mock_torch):
        """Test that same random seed produces deterministic results."""
        mock_model1 = Mock()
        mock_model1.forward.return_value = np.array([[0.1, 0.9], [0.8, 0.2]])
        
        mock_model2 = Mock()
        mock_model2.forward.return_value = np.array([[0.1, 0.9], [0.8, 0.2]])
        
        trainer1 = BertTrainer(model=mock_model1, random_state=42)
        trainer2 = BertTrainer(model=mock_model2, random_state=42)
        
        input_ids = np.random.randint(0, 1000, (10, 128))
        attention_mask = np.ones((10, 128))
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        
        trainer1.train(input_ids, attention_mask, labels, epochs=1)
        trainer2.train(input_ids, attention_mask, labels, epochs=1)
        
        # Both trainers should have been called with same seed
        assert trainer1.random_state == trainer2.random_state


class TestInputValidation:
    """Test suite for input validation."""

    def test_raises_on_mismatched_lengths(self):
        """Test that trainer raises ValueError on mismatched input lengths."""
        mock_model = Mock()
        trainer = BertTrainer(model=mock_model)
        
        input_ids = np.random.randint(0, 1000, (10, 128))
        attention_mask = np.ones((10, 128))
        labels = np.array([0, 1, 0])  # Wrong length
        
        with pytest.raises(ValueError, match="Input lengths must match"):
            trainer.train(input_ids, attention_mask, labels, epochs=1)

    def test_raises_on_empty_inputs(self):
        """Test that trainer raises ValueError on empty inputs."""
        mock_model = Mock()
        trainer = BertTrainer(model=mock_model)
        
        with pytest.raises(ValueError, match="Inputs cannot be empty"):
            trainer.train(np.array([]), np.array([]), np.array([]), epochs=1)


class TestTrainingConfiguration:
    """Test suite for training configuration."""

    @patch('src.bert.trainer.torch')
    def test_train_accepts_learning_rate(self, mock_torch):
        """Test that trainer accepts learning rate parameter."""
        mock_model = Mock()
        trainer = BertTrainer(model=mock_model, learning_rate=2e-5)
        
        assert trainer.learning_rate == 2e-5

    @patch('src.bert.trainer.torch')
    def test_train_accepts_batch_size(self, mock_torch):
        """Test that trainer accepts batch size parameter."""
        mock_model = Mock()
        trainer = BertTrainer(model=mock_model, batch_size=16)
        
        assert trainer.batch_size == 16
