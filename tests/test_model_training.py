"""
Tests for baseline model training.
"""
import pytest
import numpy as np
from scipy.sparse import csr_matrix
from src.model_training import train_model


class TestSuccessfulTraining:
    """Test suite for successful model training."""

    def test_train_returns_fitted_model(self):
        """Test that train_model returns a fitted model object."""
        X_train = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        y_train = np.array([0, 1, 1, 0])
        
        model = train_model(X_train, y_train)
        
        assert model is not None

    def test_model_has_predict_method(self):
        """Test that returned model exposes .predict() method."""
        X_train = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        y_train = np.array([0, 1, 1, 0])
        
        model = train_model(X_train, y_train)
        
        assert hasattr(model, 'predict')
        assert callable(getattr(model, 'predict'))

    def test_model_can_predict(self):
        """Test that model can make predictions after training."""
        X_train = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        y_train = np.array([0, 1, 1, 0])
        
        model = train_model(X_train, y_train)
        predictions = model.predict(X_train)
        
        assert predictions is not None
        assert len(predictions) == len(y_train)

    def test_train_with_sparse_matrix(self):
        """Test that training works with sparse matrices."""
        X_train = csr_matrix([[1, 0], [0, 1], [1, 1], [0, 0]])
        y_train = np.array([0, 1, 1, 0])
        
        model = train_model(X_train, y_train)
        
        assert model is not None
        assert hasattr(model, 'predict')

    def test_train_with_larger_dataset(self):
        """Test training with a larger synthetic dataset."""
        np.random.seed(42)
        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(0, 2, 100)
        
        model = train_model(X_train, y_train)
        
        assert model is not None


class TestInputValidation:
    """Test suite for input validation."""

    def test_raises_on_empty_X_train(self):
        """Test that train_model raises ValueError on empty X_train."""
        X_train = np.array([])
        y_train = np.array([0, 1])
        
        with pytest.raises(ValueError, match="X_train cannot be empty"):
            train_model(X_train, y_train)

    def test_raises_on_empty_y_train(self):
        """Test that train_model raises ValueError on empty y_train."""
        X_train = np.array([[1, 0], [0, 1]])
        y_train = np.array([])
        
        with pytest.raises(ValueError, match="y_train cannot be empty"):
            train_model(X_train, y_train)

    def test_raises_on_length_mismatch(self):
        """Test that train_model raises ValueError when lengths don't match."""
        X_train = np.array([[1, 0], [0, 1], [1, 1]])
        y_train = np.array([0, 1])
        
        with pytest.raises(ValueError, match="X_train and y_train must have the same length"):
            train_model(X_train, y_train)

    def test_raises_on_none_X_train(self):
        """Test that train_model raises TypeError on None X_train."""
        y_train = np.array([0, 1])
        
        with pytest.raises(TypeError, match="X_train cannot be None"):
            train_model(None, y_train)

    def test_raises_on_none_y_train(self):
        """Test that train_model raises TypeError on None y_train."""
        X_train = np.array([[1, 0], [0, 1]])
        
        with pytest.raises(TypeError, match="y_train cannot be None"):
            train_model(X_train, None)

    def test_raises_on_single_sample(self):
        """Test that train_model raises ValueError with only one sample."""
        X_train = np.array([[1, 0]])
        y_train = np.array([0])
        
        with pytest.raises(ValueError, match="Need at least 2 samples"):
            train_model(X_train, y_train)


class TestDeterminism:
    """Test suite for deterministic behavior."""

    def test_same_data_same_random_state_identical_predictions(self):
        """Test that training twice with same data and random_state yields identical predictions."""
        X_train = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1]])
        y_train = np.array([0, 1, 1, 0, 0, 1])
        
        model1 = train_model(X_train, y_train, random_state=42)
        predictions1 = model1.predict(X_train)
        
        model2 = train_model(X_train, y_train, random_state=42)
        predictions2 = model2.predict(X_train)
        
        np.testing.assert_array_equal(predictions1, predictions2)

    def test_different_random_state_may_differ(self):
        """Test that different random_state may produce different models."""
        X_train = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1]])
        y_train = np.array([0, 1, 1, 0, 0, 1])
        
        model1 = train_model(X_train, y_train, random_state=42)
        model2 = train_model(X_train, y_train, random_state=99)
        
        # Models should exist (may or may not have different predictions)
        assert model1 is not None
        assert model2 is not None

    def test_default_random_state_deterministic(self):
        """Test that default random_state produces deterministic results."""
        X_train = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        y_train = np.array([0, 1, 1, 0])
        
        model1 = train_model(X_train, y_train)
        predictions1 = model1.predict(X_train)
        
        model2 = train_model(X_train, y_train)
        predictions2 = model2.predict(X_train)
        
        np.testing.assert_array_equal(predictions1, predictions2)


class TestBinaryClassification:
    """Test suite for binary classification constraints."""

    def test_supports_binary_labels(self):
        """Test that model supports binary labels (0 and 1)."""
        X_train = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        y_train = np.array([0, 1, 1, 0])
        
        model = train_model(X_train, y_train)
        
        assert model is not None

    def test_raises_on_more_than_two_classes(self):
        """Test that train_model raises ValueError for more than 2 unique labels."""
        X_train = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        y_train = np.array([0, 1, 2, 0])
        
        with pytest.raises(ValueError, match="Only binary classification is supported"):
            train_model(X_train, y_train)

    def test_raises_on_single_class(self):
        """Test that train_model raises ValueError when only one class present."""
        X_train = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        y_train = np.array([0, 0, 0, 0])
        
        with pytest.raises(ValueError, match="Need at least 2 classes"):
            train_model(X_train, y_train)

    def test_handles_imbalanced_classes(self):
        """Test that model handles imbalanced class distribution."""
        X_train = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1], [1, 1]])
        y_train = np.array([0, 0, 0, 0, 0, 1, 1])  # 5 class 0, 2 class 1
        
        model = train_model(X_train, y_train)
        
        assert model is not None


class TestEdgeCases:
    """Test suite for edge cases."""

    def test_train_with_all_zeros_features(self):
        """Test training with all-zero feature vectors."""
        X_train = np.zeros((10, 5))
        y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        
        model = train_model(X_train, y_train)
        
        assert model is not None

    def test_train_with_high_dimensional_features(self):
        """Test training with high-dimensional feature space."""
        np.random.seed(42)
        X_train = np.random.rand(50, 1000)
        y_train = np.random.randint(0, 2, 50)
        
        model = train_model(X_train, y_train)
        
        assert model is not None

    def test_predictions_are_binary(self):
        """Test that predictions are binary (0 or 1)."""
        X_train = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        y_train = np.array([0, 1, 1, 0])
        
        model = train_model(X_train, y_train)
        predictions = model.predict(X_train)
        
        assert set(predictions).issubset({0, 1})
