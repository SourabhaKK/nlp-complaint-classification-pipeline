"""
Tests for TF-IDF vectorization.
"""
import pytest
import numpy as np
from scipy.sparse import issparse
from src.vectorizer import fit_vectorizer, transform_texts


class TestFitPhase:
    """Test suite for vectorizer fitting."""

    def test_fit_returns_vectorizer_object(self):
        """Test that fit_vectorizer returns a fitted object."""
        texts = ["hello world", "test document"]
        vectorizer = fit_vectorizer(texts)
        
        assert vectorizer is not None
        assert hasattr(vectorizer, 'transform') or hasattr(vectorizer, 'vocabulary_')

    def test_fit_raises_on_empty_input(self):
        """Test that fit_vectorizer raises ValueError on empty list."""
        with pytest.raises(ValueError, match="Input texts cannot be empty"):
            fit_vectorizer([])

    def test_fit_raises_on_none_input(self):
        """Test that fit_vectorizer raises ValueError on None input."""
        with pytest.raises(ValueError, match="Input texts cannot be None"):
            fit_vectorizer(None)

    def test_fit_raises_on_non_list_input(self):
        """Test that fit_vectorizer raises TypeError on non-list input."""
        with pytest.raises(TypeError, match="Input must be a list"):
            fit_vectorizer("not a list")

    def test_fit_raises_on_non_string_elements(self):
        """Test that fit_vectorizer raises TypeError if list contains non-strings."""
        with pytest.raises(TypeError, match="All elements must be strings"):
            fit_vectorizer(["hello", 123, "world"])

    def test_fit_with_single_document(self):
        """Test that fit_vectorizer works with single document."""
        texts = ["hello world"]
        vectorizer = fit_vectorizer(texts)
        
        assert vectorizer is not None

    def test_fit_with_multiple_documents(self):
        """Test that fit_vectorizer works with multiple documents."""
        texts = ["hello world", "test document", "another example"]
        vectorizer = fit_vectorizer(texts)
        
        assert vectorizer is not None


class TestTransformPhase:
    """Test suite for text transformation."""

    def test_transform_returns_matrix(self):
        """Test that transform_texts returns a numeric matrix."""
        texts = ["hello world", "test document"]
        vectorizer = fit_vectorizer(texts)
        result = transform_texts(vectorizer, texts)
        
        # Should be sparse matrix or numpy array
        assert issparse(result) or isinstance(result, np.ndarray)

    def test_transform_row_count_matches_input(self):
        """Test that output row count equals input length."""
        texts = ["hello world", "test document", "another text"]
        vectorizer = fit_vectorizer(texts)
        result = transform_texts(vectorizer, texts)
        
        assert result.shape[0] == len(texts)

    def test_transform_column_count_deterministic(self):
        """Test that output column count is deterministic."""
        texts = ["hello world", "test document"]
        vectorizer = fit_vectorizer(texts)
        result1 = transform_texts(vectorizer, texts)
        result2 = transform_texts(vectorizer, texts)
        
        assert result1.shape[1] == result2.shape[1]

    def test_transform_raises_on_none_vectorizer(self):
        """Test that transform_texts raises ValueError if vectorizer is None."""
        texts = ["hello world"]
        with pytest.raises(ValueError, match="Vectorizer cannot be None"):
            transform_texts(None, texts)

    def test_transform_raises_on_empty_texts(self):
        """Test that transform_texts raises ValueError on empty list."""
        texts = ["hello world"]
        vectorizer = fit_vectorizer(texts)
        
        with pytest.raises(ValueError, match="Input texts cannot be empty"):
            transform_texts(vectorizer, [])

    def test_transform_single_document(self):
        """Test transforming a single document."""
        train_texts = ["hello world", "test document"]
        vectorizer = fit_vectorizer(train_texts)
        
        test_texts = ["hello"]
        result = transform_texts(vectorizer, test_texts)
        
        assert result.shape[0] == 1

    def test_transform_new_documents(self):
        """Test transforming documents not in training set."""
        train_texts = ["hello world", "test document"]
        vectorizer = fit_vectorizer(train_texts)
        
        test_texts = ["new text", "another example"]
        result = transform_texts(vectorizer, test_texts)
        
        assert result.shape[0] == len(test_texts)


class TestLeakageSafety:
    """Test suite for data leakage prevention."""

    def test_fit_only_on_training_data(self):
        """Test that fit_vectorizer is only used for training data."""
        train_texts = ["hello world", "test document"]
        test_texts = ["new text"]
        
        vectorizer = fit_vectorizer(train_texts)
        
        # Transform should work without refitting
        result = transform_texts(vectorizer, test_texts)
        assert result is not None

    def test_transform_does_not_refit(self):
        """Test that transform_texts does not refit the vectorizer."""
        train_texts = ["hello world", "test document"]
        vectorizer = fit_vectorizer(train_texts)
        
        # Get vocabulary size
        result1 = transform_texts(vectorizer, train_texts)
        vocab_size_1 = result1.shape[1]
        
        # Transform with new text
        new_texts = ["completely different words here"]
        result2 = transform_texts(vectorizer, new_texts)
        
        # Transform original again
        result3 = transform_texts(vectorizer, train_texts)
        vocab_size_3 = result3.shape[1]
        
        # Vocabulary size should not change
        assert vocab_size_1 == vocab_size_3

    def test_transform_twice_identical_results(self):
        """Test that calling transform twice yields identical results."""
        texts = ["hello world", "test document"]
        vectorizer = fit_vectorizer(texts)
        
        result1 = transform_texts(vectorizer, texts)
        result2 = transform_texts(vectorizer, texts)
        
        # Results should be identical
        if issparse(result1):
            assert np.allclose(result1.toarray(), result2.toarray())
        else:
            assert np.allclose(result1, result2)

    def test_unseen_words_handled_correctly(self):
        """Test that unseen words in test set don't affect vocabulary."""
        train_texts = ["hello world"]
        test_texts = ["unseen vocabulary words"]
        
        vectorizer = fit_vectorizer(train_texts)
        
        # Should not raise error
        result = transform_texts(vectorizer, test_texts)
        assert result is not None


class TestDeterminism:
    """Test suite for deterministic behavior."""

    def test_same_input_same_output(self):
        """Test that same input produces same feature matrix."""
        texts = ["hello world", "test document"]
        
        vectorizer1 = fit_vectorizer(texts)
        result1 = transform_texts(vectorizer1, texts)
        
        vectorizer2 = fit_vectorizer(texts)
        result2 = transform_texts(vectorizer2, texts)
        
        # Results should be identical
        if issparse(result1):
            assert np.allclose(result1.toarray(), result2.toarray())
        else:
            assert np.allclose(result1, result2)

    def test_vocabulary_does_not_change(self):
        """Test that vocabulary does not change across transforms."""
        texts = ["hello world", "test document"]
        vectorizer = fit_vectorizer(texts)
        
        result1 = transform_texts(vectorizer, texts)
        result2 = transform_texts(vectorizer, ["hello"])
        result3 = transform_texts(vectorizer, texts)
        
        # Column count should remain the same
        assert result1.shape[1] == result3.shape[1]

    def test_fit_deterministic_vocabulary_size(self):
        """Test that fitting produces deterministic vocabulary size."""
        texts = ["hello world", "test document", "hello test"]
        
        vectorizer1 = fit_vectorizer(texts)
        result1 = transform_texts(vectorizer1, texts)
        
        vectorizer2 = fit_vectorizer(texts)
        result2 = transform_texts(vectorizer2, texts)
        
        assert result1.shape[1] == result2.shape[1]


class TestEdgeCases:
    """Test suite for edge cases."""

    def test_repeated_documents(self):
        """Test handling of repeated documents."""
        texts = ["hello world", "hello world", "test"]
        vectorizer = fit_vectorizer(texts)
        result = transform_texts(vectorizer, texts)
        
        assert result.shape[0] == 3

    def test_empty_strings_after_preprocessing(self):
        """Test handling of texts that become empty after preprocessing."""
        texts = ["hello world", "   ", "test"]
        vectorizer = fit_vectorizer(texts)
        result = transform_texts(vectorizer, texts)
        
        assert result.shape[0] == 3

    def test_single_word_documents(self):
        """Test handling of single-word documents."""
        texts = ["hello", "world", "test"]
        vectorizer = fit_vectorizer(texts)
        result = transform_texts(vectorizer, texts)
        
        assert result.shape[0] == 3

    def test_very_long_document(self):
        """Test handling of very long documents."""
        texts = ["word " * 1000, "test"]
        vectorizer = fit_vectorizer(texts)
        result = transform_texts(vectorizer, texts)
        
        assert result.shape[0] == 2

    def test_special_characters_handled(self):
        """Test that special characters are handled (should be removed by preprocessing)."""
        texts = ["hello@world", "test#document"]
        vectorizer = fit_vectorizer(texts)
        result = transform_texts(vectorizer, texts)
        
        assert result is not None
