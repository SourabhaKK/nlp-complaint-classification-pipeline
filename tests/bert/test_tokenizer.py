"""
Tests for BERT tokenizer.
"""
import pytest
import numpy as np
from src.bert.tokenizer import BertTokenizer


class TestTokenizerInitialization:
    """Test suite for tokenizer initialization."""

    def test_tokenizer_initializes_with_max_length(self):
        """Test that tokenizer initializes with max sequence length."""
        tokenizer = BertTokenizer(max_length=128)
        
        assert tokenizer is not None
        assert tokenizer.max_length == 128

    def test_tokenizer_default_max_length(self):
        """Test that tokenizer has default max length."""
        tokenizer = BertTokenizer()
        
        assert tokenizer.max_length == 512


class TestTextToTokens:
    """Test suite for text tokenization."""

    def test_tokenize_returns_input_ids(self):
        """Test that tokenization returns input_ids."""
        tokenizer = BertTokenizer(max_length=128)
        text = "This is a test complaint"
        
        result = tokenizer.tokenize(text)
        
        assert "input_ids" in result

    def test_tokenize_returns_attention_mask(self):
        """Test that tokenization returns attention_mask."""
        tokenizer = BertTokenizer(max_length=128)
        text = "This is a test complaint"
        
        result = tokenizer.tokenize(text)
        
        assert "attention_mask" in result

    def test_input_ids_shape_matches_max_length(self):
        """Test that input_ids length matches max_length."""
        tokenizer = BertTokenizer(max_length=128)
        text = "This is a test complaint"
        
        result = tokenizer.tokenize(text)
        
        assert len(result["input_ids"]) == 128

    def test_attention_mask_shape_matches_max_length(self):
        """Test that attention_mask length matches max_length."""
        tokenizer = BertTokenizer(max_length=128)
        text = "This is a test complaint"
        
        result = tokenizer.tokenize(text)
        
        assert len(result["attention_mask"]) == 128

    def test_tokenize_long_text_truncates(self):
        """Test that long text is truncated to max_length."""
        tokenizer = BertTokenizer(max_length=64)
        text = "word " * 200  # Very long text
        
        result = tokenizer.tokenize(text)
        
        assert len(result["input_ids"]) == 64


class TestBatchTokenization:
    """Test suite for batch tokenization."""

    def test_tokenize_batch_returns_dict(self):
        """Test that batch tokenization returns dictionary."""
        tokenizer = BertTokenizer(max_length=128)
        texts = ["complaint one", "complaint two", "complaint three"]
        
        result = tokenizer.tokenize_batch(texts)
        
        assert isinstance(result, dict)

    def test_batch_input_ids_shape(self):
        """Test that batch input_ids has correct shape."""
        tokenizer = BertTokenizer(max_length=128)
        texts = ["complaint one", "complaint two", "complaint three"]
        
        result = tokenizer.tokenize_batch(texts)
        
        assert result["input_ids"].shape == (3, 128)

    def test_batch_attention_mask_shape(self):
        """Test that batch attention_mask has correct shape."""
        tokenizer = BertTokenizer(max_length=128)
        texts = ["complaint one", "complaint two", "complaint three"]
        
        result = tokenizer.tokenize_batch(texts)
        
        assert result["attention_mask"].shape == (3, 128)


class TestInputValidation:
    """Test suite for input validation."""

    def test_raises_on_empty_string(self):
        """Test that tokenizer raises ValueError on empty string."""
        tokenizer = BertTokenizer(max_length=128)
        
        with pytest.raises(ValueError, match="Input text cannot be empty"):
            tokenizer.tokenize("")

    def test_raises_on_none_input(self):
        """Test that tokenizer raises ValueError on None input."""
        tokenizer = BertTokenizer(max_length=128)
        
        with pytest.raises(ValueError, match="Input text cannot be None"):
            tokenizer.tokenize(None)

    def test_raises_on_empty_batch(self):
        """Test that tokenizer raises ValueError on empty batch."""
        tokenizer = BertTokenizer(max_length=128)
        
        with pytest.raises(ValueError, match="Batch cannot be empty"):
            tokenizer.tokenize_batch([])


class TestDeterminism:
    """Test suite for deterministic behavior."""

    def test_same_text_same_tokens(self):
        """Test that same text produces same tokens."""
        tokenizer = BertTokenizer(max_length=128)
        text = "This is a test complaint"
        
        result1 = tokenizer.tokenize(text)
        result2 = tokenizer.tokenize(text)
        
        np.testing.assert_array_equal(result1["input_ids"], result2["input_ids"])
        np.testing.assert_array_equal(result1["attention_mask"], result2["attention_mask"])
