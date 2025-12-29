"""
Tests for deterministic text preprocessing.
"""
import pytest
from src.text_preprocessing import preprocess_text


class TestInputValidation:
    """Test suite for input validation."""

    def test_preprocess_none_input(self):
        """Test that preprocessing fails when input is None."""
        with pytest.raises(ValueError, match="Input text cannot be None"):
            preprocess_text(None)

    def test_preprocess_non_string_input_integer(self):
        """Test that preprocessing fails when input is an integer."""
        with pytest.raises(TypeError, match="Input must be a string"):
            preprocess_text(123)

    def test_preprocess_non_string_input_list(self):
        """Test that preprocessing fails when input is a list."""
        with pytest.raises(TypeError, match="Input must be a string"):
            preprocess_text(["hello", "world"])

    def test_preprocess_non_string_input_dict(self):
        """Test that preprocessing fails when input is a dict."""
        with pytest.raises(TypeError, match="Input must be a string"):
            preprocess_text({"text": "hello"})

    def test_preprocess_empty_string(self):
        """Test that preprocessing returns empty string for empty input."""
        result = preprocess_text("")
        assert result == ""

    def test_preprocess_whitespace_only(self):
        """Test that preprocessing returns empty string for whitespace-only input."""
        result = preprocess_text("   ")
        assert result == ""


class TestNormalization:
    """Test suite for text normalization."""

    def test_converts_to_lowercase(self):
        """Test that text is converted to lowercase."""
        result = preprocess_text("HELLO WORLD")
        assert result == "hello world"

    def test_converts_mixed_case_to_lowercase(self):
        """Test that mixed case text is converted to lowercase."""
        result = preprocess_text("HeLLo WoRLd")
        assert result == "hello world"

    def test_strips_leading_whitespace(self):
        """Test that leading whitespace is removed."""
        result = preprocess_text("   hello world")
        assert result == "hello world"

    def test_strips_trailing_whitespace(self):
        """Test that trailing whitespace is removed."""
        result = preprocess_text("hello world   ")
        assert result == "hello world"

    def test_strips_leading_and_trailing_whitespace(self):
        """Test that both leading and trailing whitespace are removed."""
        result = preprocess_text("   hello world   ")
        assert result == "hello world"

    def test_collapses_multiple_spaces(self):
        """Test that multiple consecutive spaces are collapsed to one."""
        result = preprocess_text("hello    world")
        assert result == "hello world"

    def test_collapses_multiple_spaces_complex(self):
        """Test that multiple spaces throughout text are collapsed."""
        result = preprocess_text("hello   world   this   is   a   test")
        assert result == "hello world this is a test"

    def test_handles_tabs_and_newlines(self):
        """Test that tabs and newlines are normalized to spaces."""
        result = preprocess_text("hello\tworld\nthis\ris")
        assert result == "hello world this is"


class TestNoiseRemoval:
    """Test suite for noise removal."""

    def test_removes_punctuation_periods(self):
        """Test that periods are removed."""
        result = preprocess_text("Hello. World.")
        assert result == "hello world"

    def test_removes_punctuation_commas(self):
        """Test that commas are removed."""
        result = preprocess_text("Hello, World")
        assert result == "hello world"

    def test_removes_punctuation_exclamation(self):
        """Test that exclamation marks are removed."""
        result = preprocess_text("Hello! World!")
        assert result == "hello world"

    def test_removes_punctuation_question(self):
        """Test that question marks are removed."""
        result = preprocess_text("Hello? World?")
        assert result == "hello world"

    def test_removes_all_punctuation(self):
        """Test that all common punctuation is removed."""
        result = preprocess_text("Hello, World! How are you? I'm fine.")
        assert result == "hello world how are you im fine"

    def test_removes_special_characters(self):
        """Test that special characters are removed."""
        result = preprocess_text("Hello@World#Test$")
        assert result == "helloworld test"

    def test_removes_numeric_digits(self):
        """Test that numeric digits are removed."""
        result = preprocess_text("Hello123World456")
        assert result == "helloworld"

    def test_removes_numbers_and_punctuation(self):
        """Test that both numbers and punctuation are removed."""
        result = preprocess_text("Product123 costs $99.99!")
        assert result == "product costs"

    def test_preserves_alphabetic_characters(self):
        """Test that alphabetic characters are preserved."""
        result = preprocess_text("abc XYZ")
        assert result == "abc xyz"

    def test_removes_embedded_numbers(self):
        """Test that numbers embedded in text are removed."""
        result = preprocess_text("test123data456end")
        assert result == "testdataend"


class TestDeterminism:
    """Test suite for deterministic behavior."""

    def test_same_input_same_output(self):
        """Test that same input always produces same output."""
        text = "Hello, World! 123"
        result1 = preprocess_text(text)
        result2 = preprocess_text(text)
        assert result1 == result2

    def test_deterministic_complex_text(self):
        """Test determinism with complex text."""
        text = "This is a TEST! With Numbers 456 and Symbols @#$"
        result1 = preprocess_text(text)
        result2 = preprocess_text(text)
        result3 = preprocess_text(text)
        assert result1 == result2 == result3

    def test_no_randomness(self):
        """Test that there is no randomness in preprocessing."""
        text = "Random Test 789"
        results = [preprocess_text(text) for _ in range(10)]
        assert len(set(results)) == 1  # All results should be identical


class TestIdempotency:
    """Test suite for idempotent behavior."""

    def test_idempotent_single_application(self):
        """Test that applying preprocessing twice gives same result as once."""
        text = "Hello, World! 123"
        once = preprocess_text(text)
        twice = preprocess_text(once)
        assert once == twice

    def test_idempotent_multiple_applications(self):
        """Test that multiple applications give same result."""
        text = "COMPLEX Text! With 999 Numbers"
        once = preprocess_text(text)
        twice = preprocess_text(preprocess_text(text))
        thrice = preprocess_text(preprocess_text(preprocess_text(text)))
        assert once == twice == thrice

    def test_idempotent_already_clean_text(self):
        """Test that already clean text remains unchanged."""
        text = "hello world"
        result = preprocess_text(text)
        assert result == text
        # Applying again should give same result
        assert preprocess_text(result) == text


class TestEdgeCases:
    """Test suite for edge cases."""

    def test_single_character(self):
        """Test preprocessing of single character."""
        result = preprocess_text("A")
        assert result == "a"

    def test_single_punctuation(self):
        """Test preprocessing of single punctuation mark."""
        result = preprocess_text("!")
        assert result == ""

    def test_only_numbers(self):
        """Test preprocessing of text with only numbers."""
        result = preprocess_text("123456")
        assert result == ""

    def test_only_special_characters(self):
        """Test preprocessing of text with only special characters."""
        result = preprocess_text("@#$%^&*()")
        assert result == ""

    def test_mixed_special_and_numbers(self):
        """Test preprocessing of mixed special characters and numbers."""
        result = preprocess_text("123@#$456")
        assert result == ""

    def test_unicode_characters(self):
        """Test that unicode/non-ASCII characters are handled."""
        result = preprocess_text("Café résumé")
        # Should preserve letters, remove accents is not required
        assert "caf" in result.lower()

    def test_very_long_text(self):
        """Test preprocessing of very long text."""
        text = "Hello World " * 1000
        result = preprocess_text(text)
        assert result == "hello world " * 999 + "hello world"
        assert result.strip() == ("hello world " * 1000).strip().lower()
