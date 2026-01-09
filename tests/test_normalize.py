"""Tests for text normalization."""

from embedding_cache.normalize import normalize_text


def test_normalize_lowercase():
    """Should convert text to lowercase."""
    assert normalize_text("Hello World") == "hello world"


def test_normalize_strip_whitespace():
    """Should strip leading and trailing whitespace."""
    assert normalize_text("  hello world  ") == "hello world"


def test_normalize_internal_whitespace():
    """Should preserve internal whitespace."""
    assert normalize_text("hello  world") == "hello  world"


def test_normalize_empty_string():
    """Should handle empty strings."""
    assert normalize_text("") == ""


def test_normalize_unicode():
    """Should handle unicode characters."""
    assert normalize_text("Café") == "café"
