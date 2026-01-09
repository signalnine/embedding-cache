"""Tests for utility functions."""

from embedding_cache.utils import generate_cache_key


def test_generate_cache_key_consistent():
    """Should generate consistent keys for same input."""
    key1 = generate_cache_key("hello", "model-v1")
    key2 = generate_cache_key("hello", "model-v1")
    assert key1 == key2


def test_generate_cache_key_different_text():
    """Should generate different keys for different text."""
    key1 = generate_cache_key("hello", "model-v1")
    key2 = generate_cache_key("world", "model-v1")
    assert key1 != key2


def test_generate_cache_key_different_model():
    """Should generate different keys for different models."""
    key1 = generate_cache_key("hello", "model-v1")
    key2 = generate_cache_key("hello", "model-v2")
    assert key1 != key2


def test_generate_cache_key_format():
    """Should return a hex string (SHA-256)."""
    key = generate_cache_key("hello", "model-v1")
    assert isinstance(key, str)
    assert len(key) == 64  # SHA-256 hex length
    assert all(c in '0123456789abcdef' for c in key)


def test_generate_cache_key_no_delimiter_collision():
    """Should handle delimiter characters in inputs without collision."""
    key1 = generate_cache_key("text", "model::with::colons")
    key2 = generate_cache_key("text::with::colons", "model")
    assert key1 != key2, "Different inputs must produce different keys"
