"""Tests for text normalization."""

from vector_embed_cache.normalize import normalize_text
from vector_embed_cache.utils import generate_cache_key


def test_normalize_lowercase():
    """Should convert text to lowercase."""
    assert normalize_text("Hello World") == "hello world"


def test_normalize_strip_whitespace():
    """Should strip leading and trailing whitespace."""
    assert normalize_text("  hello world  ") == "hello world"


def test_normalize_collapse_internal_spaces():
    """Should collapse runs of spaces to a single space."""
    assert normalize_text("hello  world") == "hello world"
    assert normalize_text("hello     world") == "hello world"


def test_normalize_collapse_tabs_and_newlines():
    """Should treat tabs and newlines as whitespace and collapse them."""
    assert normalize_text("hello\tworld") == "hello world"
    assert normalize_text("hello\nworld") == "hello world"
    assert normalize_text("hello \t\n world") == "hello world"


def test_normalize_whitespace_variants_produce_same_cache_key():
    """All whitespace variants should yield identical cache keys."""
    model = "nomic-ai/nomic-embed-text-v1.5"
    variants = [
        "hello world",
        "hello  world",
        "hello\tworld",
        "hello\nworld",
        "  hello   world  ",
        "Hello\tWorld",
    ]
    keys = {generate_cache_key(normalize_text(v), model) for v in variants}
    assert len(keys) == 1


def test_normalize_empty_string():
    """Should handle empty strings."""
    assert normalize_text("") == ""


def test_normalize_unicode():
    """Should handle unicode characters."""
    assert normalize_text("Café") == "café"
