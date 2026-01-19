# server/tests/test_normalize.py
import pytest
from app.normalize import normalize_text


def test_normalize_strips_whitespace():
    assert normalize_text("  hello  ") == "hello"


def test_normalize_collapses_internal_whitespace():
    assert normalize_text("hello    world") == "hello world"


def test_normalize_lowercases():
    assert normalize_text("HELLO World") == "hello world"


def test_normalize_handles_newlines():
    assert normalize_text("hello\n\nworld") == "hello world"


def test_normalize_empty_string():
    assert normalize_text("") == ""
