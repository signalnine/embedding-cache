"""Tests for exception classes."""

from vector_embed_client.errors import (
    AuthenticationError,
    NetworkError,
    RateLimitError,
    ServerError,
    ValidationError,
    VectorEmbedError,
)


def test_vector_embed_error_base():
    error = VectorEmbedError("test message", code="test_error", status=500)
    assert str(error) == "test message"
    assert error.code == "test_error"
    assert error.status == 500


def test_authentication_error_inherits():
    error = AuthenticationError("unauthorized", code="auth_error", status=401)
    assert isinstance(error, VectorEmbedError)
    assert error.status == 401


def test_rate_limit_error_retry_after():
    error = RateLimitError("too many requests", code="rate_limit", status=429, retry_after=30)
    assert error.retry_after == 30
    assert isinstance(error, VectorEmbedError)


def test_rate_limit_error_no_retry_after():
    error = RateLimitError("too many requests", code="rate_limit", status=429)
    assert error.retry_after is None


def test_validation_error():
    error = ValidationError("bad input", code="validation_error", status=400)
    assert isinstance(error, VectorEmbedError)


def test_server_error():
    error = ServerError("internal error", code="server_error", status=500)
    assert isinstance(error, VectorEmbedError)


def test_network_error():
    error = NetworkError("connection failed", code="network_error")
    assert isinstance(error, VectorEmbedError)
    assert error.status is None
