# server/tests/test_passthrough.py
import pytest
from app.passthrough import validate_endpoint, is_allowed_host, extract_embedding


def test_is_allowed_host_openai():
    """OpenAI should be allowed."""
    assert is_allowed_host("api.openai.com") is True


def test_is_allowed_host_localhost_blocked():
    """Localhost should be blocked."""
    assert is_allowed_host("localhost") is False
    assert is_allowed_host("127.0.0.1") is False


def test_is_allowed_host_private_ip_blocked():
    """Private IPs should be blocked."""
    assert is_allowed_host("192.168.1.1") is False
    assert is_allowed_host("10.0.0.1") is False


def test_validate_endpoint_valid():
    """Valid OpenAI endpoint should pass."""
    validate_endpoint("https://api.openai.com/v1/embeddings")


def test_validate_endpoint_http_rejected():
    """HTTP (not HTTPS) should be rejected."""
    with pytest.raises(ValueError, match="HTTPS required"):
        validate_endpoint("http://api.openai.com/v1/embeddings")


def test_extract_embedding_jsonpath():
    """Should extract embedding using JSONPath."""
    response = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
    result = extract_embedding(response, "$.data[0].embedding")
    assert result == [0.1, 0.2, 0.3]
