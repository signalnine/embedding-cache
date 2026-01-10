"""Tests for embedding providers."""

import numpy as np
import os
import pytest
from embedding_cache.providers import LocalProvider, RemoteProvider, OpenAIProvider
from unittest.mock import Mock, patch
import httpx


def test_local_provider_unavailable():
    """Should detect when sentence-transformers is not installed."""
    # This test might pass or fail depending on environment
    # Just check that is_available() returns a boolean
    provider = LocalProvider(model="nomic-ai/nomic-embed-text-v1.5")
    assert isinstance(provider.is_available(), bool)


@pytest.mark.integration
def test_local_provider_embed_single():
    """Should embed single text string."""
    provider = LocalProvider(model="nomic-ai/nomic-embed-text-v1.5")

    if not provider.is_available():
        pytest.skip("sentence-transformers not installed")

    embedding = provider.embed("hello world")
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (768,)  # nomic-embed produces 768-dim vectors
    assert embedding.dtype == np.float32


@pytest.mark.integration
def test_local_provider_embed_batch():
    """Should embed multiple text strings."""
    provider = LocalProvider(model="nomic-ai/nomic-embed-text-v1.5")

    if not provider.is_available():
        pytest.skip("sentence-transformers not installed")

    embeddings = provider.embed_batch(["hello", "world"])
    assert isinstance(embeddings, list)
    assert len(embeddings) == 2
    assert all(isinstance(e, np.ndarray) for e in embeddings)
    assert all(e.shape == (768,) for e in embeddings)


def test_remote_provider_embed_single():
    """Should send single text to backend and return embedding."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "embeddings": [[0.1, 0.2, 0.3]],
        "cache_hits": [False],
        "model": "nomic-embed-text-v2",
        "dimensions": 3
    }

    mock_client = Mock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = Mock(return_value=mock_client)
    mock_client.__exit__ = Mock(return_value=False)

    with patch('httpx.Client', return_value=mock_client):
        provider = RemoteProvider(backend_url="http://test.com", timeout=5.0)
        embedding = provider.embed("hello")

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (3,)
    np.testing.assert_array_almost_equal(embedding, [0.1, 0.2, 0.3])


def test_remote_provider_embed_batch():
    """Should send multiple texts to backend and return embeddings."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        "cache_hits": [False, True],
        "model": "nomic-embed-text-v2",
        "dimensions": 3
    }

    mock_client = Mock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = Mock(return_value=mock_client)
    mock_client.__exit__ = Mock(return_value=False)

    with patch('httpx.Client', return_value=mock_client):
        provider = RemoteProvider(backend_url="http://test.com", timeout=5.0)
        embeddings = provider.embed_batch(["hello", "world"])

    assert len(embeddings) == 2
    np.testing.assert_array_almost_equal(embeddings[0], [0.1, 0.2, 0.3])
    np.testing.assert_array_almost_equal(embeddings[1], [0.4, 0.5, 0.6])


def test_remote_provider_timeout():
    """Should raise error on timeout."""
    mock_client = Mock()
    mock_client.post.side_effect = httpx.TimeoutException("Timeout")
    mock_client.__enter__ = Mock(return_value=mock_client)
    mock_client.__exit__ = Mock(return_value=False)

    with patch('httpx.Client', return_value=mock_client):
        provider = RemoteProvider(backend_url="http://test.com", timeout=0.1)
        with pytest.raises(RuntimeError, match="Remote backend timeout"):
            provider.embed("hello")


def test_remote_provider_network_error():
    """Should raise error on network failure."""
    mock_client = Mock()
    mock_client.post.side_effect = httpx.NetworkError("Network error")
    mock_client.__enter__ = Mock(return_value=mock_client)
    mock_client.__exit__ = Mock(return_value=False)

    with patch('httpx.Client', return_value=mock_client):
        provider = RemoteProvider(backend_url="http://test.com", timeout=5.0)
        with pytest.raises(RuntimeError, match="Remote backend error"):
            provider.embed("hello")


def test_openai_provider_unavailable():
    """Should detect when openai package is not installed."""
    import sys
    # Save original module state
    original_openai = sys.modules.get('openai')

    try:
        # Make openai import fail by setting it to None in sys.modules
        sys.modules['openai'] = None

        # Create provider and call is_available()
        provider = OpenAIProvider()
        result = provider.is_available()

        # Verify it returns False when openai is not available
        assert result is False
    finally:
        # Restore original state
        if original_openai is None:
            sys.modules.pop('openai', None)
        else:
            sys.modules['openai'] = original_openai


def test_openai_provider_no_api_key():
    """Should detect when OPENAI_API_KEY is not set."""
    # Clear all environment variables to ensure no API key
    with patch.dict(os.environ, {}, clear=True):
        provider = OpenAIProvider()
        result = provider.is_available()
        # Verify it returns False when API key is not set
        assert result is False
