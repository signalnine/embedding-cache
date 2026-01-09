"""Tests for public API."""

import pytest
import numpy as np
import tempfile
from unittest.mock import Mock, patch

from embedding_cache import embed, EmbeddingCache


def test_embed_function_exists():
    """Test that embed function is exported."""
    assert callable(embed)


def test_embed_class_exists():
    """Test that EmbeddingCache class is exported."""
    assert EmbeddingCache is not None
    assert callable(EmbeddingCache)


def test_embed_function_simple_usage():
    """Test that embed() function works with simple function call."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock the provider to avoid requiring sentence-transformers
        test_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        with patch('embedding_cache.cache.EmbeddingCache.__init__', return_value=None) as mock_init:
            with patch('embedding_cache.cache.EmbeddingCache.embed', return_value=test_embedding) as mock_embed:
                # Need to mock the global singleton creation
                import embedding_cache
                embedding_cache._default_cache = None

                # Patch EmbeddingCache to return a mock instance
                mock_cache_instance = Mock()
                mock_cache_instance.embed = Mock(return_value=test_embedding)

                with patch('embedding_cache.EmbeddingCache') as MockCache:
                    MockCache.return_value = mock_cache_instance

                    # Call embed function
                    result = embed("Hello world")

                    # Should have created singleton
                    MockCache.assert_called_once()

                    # Should have called embed on the instance
                    mock_cache_instance.embed.assert_called_once_with("Hello world")

                    # Should return the embedding
                    np.testing.assert_array_equal(result, test_embedding)
