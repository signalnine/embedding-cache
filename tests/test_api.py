"""Tests for public API."""

import os
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
        # Reset singleton for this test
        import embedding_cache
        embedding_cache._default_cache = None

        # Patch env var to use temp directory
        with patch.dict('os.environ', {'EMBEDDING_CACHE_DIR': tmpdir}):
            # Create a mock cache instance with proper behavior
            mock_cache = Mock()
            test_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
            mock_cache.embed.return_value = test_embedding

            # Patch EmbeddingCache constructor to return our mock
            with patch('embedding_cache.EmbeddingCache', return_value=mock_cache):
                # Call embed function
                result = embed("Hello world")

                # Should return the embedding
                np.testing.assert_array_equal(result, test_embedding)

                # Should have called embed on the singleton instance
                mock_cache.embed.assert_called_once_with("Hello world")


def test_embed_singleton_persistence():
    """Test that embed() reuses the same singleton across multiple calls."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Reset singleton for this test
        import embedding_cache
        embedding_cache._default_cache = None

        # Patch env var to use temp directory
        with patch.dict('os.environ', {'EMBEDDING_CACHE_DIR': tmpdir}):
            # Create a mock cache instance
            mock_cache = Mock()
            test_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
            mock_cache.embed.return_value = test_embedding

            # Patch EmbeddingCache constructor to return our mock
            with patch('embedding_cache.EmbeddingCache', return_value=mock_cache) as MockCache:
                # Call embed function multiple times
                result1 = embed("first")
                result2 = embed("second")
                result3 = embed("third")

                # Constructor should only be called once (singleton created once)
                MockCache.assert_called_once()

                # embed should be called 3 times on the same instance
                assert mock_cache.embed.call_count == 3
                mock_cache.embed.assert_any_call("first")
                mock_cache.embed.assert_any_call("second")
                mock_cache.embed.assert_any_call("third")
