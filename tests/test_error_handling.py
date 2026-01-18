"""Tests for error handling and edge cases."""

import pytest
import numpy as np

from embedding_cache.cache import EmbeddingCache


def test_all_providers_fail(temp_cache_dir, mocker):
    """Test that error is raised when all providers fail."""
    # Create cache with both local and remote providers
    cache = EmbeddingCache(
        cache_dir=temp_cache_dir,
        remote_url="https://example.com",
        fallback_providers=["local", "remote"]
    )

    # Mock both providers to fail
    mocker.patch.object(cache.local_provider, 'is_available', return_value=False)
    mocker.patch.object(cache.remote_provider, 'embed', side_effect=RuntimeError("Remote failed"))

    # Should raise RuntimeError with helpful message
    with pytest.raises(RuntimeError, match="All embedding providers failed"):
        cache.embed("hello")


def test_empty_string_validation(temp_cache_dir):
    """Test that empty strings are rejected."""
    cache = EmbeddingCache(cache_dir=temp_cache_dir)

    # Should reject empty string
    with pytest.raises(ValueError, match="cannot be empty"):
        cache.embed("")


def test_none_input_validation(temp_cache_dir):
    """Test that None input is rejected."""
    cache = EmbeddingCache(cache_dir=temp_cache_dir)

    # Should reject None
    with pytest.raises(ValueError, match="cannot be None"):
        cache.embed(None)


def test_wrong_type_validation(temp_cache_dir):
    """Test that non-string inputs are rejected."""
    cache = EmbeddingCache(cache_dir=temp_cache_dir)

    # Should reject number
    with pytest.raises(TypeError, match="must be string or list"):
        cache.embed(123)

    # Should reject dict
    with pytest.raises(TypeError, match="must be string or list"):
        cache.embed({"text": "hello"})


def test_list_with_non_strings(temp_cache_dir):
    """Test that lists containing non-strings are rejected."""
    cache = EmbeddingCache(cache_dir=temp_cache_dir)

    # Should reject list with number
    with pytest.raises(TypeError, match="All items in list must be strings"):
        cache.embed(["hello", 123, "world"])


def test_list_with_empty_strings(temp_cache_dir):
    """Test that lists containing empty strings are rejected."""
    cache = EmbeddingCache(cache_dir=temp_cache_dir)

    # Should reject list with empty string
    with pytest.raises(ValueError, match="cannot be empty"):
        cache.embed(["hello", "", "world"])


def test_remote_fallback_on_local_failure(temp_cache_dir, mocker):
    """Test that remote provider is used when local provider fails."""
    # Create cache with remote URL configured
    cache = EmbeddingCache(
        cache_dir=temp_cache_dir,
        remote_url="https://example.com",
        fallback_providers=["local", "remote"]
    )

    # Mock local provider to raise RuntimeError
    mocker.patch.object(cache.local_provider, 'is_available', return_value=True)
    mocker.patch.object(cache.local_provider, 'embed', side_effect=RuntimeError("Local failed"))

    # Mock remote provider to succeed
    test_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    mocker.patch.object(cache.remote_provider, 'embed', return_value=test_embedding)

    # Should fall back to remote
    result = cache.embed("hello")

    # Verify remote provider was used
    assert cache.remote_provider.embed.call_count == 1

    # Verify stats tracked remote_hits
    assert cache.stats["remote_hits"] == 1

    # Verify result is correct
    np.testing.assert_array_equal(result, test_embedding)
