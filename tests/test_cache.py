"""Tests for EmbeddingCache main class."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock

from embedding_cache.cache import EmbeddingCache


def test_cache_init_creates_storage():
    """Test that EmbeddingCache initializes storage in specified directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir)

        # Storage should be initialized
        assert cache.storage is not None

        # Database file should exist
        db_path = Path(tmpdir) / "cache.db"
        assert db_path.exists()


def test_cache_embed_caches_result():
    """Test that embedding results are cached."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir)

        # Mock the provider to return a known embedding
        test_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        cache.local_provider.embed = Mock(return_value=test_embedding)

        text = "Hello world"

        # First call should compute
        result1 = cache.embed(text)
        assert cache.local_provider.embed.call_count == 1
        np.testing.assert_array_equal(result1, test_embedding)

        # Second call should hit cache (no additional compute)
        result2 = cache.embed(text)
        assert cache.local_provider.embed.call_count == 1
        np.testing.assert_array_equal(result2, test_embedding)


def test_cache_embed_normalizes_text():
    """Test that text is normalized before generating cache key."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir)

        test_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        cache.local_provider.embed = Mock(return_value=test_embedding)

        # These should all hit the same cache entry after normalization
        texts = [
            "Hello World",
            "hello world",
            "  Hello World  ",
            "HELLO WORLD"
        ]

        for text in texts:
            cache.embed(text)

        # Should only compute once (all normalize to same key)
        assert cache.local_provider.embed.call_count == 1


def test_cache_embed_batch():
    """Test that cache handles list of texts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir)

        # Mock embeddings for different texts
        embeddings_map = {
            "hello": np.array([0.1, 0.2], dtype=np.float32),
            "world": np.array([0.3, 0.4], dtype=np.float32),
            "test": np.array([0.5, 0.6], dtype=np.float32)
        }

        def mock_embed(text):
            return embeddings_map[text]

        cache.local_provider.embed = Mock(side_effect=mock_embed)

        texts = ["hello", "world", "test"]
        results = cache.embed(texts)

        # Should return list of embeddings
        assert isinstance(results, list)
        assert len(results) == 3

        # Each result should match expected embedding
        for i, text in enumerate(texts):
            np.testing.assert_array_equal(results[i], embeddings_map[text])


def test_cache_stats_tracking():
    """Test that cache tracks hits and misses."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir)

        test_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        cache.local_provider.embed = Mock(return_value=test_embedding)

        # Initial stats
        assert cache.stats["hits"] == 0
        assert cache.stats["misses"] == 0

        # First call - miss
        cache.embed("hello")
        assert cache.stats["hits"] == 0
        assert cache.stats["misses"] == 1

        # Second call - hit
        cache.embed("hello")
        assert cache.stats["hits"] == 1
        assert cache.stats["misses"] == 1

        # New text - miss
        cache.embed("world")
        assert cache.stats["hits"] == 1
        assert cache.stats["misses"] == 2

        # Existing text - hit
        cache.embed("world")
        assert cache.stats["hits"] == 2
        assert cache.stats["misses"] == 2


def test_cache_input_validation():
    """Test that cache validates input types."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir)

        # Should reject None
        with pytest.raises(ValueError):
            cache.embed(None)

        # Should reject empty string
        with pytest.raises(ValueError):
            cache.embed("")

        # Should reject number
        with pytest.raises(TypeError):
            cache.embed(123)

        # Should reject list with empty string
        with pytest.raises(ValueError):
            cache.embed(["hello", ""])

        # Should reject list with non-string
        with pytest.raises(TypeError):
            cache.embed(["hello", 123])


def test_remote_fallback_on_local_failure():
    """Test that remote provider is used when local provider fails."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create cache with remote URL configured
        cache = EmbeddingCache(
            cache_dir=tmpdir,
            remote_url="https://example.com",
            fallback_providers=["local", "remote"]
        )

        # Set local provider to fail
        cache.local_provider.is_available = Mock(return_value=False)

        # Mock remote provider to succeed
        test_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        cache.remote_provider.embed = Mock(return_value=test_embedding)

        # Should fall back to remote
        result = cache.embed("hello")

        # Verify remote was used
        assert cache.remote_provider.embed.call_count == 1
        np.testing.assert_array_equal(result, test_embedding)

        # Verify stats tracked remote usage
        assert cache.stats["remote_hits"] == 1
        assert cache.stats["misses"] == 1


def test_all_providers_fail():
    """Test that RuntimeError is raised when all providers fail."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create cache without remote URL
        cache = EmbeddingCache(cache_dir=tmpdir, remote_url=None)

        # Make local provider unavailable
        cache.local_provider.is_available = Mock(return_value=False)

        # Should raise RuntimeError with helpful message
        with pytest.raises(RuntimeError, match="All embedding providers failed"):
            cache.embed("hello")


def test_cache_selects_openai_provider():
    """Should select OpenAIProvider when model starts with 'openai:'."""
    from embedding_cache import EmbeddingCache
    from embedding_cache.providers import OpenAIProvider

    cache = EmbeddingCache(model="openai:text-embedding-3-small")

    assert isinstance(cache.local_provider, OpenAIProvider)
    assert cache.local_provider.model_name == "text-embedding-3-small"


def test_cache_selects_local_provider_for_nomic():
    """Should select LocalProvider for nomic models."""
    from embedding_cache import EmbeddingCache
    from embedding_cache.providers import LocalProvider

    cache = EmbeddingCache(model="nomic-ai/nomic-embed-text-v1.5")

    assert isinstance(cache.local_provider, LocalProvider)
    assert cache.local_provider.model_name == "nomic-ai/nomic-embed-text-v1.5"
