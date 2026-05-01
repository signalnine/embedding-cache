"""Tests for EmbeddingCache main class."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock

from vector_embed_cache.cache import EmbeddingCache


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
    """Test that embedding results are cached with float16 compression."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir)

        # Mock the provider to return a known embedding
        test_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        cache.local_provider.embed = Mock(return_value=test_embedding)

        text = "Hello world"

        # First call should compute
        result1 = cache.embed(text)
        assert cache.local_provider.embed.call_count == 1
        # Use decimal=3 for float16 precision
        np.testing.assert_array_almost_equal(result1, test_embedding, decimal=3)

        # Second call should hit cache (no additional compute)
        result2 = cache.embed(text)
        assert cache.local_provider.embed.call_count == 1
        # Cached result should be close to original (within float16 precision)
        np.testing.assert_array_almost_equal(result2, test_embedding, decimal=3)


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
        cache.preseed_storage = None  # Disable preseed for this test

        # Mock embeddings for different texts
        embeddings_map = {
            "hello": np.array([0.1, 0.2], dtype=np.float32),
            "world": np.array([0.3, 0.4], dtype=np.float32),
            "test": np.array([0.5, 0.6], dtype=np.float32)
        }

        def mock_embed_batch(texts):
            return [embeddings_map[t] for t in texts]

        cache.local_provider.embed_batch = Mock(side_effect=mock_embed_batch)

        texts = ["hello", "world", "test"]
        results = cache.embed(texts)

        # Should return list of embeddings
        assert isinstance(results, list)
        assert len(results) == 3

        # Each result should match expected embedding (with float16 precision)
        for i, text in enumerate(texts):
            np.testing.assert_array_almost_equal(results[i], embeddings_map[text], decimal=3)


def test_cache_stats_tracking():
    """Test that cache tracks hits and misses."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir)
        cache.preseed_storage = None  # Disable preseed for this test

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
        cache.preseed_storage = None  # Disable preseed for this test

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
        cache.preseed_storage = None  # Disable preseed for this test

        # Make local provider unavailable
        cache.local_provider.is_available = Mock(return_value=False)

        # Should raise RuntimeError with helpful message
        with pytest.raises(RuntimeError, match="All embedding providers failed"):
            cache.embed("hello")


def test_cache_selects_openai_provider():
    """Should select OpenAIProvider when model starts with 'openai:'."""
    from vector_embed_cache import EmbeddingCache
    from vector_embed_cache.providers import OpenAIProvider

    cache = EmbeddingCache(model="openai:text-embedding-3-small")

    assert isinstance(cache.local_provider, OpenAIProvider)
    assert cache.local_provider.model_name == "text-embedding-3-small"


def test_embed_list_uses_batched_provider_call():
    """A list of N uncached texts should produce ONE provider.embed_batch call."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir)
        cache.preseed_storage = None  # Disable preseed so all are misses

        texts = [f"unique-text-{i}" for i in range(5)]
        embeddings = [np.array([i, i + 0.1, i + 0.2], dtype=np.float32) for i in range(5)]

        cache.local_provider.embed_batch = Mock(return_value=embeddings)
        cache.local_provider.embed = Mock(side_effect=AssertionError("embed() must not be called for list input"))

        results = cache.embed(texts)

        # Exactly one batched call covering all 5 texts
        assert cache.local_provider.embed_batch.call_count == 1
        called_with = cache.local_provider.embed_batch.call_args[0][0]
        assert len(called_with) == 5
        assert results is not None
        assert len(results) == 5
        assert cache.stats["misses"] == 5
        assert cache.stats["hits"] == 0


def test_embed_list_mixed_cached_and_uncached_batches_only_misses():
    """With some cached and some uncached, embed_batch is called once with only the misses."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir)
        cache.preseed_storage = None

        # Pre-populate cache with two entries by computing them once with batch
        existing = ["already-cached-a", "already-cached-b"]
        cache.local_provider.embed_batch = Mock(
            return_value=[
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
                np.array([4.0, 5.0, 6.0], dtype=np.float32),
            ]
        )
        cache.embed(existing)
        assert cache.local_provider.embed_batch.call_count == 1

        # Now mix two cached + two new texts
        new_texts = ["fresh-1", "fresh-2"]
        cache.local_provider.embed_batch = Mock(
            return_value=[
                np.array([7.0, 8.0, 9.0], dtype=np.float32),
                np.array([10.0, 11.0, 12.0], dtype=np.float32),
            ]
        )

        mixed = [existing[0], new_texts[0], existing[1], new_texts[1]]
        results = cache.embed(mixed)

        # Exactly one batched call, containing only the two misses
        assert cache.local_provider.embed_batch.call_count == 1
        called_with = cache.local_provider.embed_batch.call_args[0][0]
        assert len(called_with) == 2
        assert len(results) == 4
        assert cache.stats["hits"] == 2
        # 2 misses from the initial seed + 2 from the mixed call
        assert cache.stats["misses"] == 4


def test_embed_list_all_cached_makes_no_provider_call():
    """When all list items are cached, the provider is not invoked."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir)
        cache.preseed_storage = None

        seed = ["a", "b", "c"]
        cache.local_provider.embed_batch = Mock(
            return_value=[np.array([float(i)], dtype=np.float32) for i in range(3)]
        )
        cache.embed(seed)
        assert cache.local_provider.embed_batch.call_count == 1

        # Reset mock; second call should not invoke provider
        cache.local_provider.embed_batch = Mock(side_effect=AssertionError("should not be called"))
        cache.local_provider.embed = Mock(side_effect=AssertionError("should not be called"))

        results = cache.embed(seed)
        assert len(results) == 3
        assert cache.stats["hits"] == 3


def test_cache_selects_local_provider_for_nomic():
    """Should select LocalProvider for nomic models."""
    from vector_embed_cache import EmbeddingCache
    from vector_embed_cache.providers import LocalProvider

    cache = EmbeddingCache(model="nomic-ai/nomic-embed-text-v1.5")

    assert isinstance(cache.local_provider, LocalProvider)
    assert cache.local_provider.model_name == "nomic-ai/nomic-embed-text-v1.5"
