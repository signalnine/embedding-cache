"""Integration tests with real components.

These tests use the actual LocalProvider with sentence-transformers
to verify end-to-end functionality. They are marked with @pytest.mark.integration
to allow selective execution.

Run with: pytest tests/test_integration.py -v -m integration
"""

import pytest
import numpy as np

from embedding_cache.cache import EmbeddingCache
from embedding_cache.providers import LocalProvider


@pytest.mark.integration
def test_local_provider_full_flow(temp_cache_dir):
    """Test full flow with real local provider.

    Verifies:
    - First embedding (cache miss) produces correct shape
    - Second embedding of same text (cache hit) returns identical array
    - Different text (cache miss) produces different embedding
    - Stats are correctly tracked
    """
    # Check if sentence-transformers is installed
    provider = LocalProvider()
    if not provider.is_available():
        pytest.skip("sentence-transformers not installed")

    # Create cache with real provider
    cache = EmbeddingCache(cache_dir=temp_cache_dir)

    # First embedding - should be cache miss
    text1 = "Hello world"
    emb1 = cache.embed(text1)

    # Verify shape (nomic-embed produces 768-dimensional embeddings)
    assert isinstance(emb1, np.ndarray)
    assert emb1.shape == (768,)
    assert emb1.dtype == np.float32

    # Verify stats
    assert cache.stats["hits"] == 0
    assert cache.stats["misses"] == 1

    # Second embedding of same text - should be cache hit
    emb2 = cache.embed(text1)

    # Verify same array returned
    assert isinstance(emb2, np.ndarray)
    assert emb2.shape == (768,)
    np.testing.assert_array_equal(emb1, emb2)

    # Verify stats
    assert cache.stats["hits"] == 1
    assert cache.stats["misses"] == 1

    # Different text - should be cache miss
    text2 = "Different text"
    emb3 = cache.embed(text2)

    # Verify different embedding
    assert isinstance(emb3, np.ndarray)
    assert emb3.shape == (768,)
    assert not np.array_equal(emb1, emb3)

    # Verify stats
    assert cache.stats["hits"] == 1
    assert cache.stats["misses"] == 2


@pytest.mark.integration
def test_batch_embedding(temp_cache_dir):
    """Test batch embedding with real provider.

    Verifies:
    - Multiple texts can be embedded in batch
    - All embeddings have correct shape
    - Different texts produce different embeddings
    """
    # Check if sentence-transformers is installed
    provider = LocalProvider()
    if not provider.is_available():
        pytest.skip("sentence-transformers not installed")

    # Create cache with real provider
    cache = EmbeddingCache(cache_dir=temp_cache_dir)

    # Embed batch of texts
    texts = ["hello", "world", "foo", "bar"]
    embeddings = cache.embed(texts)

    # Verify 4 embeddings returned
    assert isinstance(embeddings, list)
    assert len(embeddings) == 4

    # Verify all are ndarray with correct shape
    for emb in embeddings:
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (768,)
        assert emb.dtype == np.float32

    # Verify different texts have different embeddings
    # Compare all pairs
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            assert not np.array_equal(embeddings[i], embeddings[j]), \
                f"Embeddings for '{texts[i]}' and '{texts[j]}' should be different"


@pytest.mark.integration
def test_normalization_cache_behavior(temp_cache_dir):
    """Test that normalization leads to cache hits.

    Verifies:
    - Text variations that normalize to same string hit cache
    - All variants return identical embeddings
    - Stats show 1 miss and multiple hits
    """
    # Check if sentence-transformers is installed
    provider = LocalProvider()
    if not provider.is_available():
        pytest.skip("sentence-transformers not installed")

    # Create cache with real provider
    cache = EmbeddingCache(cache_dir=temp_cache_dir)

    # Test variants that should all normalize to same cache key
    variants = [
        "Hello World",           # First - cache miss
        "hello world",           # Should hit cache
        "HELLO WORLD",           # Should hit cache
        "  hello world  ",       # Should hit cache (whitespace stripped)
        "  HELLO WORLD  ",       # Should hit cache (whitespace + case)
    ]

    embeddings = []
    for text in variants:
        emb = cache.embed(text)
        embeddings.append(emb)

    # Verify all embeddings are identical (from cache)
    for i in range(1, len(embeddings)):
        np.testing.assert_array_equal(
            embeddings[0],
            embeddings[i],
            err_msg=f"Embedding for '{variants[i]}' should match '{variants[0]}'"
        )

    # Verify stats: 1 miss (first), 4 hits (rest)
    assert cache.stats["misses"] == 1
    assert cache.stats["hits"] == 4
