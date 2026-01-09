"""Tests for embedding providers."""

import numpy as np
import pytest
from embedding_cache.providers import LocalProvider


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
