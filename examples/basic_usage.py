#!/usr/bin/env python3
"""Basic usage of embedding-cache."""

from vector_embed_cache import embed, EmbeddingCache

# Simple function API (uses global singleton)
embedding = embed("Hello, world!")
print(f"Embedding shape: {embedding.shape}")  # (768,)

# Class-based API for more control
cache = EmbeddingCache(model="nomic-ai/nomic-embed-text-v1.5")
embedding = cache.embed("Hello, world!")
print(f"Cache stats: {cache.stats}")

# Second call hits the cache
embedding2 = cache.embed("Hello, world!")
print(f"After cache hit: {cache.stats}")
