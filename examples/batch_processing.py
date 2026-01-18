#!/usr/bin/env python3
"""Batch processing example for large datasets."""

from embedding_cache import EmbeddingCache

# Sample dataset
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming industries.",
    "Python is a popular programming language.",
    "Vector embeddings capture semantic meaning.",
    "Caching reduces latency and API costs.",
]

# Create cache
cache = EmbeddingCache()

# Batch embed
print(f"Embedding {len(texts)} texts...")
embeddings = cache.embed(texts)

print(f"Generated {len(embeddings)} embeddings")
print(f"Each embedding has {embeddings[0].shape[0]} dimensions")
print(f"Stats: {cache.stats}")

# Second batch call - all cache hits
embeddings2 = cache.embed(texts)
print(f"After re-embedding: {cache.stats}")
