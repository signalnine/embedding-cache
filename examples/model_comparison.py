#!/usr/bin/env python3
"""Compare embeddings from different models."""

import numpy as np
from embedding_cache import EmbeddingCache

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Test text
text = "The meaning of life is to find purpose and connection."

# Compare v1.5 and v2-moe (both local, free)
print("Comparing local models...")

cache_v15 = EmbeddingCache(model="nomic-ai/nomic-embed-text-v1.5")
cache_v2 = EmbeddingCache(model="nomic-ai/nomic-embed-text-v2-moe")

emb_v15 = cache_v15.embed(text)
emb_v2 = cache_v2.embed(text)

print(f"v1.5 shape: {emb_v15.shape}")
print(f"v2-moe shape: {emb_v2.shape}")
print(f"Cosine similarity between models: {cosine_similarity(emb_v15, emb_v2):.4f}")

# Note: OpenAI requires API key
# cache_openai = EmbeddingCache(model="openai:text-embedding-3-small")
# emb_openai = cache_openai.embed(text)
# print(f"OpenAI shape: {emb_openai.shape}")  # (1536,)
