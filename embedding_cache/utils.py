"""Utility functions for caching."""

import hashlib


def generate_cache_key(text: str, model: str) -> str:
    """Generate a cache key from text and model name.

    Uses SHA-256 hash of the concatenated text and model.

    Args:
        text: Normalized text
        model: Model name (e.g., "nomic-embed-text-v2")

    Returns:
        Hex string of SHA-256 hash
    """
    combined = f"{text}::{model}"
    return hashlib.sha256(combined.encode('utf-8')).hexdigest()
