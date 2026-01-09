"""Utility functions for caching."""

import hashlib
import json


def generate_cache_key(text: str, model: str) -> str:
    """Generate a cache key from text and model name.

    Uses SHA-256 hash of JSON-encoded text and model to prevent collisions.

    Args:
        text: Normalized text
        model: Model name (e.g., "nomic-embed-text-v2")

    Returns:
        Hex string of SHA-256 hash
    """
    combined = json.dumps({"text": text, "model": model}, sort_keys=True)
    return hashlib.sha256(combined.encode('utf-8')).hexdigest()
