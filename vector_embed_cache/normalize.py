"""Text normalization utilities for consistent cache keys."""

import re

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize text for cache key generation.

    Applies:
    - Lowercase conversion
    - Internal whitespace runs collapsed to a single space
    - Leading/trailing whitespace removal

    Args:
        text: Input text to normalize

    Returns:
        Normalized text
    """
    return _WHITESPACE_RE.sub(" ", text).lower().strip()
