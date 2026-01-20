"""Text normalization utilities for consistent cache keys."""


def normalize_text(text: str) -> str:
    """Normalize text for cache key generation.

    Applies:
    - Lowercase conversion
    - Leading/trailing whitespace removal

    Args:
        text: Input text to normalize

    Returns:
        Normalized text
    """
    return text.lower().strip()
