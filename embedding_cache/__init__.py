"""Embedding cache with local and remote fallback."""

from typing import Union, List
import numpy as np

from .cache import EmbeddingCache

__version__ = "0.1.0"

# Global singleton for simple API
_default_cache = None


def embed(text: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
    """Embed text using a singleton EmbeddingCache instance.

    This is a convenience function for simple use cases. For more control
    over cache configuration, create your own EmbeddingCache instance.

    Args:
        text: Single text string or list of text strings

    Returns:
        Single embedding or list of embeddings

    Example:
        >>> from embedding_cache import embed
        >>> embedding = embed("Hello world")
        >>> embeddings = embed(["Hello", "World"])
    """
    global _default_cache

    if _default_cache is None:
        _default_cache = EmbeddingCache()

    return _default_cache.embed(text)


__all__ = ["embed", "EmbeddingCache"]
