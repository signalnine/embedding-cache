"""Embedding cache with local and remote fallback."""

from typing import Union, List
import threading
import logging
import numpy as np

from .cache import EmbeddingCache

__version__ = "0.1.0"

logger = logging.getLogger(__name__)

# Global singleton for simple API
_default_cache = None
_default_cache_lock = threading.Lock()


def embed(text: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
    """Embed text using a singleton EmbeddingCache instance.

    This is a convenience function for simple use cases with default configuration.
    The singleton uses the default model (nomic-embed-text-v1.5) and cache directory
    (~/.cache/embedding-cache or EMBEDDING_CACHE_DIR env var).

    For more control over configuration (custom models, remote backend, fallback order),
    create your own EmbeddingCache instance.

    Args:
        text: Single text string or list of text strings

    Returns:
        Single embedding array or list of embedding arrays

    Example:
        >>> from embedding_cache import embed
        >>> embedding = embed("Hello world")
        >>> embeddings = embed(["Hello", "World"])

    Note:
        This function is thread-safe. The singleton is created once and reused
        across all calls.
    """
    global _default_cache

    if _default_cache is None:
        with _default_cache_lock:
            # Double-check pattern to avoid race condition
            if _default_cache is None:
                logger.debug("Creating default EmbeddingCache singleton")
                _default_cache = EmbeddingCache()

    return _default_cache.embed(text)


__all__ = ["embed", "EmbeddingCache"]
