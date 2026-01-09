"""Main EmbeddingCache class with fallback chain logic."""

import os
import logging
from pathlib import Path
from typing import Union, List, Optional

import numpy as np

from .normalize import normalize_text
from .utils import generate_cache_key
from .storage import EmbeddingStorage
from .providers import LocalProvider, RemoteProvider

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Main cache class that ties together normalization, storage, and providers."""

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        model: str = "nomic-ai/nomic-embed-text-v1.5",
        remote_url: Optional[str] = None
    ):
        """Initialize EmbeddingCache.

        Args:
            cache_dir: Cache directory path (defaults to ~/.cache/embedding-cache or EMBEDDING_CACHE_DIR)
            model: Model name for embeddings
            remote_url: Optional remote backend URL
        """
        # Set up cache directory
        if cache_dir is None:
            cache_dir = os.environ.get(
                "EMBEDDING_CACHE_DIR",
                str(Path.home() / ".cache" / "embedding-cache")
            )
        self.cache_dir = cache_dir
        self.model = model

        # Initialize storage
        db_path = Path(cache_dir) / "cache.db"
        self.storage = EmbeddingStorage(str(db_path))

        # Initialize providers
        self.local_provider = LocalProvider(model=model)
        self.remote_provider = RemoteProvider(remote_url, model=model) if remote_url else None

        # Set up fallback chain
        if remote_url:
            self.fallback_providers = ["local", "remote"]
        else:
            self.fallback_providers = ["local"]

        # Initialize stats
        self.stats = {
            "hits": 0,
            "misses": 0,
            "remote_hits": 0
        }

    def _validate_input(self, text: Union[str, List[str]]):
        """Validate input text.

        Args:
            text: String or list of strings to validate

        Raises:
            ValueError: If text is None or empty string
            TypeError: If text is not string or list of strings
        """
        if text is None:
            raise ValueError("Text cannot be None")

        if isinstance(text, str):
            if text == "":
                raise ValueError("Text cannot be empty string")
        elif isinstance(text, list):
            for item in text:
                if not isinstance(item, str):
                    raise TypeError(f"All items in list must be strings, got {type(item).__name__}")
                if item == "":
                    raise ValueError("Text cannot be empty string")
        else:
            raise TypeError(f"Text must be string or list of strings, got {type(text).__name__}")

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text (with cache lookup).

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        # Normalize text
        normalized = normalize_text(text)

        # Generate cache key
        cache_key = generate_cache_key(normalized, self.model)

        # Check cache
        cached = self.storage.get(cache_key)
        if cached is not None:
            self.stats["hits"] += 1
            return cached

        # Cache miss - compute embedding
        self.stats["misses"] += 1
        embedding = self._compute_embedding(normalized)

        # Store in cache
        self.storage.set(cache_key, self.model, embedding)

        return embedding

    def _compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding using fallback chain.

        Args:
            text: Normalized text

        Returns:
            Embedding vector

        Raises:
            RuntimeError: If all providers fail
        """
        for provider_name in self.fallback_providers:
            try:
                if provider_name == "local":
                    if self.local_provider.is_available():
                        logger.debug("Using local provider")
                        return self.local_provider.embed(text)
                    else:
                        logger.warning("Local provider not available (sentence-transformers not installed)")
                        continue

                elif provider_name == "remote":
                    if self.remote_provider is not None:
                        logger.debug("Using remote provider")
                        self.stats["remote_hits"] += 1
                        return self.remote_provider.embed(text)
                    else:
                        logger.warning("Remote provider not configured")
                        continue

            except Exception as e:
                logger.warning(f"Provider {provider_name} failed: {e}")
                continue

        # All providers failed
        raise RuntimeError(
            "All embedding providers failed. "
            "Install sentence-transformers (pip install embedding-cache[local]) "
            "or configure a remote backend."
        )

    def embed(self, text: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """Embed text or list of texts.

        Args:
            text: Single text string or list of text strings

        Returns:
            Single embedding or list of embeddings

        Raises:
            ValueError: If text is None or empty
            TypeError: If text is not string or list of strings
        """
        self._validate_input(text)

        if isinstance(text, str):
            return self._get_embedding(text)
        else:
            return [self._get_embedding(t) for t in text]
