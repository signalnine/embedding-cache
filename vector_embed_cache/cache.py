"""Main EmbeddingCache class with fallback chain logic."""

import os
import logging
import threading
from pathlib import Path
from typing import Union, List, Optional

import numpy as np

from .normalize import normalize_text
from .utils import generate_cache_key
from .storage import EmbeddingStorage
from .providers import LocalProvider, RemoteProvider, OpenAIProvider
from .preseed import get_preseed_db_path, preseed_db_exists, PreseedStorage

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Main cache class that ties together normalization, storage, and providers.

    Note: Stats tracking is thread-safe. The underlying storage layer is also thread-safe.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        model: str = "nomic-ai/nomic-embed-text-v1.5",
        remote_url: Optional[str] = None,
        fallback_providers: Optional[List[str]] = None,
        timeout: float = 5.0
    ):
        """Initialize EmbeddingCache.

        Args:
            cache_dir: Cache directory path (defaults to EMBEDDING_CACHE_DIR env var or ~/.cache/embedding-cache)
            model: Model name for embeddings
            remote_url: Optional remote backend URL
            fallback_providers: Provider fallback order (default: ["local", "remote"] if remote_url else ["local"])
            timeout: Remote request timeout in seconds
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

        # Initialize preseed storage if available
        self.preseed_storage = None
        if preseed_db_exists():
            self.preseed_storage = PreseedStorage(get_preseed_db_path())

        # Initialize providers based on model prefix
        if model.startswith("openai:"):
            # Extract actual model name: "openai:text-embedding-3-small" -> "text-embedding-3-small"
            actual_model = model.split(":", 1)[1]
            self.local_provider = OpenAIProvider(model=actual_model)
        else:
            # Local model (nomic-embed or other sentence-transformers model)
            self.local_provider = LocalProvider(model=model)

        self.remote_provider = RemoteProvider(remote_url, timeout=timeout, model=model) if remote_url else None

        # Set up fallback chain
        if fallback_providers is None:
            fallback_providers = ["local", "remote"] if remote_url else ["local"]
        self.fallback_providers = fallback_providers

        # Initialize stats with thread-safe lock
        self._stats_lock = threading.Lock()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "remote_hits": 0,
            "preseed_hits": 0
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

        # Check user cache first
        cached = self.storage.get(cache_key)
        if cached is not None:
            with self._stats_lock:
                self.stats["hits"] += 1
            return cached

        # Check preseed cache (fallback)
        if self.preseed_storage is not None:
            preseed_result = self.preseed_storage.get(cache_key)
            if preseed_result is not None:
                with self._stats_lock:
                    self.stats["preseed_hits"] += 1
                return preseed_result

        # Cache miss - compute embedding
        with self._stats_lock:
            self.stats["misses"] += 1
        embedding = self._compute_embedding(normalized)

        # Store in user cache
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
                        embedding = self.remote_provider.embed(text)
                        with self._stats_lock:
                            self.stats["remote_hits"] += 1
                        return embedding
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
