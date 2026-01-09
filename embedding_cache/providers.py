"""Embedding provider implementations."""

import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class LocalProvider:
    """Local embedding provider using sentence-transformers."""

    def __init__(self, model: str = "nomic-ai/nomic-embed-text-v1.5"):
        """Initialize local provider.

        Args:
            model: HuggingFace model name
        """
        self.model_name = model
        self._model: Optional[object] = None

    def is_available(self) -> bool:
        """Check if sentence-transformers is installed."""
        try:
            import sentence_transformers
            return True
        except ImportError:
            return False

    def _load_model(self):
        """Lazy load the sentence-transformers model."""
        if self._model is not None:
            return

        if not self.is_available():
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Install with: pip install embedding-cache[local]"
            )

        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading model {self.model_name}...")
        self._model = SentenceTransformer(
            self.model_name,
            trust_remote_code=True  # Required for nomic-embed models
        )
        logger.info(f"Model {self.model_name} loaded successfully")

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        self._load_model()
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple text strings.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        self._load_model()
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return [emb.astype(np.float32) for emb in embeddings]
