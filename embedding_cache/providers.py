"""Embedding provider implementations."""

import numpy as np
from typing import List, Optional
import logging
import httpx
import os

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


class RemoteProvider:
    """Remote embedding provider via HTTP backend."""

    def __init__(self, backend_url: str, timeout: float = 5.0, model: str = "nomic-embed-text-v2"):
        """Initialize remote provider.

        Args:
            backend_url: Backend service URL
            timeout: Request timeout in seconds
            model: Model name to request
        """
        self.backend_url = backend_url.rstrip('/')
        self.timeout = timeout
        self.model = model

    def _call_backend(self, texts: List[str]) -> List[np.ndarray]:
        """Call backend API with texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings
        """
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.backend_url}/embed",
                    json={
                        "texts": texts,
                        "model": self.model,
                        "normalized": True
                    }
                )
                response.raise_for_status()
                data = response.json()

                embeddings = [np.array(emb, dtype=np.float32) for emb in data["embeddings"]]
                return embeddings

        except httpx.TimeoutException as e:
            raise RuntimeError(f"Remote backend timeout: {e}")
        except httpx.NetworkError as e:
            raise RuntimeError(f"Remote backend error: {e}")
        except Exception as e:
            raise RuntimeError(f"Remote backend failed: {e}")

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        embeddings = self._call_backend([text])
        return embeddings[0]

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple text strings.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        return self._call_backend(texts)


class OpenAIProvider:
    """OpenAI embedding provider using OpenAI API."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        """Initialize OpenAI provider.

        Args:
            model: OpenAI model name
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client: Optional[object] = None

    def is_available(self) -> bool:
        """Check if openai package is installed and API key is set."""
        try:
            import openai
            return self.api_key is not None
        except ImportError:
            return False

    def _load_client(self):
        """Lazy load the OpenAI client."""
        if self._client is not None:
            return

        if not self.is_available():
            raise RuntimeError(
                "OpenAI not available. "
                "Install with: pip install openai, "
                "and set OPENAI_API_KEY environment variable"
            )

        from openai import OpenAI
        logger.info(f"Initializing OpenAI client with model {self.model}...")
        self._client = OpenAI(api_key=self.api_key)
        logger.info(f"OpenAI client initialized successfully")

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        self._load_client()
        response = self._client.embeddings.create(
            input=[text],
            model=self.model
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        return embedding

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple text strings.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        self._load_client()
        response = self._client.embeddings.create(
            input=texts,
            model=self.model
        )
        embeddings = [np.array(item.embedding, dtype=np.float32) for item in response.data]
        return embeddings
