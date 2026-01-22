"""Python client for vector-embed-cache API."""

from vector_embed_client.errors import (
    AuthenticationError,
    NetworkError,
    RateLimitError,
    ServerError,
    ValidationError,
    VectorEmbedError,
)

__version__ = "0.1.0"

__all__: list[str] = [
    "VectorEmbedError",
    "AuthenticationError",
    "RateLimitError",
    "ValidationError",
    "ServerError",
    "NetworkError",
]
