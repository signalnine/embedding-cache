"""Python client for vector-embed-cache API."""

from vector_embed_client.async_client import AsyncClient
from vector_embed_client.client import Client
from vector_embed_client.errors import (
    AuthenticationError,
    NetworkError,
    RateLimitError,
    ServerError,
    ValidationError,
    VectorEmbedError,
)
from vector_embed_client.types import (
    EmbedResponse,
    SearchResponse,
    SearchResult,
    StatsResponse,
)

__version__ = "0.1.0"

__all__: list[str] = [
    "AsyncClient",
    "Client",
    "VectorEmbedError",
    "AuthenticationError",
    "RateLimitError",
    "ValidationError",
    "ServerError",
    "NetworkError",
    "EmbedResponse",
    "SearchResult",
    "SearchResponse",
    "StatsResponse",
]
