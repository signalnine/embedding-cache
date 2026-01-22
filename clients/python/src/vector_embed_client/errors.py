"""Exception classes for vector-embed-cache-client."""

from __future__ import annotations


class VectorEmbedError(Exception):
    """Base exception for all vector-embed-cache-client errors."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


class AuthenticationError(VectorEmbedError):
    """Raised for 401 Unauthorized responses."""

    pass


class RateLimitError(VectorEmbedError):
    """Raised for 429 Too Many Requests responses."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        status: int | None = None,
        retry_after: int | None = None,
    ) -> None:
        super().__init__(message, code=code, status=status)
        self.retry_after = retry_after


class ValidationError(VectorEmbedError):
    """Raised for 400 Bad Request responses."""

    pass


class ServerError(VectorEmbedError):
    """Raised for 5xx server errors."""

    pass


class NetworkError(VectorEmbedError):
    """Raised for connection failures, timeouts, etc."""

    pass
