"""HTTP layer with retry logic for vector-embed-cache-client."""

from __future__ import annotations

import contextlib
import random
import sys
from typing import Any

from vector_embed_client.errors import (
    AuthenticationError,
    RateLimitError,
    ServerError,
    ValidationError,
    VectorEmbedError,
)

VERSION = "0.1.0"

RETRY_CONFIG: dict[str, Any] = {
    "base_delay": 1.0,  # 1 second
    "max_delay": 30.0,  # 30 seconds cap
    "jitter_factor": 0.2,  # ±20% randomization
    "retryable_statuses": [429, 500, 502, 503, 504],
}


def calculate_delay(attempt: int) -> float:
    """Calculate delay for retry attempt with exponential backoff and jitter.

    Args:
        attempt: The retry attempt number (0-indexed).

    Returns:
        Delay in seconds with jitter applied.
    """
    base_delay: float = RETRY_CONFIG["base_delay"]
    max_delay: float = RETRY_CONFIG["max_delay"]
    jitter_factor: float = RETRY_CONFIG["jitter_factor"]

    # Exponential backoff: base_delay * 2^attempt
    delay: float = base_delay * (2**attempt)

    # Cap at max_delay
    delay = min(delay, max_delay)

    # Apply jitter: ±jitter_factor
    jitter: float = delay * jitter_factor * (2 * random.random() - 1)
    delay = delay + jitter

    return delay


def map_status_to_error(
    status: int,
    body: dict[str, Any],
    headers: dict[str, str],
) -> VectorEmbedError:
    """Map HTTP status code to appropriate exception.

    Args:
        status: HTTP status code.
        body: Response body as dict.
        headers: Response headers.

    Returns:
        Appropriate VectorEmbedError subclass instance.
    """
    message = body.get("error", "Unknown error")

    if status == 400:
        return ValidationError(
            message=message,
            code="validation_error",
            status=status,
        )
    elif status == 401:
        return AuthenticationError(
            message=message,
            code="authentication_error",
            status=status,
        )
    elif status == 429:
        retry_after: int | None = None
        retry_after_header = headers.get("retry-after")
        if retry_after_header is not None:
            with contextlib.suppress(ValueError):
                retry_after = int(retry_after_header)
        return RateLimitError(
            message=message,
            code="rate_limit_error",
            status=status,
            retry_after=retry_after,
        )
    elif status >= 500:
        return ServerError(
            message=message,
            code="server_error",
            status=status,
        )
    else:
        return VectorEmbedError(
            message=message,
            code="unknown_error",
            status=status,
        )


def build_headers(api_key: str) -> dict[str, str]:
    """Build HTTP request headers.

    Args:
        api_key: API key for authentication.

    Returns:
        Dict of headers to include in requests.
    """
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    user_agent = f"vector-embed-cache-client/{VERSION} python/{python_version}"

    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": user_agent,
    }
