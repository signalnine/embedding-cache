"""Sync Client for vector-embed-cache API."""

from __future__ import annotations

import logging
import os
import time
from types import TracebackType
from typing import Any

import httpx

from vector_embed_client.errors import NetworkError, VectorEmbedError
from vector_embed_client.http import (
    RETRY_CONFIG,
    build_headers,
    calculate_delay,
    map_status_to_error,
)
from vector_embed_client.types import (
    EmbedResponse,
    SearchResponse,
    SearchResult,
    StatsResponse,
)

logger = logging.getLogger(__name__)


class Client:
    """Synchronous client for vector-embed-cache API.

    Usage:
        with Client(api_key="your-api-key") as client:
            response = client.embed("hello world")
            print(response.vector)

    Args:
        api_key: API key for authentication. Falls back to VECTOR_EMBED_API_KEY env var.
        base_url: Base URL for the API. Defaults to https://api.vector-embed-cache.com
        model: Default model to use for embeddings. Defaults to "nomic-v1.5".
        timeout: Request timeout in seconds. Defaults to 30.0.
        retries: Number of retry attempts for retryable errors. Defaults to 3.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.vector-embed-cache.com",
        model: str = "nomic-v1.5",
        timeout: float = 30.0,
        retries: int = 3,
    ) -> None:
        """Initialize the client."""
        resolved_api_key = api_key or os.environ.get("VECTOR_EMBED_API_KEY")
        if not resolved_api_key:
            msg = "API key required. Provide api_key or set VECTOR_EMBED_API_KEY env var."
            raise ValueError(msg)

        self._api_key = resolved_api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._retries = retries
        self._http_client = httpx.Client(
            timeout=timeout,
            headers=build_headers(self._api_key),
        )

    def __enter__(self) -> Client:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager."""
        self.close()

    def close(self) -> None:
        """Close the HTTP client."""
        self._http_client.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
    ) -> Any:
        """Make an HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API path (e.g., "/v1/embed")
            json: Request body as dict

        Returns:
            Response JSON

        Raises:
            VectorEmbedError: For API errors
            NetworkError: For connection/timeout errors
        """
        url = f"{self._base_url}{path}"
        last_error: VectorEmbedError | None = None

        for attempt in range(self._retries + 1):
            try:
                response = self._http_client.request(method, url, json=json)

                if response.status_code >= 400:
                    try:
                        body = response.json()
                    except Exception:
                        body = {"error": response.text or "Unknown error"}

                    headers = dict(response.headers)
                    error = map_status_to_error(response.status_code, body, headers)

                    # Check if this is a retryable status
                    if response.status_code in RETRY_CONFIG["retryable_statuses"]:
                        last_error = error
                        if attempt < self._retries:
                            delay = calculate_delay(attempt)
                            logger.debug(
                                "Retrying request after %.2fs (attempt %d/%d): %s",
                                delay,
                                attempt + 1,
                                self._retries,
                                str(error),
                            )
                            time.sleep(delay)
                            continue
                    # Non-retryable error, raise immediately
                    raise error

                return response.json()

            except httpx.ConnectError as e:
                last_error = NetworkError(
                    message=f"Connection error: {e}",
                    code="network_error",
                )
                if attempt < self._retries:
                    delay = calculate_delay(attempt)
                    logger.debug(
                        "Retrying after connection error (attempt %d/%d)",
                        attempt + 1,
                        self._retries,
                    )
                    time.sleep(delay)
                    continue
                raise last_error from e

            except httpx.TimeoutException as e:
                last_error = NetworkError(
                    message=f"Request timeout: {e}",
                    code="network_error",
                )
                if attempt < self._retries:
                    delay = calculate_delay(attempt)
                    logger.debug(
                        "Retrying after timeout (attempt %d/%d)",
                        attempt + 1,
                        self._retries,
                    )
                    time.sleep(delay)
                    continue
                raise last_error from e

        # Should only reach here if retries exhausted
        if last_error is not None:
            raise last_error
        # This should never happen, but satisfy type checker
        raise RuntimeError("Unexpected state: no error but request did not succeed")

    def embed(self, text: str, *, model: str | None = None) -> EmbedResponse:
        """Compute embedding for a single text.

        Args:
            text: Text to embed.
            model: Model to use. Defaults to client's default model.

        Returns:
            EmbedResponse with vector, cached status, and dimensions.
        """
        response = self._request(
            "POST",
            "/v1/embed",
            json={
                "text": text,
                "model": model or self._model,
            },
        )
        return EmbedResponse(
            vector=response["vector"],
            cached=response["cached"],
            dimensions=response["dimensions"],
        )

    def embed_batch(
        self, texts: list[str], *, model: str | None = None
    ) -> list[EmbedResponse]:
        """Compute embeddings for multiple texts.

        Args:
            texts: List of texts to embed.
            model: Model to use. Defaults to client's default model.

        Returns:
            List of EmbedResponse objects.
        """
        response = self._request(
            "POST",
            "/v1/embed/batch",
            json={
                "texts": texts,
                "model": model or self._model,
            },
        )
        return [
            EmbedResponse(
                vector=item["vector"],
                cached=item["cached"],
                dimensions=item["dimensions"],
            )
            for item in response
        ]

    def search(
        self,
        text: str,
        *,
        top_k: int = 10,
        min_score: float = 0.0,
        include_text: bool = False,
    ) -> SearchResponse:
        """Search for similar embeddings.

        Args:
            text: Query text.
            top_k: Maximum number of results to return.
            min_score: Minimum similarity score threshold.
            include_text: Whether to include original text in results.

        Returns:
            SearchResponse with results and metadata.
        """
        response = self._request(
            "POST",
            "/v1/search",
            json={
                "text": text,
                "top_k": top_k,
                "min_score": min_score,
                "include_text": include_text,
            },
        )
        results = [
            SearchResult(
                text_hash=item["text_hash"],
                score=item["score"],
                text=item.get("text"),
                model=item["model"],
                hit_count=item["hit_count"],
            )
            for item in response["results"]
        ]
        return SearchResponse(
            results=results,
            total=response["total"],
            search_time_ms=response["search_time_ms"],
        )

    def stats(self) -> StatsResponse:
        """Get cache statistics.

        Returns:
            StatsResponse with cache hit/miss counts.
        """
        response = self._request("GET", "/v1/stats")
        return StatsResponse(
            cache_hits=response["cache_hits"],
            cache_misses=response["cache_misses"],
            total_cached=response["total_cached"],
        )
