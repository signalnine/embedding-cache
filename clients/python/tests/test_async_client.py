"""Tests for AsyncClient."""

from __future__ import annotations

import os

import pytest
from pytest_httpx import HTTPXMock

from vector_embed_client import AsyncClient
from vector_embed_client.errors import (
    AuthenticationError,
    RateLimitError,
)
from vector_embed_client.types import EmbedResponse, SearchResponse, StatsResponse


@pytest.mark.asyncio
async def test_async_client_requires_api_key() -> None:
    """AsyncClient should raise ValueError when no API key provided."""
    os.environ.pop("VECTOR_EMBED_API_KEY", None)
    with pytest.raises(ValueError, match="API key required"):
        AsyncClient()


@pytest.mark.asyncio
async def test_async_client_uses_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """AsyncClient should use VECTOR_EMBED_API_KEY env var."""
    monkeypatch.setenv("VECTOR_EMBED_API_KEY", "env_test_key")
    client = AsyncClient()
    assert client._api_key == "env_test_key"
    await client.close()


@pytest.mark.asyncio
async def test_async_client_context_manager(httpx_mock: HTTPXMock) -> None:
    """AsyncClient should work as async context manager."""
    httpx_mock.add_response(
        url="https://api.vector-embed-cache.com/v1/stats",
        method="GET",
        json={"cache_hits": 1, "cache_misses": 0, "total_cached": 1},
    )
    async with AsyncClient(api_key="test") as client:
        await client.stats()


@pytest.mark.asyncio
async def test_async_embed_success(httpx_mock: HTTPXMock) -> None:
    """embed() should return EmbedResponse on success."""
    httpx_mock.add_response(
        url="https://api.vector-embed-cache.com/v1/embed",
        method="POST",
        json={"vector": [0.1, 0.2, 0.3], "cached": True, "dimensions": 3},
    )

    async with AsyncClient(api_key="test") as client:
        response = await client.embed("hello world")
        assert isinstance(response, EmbedResponse)
        assert response.vector == [0.1, 0.2, 0.3]
        assert response.cached is True
        assert response.dimensions == 3


@pytest.mark.asyncio
async def test_async_embed_with_model_override(httpx_mock: HTTPXMock) -> None:
    """embed() should allow model override."""
    httpx_mock.add_response(
        url="https://api.vector-embed-cache.com/v1/embed",
        method="POST",
        json={"vector": [0.1], "cached": False, "dimensions": 1},
    )

    async with AsyncClient(api_key="test", model="default-model") as client:
        await client.embed("test", model="override-model")
        request = httpx_mock.get_requests()[0]
        import json

        body = json.loads(request.content)
        assert body["model"] == "override-model"


@pytest.mark.asyncio
async def test_async_embed_batch_success(httpx_mock: HTTPXMock) -> None:
    """embed_batch() should return list of EmbedResponse."""
    httpx_mock.add_response(
        url="https://api.vector-embed-cache.com/v1/embed/batch",
        method="POST",
        json=[
            {"vector": [0.1], "cached": True, "dimensions": 1},
            {"vector": [0.2], "cached": False, "dimensions": 1},
        ],
    )

    async with AsyncClient(api_key="test") as client:
        responses = await client.embed_batch(["hello", "world"])
        assert len(responses) == 2
        assert all(isinstance(r, EmbedResponse) for r in responses)
        assert responses[0].vector == [0.1]
        assert responses[0].cached is True
        assert responses[1].vector == [0.2]
        assert responses[1].cached is False


@pytest.mark.asyncio
async def test_async_search_success(httpx_mock: HTTPXMock) -> None:
    """search() should return SearchResponse."""
    httpx_mock.add_response(
        url="https://api.vector-embed-cache.com/v1/search",
        method="POST",
        json={
            "results": [
                {
                    "text_hash": "abc",
                    "score": 0.95,
                    "text": None,
                    "model": "m",
                    "hit_count": 1,
                }
            ],
            "total": 1,
            "search_time_ms": 5.0,
        },
    )

    async with AsyncClient(api_key="test") as client:
        response = await client.search("query")
        assert isinstance(response, SearchResponse)
        assert len(response.results) == 1
        assert response.results[0].text_hash == "abc"
        assert response.results[0].score == 0.95
        assert response.total == 1
        assert response.search_time_ms == 5.0


@pytest.mark.asyncio
async def test_async_search_with_options(httpx_mock: HTTPXMock) -> None:
    """search() should pass options correctly."""
    httpx_mock.add_response(
        url="https://api.vector-embed-cache.com/v1/search",
        method="POST",
        json={
            "results": [],
            "total": 0,
            "search_time_ms": 1.0,
        },
    )

    async with AsyncClient(api_key="test") as client:
        await client.search("query", top_k=5, min_score=0.5, include_text=True)
        request = httpx_mock.get_requests()[0]
        import json

        body = json.loads(request.content)
        assert body["top_k"] == 5
        assert body["min_score"] == 0.5
        assert body["include_text"] is True


@pytest.mark.asyncio
async def test_async_stats_success(httpx_mock: HTTPXMock) -> None:
    """stats() should return StatsResponse."""
    httpx_mock.add_response(
        url="https://api.vector-embed-cache.com/v1/stats",
        method="GET",
        json={"cache_hits": 100, "cache_misses": 10, "total_cached": 110},
    )

    async with AsyncClient(api_key="test") as client:
        response = await client.stats()
        assert isinstance(response, StatsResponse)
        assert response.cache_hits == 100
        assert response.cache_misses == 10
        assert response.total_cached == 110


@pytest.mark.asyncio
async def test_async_authentication_error(httpx_mock: HTTPXMock) -> None:
    """401 response should raise AuthenticationError."""
    httpx_mock.add_response(
        url="https://api.vector-embed-cache.com/v1/embed",
        method="POST",
        status_code=401,
        json={"error": "unauthorized"},
    )

    async with AsyncClient(api_key="bad_key") as client:
        with pytest.raises(AuthenticationError):
            await client.embed("test")


@pytest.mark.asyncio
async def test_async_rate_limit_error(httpx_mock: HTTPXMock) -> None:
    """429 response should raise RateLimitError with retry_after."""
    httpx_mock.add_response(
        url="https://api.vector-embed-cache.com/v1/embed",
        method="POST",
        status_code=429,
        json={"error": "rate limited"},
        headers={"retry-after": "30"},
    )

    async with AsyncClient(api_key="test", retries=0) as client:
        with pytest.raises(RateLimitError) as exc_info:
            await client.embed("test")
        assert exc_info.value.retry_after == 30


@pytest.mark.asyncio
async def test_async_retry_on_server_error(httpx_mock: HTTPXMock) -> None:
    """Client should retry on 500 errors."""
    # First request fails with 500, second succeeds
    httpx_mock.add_response(status_code=500, json={"error": "server error"})
    httpx_mock.add_response(json={"vector": [0.1], "cached": False, "dimensions": 1})

    async with AsyncClient(api_key="test", retries=1) as client:
        response = await client.embed("test")
        assert response.cached is False
        # Should have made 2 requests
        assert len(httpx_mock.get_requests()) == 2


@pytest.mark.asyncio
async def test_async_custom_base_url(httpx_mock: HTTPXMock) -> None:
    """Client should use custom base_url."""
    httpx_mock.add_response(
        url="https://custom.api.com/v1/stats",
        method="GET",
        json={"cache_hits": 0, "cache_misses": 0, "total_cached": 0},
    )

    async with AsyncClient(api_key="test", base_url="https://custom.api.com") as client:
        await client.stats()
        request = httpx_mock.get_requests()[0]
        assert str(request.url) == "https://custom.api.com/v1/stats"


@pytest.mark.asyncio
async def test_async_headers_include_auth(httpx_mock: HTTPXMock) -> None:
    """Requests should include Authorization header."""
    httpx_mock.add_response(
        url="https://api.vector-embed-cache.com/v1/stats",
        method="GET",
        json={"cache_hits": 0, "cache_misses": 0, "total_cached": 0},
    )

    async with AsyncClient(api_key="my_secret_key") as client:
        await client.stats()
        request = httpx_mock.get_requests()[0]
        assert request.headers["authorization"] == "Bearer my_secret_key"
