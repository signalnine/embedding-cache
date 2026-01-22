"""Tests for sync Client class."""

from __future__ import annotations

import os

import pytest
from pytest_httpx import HTTPXMock

from vector_embed_client import Client
from vector_embed_client.errors import (
    AuthenticationError,
    NetworkError,
    RateLimitError,
    ServerError,
    ValidationError,
)
from vector_embed_client.types import EmbedResponse, SearchResponse, StatsResponse


class TestClientInitialization:
    """Tests for Client initialization and API key handling."""

    def test_client_requires_api_key(self):
        """Client should raise ValueError if no API key is provided."""
        # Ensure env var is not set
        os.environ.pop("VECTOR_EMBED_API_KEY", None)
        with pytest.raises(ValueError, match="API key required"):
            Client()

    def test_client_uses_env_var(self, monkeypatch: pytest.MonkeyPatch):
        """Client should fall back to environment variable for API key."""
        monkeypatch.setenv("VECTOR_EMBED_API_KEY", "env_test_key")
        client = Client()
        assert client._api_key == "env_test_key"
        client.close()

    def test_client_explicit_key_overrides_env(self, monkeypatch: pytest.MonkeyPatch):
        """Explicit API key should override environment variable."""
        monkeypatch.setenv("VECTOR_EMBED_API_KEY", "env_key")
        client = Client(api_key="explicit_key")
        assert client._api_key == "explicit_key"
        client.close()

    def test_client_default_base_url(self):
        """Client should use default base URL if not specified."""
        client = Client(api_key="test")
        assert client._base_url == "https://api.vector-embed-cache.com"
        client.close()

    def test_client_custom_base_url(self):
        """Client should accept custom base URL."""
        client = Client(api_key="test", base_url="https://custom.example.com")
        assert client._base_url == "https://custom.example.com"
        client.close()

    def test_client_default_model(self):
        """Client should use default model if not specified."""
        client = Client(api_key="test")
        assert client._model == "nomic-v1.5"
        client.close()

    def test_client_custom_model(self):
        """Client should accept custom model."""
        client = Client(api_key="test", model="openai:text-embedding-3-small")
        assert client._model == "openai:text-embedding-3-small"
        client.close()


class TestClientContextManager:
    """Tests for Client context manager support."""

    def test_client_context_manager(self):
        """Client should support context manager protocol."""
        with Client(api_key="test") as client:
            assert client._http_client is not None
        # After exit, client should be closed
        assert client._http_client.is_closed

    def test_client_close(self):
        """Client.close() should close the HTTP client."""
        client = Client(api_key="test")
        assert not client._http_client.is_closed
        client.close()
        assert client._http_client.is_closed


class TestEmbed:
    """Tests for embed method."""

    def test_embed_success(self, httpx_mock: HTTPXMock):
        """Successful embed should return EmbedResponse."""
        httpx_mock.add_response(
            url="https://api.vector-embed-cache.com/v1/embed",
            method="POST",
            json={"vector": [0.1, 0.2, 0.3], "cached": True, "dimensions": 3},
        )

        with Client(api_key="test") as client:
            response = client.embed("hello world")
            assert isinstance(response, EmbedResponse)
            assert response.vector == [0.1, 0.2, 0.3]
            assert response.cached is True
            assert response.dimensions == 3

    def test_embed_uses_default_model(self, httpx_mock: HTTPXMock):
        """Embed should use the client's default model."""
        httpx_mock.add_response(
            url="https://api.vector-embed-cache.com/v1/embed",
            method="POST",
            json={"vector": [0.1], "cached": True, "dimensions": 1},
        )

        with Client(api_key="test", model="custom-model") as client:
            client.embed("test")

        request = httpx_mock.get_request()
        assert request is not None
        import json

        body = json.loads(request.content)
        assert body["model"] == "custom-model"

    def test_embed_model_override(self, httpx_mock: HTTPXMock):
        """Model parameter should override client's default model."""
        httpx_mock.add_response(
            url="https://api.vector-embed-cache.com/v1/embed",
            method="POST",
            json={"vector": [0.1], "cached": True, "dimensions": 1},
        )

        with Client(api_key="test", model="default-model") as client:
            client.embed("test", model="override-model")

        request = httpx_mock.get_request()
        assert request is not None
        import json

        body = json.loads(request.content)
        assert body["model"] == "override-model"

    def test_embed_sends_correct_request(self, httpx_mock: HTTPXMock):
        """Embed should send correct request body."""
        httpx_mock.add_response(
            url="https://api.vector-embed-cache.com/v1/embed",
            method="POST",
            json={"vector": [0.1], "cached": True, "dimensions": 1},
        )

        with Client(api_key="test_key") as client:
            client.embed("hello world")

        request = httpx_mock.get_request()
        assert request is not None
        assert request.headers["Authorization"] == "Bearer test_key"
        assert request.headers["Content-Type"] == "application/json"
        import json

        body = json.loads(request.content)
        assert body["text"] == "hello world"


class TestEmbedBatch:
    """Tests for embed_batch method."""

    def test_embed_batch_success(self, httpx_mock: HTTPXMock):
        """Successful embed_batch should return list of EmbedResponse."""
        httpx_mock.add_response(
            url="https://api.vector-embed-cache.com/v1/embed/batch",
            method="POST",
            json=[
                {"vector": [0.1], "cached": True, "dimensions": 1},
                {"vector": [0.2], "cached": False, "dimensions": 1},
            ],
        )

        with Client(api_key="test") as client:
            responses = client.embed_batch(["hello", "world"])
            assert len(responses) == 2
            assert all(isinstance(r, EmbedResponse) for r in responses)
            assert responses[0].cached is True
            assert responses[1].cached is False

    def test_embed_batch_sends_correct_request(self, httpx_mock: HTTPXMock):
        """Embed_batch should send correct request body."""
        httpx_mock.add_response(
            url="https://api.vector-embed-cache.com/v1/embed/batch",
            method="POST",
            json=[{"vector": [0.1], "cached": True, "dimensions": 1}],
        )

        with Client(api_key="test") as client:
            client.embed_batch(["text1", "text2", "text3"])

        request = httpx_mock.get_request()
        assert request is not None
        import json

        body = json.loads(request.content)
        assert body["texts"] == ["text1", "text2", "text3"]


class TestSearch:
    """Tests for search method."""

    def test_search_success(self, httpx_mock: HTTPXMock):
        """Successful search should return SearchResponse."""
        httpx_mock.add_response(
            url="https://api.vector-embed-cache.com/v1/search",
            method="POST",
            json={
                "results": [
                    {
                        "text_hash": "abc",
                        "score": 0.95,
                        "text": None,
                        "model": "nomic-v1.5",
                        "hit_count": 1,
                    }
                ],
                "total": 1,
                "search_time_ms": 5.0,
            },
        )

        with Client(api_key="test") as client:
            response = client.search("query", top_k=5)
            assert isinstance(response, SearchResponse)
            assert len(response.results) == 1
            assert response.results[0].score == 0.95
            assert response.total == 1

    def test_search_with_include_text(self, httpx_mock: HTTPXMock):
        """Search with include_text should return text in results."""
        httpx_mock.add_response(
            url="https://api.vector-embed-cache.com/v1/search",
            method="POST",
            json={
                "results": [
                    {
                        "text_hash": "abc",
                        "score": 0.95,
                        "text": "matching text",
                        "model": "nomic-v1.5",
                        "hit_count": 5,
                    }
                ],
                "total": 1,
                "search_time_ms": 3.0,
            },
        )

        with Client(api_key="test") as client:
            response = client.search("query", include_text=True)
            assert response.results[0].text == "matching text"

    def test_search_sends_correct_request(self, httpx_mock: HTTPXMock):
        """Search should send correct request body."""
        httpx_mock.add_response(
            url="https://api.vector-embed-cache.com/v1/search",
            method="POST",
            json={"results": [], "total": 0, "search_time_ms": 1.0},
        )

        with Client(api_key="test") as client:
            client.search("query text", top_k=20, min_score=0.5, include_text=True)

        request = httpx_mock.get_request()
        assert request is not None
        import json

        body = json.loads(request.content)
        assert body["text"] == "query text"
        assert body["top_k"] == 20
        assert body["min_score"] == 0.5
        assert body["include_text"] is True


class TestStats:
    """Tests for stats method."""

    def test_stats_success(self, httpx_mock: HTTPXMock):
        """Successful stats should return StatsResponse."""
        httpx_mock.add_response(
            url="https://api.vector-embed-cache.com/v1/stats",
            method="GET",
            json={"cache_hits": 100, "cache_misses": 10, "total_cached": 110},
        )

        with Client(api_key="test") as client:
            response = client.stats()
            assert isinstance(response, StatsResponse)
            assert response.cache_hits == 100
            assert response.cache_misses == 10
            assert response.total_cached == 110


class TestErrorHandling:
    """Tests for error handling."""

    def test_authentication_error(self, httpx_mock: HTTPXMock):
        """401 response should raise AuthenticationError."""
        httpx_mock.add_response(
            url="https://api.vector-embed-cache.com/v1/embed",
            method="POST",
            status_code=401,
            json={"error": "unauthorized"},
        )

        with Client(api_key="bad_key") as client:
            with pytest.raises(AuthenticationError) as exc_info:
                client.embed("test")
            assert exc_info.value.status == 401

    def test_validation_error(self, httpx_mock: HTTPXMock):
        """400 response should raise ValidationError."""
        httpx_mock.add_response(
            url="https://api.vector-embed-cache.com/v1/embed",
            method="POST",
            status_code=400,
            json={"error": "invalid text"},
        )

        with Client(api_key="test") as client:
            with pytest.raises(ValidationError) as exc_info:
                client.embed("test")
            assert exc_info.value.status == 400

    def test_rate_limit_error(self, httpx_mock: HTTPXMock):
        """429 response should raise RateLimitError with retry_after."""
        httpx_mock.add_response(
            url="https://api.vector-embed-cache.com/v1/embed",
            method="POST",
            status_code=429,
            json={"error": "rate limited"},
            headers={"retry-after": "30"},
        )

        with Client(api_key="test", retries=0) as client:
            with pytest.raises(RateLimitError) as exc_info:
                client.embed("test")
            assert exc_info.value.retry_after == 30

    def test_server_error(self, httpx_mock: HTTPXMock):
        """500 response should raise ServerError after retries exhausted."""
        # Add multiple responses for retry attempts
        for _ in range(4):  # Initial + 3 retries
            httpx_mock.add_response(
                url="https://api.vector-embed-cache.com/v1/embed",
                method="POST",
                status_code=500,
                json={"error": "internal server error"},
            )

        with Client(api_key="test", retries=3) as client:
            with pytest.raises(ServerError) as exc_info:
                client.embed("test")
            assert exc_info.value.status == 500


class TestRetryLogic:
    """Tests for retry logic."""

    def test_retry_on_server_error(self, httpx_mock: HTTPXMock):
        """Client should retry on 5xx errors."""
        # First request fails, second succeeds
        httpx_mock.add_response(
            url="https://api.vector-embed-cache.com/v1/embed",
            method="POST",
            status_code=503,
            json={"error": "service unavailable"},
        )
        httpx_mock.add_response(
            url="https://api.vector-embed-cache.com/v1/embed",
            method="POST",
            json={"vector": [0.1], "cached": True, "dimensions": 1},
        )

        with Client(api_key="test", retries=1) as client:
            response = client.embed("test")
            assert response.vector == [0.1]

        # Should have made 2 requests
        assert len(httpx_mock.get_requests()) == 2

    def test_retry_on_rate_limit(self, httpx_mock: HTTPXMock):
        """Client should retry on 429 errors."""
        httpx_mock.add_response(
            url="https://api.vector-embed-cache.com/v1/embed",
            method="POST",
            status_code=429,
            json={"error": "rate limited"},
            headers={"retry-after": "1"},
        )
        httpx_mock.add_response(
            url="https://api.vector-embed-cache.com/v1/embed",
            method="POST",
            json={"vector": [0.1], "cached": True, "dimensions": 1},
        )

        with Client(api_key="test", retries=1) as client:
            response = client.embed("test")
            assert response.vector == [0.1]

    def test_no_retry_on_4xx(self, httpx_mock: HTTPXMock):
        """Client should not retry on non-retryable 4xx errors."""
        httpx_mock.add_response(
            url="https://api.vector-embed-cache.com/v1/embed",
            method="POST",
            status_code=400,
            json={"error": "bad request"},
        )

        with Client(api_key="test", retries=3) as client, pytest.raises(ValidationError):
            client.embed("test")

        # Should only make 1 request
        assert len(httpx_mock.get_requests()) == 1

    def test_no_retry_on_401(self, httpx_mock: HTTPXMock):
        """Client should not retry on authentication errors."""
        httpx_mock.add_response(
            url="https://api.vector-embed-cache.com/v1/embed",
            method="POST",
            status_code=401,
            json={"error": "unauthorized"},
        )

        with Client(api_key="test", retries=3) as client, pytest.raises(AuthenticationError):
            client.embed("test")

        # Should only make 1 request
        assert len(httpx_mock.get_requests()) == 1

    def test_retries_zero_disables_retry(self, httpx_mock: HTTPXMock):
        """Setting retries=0 should disable retry logic."""
        httpx_mock.add_response(
            url="https://api.vector-embed-cache.com/v1/embed",
            method="POST",
            status_code=503,
            json={"error": "service unavailable"},
        )

        with Client(api_key="test", retries=0) as client, pytest.raises(ServerError):
            client.embed("test")

        # Should only make 1 request
        assert len(httpx_mock.get_requests()) == 1


class TestNetworkErrors:
    """Tests for network error handling."""

    def test_connection_error(self, httpx_mock: HTTPXMock):
        """Connection errors should raise NetworkError."""
        import httpx

        httpx_mock.add_exception(
            httpx.ConnectError("Failed to connect"),
            url="https://api.vector-embed-cache.com/v1/embed",
        )

        with Client(api_key="test", retries=0) as client:
            with pytest.raises(NetworkError) as exc_info:
                client.embed("test")
            assert "connection" in str(exc_info.value).lower()

    def test_timeout_error(self, httpx_mock: HTTPXMock):
        """Timeout errors should raise NetworkError."""
        import httpx

        httpx_mock.add_exception(
            httpx.TimeoutException("Request timed out"),
            url="https://api.vector-embed-cache.com/v1/embed",
        )

        with Client(api_key="test", retries=0) as client:
            with pytest.raises(NetworkError) as exc_info:
                client.embed("test")
            assert "timeout" in str(exc_info.value).lower()
