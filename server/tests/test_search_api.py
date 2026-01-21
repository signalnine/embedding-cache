# server/tests/test_search_api.py
"""Tests for /v1/search endpoint."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime


@pytest.fixture
def mock_api_key():
    """Create mock API key object."""
    mock = MagicMock()
    mock.user_id = "test-user-123"
    mock.tier = "paid"
    mock.last_used_at = None
    return mock


@pytest.fixture
def client(mock_api_key):
    """Create test client with mocked authentication."""
    from app.main import app
    from app.auth import get_current_api_key
    from app.database import get_db

    # Override auth dependency
    app.dependency_overrides[get_current_api_key] = lambda: mock_api_key

    # Override database dependency
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db

    yield TestClient(app)

    # Clean up
    app.dependency_overrides.clear()


@pytest.fixture
def unauthenticated_client():
    """Create test client without auth override."""
    from app.main import app

    # Clear any overrides
    app.dependency_overrides.clear()
    return TestClient(app)


class TestSearchValidation:
    """Test request validation."""

    def test_search_rejects_both_text_and_vector(self, client):
        """Cannot provide both query_text and query_vector."""
        response = client.post(
            "/v1/search",
            json={
                "query_text": "hello",
                "query_vector": [0.1] * 768,
                "model": "nomic-v1.5",
            },
            headers={"Authorization": "Bearer vec_test"},
        )

        assert response.status_code == 422  # Validation error

    def test_search_requires_either_text_or_vector(self, client):
        """Must provide query_text or query_vector."""
        response = client.post(
            "/v1/search",
            json={"model": "nomic-v1.5"},
            headers={"Authorization": "Bearer vec_test"},
        )

        assert response.status_code == 422

    def test_search_requires_model(self, client):
        """Model field is required."""
        response = client.post(
            "/v1/search",
            json={"query_text": "hello"},
            headers={"Authorization": "Bearer vec_test"},
        )

        assert response.status_code == 422

    def test_search_rejects_unknown_model(self, client):
        """Unknown model returns 400."""
        response = client.post(
            "/v1/search",
            json={"query_text": "hello", "model": "unknown-model-xyz"},
            headers={"Authorization": "Bearer vec_test"},
        )

        assert response.status_code == 400
        assert "Unknown model" in response.json()["detail"]


class TestSearchWithVector:
    """Test search with pre-computed vector."""

    def test_search_with_vector_validates_dimensions(self, client):
        """Vector dimension must match model."""
        # nomic-v1.5 expects 768 dimensions, provide 384
        response = client.post(
            "/v1/search",
            json={
                "query_vector": [0.1] * 384,
                "model": "nomic-v1.5",
            },
            headers={"Authorization": "Bearer vec_test"},
        )

        assert response.status_code == 400
        assert "dimension" in response.json()["detail"].lower()

    def test_search_with_vector_returns_results(self, client):
        """Search with valid vector returns results."""
        mock_results = [
            {
                "text_hash": "abc123",
                "original_text": "similar text",
                "model": "nomic-v1.5",
                "score": 0.85,
                "hit_count": 3,
            }
        ]

        with patch("app.routes.search.similarity_search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_results

            response = client.post(
                "/v1/search",
                json={
                    "query_vector": [0.1] * 768,
                    "model": "nomic-v1.5",
                },
                headers={"Authorization": "Bearer vec_test"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "nomic-v1.5"
        assert data["dimensions"] == 768
        assert len(data["results"]) == 1
        assert data["results"][0]["text_hash"] == "abc123"
        assert data["results"][0]["score"] == 0.85
        assert "search_time_ms" in data


class TestSearchWithText:
    """Test search with text query."""

    def test_search_with_text_computes_embedding(self, client):
        """Text query computes embedding then searches."""
        mock_embedding = [0.1] * 768
        mock_results = [
            {
                "text_hash": "xyz789",
                "original_text": "found text",
                "model": "nomic-v1.5",
                "score": 0.92,
                "hit_count": 1,
            }
        ]

        with patch("app.routes.search.compute_embedding", new_callable=AsyncMock) as mock_compute, \
             patch("app.routes.search.similarity_search", new_callable=AsyncMock) as mock_search:
            mock_compute.return_value = mock_embedding
            mock_search.return_value = mock_results

            response = client.post(
                "/v1/search",
                json={
                    "query_text": "find similar",
                    "model": "nomic-v1.5",
                },
                headers={"Authorization": "Bearer vec_test"},
            )

        assert response.status_code == 200
        mock_compute.assert_called_once_with("find similar", "nomic-v1.5")
        data = response.json()
        assert len(data["results"]) == 1


class TestSearchOptions:
    """Test search options (top_k, min_score, etc.)."""

    def test_top_k_limit(self, client):
        """Respects top_k parameter."""
        with patch("app.routes.search.similarity_search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = []

            response = client.post(
                "/v1/search",
                json={
                    "query_vector": [0.1] * 768,
                    "model": "nomic-v1.5",
                    "top_k": 5,
                },
                headers={"Authorization": "Bearer vec_test"},
            )

        assert response.status_code == 200
        # Verify top_k was passed to similarity_search
        call_args = mock_search.call_args
        assert call_args.kwargs["params"].top_k == 5

    def test_top_k_max_validation(self, client):
        """top_k has maximum value of 100."""
        response = client.post(
            "/v1/search",
            json={
                "query_vector": [0.1] * 768,
                "model": "nomic-v1.5",
                "top_k": 200,
            },
            headers={"Authorization": "Bearer vec_test"},
        )

        assert response.status_code == 422

    def test_include_text_option(self, client):
        """include_text controls whether text is returned."""
        mock_results = [
            {
                "text_hash": "abc123",
                "original_text": "the actual text",
                "model": "nomic-v1.5",
                "score": 0.85,
                "hit_count": 3,
            }
        ]

        with patch("app.routes.search.similarity_search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_results

            # With include_text=True (default)
            response = client.post(
                "/v1/search",
                json={
                    "query_vector": [0.1] * 768,
                    "model": "nomic-v1.5",
                    "include_text": True,
                },
                headers={"Authorization": "Bearer vec_test"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["results"][0]["text"] == "the actual text"

    def test_exclude_text_option(self, client):
        """include_text=False excludes text from results."""
        mock_results = [
            {
                "text_hash": "abc123",
                "original_text": "the actual text",
                "model": "nomic-v1.5",
                "score": 0.85,
                "hit_count": 3,
            }
        ]

        with patch("app.routes.search.similarity_search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_results

            response = client.post(
                "/v1/search",
                json={
                    "query_vector": [0.1] * 768,
                    "model": "nomic-v1.5",
                    "include_text": False,
                },
                headers={"Authorization": "Bearer vec_test"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["results"][0]["text"] is None


class TestSearchAuthentication:
    """Test authentication requirements."""

    def test_search_requires_auth(self, unauthenticated_client):
        """Search endpoint requires authentication."""
        response = unauthenticated_client.post(
            "/v1/search",
            json={
                "query_vector": [0.1] * 768,
                "model": "nomic-v1.5",
            },
        )

        # Without Authorization header, should fail
        assert response.status_code in [401, 422]
