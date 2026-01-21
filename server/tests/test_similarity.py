import math
import pytest
from unittest.mock import AsyncMock, MagicMock
from app.similarity import (
    calculate_similarity_score,
    SUPPORTED_DIMENSIONS,
    similarity_search,
    SearchParams,
    validate_vector_normalization,
)


def make_normalized_vector(dimensions: int) -> list[float]:
    """Create a normalized unit vector of given dimensions."""
    component = 1.0 / math.sqrt(dimensions)
    return [component] * dimensions


class TestScoreCalculation:
    def test_identical_vectors_score_one(self):
        # Inner product of identical normalized vectors = 1
        # pgvector <#> returns -1 for identical vectors
        raw_distance = -1.0
        score = calculate_similarity_score(raw_distance)
        assert score == pytest.approx(1.0)

    def test_opposite_vectors_score_zero(self):
        # Inner product of opposite vectors = -1
        # pgvector <#> returns 1 for opposite vectors
        raw_distance = 1.0
        score = calculate_similarity_score(raw_distance)
        assert score == pytest.approx(0.0)

    def test_orthogonal_vectors_score_half(self):
        # Inner product of orthogonal vectors = 0
        # pgvector <#> returns 0 for orthogonal vectors
        raw_distance = 0.0
        score = calculate_similarity_score(raw_distance)
        assert score == pytest.approx(0.5)


class TestSupportedDimensions:
    def test_768_supported(self):
        assert 768 in SUPPORTED_DIMENSIONS

    def test_1536_supported(self):
        assert 1536 in SUPPORTED_DIMENSIONS

    def test_384_supported(self):
        assert 384 in SUPPORTED_DIMENSIONS

    def test_3072_not_supported(self):
        assert 3072 not in SUPPORTED_DIMENSIONS


class TestSimilaritySearch:
    @pytest.mark.asyncio
    async def test_search_builds_correct_query(self):
        # Mock database connection
        mock_db = AsyncMock()
        mock_db.fetch = AsyncMock(return_value=[
            {
                "text_hash": "abc123",
                "original_text": "hello world",
                "model": "nomic-v1.5",
                "score": 0.92,
                "hit_count": 5
            }
        ])

        # Mock transaction context manager
        mock_transaction = AsyncMock()
        mock_transaction.__aenter__ = AsyncMock(return_value=mock_transaction)
        mock_transaction.__aexit__ = AsyncMock(return_value=None)
        mock_db.transaction = MagicMock(return_value=mock_transaction)

        params = SearchParams(
            query_vector=make_normalized_vector(768),
            tenant_id="tenant-123",
            model="nomic-v1.5",
            dimensions=768,
            top_k=10,
            min_score=None
        )

        results = await similarity_search(mock_db, params)

        # Verify SET LOCAL was called
        mock_db.execute.assert_called()
        call_args = str(mock_db.execute.call_args)
        assert "SET LOCAL hnsw.ef_search" in call_args

        # Verify query structure
        fetch_call = mock_db.fetch.call_args
        query = fetch_call[0][0]
        assert "vector::vector(768)" in query
        assert "dimensions" in query
        assert "tenant_id" in query
        assert "ORDER BY" in query
        assert "LIMIT" in query

    @pytest.mark.asyncio
    async def test_search_rejects_unsupported_dimensions(self):
        mock_db = AsyncMock()

        params = SearchParams(
            query_vector=[0.1] * 3072,
            tenant_id="tenant-123",
            model="unknown-model",
            dimensions=3072,
            top_k=10,
            min_score=None
        )

        with pytest.raises(ValueError) as exc:
            await similarity_search(mock_db, params)
        assert "Unsupported dimension" in str(exc.value)


class TestVectorNormalization:
    def test_accepts_normalized_vector(self):
        vec = [1.0, 0.0, 0.0]  # Unit vector
        validate_vector_normalization(vec)  # Should not raise

    def test_accepts_approximately_normalized(self):
        # Normalized within tolerance
        vec = [0.577, 0.577, 0.577]  # ~unit vector
        validate_vector_normalization(vec)  # Should not raise

    def test_rejects_unnormalized_vector(self):
        vec = [1.0, 1.0, 1.0]  # norm = sqrt(3) ~ 1.73
        with pytest.raises(ValueError) as exc:
            validate_vector_normalization(vec)
        assert "normalized" in str(exc.value).lower()

    def test_rejects_zero_vector(self):
        vec = [0.0, 0.0, 0.0]
        with pytest.raises(ValueError):
            validate_vector_normalization(vec)
