import pytest
from app.similarity import calculate_similarity_score, SUPPORTED_DIMENSIONS


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
