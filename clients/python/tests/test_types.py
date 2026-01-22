"""Tests for response types."""

from vector_embed_client.types import (
    EmbedResponse,
    SearchResponse,
    SearchResult,
    StatsResponse,
)


def test_embed_response():
    response = EmbedResponse(vector=[0.1, 0.2, 0.3], cached=True, dimensions=3)
    assert response.vector == [0.1, 0.2, 0.3]
    assert response.cached is True
    assert response.dimensions == 3


def test_embed_response_frozen():
    response = EmbedResponse(vector=[0.1], cached=False, dimensions=1)
    # Should raise if we try to modify (frozen)
    try:
        response.cached = True  # type: ignore
        raise AssertionError("Should have raised")
    except AttributeError:
        pass


def test_search_result():
    result = SearchResult(
        text_hash="abc123",
        score=0.95,
        text="hello world",
        model="nomic-v1.5",
        hit_count=5,
    )
    assert result.text_hash == "abc123"
    assert result.score == 0.95
    assert result.text == "hello world"


def test_search_result_optional_text():
    result = SearchResult(
        text_hash="abc123",
        score=0.95,
        text=None,
        model="nomic-v1.5",
        hit_count=5,
    )
    assert result.text is None


def test_search_response():
    results = [
        SearchResult(text_hash="a", score=0.9, text=None, model="m", hit_count=1),
    ]
    response = SearchResponse(results=results, total=1, search_time_ms=5.5)
    assert len(response.results) == 1
    assert response.total == 1
    assert response.search_time_ms == 5.5


def test_stats_response():
    stats = StatsResponse(cache_hits=100, cache_misses=10, total_cached=110)
    assert stats.cache_hits == 100
    assert stats.cache_misses == 10
    assert stats.total_cached == 110
