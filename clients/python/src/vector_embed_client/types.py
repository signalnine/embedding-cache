"""Response types for vector-embed-cache-client."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbedResponse:
    """Response from embed endpoint."""

    vector: list[float]
    cached: bool
    dimensions: int


@dataclass(frozen=True)
class SearchResult:
    """Individual search result."""

    text_hash: str
    score: float
    text: str | None
    model: str
    hit_count: int


@dataclass(frozen=True)
class SearchResponse:
    """Response from search endpoint."""

    results: list[SearchResult]
    total: int
    search_time_ms: float


@dataclass(frozen=True)
class StatsResponse:
    """Response from stats endpoint."""

    cache_hits: int
    cache_misses: int
    total_cached: int
