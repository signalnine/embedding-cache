# server/tests/test_schemas.py
import pytest
from pydantic import ValidationError
from app.schemas import (
    BatchEmbedRequest,
    MAX_TEXT_LENGTH,
    SearchRequest,
    SearchResponse,
    SearchResult,
)


class TestBatchEmbedRequest:
    def test_accepts_valid_texts(self):
        req = BatchEmbedRequest(texts=["hello", "world"])
        assert req.texts == ["hello", "world"]

    def test_rejects_empty_text_in_list(self):
        with pytest.raises(ValidationError):
            BatchEmbedRequest(texts=["hello", ""])

    def test_rejects_oversize_text_in_list(self):
        oversize = "a" * (MAX_TEXT_LENGTH + 1)
        with pytest.raises(ValidationError) as exc:
            BatchEmbedRequest(texts=["ok", oversize])
        assert "max length" in str(exc.value).lower()

    def test_accepts_text_at_max_length(self):
        at_limit = "a" * MAX_TEXT_LENGTH
        req = BatchEmbedRequest(texts=[at_limit])
        assert len(req.texts[0]) == MAX_TEXT_LENGTH

    def test_rejects_too_few_items(self):
        with pytest.raises(ValidationError):
            BatchEmbedRequest(texts=[])

    def test_rejects_too_many_items(self):
        with pytest.raises(ValidationError):
            BatchEmbedRequest(texts=["x"] * 101)


class TestSearchRequest:
    def test_valid_text_query(self):
        req = SearchRequest(query_text="hello world", model="nomic-v1.5")
        assert req.query_text == "hello world"
        assert req.query_vector is None
        assert req.top_k == 10  # default
        assert req.min_score is None
        assert req.include_vectors is False
        assert req.include_text is True

    def test_valid_vector_query(self):
        vec = [0.1] * 768
        req = SearchRequest(query_vector=vec, model="nomic-v1.5")
        assert req.query_vector == vec
        assert req.query_text is None

    def test_rejects_both_text_and_vector(self):
        with pytest.raises(ValidationError) as exc:
            SearchRequest(query_text="hello", query_vector=[0.1] * 768, model="nomic-v1.5")
        assert "mutually exclusive" in str(exc.value).lower()

    def test_rejects_neither_text_nor_vector(self):
        with pytest.raises(ValidationError) as exc:
            SearchRequest(model="nomic-v1.5")
        assert "required" in str(exc.value).lower()

    def test_model_required(self):
        with pytest.raises(ValidationError):
            SearchRequest(query_text="hello")

    def test_top_k_bounds(self):
        with pytest.raises(ValidationError):
            SearchRequest(query_text="hello", model="nomic-v1.5", top_k=0)
        with pytest.raises(ValidationError):
            SearchRequest(query_text="hello", model="nomic-v1.5", top_k=101)

    def test_min_score_bounds(self):
        with pytest.raises(ValidationError):
            SearchRequest(query_text="hello", model="nomic-v1.5", min_score=-0.1)
        with pytest.raises(ValidationError):
            SearchRequest(query_text="hello", model="nomic-v1.5", min_score=1.1)


class TestSearchResponse:
    def test_valid_response(self):
        result = SearchResult(
            text_hash="abc123",
            score=0.92,
            text="hello world",
            model="nomic-v1.5",
            hit_count=5,
        )
        resp = SearchResponse(
            results=[result],
            total=1,
            search_time_ms=12,
            model="nomic-v1.5",
            dimensions=768,
        )
        assert resp.total == 1
        assert resp.results[0].score == 0.92
