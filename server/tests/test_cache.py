# server/tests/test_cache.py
import pytest
import struct
from unittest.mock import MagicMock
from app.cache import (
    get_cached_embedding,
    store_embedding,
    vector_to_bytes,
    bytes_to_vector,
)


def test_vector_to_bytes_roundtrip():
    """Vector should survive bytes conversion."""
    original = [0.1, 0.2, 0.3, 0.4]
    as_bytes = vector_to_bytes(original)
    recovered = bytes_to_vector(as_bytes)
    assert recovered == pytest.approx(original)


def test_vector_to_bytes_length():
    """Bytes should be 4 * len(vector) for float32."""
    vector = [0.1] * 768
    as_bytes = vector_to_bytes(vector)
    assert len(as_bytes) == 768 * 4


def test_get_cached_embedding_miss():
    """Cache miss should return None."""
    mock_db = MagicMock()
    # Mock the full query chain: query().filter().filter().order_by().first()
    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query.order_by.return_value = mock_query
    mock_query.first.return_value = None
    mock_db.query.return_value = mock_query

    result = get_cached_embedding(
        db=mock_db,
        text_hash="abc123",
        model="nomic-v1.5",
        model_version="1.0.0",
        tenant_id="usr_123"
    )
    assert result is None
