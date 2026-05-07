# server/tests/test_compute.py
import pytest
import numpy as np
from unittest.mock import patch, MagicMock


def test_compute_embedding_returns_list():
    """Compute should return list of floats."""
    # Mock sentence-transformers to avoid GPU requirement in tests
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1] * 768])

    with patch("app.compute._get_model", return_value=mock_model):
        from app.compute import compute_embedding_sync
        result = compute_embedding_sync("hello world", "nomic-v1.5")
        assert isinstance(result, list)
        assert len(result) == 768


def test_compute_batch_returns_multiple():
    """Batch compute should return multiple embeddings."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1] * 768, [0.2] * 768])

    with patch("app.compute._get_model", return_value=mock_model):
        from app.compute import compute_batch_sync
        result = compute_batch_sync(["hello", "world"], "nomic-v1.5")
        assert len(result) == 2
        assert len(result[0]) == 768


# bd: embedding-cache-5lf -- nomic models distinguish documents from queries
# via 'search_document:' / 'search_query:' prefixes. Other encoders do not
# use this convention; applying the prefix to them embeds it as literal text.

def test_nomic_document_role_uses_document_prefix():
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.0] * 768])
    with patch("app.compute._get_model", return_value=mock_model):
        from app.compute import compute_embedding_sync
        compute_embedding_sync("hello", "nomic-v1.5", role="document")
    mock_model.encode.assert_called_once_with(["search_document: hello"])


def test_nomic_query_role_uses_query_prefix():
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.0] * 768])
    with patch("app.compute._get_model", return_value=mock_model):
        from app.compute import compute_embedding_sync
        compute_embedding_sync("hello", "nomic-v1.5", role="query")
    mock_model.encode.assert_called_once_with(["search_query: hello"])


def test_non_nomic_model_does_not_get_prefix():
    """all-MiniLM-L6-v2 doesn't use nomic prefixes; raw text must be passed."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.0] * 384])
    with patch("app.compute._get_model", return_value=mock_model):
        from app.compute import compute_embedding_sync
        compute_embedding_sync("hello", "all-MiniLM-L6-v2", role="document")
    mock_model.encode.assert_called_once_with(["hello"])


def test_compute_batch_applies_role_to_all_texts():
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.0] * 768, [0.0] * 768])
    with patch("app.compute._get_model", return_value=mock_model):
        from app.compute import compute_batch_sync
        compute_batch_sync(["a", "b"], "nomic-v1.5", role="query")
    mock_model.encode.assert_called_once_with(["search_query: a", "search_query: b"])


def test_compute_default_role_is_document_for_backwards_compat():
    """Existing /v1/embed callers don't pass role; document is the default."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.0] * 768])
    with patch("app.compute._get_model", return_value=mock_model):
        from app.compute import compute_embedding_sync
        compute_embedding_sync("hello", "nomic-v1.5")
    mock_model.encode.assert_called_once_with(["search_document: hello"])
