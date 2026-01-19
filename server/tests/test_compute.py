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
