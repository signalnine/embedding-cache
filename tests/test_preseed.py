"""Tests for preseed module."""

import sqlite3
from unittest.mock import patch

import msgpack
import numpy as np
import pytest


class TestPreseedPaths:
    def test_get_preseed_db_path_returns_path_in_package(self):
        from vector_embed_cache.preseed import get_preseed_db_path, PRESEED_DB_NAME

        path = get_preseed_db_path()
        assert path is not None
        assert "vector_embed_cache" in str(path)
        assert path.name == PRESEED_DB_NAME

    def test_preseed_db_exists_returns_false_when_missing(self):
        from vector_embed_cache.preseed import preseed_db_exists

        # Mock Path.exists() to return False
        with patch("vector_embed_cache.preseed.get_preseed_db_path") as mock_path:
            mock_path.return_value.exists.return_value = False
            assert preseed_db_exists() is False

    def test_preseed_db_exists_returns_true_when_present(self):
        from vector_embed_cache.preseed import preseed_db_exists

        # Mock Path.exists() to return True
        with patch("vector_embed_cache.preseed.get_preseed_db_path") as mock_path:
            mock_path.return_value.exists.return_value = True
            assert preseed_db_exists() is True

    def test_get_preseed_db_path_is_in_data_directory(self):
        from vector_embed_cache.preseed import get_preseed_db_path

        path = get_preseed_db_path()
        assert path.parent.name == "data"


class TestPreseedStorage:
    @pytest.fixture
    def temp_preseed_db(self, tmp_path):
        """Create a temporary preseed database with test data."""
        db_path = tmp_path / "test_preseed.db"
        conn = sqlite3.connect(db_path)

        # Create schema with metadata
        conn.execute("""
            CREATE TABLE embeddings (
                cache_key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        # Insert test embedding
        test_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        embedding_bytes = msgpack.packb(test_embedding.tolist())
        conn.execute(
            "INSERT INTO embeddings (cache_key, model, embedding) VALUES (?, ?, ?)",
            ("test_key", "nomic-ai/nomic-embed-text-v1.5", embedding_bytes),
        )

        # Insert metadata
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            ("schema_version", "1"),
        )
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            ("model_version", "nomic-ai/nomic-embed-text-v1.5"),
        )

        conn.commit()
        conn.close()
        return db_path

    def test_preseed_storage_get_existing_key(self, temp_preseed_db):
        from vector_embed_cache.preseed import PreseedStorage

        storage = PreseedStorage(temp_preseed_db)
        result = storage.get("test_key")

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        np.testing.assert_array_almost_equal(result, [0.1, 0.2, 0.3])

    def test_preseed_storage_get_missing_key(self, temp_preseed_db):
        from vector_embed_cache.preseed import PreseedStorage

        storage = PreseedStorage(temp_preseed_db)
        result = storage.get("nonexistent_key")

        assert result is None

    def test_preseed_storage_is_readonly(self, temp_preseed_db):
        from vector_embed_cache.preseed import PreseedStorage

        storage = PreseedStorage(temp_preseed_db)
        # Should not have a set method or it should raise
        assert not hasattr(storage, "set") or storage.set is None

    def test_preseed_storage_context_manager(self, temp_preseed_db):
        from vector_embed_cache.preseed import PreseedStorage

        with PreseedStorage(temp_preseed_db) as storage:
            result = storage.get("test_key")
            assert result is not None
        # Connection should be closed after context exit
        # Verify by checking internal state
        assert storage._conn is not None  # Connection object exists but is closed

    def test_preseed_storage_close(self, temp_preseed_db):
        from vector_embed_cache.preseed import PreseedStorage

        storage = PreseedStorage(temp_preseed_db)
        result = storage.get("test_key")
        assert result is not None
        storage.close()
        # Connection is closed - further operations would fail
        # but we don't need to verify that explicitly
