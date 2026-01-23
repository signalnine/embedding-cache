"""Tests for SQLite storage layer."""

import numpy as np
from pathlib import Path
from vector_embed_cache.storage import EmbeddingStorage


def test_storage_init_creates_database(temp_cache_dir):
    """Should create database file on initialization."""
    db_path = Path(temp_cache_dir) / "test.db"
    _ = EmbeddingStorage(str(db_path))  # Side effect: creates database
    assert db_path.exists()


def test_storage_init_creates_table(temp_cache_dir):
    """Should create embeddings table with correct schema."""
    db_path = Path(temp_cache_dir) / "test.db"
    storage = EmbeddingStorage(str(db_path))

    # Query sqlite_master to verify table exists
    cursor = storage._conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'")
    assert cursor.fetchone() is not None


def test_storage_get_nonexistent(temp_cache_dir):
    """Should return None for nonexistent cache key."""
    db_path = Path(temp_cache_dir) / "test.db"
    storage = EmbeddingStorage(str(db_path))
    result = storage.get("nonexistent_key")
    assert result is None


def test_storage_set_and_get(temp_cache_dir):
    """Should store and retrieve embeddings with float16 compression."""
    db_path = Path(temp_cache_dir) / "test.db"
    storage = EmbeddingStorage(str(db_path))

    cache_key = "test_key"
    model = "nomic-embed-text-v2"
    embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    storage.set(cache_key, model, embedding)
    result = storage.get(cache_key)

    assert result is not None
    # Use decimal=3 for float16 precision (3-4 significant digits)
    np.testing.assert_array_almost_equal(result, embedding, decimal=3)


def test_storage_updates_access_count(temp_cache_dir):
    """Should increment access count on repeated gets."""
    db_path = Path(temp_cache_dir) / "test.db"
    storage = EmbeddingStorage(str(db_path))

    cache_key = "test_key"
    model = "nomic-embed-text-v2"
    embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    storage.set(cache_key, model, embedding)
    storage.get(cache_key)
    storage.get(cache_key)

    # Query access_count directly
    cursor = storage._conn.cursor()
    cursor.execute("SELECT access_count FROM embeddings WHERE cache_key = ?", (cache_key,))
    access_count = cursor.fetchone()[0]
    assert access_count == 3  # 1 from set + 2 from gets


def test_storage_preserves_access_count_on_update(temp_cache_dir):
    """Should preserve access_count when updating existing entry."""
    db_path = Path(temp_cache_dir) / "test.db"
    storage = EmbeddingStorage(str(db_path))

    cache_key = "test_key"
    model = "nomic-embed-text-v2"
    embedding1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    embedding2 = np.array([0.4, 0.5, 0.6], dtype=np.float32)

    # Initial set
    storage.set(cache_key, model, embedding1)
    storage.get(cache_key)  # access_count now 2

    # Update with new embedding
    storage.set(cache_key, model, embedding2)

    # Verify access_count preserved
    cursor = storage._conn.cursor()
    cursor.execute("SELECT access_count FROM embeddings WHERE cache_key = ?", (cache_key,))
    access_count = cursor.fetchone()[0]
    assert access_count == 2, "access_count should be preserved on update"
