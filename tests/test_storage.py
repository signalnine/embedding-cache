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


# ============================================================================
# Compression-specific tests
# ============================================================================


def test_storage_new_format_stores_metadata(temp_cache_dir):
    """New entries should have dimensions and dtype set."""
    db_path = Path(temp_cache_dir) / "test.db"
    storage = EmbeddingStorage(str(db_path))

    embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    storage.set("key1", "model1", embedding)

    cursor = storage._conn.execute(
        "SELECT dimensions, dtype FROM embeddings WHERE cache_key = 'key1'"
    )
    dimensions, dtype = cursor.fetchone()

    assert dimensions == 5
    assert dtype == "float16"


def test_storage_new_format_blob_size(temp_cache_dir):
    """New format blob should be dimensions * 2 bytes (float16)."""
    db_path = Path(temp_cache_dir) / "test.db"
    storage = EmbeddingStorage(str(db_path))

    embedding = np.array([0.1] * 768, dtype=np.float32)
    storage.set("key1", "model1", embedding)

    cursor = storage._conn.execute(
        "SELECT embedding FROM embeddings WHERE cache_key = 'key1'"
    )
    blob = cursor.fetchone()[0]

    # 768 dimensions * 2 bytes per float16 = 1536 bytes
    assert len(blob) == 768 * 2


def test_storage_roundtrip_preserves_cosine_similarity(temp_cache_dir):
    """Float16 compression should preserve cosine similarity."""
    db_path = Path(temp_cache_dir) / "test.db"
    storage = EmbeddingStorage(str(db_path))

    # Use realistic embedding values
    np.random.seed(42)
    original = np.random.randn(768).astype(np.float32)
    original = original / np.linalg.norm(original)  # Normalize

    storage.set("key1", "model1", original)
    retrieved = storage.get("key1")

    # Compute cosine similarity
    cosine_sim = np.dot(original, retrieved) / (
        np.linalg.norm(original) * np.linalg.norm(retrieved)
    )

    # Should be very close to 1.0
    assert cosine_sim > 0.9999, f"Cosine similarity {cosine_sim} too low"


def test_storage_validation_blob_size_mismatch(temp_cache_dir):
    """Should raise ValueError on blob size mismatch."""
    import pytest
    db_path = Path(temp_cache_dir) / "test.db"
    storage = EmbeddingStorage(str(db_path))

    # Manually insert corrupted entry
    storage._conn.execute("""
        INSERT INTO embeddings
        (cache_key, model, embedding, dimensions, dtype, created_at, access_count, last_accessed)
        VALUES ('corrupt', 'model', X'0102030405', 768, 'float16', 0, 1, 0)
    """)
    storage._conn.commit()

    with pytest.raises(ValueError, match="Blob size mismatch"):
        storage.get("corrupt")


def test_storage_validation_unknown_dtype(temp_cache_dir):
    """Should raise ValueError on unknown dtype."""
    import pytest
    db_path = Path(temp_cache_dir) / "test.db"
    storage = EmbeddingStorage(str(db_path))

    # Manually insert entry with bad dtype
    storage._conn.execute("""
        INSERT INTO embeddings
        (cache_key, model, embedding, dimensions, dtype, created_at, access_count, last_accessed)
        VALUES ('bad_dtype', 'model', X'00000000', 2, 'float128', 0, 1, 0)
    """)
    storage._conn.commit()

    with pytest.raises(ValueError, match="Unknown dtype"):
        storage.get("bad_dtype")


def test_storage_schema_migration(temp_cache_dir):
    """Opening old database should add new columns."""
    import sqlite3
    db_path = Path(temp_cache_dir) / "test.db"

    # Create old schema manually
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE embeddings (
            cache_key TEXT PRIMARY KEY,
            model TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at INTEGER NOT NULL,
            access_count INTEGER DEFAULT 1,
            last_accessed INTEGER NOT NULL
        )
    """)
    conn.commit()
    conn.close()

    # Open with EmbeddingStorage (should migrate)
    storage = EmbeddingStorage(str(db_path))

    # Verify new columns exist
    cursor = storage._conn.execute("PRAGMA table_info(embeddings)")
    columns = {row[1] for row in cursor.fetchall()}

    assert "dimensions" in columns
    assert "dtype" in columns


def test_storage_legacy_format_read(temp_cache_dir):
    """Should read legacy msgpack format entries."""
    import sqlite3
    import msgpack
    db_path = Path(temp_cache_dir) / "test.db"

    # Create database with schema that includes new columns
    storage = EmbeddingStorage(str(db_path))

    # Insert legacy format entry (msgpack, no dimensions/dtype)
    legacy_embedding = [0.1, 0.2, 0.3]
    legacy_blob = msgpack.packb(legacy_embedding)

    storage._conn.execute("""
        INSERT INTO embeddings
        (cache_key, model, embedding, dimensions, dtype, created_at, access_count, last_accessed)
        VALUES ('legacy', 'model', ?, NULL, NULL, 0, 1, 0)
    """, (legacy_blob,))
    storage._conn.commit()

    # Should read legacy format
    result = storage.get("legacy")

    assert result is not None
    np.testing.assert_array_almost_equal(result, [0.1, 0.2, 0.3], decimal=5)
