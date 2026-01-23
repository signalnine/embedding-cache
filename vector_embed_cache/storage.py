"""SQLite storage layer for embeddings."""

import sqlite3
import threading
import numpy as np
import msgpack
import time
from pathlib import Path
from typing import Optional


class EmbeddingStorage:
    """SQLite-based storage for embedding vectors."""

    def __init__(self, db_path: str):
        """Initialize storage and create schema.

        Args:
            db_path: Path to SQLite database file
        """
        # Ensure parent directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._create_schema()

    def _create_schema(self):
        """Create embeddings table if it doesn't exist, and migrate if needed."""
        # Create table with new schema (includes dimensions and dtype columns)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                cache_key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                embedding BLOB NOT NULL,
                dimensions INTEGER,
                dtype TEXT,
                created_at INTEGER NOT NULL,
                access_count INTEGER DEFAULT 1,
                last_accessed INTEGER NOT NULL
            )
        """)
        self._conn.commit()

        # Migrate existing tables that don't have the new columns
        self._migrate_schema()

    def _migrate_schema(self):
        """Add dimensions and dtype columns to existing tables if missing."""
        cursor = self._conn.execute("PRAGMA table_info(embeddings)")
        columns = {row[1] for row in cursor.fetchall()}

        if "dimensions" not in columns:
            self._conn.execute("ALTER TABLE embeddings ADD COLUMN dimensions INTEGER")

        if "dtype" not in columns:
            self._conn.execute("ALTER TABLE embeddings ADD COLUMN dtype TEXT")

        self._conn.commit()

    def _deserialize_embedding(
        self, blob: bytes, dimensions: Optional[int], dtype: Optional[str], cache_key: str
    ) -> np.ndarray:
        """Deserialize embedding with validation. Always returns float32.

        Args:
            blob: Raw embedding bytes
            dimensions: Expected dimensions (None for legacy format)
            dtype: Storage dtype string (None for legacy format)
            cache_key: Cache key for error messages

        Returns:
            Embedding array as float32

        Raises:
            ValueError: On unknown dtype, size mismatch, or parse failure
        """
        # New format: has dimensions and dtype
        if dimensions is not None and dtype is not None:
            # Map dtype string to numpy dtype (little-endian)
            dtype_map = {"float16": "<f2", "float32": "<f4"}
            if dtype not in dtype_map:
                raise ValueError(f"Unknown dtype '{dtype}' for cache_key={cache_key}")

            np_dtype = np.dtype(dtype_map[dtype])
            expected_size = dimensions * np_dtype.itemsize

            # Validate blob size matches expected dimensions
            if len(blob) != expected_size:
                raise ValueError(
                    f"Blob size mismatch for cache_key={cache_key}: "
                    f"expected {expected_size} bytes ({dimensions} × {np_dtype.itemsize}), "
                    f"got {len(blob)} bytes"
                )

            embedding = np.frombuffer(blob, dtype=np_dtype)
            # Always upcast to float32 for computation
            return embedding.astype(np.float32)

        # Legacy format: msgpack (no fallback - fail explicitly)
        try:
            embedding_list = msgpack.unpackb(blob)
            return np.array(embedding_list, dtype=np.float32)
        except Exception as e:
            raise ValueError(
                f"Failed to deserialize legacy format for cache_key={cache_key}: {e}"
            )

    def get(self, cache_key: str) -> Optional[np.ndarray]:
        """Retrieve embedding from cache.

        Updates access_count and last_accessed on hit.

        Args:
            cache_key: Cache key to look up

        Returns:
            Embedding array (always float32) or None if not found
        """
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT embedding, dimensions, dtype FROM embeddings WHERE cache_key = ?",
                (cache_key,)
            )
            row = cursor.fetchone()

            if row is None:
                return None

            # Update access statistics
            now = int(time.time())
            self._conn.execute("""
                UPDATE embeddings
                SET access_count = access_count + 1, last_accessed = ?
                WHERE cache_key = ?
            """, (now, cache_key))
            self._conn.commit()

            # Deserialize embedding with format detection
            embedding_bytes, dimensions, dtype = row
            return self._deserialize_embedding(embedding_bytes, dimensions, dtype, cache_key)

    def set(self, cache_key: str, model: str, embedding: np.ndarray):
        """Store embedding in cache using float16 binary format.

        Args:
            cache_key: Cache key
            model: Model name
            embedding: Embedding vector (any float dtype, will be converted to float16)
        """
        with self._lock:
            now = int(time.time())

            # Serialize embedding as little-endian float16
            embedding_f16 = embedding.astype("<f2")
            embedding_bytes = embedding_f16.tobytes()
            dimensions = len(embedding)

            self._conn.execute("""
                INSERT INTO embeddings
                (cache_key, model, embedding, dimensions, dtype, created_at, access_count, last_accessed)
                VALUES (?, ?, ?, ?, 'float16', ?, 1, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    model = excluded.model,
                    embedding = excluded.embedding,
                    dimensions = excluded.dimensions,
                    dtype = excluded.dtype,
                    last_accessed = excluded.last_accessed
            """, (cache_key, model, embedding_bytes, dimensions, now, now))
            self._conn.commit()

    def close(self):
        """Close database connection."""
        self._conn.close()
