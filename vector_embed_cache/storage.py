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
        """Create embeddings table if it doesn't exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                cache_key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at INTEGER NOT NULL,
                access_count INTEGER DEFAULT 1,
                last_accessed INTEGER NOT NULL
            )
        """)
        self._conn.commit()

    def get(self, cache_key: str) -> Optional[np.ndarray]:
        """Retrieve embedding from cache.

        Updates access_count and last_accessed on hit.

        Args:
            cache_key: Cache key to look up

        Returns:
            Embedding array or None if not found
        """
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT embedding FROM embeddings WHERE cache_key = ?",
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

            # Deserialize embedding
            embedding_bytes = row[0]
            embedding_list = msgpack.unpackb(embedding_bytes)
            return np.array(embedding_list, dtype=np.float32)

    def set(self, cache_key: str, model: str, embedding: np.ndarray):
        """Store embedding in cache.

        Args:
            cache_key: Cache key
            model: Model name
            embedding: Embedding vector
        """
        with self._lock:
            now = int(time.time())

            # Serialize embedding
            embedding_bytes = msgpack.packb(embedding.tolist())

            self._conn.execute("""
                INSERT INTO embeddings
                (cache_key, model, embedding, created_at, access_count, last_accessed)
                VALUES (?, ?, ?, ?, 1, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    model = excluded.model,
                    embedding = excluded.embedding,
                    last_accessed = excluded.last_accessed
            """, (cache_key, model, embedding_bytes, now, now))
            self._conn.commit()

    def close(self):
        """Close database connection."""
        self._conn.close()
