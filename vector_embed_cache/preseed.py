"""Pre-seeded embeddings lookup."""

import sqlite3
from pathlib import Path
from typing import Optional

import msgpack
import numpy as np

# Database filename constant
PRESEED_DB_NAME = "preseed_v1.5.db"


def get_preseed_db_path() -> Path:
    """Get path to bundled preseed database.

    Returns:
        Path to preseed DB location (may or may not exist)
    """
    # Get path relative to this module
    module_dir = Path(__file__).parent
    return module_dir / "data" / PRESEED_DB_NAME


def preseed_db_exists() -> bool:
    """Check if preseed database exists.

    Returns:
        True if preseed DB file exists, False otherwise
    """
    return get_preseed_db_path().exists()


class PreseedStorage:
    """Read-only storage for pre-seeded embeddings."""

    def __init__(self, db_path: Path):
        """Initialize read-only connection to preseed database.

        Args:
            db_path: Path to preseed SQLite database
        """
        # Open in read-only mode
        self._conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self.set = None  # Explicitly disable writes

    def get(self, cache_key: str) -> Optional[np.ndarray]:
        """Retrieve embedding from preseed cache.

        Args:
            cache_key: Cache key to look up

        Returns:
            Embedding array or None if not found
        """
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT embedding FROM embeddings WHERE cache_key = ?",
            (cache_key,),
        )
        row = cursor.fetchone()

        if row is None:
            return None

        # Deserialize embedding
        embedding_bytes = row[0]
        embedding_list = msgpack.unpackb(embedding_bytes)
        return np.array(embedding_list, dtype=np.float32)

    def close(self):
        """Close database connection."""
        self._conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close connection."""
        self.close()
        return False
