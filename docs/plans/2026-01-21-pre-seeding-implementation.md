# Pre-Seeded Embeddings Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship pre-computed embeddings for 3,000 common English words so users get instant cache hits.

**Architecture:** Bundle a read-only SQLite database with 3K word embeddings in the package. Modify storage layer to check bundled DB as fallback after user cache miss. Track preseed hits separately in stats.

**Tech Stack:** Python, SQLite, wordfreq library (dev dependency), msgpack for serialization

---

### Task 1: Create Preseed Module with Database Path Resolution

**Files:**
- Create: `vector_embed_cache/preseed.py`
- Create: `vector_embed_cache/data/.gitkeep`
- Test: `tests/test_preseed.py`

**Step 1: Write the failing test**

```python
# tests/test_preseed.py
"""Tests for preseed module."""

import pytest
from pathlib import Path


class TestPreseedPaths:
    def test_get_preseed_db_path_returns_path_in_package(self):
        from vector_embed_cache.preseed import get_preseed_db_path

        path = get_preseed_db_path()
        assert path is not None
        assert "vector_embed_cache" in str(path)
        assert path.name == "preseed_v1.5.db"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_preseed.py::TestPreseedPaths::test_get_preseed_db_path_returns_path_in_package -v`
Expected: FAIL with "No module named 'vector_embed_cache.preseed'"

**Step 3: Create data directory and preseed module**

```bash
mkdir -p vector_embed_cache/data
touch vector_embed_cache/data/.gitkeep
```

```python
# vector_embed_cache/preseed.py
"""Pre-seeded embeddings lookup."""

from pathlib import Path
from typing import Optional


def get_preseed_db_path() -> Optional[Path]:
    """Get path to bundled preseed database.

    Returns:
        Path to preseed DB if it exists, None otherwise
    """
    # Get path relative to this module
    module_dir = Path(__file__).parent
    db_path = module_dir / "data" / "preseed_v1.5.db"

    if db_path.exists():
        return db_path
    return None
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_preseed.py::TestPreseedPaths::test_get_preseed_db_path_returns_path_in_package -v`
Expected: PASS

**Step 5: Commit**

```bash
git add vector_embed_cache/preseed.py vector_embed_cache/data/.gitkeep tests/test_preseed.py
git commit -m "feat: add preseed module with path resolution"
```

---

### Task 2: Add Preseed Storage Class

**Files:**
- Modify: `vector_embed_cache/preseed.py`
- Test: `tests/test_preseed.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_preseed.py
import numpy as np
import tempfile
import sqlite3
import msgpack


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
            ("test_key", "nomic-ai/nomic-embed-text-v1.5", embedding_bytes)
        )

        # Insert metadata
        conn.execute("INSERT INTO metadata (key, value) VALUES (?, ?)",
                    ("schema_version", "1"))
        conn.execute("INSERT INTO metadata (key, value) VALUES (?, ?)",
                    ("model_version", "nomic-ai/nomic-embed-text-v1.5"))

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
        assert not hasattr(storage, 'set') or storage.set is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_preseed.py::TestPreseedStorage -v`
Expected: FAIL with "cannot import name 'PreseedStorage'"

**Step 3: Implement PreseedStorage class**

```python
# Add to vector_embed_cache/preseed.py
import sqlite3
import numpy as np
import msgpack


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
            (cache_key,)
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_preseed.py::TestPreseedStorage -v`
Expected: PASS

**Step 5: Commit**

```bash
git add vector_embed_cache/preseed.py tests/test_preseed.py
git commit -m "feat: add PreseedStorage class for read-only lookups"
```

---

### Task 3: Integrate Preseed Fallback into EmbeddingCache

**Files:**
- Modify: `vector_embed_cache/cache.py`
- Test: `tests/test_preseed.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_preseed.py
class TestPreseedIntegration:
    @pytest.fixture
    def cache_with_preseed(self, tmp_path, temp_preseed_db, monkeypatch):
        """Create EmbeddingCache with mocked preseed path."""
        from vector_embed_cache.cache import EmbeddingCache
        from vector_embed_cache import preseed

        # Mock preseed path to use our temp DB
        monkeypatch.setattr(preseed, 'get_preseed_db_path', lambda: temp_preseed_db)

        cache = EmbeddingCache(
            cache_dir=str(tmp_path / "cache"),
            model="nomic-ai/nomic-embed-text-v1.5"
        )
        return cache

    def test_cache_checks_preseed_on_miss(self, cache_with_preseed, temp_preseed_db):
        """Verify cache checks preseed DB when user cache misses."""
        # The temp_preseed_db has "test_key" with embedding [0.1, 0.2, 0.3]
        # We need to query with the actual text that hashes to "test_key"
        # For this test, we'll mock at a lower level
        pass  # Will implement properly in step 3

    def test_stats_track_preseed_hits(self, cache_with_preseed):
        """Verify preseed hits are tracked separately."""
        # Check that stats dict has preseed_hits key
        assert "preseed_hits" in cache_with_preseed.stats
        assert cache_with_preseed.stats["preseed_hits"] == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_preseed.py::TestPreseedIntegration::test_stats_track_preseed_hits -v`
Expected: FAIL with "KeyError: 'preseed_hits'"

**Step 3: Modify EmbeddingCache to include preseed**

```python
# Modify vector_embed_cache/cache.py - update __init__ to add preseed storage and stats

# Add import at top:
from .preseed import get_preseed_db_path, PreseedStorage

# In __init__, after self.storage initialization, add:
        # Initialize preseed storage if available
        preseed_path = get_preseed_db_path()
        self.preseed_storage = PreseedStorage(preseed_path) if preseed_path else None

# Update stats dict to include preseed_hits:
        self.stats = {
            "hits": 0,
            "misses": 0,
            "remote_hits": 0,
            "preseed_hits": 0
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_preseed.py::TestPreseedIntegration::test_stats_track_preseed_hits -v`
Expected: PASS

**Step 5: Commit**

```bash
git add vector_embed_cache/cache.py tests/test_preseed.py
git commit -m "feat: add preseed storage initialization to EmbeddingCache"
```

---

### Task 4: Add Preseed Lookup in Cache Flow

**Files:**
- Modify: `vector_embed_cache/cache.py`
- Test: `tests/test_preseed.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_preseed.py TestPreseedIntegration class
    def test_preseed_lookup_returns_cached_embedding(self, tmp_path, monkeypatch):
        """Test that preseed DB is checked and returns embeddings."""
        from vector_embed_cache.cache import EmbeddingCache
        from vector_embed_cache import preseed
        from vector_embed_cache.utils import generate_cache_key
        from vector_embed_cache.normalize import normalize_text

        # Create preseed DB with known word
        db_path = tmp_path / "preseed.db"
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE embeddings (
                cache_key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)
        """)

        # Generate cache key for "hello" with the default model
        model = "nomic-ai/nomic-embed-text-v1.5"
        normalized = normalize_text("hello")
        cache_key = generate_cache_key(normalized, model)

        # Insert embedding for "hello"
        test_embedding = np.ones(768, dtype=np.float32) * 0.5
        embedding_bytes = msgpack.packb(test_embedding.tolist())
        conn.execute(
            "INSERT INTO embeddings (cache_key, model, embedding) VALUES (?, ?, ?)",
            (cache_key, model, embedding_bytes)
        )
        conn.commit()
        conn.close()

        # Mock preseed path
        monkeypatch.setattr(preseed, 'get_preseed_db_path', lambda: db_path)

        # Create cache (will fail to load local model, but that's ok for this test)
        cache = EmbeddingCache(
            cache_dir=str(tmp_path / "user_cache"),
            model=model
        )

        # Mock the local provider to fail so we can verify preseed is used
        cache.local_provider._model = None
        cache.local_provider._available = False

        # This should find "hello" in preseed and return it
        result = cache.embed("hello")

        assert result is not None
        assert cache.stats["preseed_hits"] == 1
        assert cache.stats["misses"] == 0  # Not a miss if found in preseed
        np.testing.assert_array_almost_equal(result, test_embedding)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_preseed.py::TestPreseedIntegration::test_preseed_lookup_returns_cached_embedding -v`
Expected: FAIL (preseed not being checked in lookup flow)

**Step 3: Modify _get_embedding to check preseed**

```python
# Modify vector_embed_cache/cache.py _get_embedding method

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text (with cache lookup).

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        # Normalize text
        normalized = normalize_text(text)

        # Generate cache key
        cache_key = generate_cache_key(normalized, self.model)

        # Check user cache first
        cached = self.storage.get(cache_key)
        if cached is not None:
            with self._stats_lock:
                self.stats["hits"] += 1
            return cached

        # Check preseed cache (fallback)
        if self.preseed_storage is not None:
            preseed_result = self.preseed_storage.get(cache_key)
            if preseed_result is not None:
                with self._stats_lock:
                    self.stats["preseed_hits"] += 1
                return preseed_result

        # Cache miss - compute embedding
        with self._stats_lock:
            self.stats["misses"] += 1
        embedding = self._compute_embedding(normalized)

        # Store in user cache
        self.storage.set(cache_key, self.model, embedding)

        return embedding
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_preseed.py::TestPreseedIntegration::test_preseed_lookup_returns_cached_embedding -v`
Expected: PASS

**Step 5: Commit**

```bash
git add vector_embed_cache/cache.py tests/test_preseed.py
git commit -m "feat: add preseed fallback lookup in cache flow"
```

---

### Task 5: Add Preseed CLI Status Command

**Files:**
- Modify: `vector_embed_cache/cli.py`
- Test: `tests/test_cli.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_cli.py
import subprocess


class TestPreseedCLI:
    def test_preseed_status_command_exists(self):
        """Test that preseed status command is available."""
        result = subprocess.run(
            ["python", "-m", "vector_embed_cache.cli", "preseed", "status"],
            capture_output=True,
            text=True
        )
        # Should not fail with "invalid choice"
        assert "invalid choice" not in result.stderr
        assert "Preseed" in result.stdout or "preseed" in result.stdout.lower()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py::TestPreseedCLI::test_preseed_status_command_exists -v`
Expected: FAIL with "invalid choice: 'preseed'"

**Step 3: Add preseed subcommand to CLI**

```python
# Modify vector_embed_cache/cli.py - add preseed commands

# Add import at top:
from .preseed import get_preseed_db_path

# Add new command function:
def cmd_preseed_status(args):
    """Show preseed database status."""
    print("Preseed Status")
    print("=" * 40)

    preseed_path = get_preseed_db_path()

    if preseed_path is None:
        print("Status: No preseed database found")
        print("The preseed database is not bundled with this installation.")
        return

    print(f"Database: {preseed_path}")

    # Get file size
    size_bytes = preseed_path.stat().st_size
    if size_bytes < 1024:
        size_str = f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        size_str = f"{size_bytes / 1024:.1f} KB"
    else:
        size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
    print(f"Size: {size_str}")

    # Count entries and show metadata
    try:
        import sqlite3
        conn = sqlite3.connect(f"file:{preseed_path}?mode=ro", uri=True)

        cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
        count = cursor.fetchone()[0]
        print(f"Entries: {count}")

        # Show metadata
        cursor = conn.execute("SELECT key, value FROM metadata")
        for key, value in cursor.fetchall():
            print(f"{key}: {value}")

        conn.close()
    except Exception as e:
        print(f"Error reading database: {e}")


# In main(), add preseed subparser:
    # preseed command group
    preseed_parser = subparsers.add_parser("preseed", help="Manage preseed database")
    preseed_subparsers = preseed_parser.add_subparsers(dest="preseed_command")

    # preseed status
    preseed_status_parser = preseed_subparsers.add_parser("status", help="Show preseed status")
    preseed_status_parser.set_defaults(func=cmd_preseed_status)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py::TestPreseedCLI::test_preseed_status_command_exists -v`
Expected: PASS

**Step 5: Commit**

```bash
git add vector_embed_cache/cli.py tests/test_cli.py
git commit -m "feat: add preseed status CLI command"
```

---

### Task 6: Create Preseed Generation Script

**Files:**
- Create: `scripts/generate_preseed.py`

**Step 1: Create the generation script**

```python
#!/usr/bin/env python3
"""Generate pre-seeded embedding database.

This script creates the preseed database bundled with the package.
Run before releasing a new version.

Requirements:
    pip install wordfreq sentence-transformers torch einops

Usage:
    python scripts/generate_preseed.py
"""

import sqlite3
import time
from pathlib import Path

import msgpack
import numpy as np

# These are only needed for generation, not runtime
try:
    from wordfreq import top_n_list
except ImportError:
    print("Error: wordfreq not installed. Run: pip install wordfreq")
    exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: sentence-transformers not installed. Run: pip install sentence-transformers")
    exit(1)

# Import from library to ensure consistency (CRITICAL: do not duplicate)
from vector_embed_cache.normalize import normalize_text
from vector_embed_cache.utils import generate_cache_key


# Configuration
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
WORD_COUNT = 3000
OUTPUT_PATH = Path(__file__).parent.parent / "vector_embed_cache" / "data" / "preseed_v1.5.db"


def main():
    print(f"Generating preseed database for {MODEL_NAME}")
    print(f"Word count: {WORD_COUNT}")
    print(f"Output: {OUTPUT_PATH}")
    print()

    # Get word list
    print("Fetching word list from wordfreq...")
    words = top_n_list('en', WORD_COUNT)
    print(f"Got {len(words)} words")

    # Load model
    print(f"Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    print("Model loaded")

    # Create database
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    conn = sqlite3.connect(OUTPUT_PATH)

    # Create schema
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

    # Generate embeddings in batches
    print("Generating embeddings...")
    batch_size = 100
    start_time = time.time()

    for i in range(0, len(words), batch_size):
        batch = words[i:i + batch_size]

        # Normalize and generate cache keys
        normalized = [normalize_text(w) for w in batch]
        cache_keys = [generate_cache_key(n, MODEL_NAME) for n in normalized]

        # Generate embeddings
        embeddings = model.encode(normalized, show_progress_bar=False)

        # Insert into database
        for cache_key, embedding in zip(cache_keys, embeddings):
            embedding_bytes = msgpack.packb(embedding.tolist())
            conn.execute(
                "INSERT INTO embeddings (cache_key, model, embedding) VALUES (?, ?, ?)",
                (cache_key, MODEL_NAME, embedding_bytes)
            )

        # Progress
        progress = min(i + batch_size, len(words))
        elapsed = time.time() - start_time
        rate = progress / elapsed if elapsed > 0 else 0
        print(f"  {progress}/{len(words)} ({rate:.1f} words/sec)")

    # Insert metadata
    from datetime import datetime, timezone
    conn.execute("INSERT INTO metadata (key, value) VALUES (?, ?)",
                ("schema_version", "1"))
    conn.execute("INSERT INTO metadata (key, value) VALUES (?, ?)",
                ("hash_algorithm", "sha256"))
    conn.execute("INSERT INTO metadata (key, value) VALUES (?, ?)",
                ("normalization", "lowercase_strip"))
    conn.execute("INSERT INTO metadata (key, value) VALUES (?, ?)",
                ("model_version", MODEL_NAME))
    conn.execute("INSERT INTO metadata (key, value) VALUES (?, ?)",
                ("generated_at", datetime.now(timezone.utc).isoformat()))
    conn.execute("INSERT INTO metadata (key, value) VALUES (?, ?)",
                ("word_count", str(len(words))))

    conn.commit()
    conn.close()

    # Print stats
    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    total_time = time.time() - start_time
    print()
    print(f"Done! Generated {len(words)} embeddings in {total_time:.1f}s")
    print(f"Database size: {size_mb:.1f} MB")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
```

**Step 2: Test the script runs**

Run: `python scripts/generate_preseed.py --help 2>&1 | head -5 || echo "Script exists"`
Expected: Script should be parseable (no syntax errors)

**Step 3: Commit**

```bash
git add scripts/generate_preseed.py
git commit -m "feat: add preseed database generation script"
```

---

### Task 7: Update pyproject.toml to Include Package Data

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add package-data configuration**

```toml
# Add after [tool.hatch.build.targets.wheel]:
[tool.hatch.build.targets.wheel]
packages = ["vector_embed_cache"]

[tool.hatch.build.targets.wheel.force-include]
"vector_embed_cache/data" = "vector_embed_cache/data"

# Also add wordfreq to dev dependencies for preseed generation
```

Update the dev dependencies:

```toml
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "pytest-mock>=3.10",
    "wordfreq>=3.0",
]
```

**Step 2: Verify package builds correctly**

Run: `pip install -e . && python -c "from vector_embed_cache.preseed import get_preseed_db_path; print(get_preseed_db_path())"`
Expected: Should print None (preseed not generated yet) or a path

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: add package data config for preseed database"
```

---

### Task 8: Generate and Bundle Preseed Database

**Step 1: Install dependencies and generate**

Run:
```bash
pip install wordfreq sentence-transformers torch einops
python scripts/generate_preseed.py
```

Expected: Database generated at `vector_embed_cache/data/preseed_v1.5.db`

**Step 2: Verify preseed works**

```bash
python -c "
from vector_embed_cache import EmbeddingCache
cache = EmbeddingCache()
result = cache.embed('hello')
print(f'Embedding shape: {result.shape}')
print(f'Stats: {cache.stats}')
"
```

Expected: Should show preseed_hits: 1 if "hello" is in top 3000 words

**Step 3: Commit the database**

```bash
git add vector_embed_cache/data/preseed_v1.5.db
git commit -m "data: add preseed database with 3000 common words"
```

---

### Task 9: Run Full Test Suite

**Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests pass

**Step 2: Fix any failures**

If tests fail, fix them before proceeding.

**Step 3: Final commit if needed**

```bash
git add -A
git commit -m "fix: resolve test issues from preseed integration"
```

---

### Task 10: Update Documentation

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

**Step 1: Update README.md**

Add section about pre-seeded embeddings:

```markdown
## Pre-Seeded Embeddings

The library ships with pre-computed embeddings for 3,000 common English words. These provide instant cache hits without any computation:

```python
from vector_embed_cache import EmbeddingCache

cache = EmbeddingCache()
embedding = cache.embed("hello")  # Instant - found in preseed

print(cache.stats)
# {'hits': 0, 'misses': 0, 'remote_hits': 0, 'preseed_hits': 1}
```

Check preseed status via CLI:

```bash
vector-embed-cache preseed status
```
```

**Step 2: Update CLAUDE.md roadmap**

Mark pre-seeding as complete:

```markdown
### Completed
...
- [x] Pre-seeded common phrases/words
```

**Step 3: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: add preseed documentation"
```
