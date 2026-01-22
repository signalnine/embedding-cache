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
