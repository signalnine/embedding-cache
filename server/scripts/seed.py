#!/usr/bin/env python3
# server/scripts/seed.py
"""Pre-seed the embedding cache with common words."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.database import SessionLocal
from app.cache import store_embedding
from app.compute import compute_batch_sync


# Common English words to pre-seed
COMMON_WORDS = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "hello", "world", "yes", "no", "please", "thank", "sorry", "help", "good", "bad",
]


def seed_common_words(model: str = "nomic-v1.5", batch_size: int = 50):
    """Seed cache with common words."""
    print(f"Seeding {len(COMMON_WORDS)} common words...")

    db = SessionLocal()
    try:
        # Process in batches
        for i in range(0, len(COMMON_WORDS), batch_size):
            batch = COMMON_WORDS[i:i + batch_size]
            print(f"  Computing batch {i // batch_size + 1}...")

            embeddings = compute_batch_sync(batch, model)

            for word, embedding in zip(batch, embeddings):
                store_embedding(
                    db=db,
                    text=word,
                    model=model,
                    vector=embedding,
                    tenant_id="public",  # Shared pool
                    public=True,
                )

            print(f"  Stored {len(batch)} embeddings")

        print(f"Done! Seeded {len(COMMON_WORDS)} words to public cache.")

    finally:
        db.close()


if __name__ == "__main__":
    seed_common_words()
