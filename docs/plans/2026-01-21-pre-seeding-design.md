# Pre-Seeded Embeddings Design

## Overview

Ship vector-embed-cache with pre-computed embeddings for common English words. Users get instant cache hits on first use without computing anything.

**Goal**: Zero-latency lookups for the most frequently used words, offline-first experience.

**Use case**: Single-word embeddings for vocabulary matching, word similarity, and as building blocks. Phrase/sentence pre-seeding is a future enhancement.

## Architecture

### Core Bundle (Included in PyPI Package)

- **3,000 most common English words** from `wordfreq` library
- **Model**: nomic-ai/nomic-embed-text-v1.5 only (the default)
- **Format**: SQLite database bundled in package
- **Size**: ~9MB (3,000 words × 768 dims × 4 bytes + overhead)

### Optional Layers (Downloaded on Demand)

Available via GitHub releases, downloaded through CLI:
- `phrases` - Common phrases and expressions
- `programming` - Programming terms and keywords
- `scientific` - Scientific vocabulary
- Other models (v2-moe, OpenAI) when needed

### Cache Lookup Order

```
get_embedding(text):
  1. Check user cache (~/.cache/embedding-cache/)
  2. On miss, check bundled DB (read-only, in site-packages)
  3. On miss, compute and store in user cache
```

User cache always takes precedence. Bundled data is fallback-only.

## Technical Details

### Word List Generation

```python
from wordfreq import top_n_list

# Pin wordfreq version in pyproject.toml
words = top_n_list('en', 3000)
```

Include all words including stopwords. They're frequently queried and storage is negligible.

### Package Structure

```
vector_embed_cache/
├── data/
│   └── preseed_v1.5.db      # Bundled SQLite (~9MB)
├── preseed.py               # Preseed lookup logic
├── storage.py               # Updated to check preseed fallback
└── ...
```

### Database Schema

Extended from user cache with version metadata:

```sql
CREATE TABLE embeddings (
    text_hash TEXT NOT NULL,
    model TEXT NOT NULL,
    embedding BLOB NOT NULL,
    PRIMARY KEY (text_hash, model)
);

CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Metadata entries:
-- schema_version: "1"
-- hash_algorithm: "sha256"
-- normalization: "lowercase_strip_whitespace"
-- model_version: "nomic-ai/nomic-embed-text-v1.5"
-- generated_at: "2026-01-21T00:00:00Z"
-- word_count: "3000"
```

### Normalization Spec (Frozen)

All text normalized before hashing:
1. Strip leading/trailing whitespace
2. Collapse internal whitespace to single space
3. Lowercase

This matches the existing `normalize.py` behavior and is frozen for preseed compatibility.

### Storage Integration

Modify `SQLiteStorage` to check bundled DB on miss:

```python
class SQLiteStorage:
    def __init__(self, cache_dir, preseed_db_path=None):
        self.user_db = self._connect(cache_dir / "embeddings.db")
        self.preseed_db = self._connect_readonly(preseed_db_path) if preseed_db_path else None

    def get(self, text_hash: str, model: str) -> Optional[np.ndarray]:
        # Check user cache first
        result = self._query(self.user_db, text_hash, model)
        if result:
            return result

        # Fallback to preseed
        if self.preseed_db:
            return self._query(self.preseed_db, text_hash, model)

        return None
```

### CLI Commands

Extend existing CLI:

```bash
# Show preseed status
embedding-cache preseed status

# Download optional layer
embedding-cache preseed download phrases

# List available layers
embedding-cache preseed list
```

### Build Process

New script `scripts/generate_preseed.py`:

```python
"""Generate pre-seeded embedding database."""
from wordfreq import top_n_list
from vector_embed_cache import EmbeddingCache

def generate():
    words = top_n_list('en', 3000)
    cache = EmbeddingCache(model="nomic-ai/nomic-embed-text-v1.5")

    # Embed all words
    cache.embed_batch(words)

    # Copy to package data location
    shutil.copy(cache.db_path, "vector_embed_cache/data/preseed_v1.5.db")
```

Run before package release. Add `wordfreq` to dev dependencies only.

### pyproject.toml Changes

```toml
[tool.setuptools.package-data]
vector_embed_cache = ["data/*.db"]
```

## Versioning

- Bundle version tied to package version
- Model version embedded in filename (`preseed_v1.5.db`)
- If model updates, generate new preseed and bump package version

## Testing

1. **Preseed lookup**: Query bundled word, verify instant return
2. **Fallback order**: User cache entry overrides preseed
3. **Missing word**: Word not in preseed triggers normal computation
4. **Stats tracking**: Preseed hits counted separately from user cache hits

## Migration

No migration needed. Feature is additive:
- Existing caches continue working
- Bundled DB checked only as fallback
- No user action required

## Future Layers (Post-MVP)

Downloadable layers stored in user cache directory:

```
~/.cache/embedding-cache/
├── embeddings.db          # User cache
├── layers/
│   ├── phrases.db         # Downloaded layer
│   └── programming.db     # Downloaded layer
```

Layers checked after preseed, before computation.

### Download Integrity (Future Layers)

GitHub releases include SHA256 checksums:

```
layers/
├── phrases.db
├── phrases.db.sha256
├── programming.db
└── programming.db.sha256
```

CLI verifies checksum before accepting download:

```python
def download_layer(name: str):
    db_url = f"{GITHUB_RELEASES_URL}/{name}.db"
    checksum_url = f"{GITHUB_RELEASES_URL}/{name}.db.sha256"

    # Download and verify
    expected_hash = fetch(checksum_url).strip()
    db_content = fetch(db_url)
    actual_hash = hashlib.sha256(db_content).hexdigest()

    if actual_hash != expected_hash:
        raise IntegrityError(f"Checksum mismatch for {name}")

    # Save to layers directory
    save(db_content, layers_dir / f"{name}.db")
```
