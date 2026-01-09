# Python Embedding Cache Client Library - Design Document

**Date:** 2026-01-09
**Status:** Design Complete
**Target:** MVP

## Overview

A Python client library that caches embedding vectors locally with fallback to a hosted backend. Local caching eliminates API costs and latency while smart fallbacks maintain reliability.

## Core Architecture

### Hybrid Caching Strategy

**Cache lookup flow:**
1. Normalize input text (lowercase + strip whitespace)
2. Generate cache key from hash of (normalized_text, model_name)
3. Check local SQLite cache - return immediately on hit
4. Check remote backend cache - store locally and return on hit
5. Compute embedding via fallback chain on miss
6. Store result in local cache, optionally push to remote

**Fallback chain:** Local model → Hosted backend

- **Local model:** nomic-embed-text-v2 via sentence-transformers (768 dimensions)
- **Backend:** Your hosted service running nomic-embed-text-v2 with shared cache
- **Key benefit:** No per-request API costs, you control the full stack

## API Design

### Simple Function API

```python
from embedding_cache import embed

# Single string - returns list of floats
vector = embed("hello world")

# Multiple strings - returns list of lists
vectors = embed(["hello", "world", "foo"])
```

### Class-Based API

```python
from embedding_cache import EmbeddingCache

cache = EmbeddingCache(
    model="nomic-embed-text-v2",              # default
    cache_dir="~/.cache/embedding-cache",     # default
    remote_url=None,                          # optional backend URL
    fallback_providers=["local", "remote"],   # default chain
)

vector = cache.embed("hello")
vectors = cache.embed(["hello", "world"])

# Access cache stats
print(cache.stats)  # {"hits": 42, "misses": 10, "remote_hits": 5}
```

The simple function wraps a singleton EmbeddingCache with default settings.

## Local Cache Storage

### SQLite Schema

```sql
CREATE TABLE embeddings (
  cache_key TEXT PRIMARY KEY,      -- SHA-256 of (normalized_text, model)
  model TEXT NOT NULL,              -- e.g., "nomic-embed-text-v2"
  embedding BLOB NOT NULL,          -- msgpack-encoded float array
  created_at INTEGER NOT NULL,      -- Unix timestamp
  access_count INTEGER DEFAULT 1,   -- For LRU tracking
  last_accessed INTEGER NOT NULL    -- Unix timestamp
);
```

### Cache Location

- **Default:** `~/.cache/embedding-cache/cache.db`
- **Override:** `cache_dir` parameter or `EMBEDDING_CACHE_DIR` environment variable
- **Versioning:** Cache keys include model name to prevent version conflicts

## Embedding Computation

### Local Model (nomic-embed-text-v2)

- Uses `sentence-transformers` library
- Downloads model on first use (~400MB) to `~/.cache/huggingface/`
- CPU inference supported (slower but works everywhere)
- GPU acceleration if PyTorch CUDA available
- Produces 768-dimensional embeddings

### Remote Backend Fallback

- Hosted service runs GPU-accelerated nomic-embed-text-v2
- Handles cache lookups and compute on cache miss
- MVP requires no API key
- Backend cache creates network effects across users

### Fallback Implementation

```python
def _compute_embedding(self, text: str) -> np.ndarray:
    for provider in self.fallback_providers:
        try:
            if provider == "local":
                return self._embed_local(text)
            elif provider == "remote":
                return self._embed_remote(text)
        except Exception as e:
            logger.warning(f"{provider} failed: {e}, trying next...")
            continue
    raise RuntimeError("All embedding providers failed")
```

## Backend Protocol

### Endpoint: POST /embed

**Request (msgpack-encoded):**
```python
{
    "texts": ["hello world", "foo bar"],  # Always list, even for single
    "model": "nomic-embed-text-v2",       # Optional, defaults to nomic-v2
    "normalized": True                     # Client already normalized
}
```

**Response (msgpack-encoded):**
```python
{
    "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],  # List of float arrays
    "cache_hits": [True, False],           # Which were cached
    "model": "nomic-embed-text-v2",        # Model used
    "dimensions": 768                      # Vector dimensions
}
```

### Why msgpack?

- Reduces 768 floats from ~6KB JSON to ~3KB binary
- Serializes and deserializes faster than JSON
- Uses Content-Type: `application/msgpack`

### Client Behavior

- 5-second timeout (configurable)
- Falls back to local on network error or timeout
- No retries needed - just fall back

### Backend Configuration

- **Default:** `https://embed-cache.example.com`
- **Override:** `remote_url` parameter or `EMBEDDING_CACHE_URL` environment variable
- **Disable:** Set `remote_url=None` for local-only mode

## Package Structure

```
embedding-cache/
├── embedding_cache/
│   ├── __init__.py           # Public API (embed, EmbeddingCache)
│   ├── cache.py              # EmbeddingCache class, cache logic
│   ├── storage.py            # SQLite operations
│   ├── providers.py          # LocalProvider, RemoteProvider
│   ├── normalize.py          # Text normalization
│   └── utils.py              # Hashing, msgpack helpers
├── tests/
│   ├── test_cache.py
│   ├── test_providers.py
│   └── test_integration.py
├── pyproject.toml
├── README.md
└── LICENSE
```

## Dependencies

```toml
[project]
dependencies = [
    "numpy>=1.20",
    "msgpack>=1.0",
    "httpx>=0.24",           # Modern HTTP client
]

[project.optional-dependencies]
local = [
    "sentence-transformers>=2.2",
    "torch>=2.0",
]
```

### Installation Options

- `pip install embedding-cache` - Remote-only, lightweight
- `pip install embedding-cache[local]` - Includes local model support

**Python version:** 3.8+

## Error Handling

### Graceful Degradation

1. **No local model + backend down**
   - Raise error: "Cannot compute embeddings: install local model (pip install embedding-cache[local]) or check backend availability"

2. **Local model OOM**
   - Log warning and fall back to remote
   - Split large batches into smaller chunks recursively

3. **Backend timeout/network error**
   - Fall back to local model (debug logging only)
   - Raise error if no local model available

4. **Model version mismatch**
   - Return error with supported models
   - MVP supports nomic-v2 only

### Input Validation

- Empty string → ValueError
- Null/None → ValueError
- Very long text (>8k tokens) → Error for MVP
- Non-string input → TypeError with clear message

### Cache Corruption

- SQLite file corrupted → Delete and recreate, log warning
- Invalid embedding blob → Skip entry, recompute

## Testing Strategy

### Unit Tests

- `test_normalize.py` - Text normalization edge cases
- `test_storage.py` - SQLite operations, corruption recovery
- `test_cache_keys.py` - Hash generation, model versioning
- `test_providers.py` - Mock local/remote providers

### Integration Tests

- `test_local_provider.py` - Real sentence-transformers
- `test_remote_provider.py` - Mock HTTP server
- `test_fallback_chain.py` - Local → remote fallback
- `test_end_to_end.py` - Full flow with real model

### Mock Strategy

- Mock sentence-transformers by default for fast tests
- Mark real model tests with `pytest.mark.integration`
- Mock httpx for backend tests

### CI Requirements

- Fast tests run on every commit without model downloads
- Integration tests run nightly or on release branches
- Test Python 3.8 through 3.12

## MVP Scope

### In Scope

- ✅ Simple `embed()` function + `EmbeddingCache` class
- ✅ Local SQLite cache with basic normalization
- ✅ Optional local model (nomic-embed-text-v2)
- ✅ Remote backend fallback (REST + msgpack)
- ✅ Smart fallback chain (local → remote)
- ✅ Cache stats tracking
- ✅ Configurable cache directory
- ✅ Single model support (nomic-v2 only)

### Out of Scope

- ❌ Multiple model support
- ❌ Similarity search on cached embeddings
- ❌ Federation/DHT distributed network
- ❌ JavaScript/TypeScript client
- ❌ OpenAI or other providers
- ❌ Authentication or API keys
- ❌ Rate limiting or quotas

## Success Metrics

- Test with semantic-tarot as first user
- Achieve >30% cache hit rate after warm-up
- Confirm local cache responds in <10ms
- Collect API ergonomics feedback

## Post-MVP Priorities

1. Deploy backend service (FastAPI + nomic-embed + Redis)
2. Monitor real-world cache hit rates
3. Add opt-in telemetry for usage patterns
4. Add models based on user demand
