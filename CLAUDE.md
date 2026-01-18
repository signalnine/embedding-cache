# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Vector Embedding Cache** - A Python library for caching string-to-vector embeddings. Reduces API costs and latency by caching pre-computed embeddings locally.

**Current Status**: Implemented and working. Tested with semantic-tarot as the first real-world client.

## Architecture

### Core Flow
1. Hash incoming string (SHA-256) combined with model name
2. Check if embedding exists in JSON cache file
3. Cache hit → return cached vectors (instant, free)
4. Cache miss → compute via embedding model, cache result, return

### Implemented Features
- **Multiple model support**: Local models (nomic-embed-text-v1.5, v2-moe) and OpenAI
- **Model prefix routing**: `"openai:text-embedding-3-small"` routes to OpenAI provider
- **JSON file storage**: Simple, portable cache files
- **Batch operations**: `embed_batch()` for multiple texts
- **Thread-safe**: Safe for concurrent use
- **Zero cost for local models**: No API fees with sentence-transformers

### Tech Stack
- **Language**: Python 3.8+
- **Storage**: JSON file cache (default: `~/.cache/embedding_cache/`)
- **Local Models**: sentence-transformers with nomic-embed-text-v1.5 (recommended)
- **Optional**: OpenAI API for cloud embeddings

## Project Structure

```
embedding_cache/
├── __init__.py          # Public API: EmbeddingCache class
├── cache.py             # Main EmbeddingCache implementation
└── providers.py         # LocalProvider, RemoteProvider, OpenAIProvider

tests/
├── test_cache.py        # Cache functionality tests
└── test_providers.py    # Provider tests including OpenAI mocks
```

## Usage

```python
from embedding_cache import EmbeddingCache

# Local model (recommended - zero cost)
cache = EmbeddingCache(model="nomic-ai/nomic-embed-text-v1.5")
embedding = cache.embed("hello world")  # Returns numpy array

# Batch embeddings
embeddings = cache.embed_batch(["hello", "world", "foo"])

# OpenAI model (requires API key)
cache = EmbeddingCache(model="openai:text-embedding-3-small")
```

## Model Recommendations

| Model | Dimensions | Cost | Use Case |
|-------|-----------|------|----------|
| nomic-ai/nomic-embed-text-v1.5 | 768 | $0 | **Default choice** - stable, precise |
| nomic-ai/nomic-embed-text-v2-moe | 768 | $0 | Alternative - broader associations |
| openai:text-embedding-3-small | 1536 | $0.0001/1K tokens | Highest quality, requires API key |

**v1.5 is recommended** based on testing with semantic-tarot. See `docs/plans/` for comparison details.

## Development

### Running Tests
```bash
pytest tests/ -v
```

### Installation
```bash
# Local models (recommended)
pip install -e ".[local]"

# With OpenAI support
pip install -e ".[openai]"
```

## Key Design Decisions

**Cache Key Format**: JSON-encoded tuple of `[model_name, text]` hashed with SHA-256. This ensures model-specific caching.

**Provider Pattern**: `LocalProvider` (sentence-transformers), `RemoteProvider` (future HTTP API), `OpenAIProvider` (OpenAI API). All share the same interface.

**Lazy Loading**: Models are loaded only when first needed via `_load_model()` / `_load_client()`.

**No Normalization**: Text is cached as-is. Normalization is the caller's responsibility.

## Integration Example: semantic-tarot

The library is tested with [semantic-tarot](../semantic-tarot/), which uses it for:
- 780 card embeddings (78 cards × 2 positions × 5 systems)
- Semantic search queries
- Zero-cost regeneration via cache hits

See `semantic-tarot/EMBEDDING_CACHE_INTEGRATION.md` for integration details.

## Future Extensions

- [ ] Remote cache server (REST API)
- [ ] Similarity search on cached embeddings
- [ ] Pre-seeded common phrases/words
- [ ] Redis cache layer for hot embeddings
- [ ] Client libraries (JS, Go)
