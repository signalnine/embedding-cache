# vector-embed-cache

[![CI](https://github.com/signalnine/embedding-cache/actions/workflows/ci.yml/badge.svg)](https://github.com/signalnine/embedding-cache/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/vector-embed-cache.svg)](https://badge.fury.io/py/vector-embed-cache)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python library and self-hosted backend for caching embedding vectors. Eliminates API costs and latency through local caching while maintaining reliability.

## Features

🚀 **Local SQLite cache for zero-latency lookups** - Cache all embeddings in a local database for instant retrieval

🔄 **Smart fallback chain: local model → remote backend** - Automatically falls back to remote if local fails

💰 **Zero API costs using nomic-embed-text-v1.5** - Run embeddings locally with open-source models

📊 **Cache statistics tracking hits/misses** - Monitor cache performance and efficiency

🎯 **Simple API with advanced options for power users** - Start simple, scale to complex use cases

🖥️ **Self-hosted backend available** - FastAPI server with multi-tenant caching, rate limiting, and BYOK support

## Why vector-embed-cache?

### vs RedisVL / Remote Caches
- **No Redis setup required** - Just SQLite, no external dependencies
- **Works offline** - Local model runs without internet
- **Zero network latency** - Cache lookups are disk reads, not network calls
- **Computes embeddings** - Built-in model support, not just caching

### vs GPTCache / Semantic Caches
- **Exact matching** - Deterministic cache keys, no similarity search needed
- **Simpler setup** - Single SQLite file vs vector stores + embedding + eviction
- **Text normalization** - "Hello" and "hello" share the same cache entry
- **Embedding-focused** - Designed for vectors, not LLM response caching

### vs Generic Caches (diskcache, cachew)
- **Embedding-aware** - Automatic normalization and provider fallback
- **Stats tracking** - Built-in monitoring for cache hits/misses
- **Smart fallback** - Local model → remote backend → error
- **Zero config** - Works out of the box with sensible defaults

### When to use vector-embed-cache
✅ Prototyping with embeddings locally
✅ Reducing API costs for embedding computation
✅ Offline or air-gapped environments
✅ Projects that need fallback to remote compute
✅ Simple single-file cache without infrastructure

### When to use alternatives
- Need semantic similarity matching → **GPTCache**
- Already have Redis infrastructure → **RedisVL**
- Caching non-embedding data → **diskcache**
- Need distributed caching → **Redis/Memcached**

## Installation

Basic installation:

```bash
pip install vector-embed-cache
```

With local model support:

```bash
pip install vector-embed-cache[local]
```

With OpenAI support:

```bash
pip install vector-embed-cache[openai]
```

With both local and OpenAI support:

```bash
pip install vector-embed-cache[local,openai]
```

Development installation:

```bash
pip install -e .[dev,local]
```

## Usage with OpenAI

Set your API key:

```bash
export OPENAI_API_KEY=your-key-here
```

Use OpenAI embeddings:

```python
from vector_embed_cache import EmbeddingCache

# Create cache with OpenAI model (note the "openai:" prefix)
cache = EmbeddingCache(model="openai:text-embedding-3-small")

# Use it like any other provider
vector = cache.embed("hello world")
print(vector.shape)  # (1536,)

# Subsequent calls hit the cache
vector2 = cache.embed("hello world")
print(cache.stats)
# {'hits': 1, 'misses': 1, 'remote_hits': 0}
```

The OpenAI provider:
- Uses the OpenAI API with automatic retries
- Caches embeddings locally just like other providers
- Requires `OPENAI_API_KEY` environment variable
- Supports batch embedding (up to 2048 texts per request)
- Returns 1536-dimensional embeddings for text-embedding-3-small

## Model Comparison

embedding-cache supports multiple embedding models. Here's how they compare:

| Model | Dimensions | Provider | Cost | Speed | Use Case |
|-------|-----------|----------|------|-------|----------|
| nomic-ai/nomic-embed-text-v1.5 | 768 | Local | $0 | Fast (cached) | General purpose, offline |
| nomic-ai/nomic-embed-text-v2-moe | 768 | Local | $0 | Medium (MoE) | Higher quality, offline |
| openai:text-embedding-3-small | 1536 | API | $0.0001/1K tokens | Fast (API) | Highest quality, online |

### Switching Models

```python
from vector_embed_cache import EmbeddingCache

# Use v1.5 (default, fast and reliable)
cache_v15 = EmbeddingCache(model="nomic-ai/nomic-embed-text-v1.5")

# Use v2-moe (newer, potentially higher quality)
cache_v2 = EmbeddingCache(model="nomic-ai/nomic-embed-text-v2-moe")

# Use OpenAI (highest quality, requires API key)
cache_openai = EmbeddingCache(model="openai:text-embedding-3-small")
```

All models benefit from the same caching layer, so repeated queries are instant regardless of which model you choose.

## Quick Start

### Simple Function API

The simplest way to use embedding-cache is with the `embed()` function:

```python
from vector_embed_cache import embed

# Single string
vector = embed("hello world")
print(vector.shape)  # (768,)

# Multiple strings
vectors = embed(["hello", "world"])
print(len(vectors))  # 2
print(vectors[0].shape)  # (768,)
```

### Advanced Class-Based API

For more control over configuration, use the `EmbeddingCache` class:

```python
from vector_embed_cache import EmbeddingCache

# Create cache with custom settings
cache = EmbeddingCache(
    cache_dir="~/.my-cache",
    model="nomic-ai/nomic-embed-text-v1.5",
    remote_url="https://api.example.com/embed",
    fallback_providers=["local", "remote"],
    timeout=10.0
)

# Use the cache
vector = cache.embed("hello world")

# Access statistics
print(cache.stats)
# {'hits': 0, 'misses': 1, 'remote_hits': 0}

# Subsequent calls will hit the cache
vector2 = cache.embed("hello world")
print(cache.stats)
# {'hits': 1, 'misses': 1, 'remote_hits': 0}
```

## Configuration

### Environment Variables

- **EMBEDDING_CACHE_DIR**: Set custom cache directory location
  ```bash
  export EMBEDDING_CACHE_DIR=/path/to/cache
  ```

- **EMBEDDING_CACHE_URL**: Set default remote backend URL
  ```bash
  export EMBEDDING_CACHE_URL=https://api.example.com/embed
  ```

### Cache Location

By default, embeddings are cached in:
- Linux/macOS: `~/.cache/embedding-cache/`
- Windows: `C:\Users\<username>\.cache\embedding-cache\`

Override with `EMBEDDING_CACHE_DIR` environment variable or the `cache_dir` parameter.

### Override Methods

You can override cache behavior using the `EmbeddingCache` constructor:

```python
cache = EmbeddingCache(
    cache_dir="/custom/path",           # Custom cache location
    model="nomic-ai/nomic-embed-text-v1.5",  # Custom model
    remote_url="https://api.example.com",    # Remote backend
    fallback_providers=["local", "remote"],  # Provider order
    timeout=5.0                          # Remote timeout (seconds)
)
```

## Architecture

### Caching Flow

1. **Input Normalization**: Text is normalized (whitespace, lowercasing) to maximize cache hits
2. **Cache Key Generation**: SHA-256 hash generated from (normalized_text + model_name)
3. **Cache Lookup**: Check SQLite database for existing embedding
4. **Cache Hit**: Return cached embedding immediately (zero latency)
5. **Cache Miss**: Generate embedding using fallback chain
6. **Cache Store**: Save new embedding to SQLite for future lookups

### Models

The default model is **nomic-embed-text-v1.5** from nomic-ai:

- **Dimensions**: 768
- **Max Tokens**: 8192
- **Performance**: MTEB score of 62.39
- **License**: Open-source (Apache 2.0)
- **Requirements**: sentence-transformers, torch, einops

Install with: `pip install vector-embed-cache[local]`

## Testing

Run unit tests:

```bash
pytest tests/ -m "not integration"
```

Run integration tests:

```bash
pytest tests/ -m integration
```

Run all tests:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest tests/ --cov=vector_embed_cache --cov-report=html
```

## Development

Clone the repository:

```bash
git clone https://github.com/signalnine/embedding-cache.git
cd embedding-cache
```

Install in development mode:

```bash
pip install -e .[dev,local]
```

Run tests:

```bash
pytest
```

Run tests with coverage:

```bash
pytest --cov=vector_embed_cache --cov-report=html
open htmlcov/index.html  # macOS
# or: xdg-open htmlcov/index.html  # Linux
# or: start htmlcov/index.html     # Windows
```

## Error Handling

The library provides clear error messages for common issues:

- **ValueError**: Empty string or None input
  ```python
  embed("")  # ValueError: Text cannot be empty string
  embed(None)  # ValueError: Text cannot be None
  ```

- **TypeError**: Invalid input type
  ```python
  embed(123)  # TypeError: Text must be string or list of strings, got int
  embed(["hello", 123])  # TypeError: All items in list must be strings, got int
  ```

- **RuntimeError**: All providers fail
  ```python
  # When no providers are available:
  # RuntimeError: All embedding providers failed. Install sentence-transformers...
  ```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Here's how to contribute:

1. **Fork the repository** on GitHub
2. **Create a feature branch**: `git checkout -b feature/my-new-feature`
3. **Make your changes** and add tests
4. **Run the test suite**: `pytest tests/`
5. **Submit a pull request** with a clear description of your changes

Please ensure your code:
- Passes all existing tests
- Includes tests for new functionality
- Follows the existing code style
- Includes docstrings for new functions/classes

## Self-Hosted Backend

For teams needing centralized caching with multi-tenant support, see the [server documentation](server/README.md).

**Features:**
- Multi-tenant embedding cache with PostgreSQL
- Similarity search via pgvector (HNSW indexes)
- Hybrid compute: BYOK (free tier) or server GPU (paid tier)
- JWT authentication with API keys
- Rate limiting with Redis
- Prometheus metrics endpoint

**Quick start:**
```bash
cd server
pip install -r requirements.txt
export DATABASE_URL="postgresql://user:pass@localhost/embeddings"
export JWT_SECRET="your-secret"
export ENCRYPTION_KEY="your-key"
uvicorn app.main:app --port 8000
```

## Roadmap

### Completed
- [x] Local SQLite caching with CLI management
- [x] Multiple model support (nomic v1.5, v2-moe, OpenAI)
- [x] Self-hosted backend with FastAPI
- [x] BYOK provider support for free tier
- [x] JWT authentication and API keys
- [x] Rate limiting with Redis
- [x] Similarity search on cached embeddings (pgvector)

### Future
- [ ] Pre-seeded common phrases/words
- [ ] Client libraries (JavaScript, Go)
- [ ] Admin dashboard
- [ ] Cache compression to reduce storage
