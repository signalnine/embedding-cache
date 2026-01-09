# embedding-cache

A Python client library that caches embedding vectors locally with smart fallback to a hosted backend. Eliminates API costs and latency through local caching while maintaining reliability.

## Features

🚀 **Local SQLite cache for zero-latency lookups** - Cache all embeddings in a local database for instant retrieval

🔄 **Smart fallback chain: local model → remote backend** - Automatically falls back to remote if local fails

💰 **Zero API costs using nomic-embed-text-v2** - Run embeddings locally with open-source models

📊 **Cache statistics tracking hits/misses** - Monitor cache performance and efficiency

🎯 **Simple API with advanced options for power users** - Start simple, scale to complex use cases

## Installation

Basic installation:

```bash
pip install embedding-cache
```

With local model support:

```bash
pip install embedding-cache[local]
```

Development installation:

```bash
pip install -e .[dev,local]
```

## Quick Start

### Simple Function API

The simplest way to use embedding-cache is with the `embed()` function:

```python
from embedding_cache import embed

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
from embedding_cache import EmbeddingCache

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
- Windows: `%LOCALAPPDATA%\embedding-cache\`

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

Install with: `pip install embedding-cache[local]`

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
pytest tests/ --cov=embedding_cache --cov-report=html
```

## Development

Clone the repository:

```bash
git clone https://github.com/yourusername/embedding-cache.git
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
pytest --cov=embedding_cache --cov-report=html
open htmlcov/index.html
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

## Roadmap

Future features under consideration:

- [ ] Support for additional embedding models (OpenAI, Cohere, etc.)
- [ ] Batch embedding optimization for large datasets
- [ ] Cache compression to reduce disk usage
- [ ] Async API support for concurrent embedding requests
- [ ] Cache export/import for sharing between systems
