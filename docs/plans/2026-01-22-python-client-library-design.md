# Python API Client Library Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Python client library for the vector-embed-cache API with both sync and async support.

**Architecture:** Separate `Client` and `AsyncClient` classes using httpx. Typed exception hierarchy for errors. Pythonic API with context managers.

**Tech Stack:** Python 3.9+, httpx, pytest

---

## Design Decisions (Consensus-Driven)

| Decision | Choice | Reasoning |
|----------|--------|-----------|
| Package structure | Separate package `clients/python/` | Clean separation, mirrors JS client structure |
| HTTP library | httpx | Dual sync/async support, requests-like API |
| Sync/async API | Separate `Client`/`AsyncClient` | Python idioms, matches httpx pattern |
| Error handling | Typed exception hierarchy | EAFP, matches JS client, enables error-specific attributes |
| Package name | `vector-embed-cache-client` | Explicit naming prevents confusion (2-1 consensus) |
| API design | Direct class instantiation | Pythonic, matches httpx/requests/boto3/stripe patterns |
| Retry logic | Optional `retries=N`, default 3 | Handles 429/5xx, configurable for power users |

---

## Public API

### Basic Usage

```python
from vector_embed_client import Client, AsyncClient

# Sync usage
with Client(api_key="vec_your_api_key") as client:
    # Single embedding
    embedding = client.embed("hello world")
    print(embedding.vector)   # list[float]
    print(embedding.cached)   # bool

    # Batch embeddings
    results = client.embed_batch(["hello", "world"])

    # Similarity search
    matches = client.search(text="find similar", top_k=10, min_score=0.5)

    # Statistics
    stats = client.stats()

# Async usage
async with AsyncClient(api_key="vec_your_api_key") as client:
    embedding = await client.embed("hello world")
    results = await client.embed_batch(["hello", "world"])
```

### Configuration Options

```python
import os

class Client:
    def __init__(
        self,
        api_key: str | None = None,      # Falls back to VECTOR_EMBED_API_KEY env var
        base_url: str = "https://api.vector-embed-cache.com",  # API base URL
        model: str = "nomic-v1.5",       # Default embedding model
        timeout: float = 30.0,           # Request timeout in seconds
        retries: int = 3,                # Retry count (0 to disable)
    ) -> None:
        # Environment variable fallback (standard Python API client pattern)
        self.api_key = api_key or os.environ.get("VECTOR_EMBED_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Pass api_key= or set VECTOR_EMBED_API_KEY environment variable."
            )
        ...

# Same options for AsyncClient
```

### Response Types

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class EmbedResponse:
    vector: list[float]
    cached: bool
    dimensions: int

@dataclass
class SearchResult:
    text_hash: str
    score: float
    text: Optional[str]       # Present only if include_text=True
    model: str
    hit_count: int

@dataclass
class SearchResponse:
    results: list[SearchResult]
    total: int
    search_time_ms: float

@dataclass
class StatsResponse:
    cache_hits: int
    cache_misses: int
    total_cached: int
```

---

## Error Handling

```python
class VectorEmbedError(Exception):
    """Base exception for all vector-embed-cache-client errors."""
    code: str
    status: Optional[int]

class AuthenticationError(VectorEmbedError):
    """Raised for 401 Unauthorized responses."""
    pass

class RateLimitError(VectorEmbedError):
    """Raised for 429 Too Many Requests responses."""
    retry_after: Optional[int]  # Seconds to wait before retry

class ValidationError(VectorEmbedError):
    """Raised for 400 Bad Request responses."""
    pass

class ServerError(VectorEmbedError):
    """Raised for 5xx server errors."""
    pass

class NetworkError(VectorEmbedError):
    """Raised for connection failures, timeouts, etc."""
    pass
```

Usage:

```python
from vector_embed_client import Client, RateLimitError, AuthenticationError

with Client(api_key="...") as client:
    try:
        result = client.embed("text")
    except RateLimitError as e:
        print(f"Rate limited. Retry after {e.retry_after}s")
    except AuthenticationError:
        print("Invalid API key")
```

---

## Module Exports

Explicit `__all__` in `__init__.py` for API surface control:

```python
__all__ = [
    # Clients
    "Client",
    "AsyncClient",
    # Response types
    "EmbedResponse",
    "SearchResult",
    "SearchResponse",
    "StatsResponse",
    # Errors
    "VectorEmbedError",
    "AuthenticationError",
    "RateLimitError",
    "ValidationError",
    "ServerError",
    "NetworkError",
]
```

---

## Project Structure

```
clients/python/
├── src/
│   └── vector_embed_client/
│       ├── __init__.py       # Public exports with __all__
│       ├── client.py         # Client class (sync)
│       ├── async_client.py   # AsyncClient class
│       ├── http.py           # Shared HTTP logic (auth, retries, errors)
│       ├── errors.py         # Exception classes
│       ├── types.py          # Response dataclasses
│       └── py.typed          # PEP 561 marker
├── tests/
│   ├── test_client.py        # Sync client tests
│   ├── test_async_client.py  # Async client tests
│   ├── test_errors.py        # Error handling tests
│   └── test_retry.py         # Retry logic tests
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## Build Configuration

### pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "vector-embed-cache-client"
version = "0.1.0"
description = "Python client for vector-embed-cache API"
readme = "README.md"
license = "MIT"
requires-python = ">=3.9"
authors = [
    { name = "Your Name", email = "you@example.com" }
]
keywords = ["embeddings", "vector", "cache", "ai", "ml", "vector-embed-cache"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Typing :: Typed",
]
dependencies = [
    "httpx>=0.25.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-httpx>=0.21.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
]

[project.urls]
Homepage = "https://github.com/your-org/vector-embed-cache"
Documentation = "https://github.com/your-org/vector-embed-cache#readme"
Repository = "https://github.com/your-org/vector-embed-cache"

[tool.hatch.build.targets.wheel]
packages = ["src/vector_embed_client"]

[tool.ruff]
line-length = 100
target-version = "py39"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.mypy]
python_version = "3.9"
strict = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

---

## Retry Logic

### Retry Configuration

```python
RETRY_CONFIG = {
    "base_delay": 1.0,           # 1 second
    "max_delay": 30.0,           # 30 seconds cap
    "jitter_factor": 0.2,        # ±20% randomization
    "retryable_statuses": [429, 500, 502, 503, 504],
}

def calculate_delay(attempt: int) -> float:
    """Calculate delay with exponential backoff + jitter."""
    exponential = RETRY_CONFIG["base_delay"] * (2 ** attempt)
    capped = min(exponential, RETRY_CONFIG["max_delay"])
    jitter = capped * RETRY_CONFIG["jitter_factor"] * (random.random() * 2 - 1)
    return capped + jitter
```

### Retry Behavior

- Respects `Retry-After` header when present (429 responses)
- Does NOT retry 4xx errors (except 429) - these are client errors
- Timeout applies per-attempt, not total
- Retries can be disabled with `retries=0`

---

## HTTP Headers

All requests include standard headers:

```python
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": f"vector-embed-cache-client/{VERSION} python/{python_version}",
}
```

---

## Logging

Use Python standard logging for observability:

```python
import logging

logger = logging.getLogger("vector_embed_client")

# Log retry attempts at DEBUG level
logger.debug("Retry attempt %d/%d after %.2fs delay", attempt, max_retries, delay)

# Log HTTP errors at WARNING level
logger.warning("Request failed with status %d: %s", status, message)
```

Users can configure logging level:

```python
import logging
logging.getLogger("vector_embed_client").setLevel(logging.DEBUG)
```

---

## CI Integration

Add to `.github/workflows/ci.yml`:

```yaml
  test-python-client:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: clients/python
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11', '3.12']
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
      - name: Lint
        run: ruff check .
      - name: Type check
        run: mypy src/
      - name: Test
        run: pytest -v
```

---

## Batch Semantics

- **Max batch size**: 100 texts (matches server limit)
- **Order guaranteed**: Results list matches input order
- **Partial failure**: All-or-nothing (server rejects entire batch on validation error)
- **Client validation**: Validates non-empty strings and batch size before sending

---

## Testing Strategy

### Unit Tests (pytest + pytest-httpx)

- **HTTP mocking**: Use pytest-httpx for mocking httpx requests
- **Error mapping**: Test all HTTP status codes map to correct error types
- **Retry logic**: Test exponential backoff, jitter, Retry-After header handling
- **Both clients**: Test sync and async clients with same test cases where possible
- **Input validation**: Test empty strings, batch size limits, invalid API keys

### Integration Tests

- Run against local server instance (started in CI)
- Test full request/response cycle
- Verify response types match dataclasses

---

## Comparison with JS Client

| Feature | JS Client | Python Client |
|---------|-----------|---------------|
| Factory pattern | `createClient({...})` | `Client(...)` (direct) |
| Async support | Promise-based | Separate `AsyncClient` |
| Error handling | Typed subclasses | Typed subclasses |
| Context managers | N/A | `with Client()` / `async with` |
| Retry default | 3 | 3 |
| HTTP library | Native fetch | httpx |
| Type system | TypeScript | Type hints + py.typed |

---

## Future Considerations

1. **Streaming support** - Not needed for current API, but design allows adding later
2. **Connection pooling** - httpx handles this automatically
3. **Middleware/hooks** - Could add request/response hooks if needed
4. **CLI wrapper** - Could add `python -m vector_embed_client` for quick testing
