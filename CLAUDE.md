# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Vector Embedding Cache** - A Python library and hosted backend for caching string-to-vector embeddings. Reduces API costs and latency through local or remote caching of pre-computed embeddings.

**Current Status**:
- **Library**: Production-ready, published to PyPI as `vector-embed-cache`
- **Server**: Implemented and tested, ready for deployment

## Architecture

### Library (vector_embed_cache/)

**Core Flow:**
1. Hash incoming string (SHA-256) combined with model name
2. Check if embedding exists in user's SQLite cache file
3. User cache hit → return cached vectors (instant, free)
4. Check bundled preseed database for common words
5. Preseed hit → return pre-computed vectors (instant, free)
6. Cache miss → compute via embedding model, cache result, return

**Features:**
- Multiple model support: Local models (nomic-embed-text-v1.5, v2-moe) and OpenAI
- Model prefix routing: `"openai:text-embedding-3-small"` routes to OpenAI provider
- SQLite file storage: Simple, portable cache files
- Batch operations: `embed_batch()` for multiple texts
- Thread-safe: Safe for concurrent use
- CLI tool: `embedding-cache stats`, `embedding-cache clear`, `embedding-cache migrate`
- Cache compression: Float16 quantization (~4x storage reduction)

### Server (server/)

**Hybrid Compute Model:**
- **Free tier**: BYOK (Bring Your Own Key) - users configure their own API provider
- **Paid tier**: Server-side GPU compute using nomic-embed-text-v1.5

**Tech Stack:**
- FastAPI with async support
- PostgreSQL for embedding storage
- Redis for rate limiting (optional, gracefully degrades)
- ProcessPoolExecutor for GPU compute isolation
- Fernet encryption for API keys at rest
- JWT authentication with bcrypt password hashing

## Project Structure

```
vector_embed_cache/          # Python library (PyPI package)
├── __init__.py              # Public API: EmbeddingCache, embed()
├── cache.py                 # Main EmbeddingCache implementation
├── storage.py               # SQLite storage layer
├── normalize.py             # Text normalization
├── providers.py             # LocalProvider, OpenAIProvider
├── preseed.py               # Pre-seeded embeddings lookup
├── cli.py                   # CLI commands
├── utils.py                 # Utility functions
└── data/
    └── preseed_v1.5.db      # Bundled preseed database (3K words)

scripts/                     # Development scripts
└── generate_preseed.py      # Generate preseed database from wordfreq

server/                      # Hosted backend (FastAPI)
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Pydantic settings
│   ├── models.py            # SQLAlchemy ORM models
│   ├── schemas.py           # Pydantic request/response schemas
│   ├── similarity.py        # pgvector similarity search logic
│   ├── auth.py              # JWT + bcrypt authentication
│   ├── admin_auth.py        # Admin JWT cookies + CSRF tokens
│   ├── cli.py               # CLI commands (create-admin)
│   ├── crypto.py            # Fernet encryption for API keys
│   ├── database.py          # Database connection
│   ├── redis_client.py      # Redis connection (optional)
│   ├── rate_limit.py        # Rate limiting middleware
│   ├── cache.py             # Embedding cache operations
│   ├── compute.py           # GPU embedding computation
│   ├── normalize.py         # Text normalization
│   ├── passthrough.py       # BYOK provider passthrough
│   ├── templates/admin/     # Jinja2 templates for admin UI
│   ├── static/              # CSS, JS (htmx, frappe-charts)
│   └── routes/
│       ├── users.py         # Auth endpoints
│       ├── embed.py         # Embedding endpoints
│       ├── providers.py     # Provider config endpoints
│       ├── search.py        # Similarity search endpoint
│       └── admin.py         # Admin dashboard routes
├── tests/                   # Server unit tests (110+ tests)
├── alembic/                 # Database migrations
├── scripts/
│   ├── seed.py              # Pre-seeding script
│   ├── create_hnsw_indexes.py   # pgvector HNSW index creation
│   ├── migrate_to_pgvector.py   # Data migration with checkpointing
│   └── swap_tables.py           # Atomic table cutover
└── requirements.txt         # Server dependencies

tests/                       # Library unit tests
docs/plans/                  # Design documents and implementation plans

clients/                     # Client libraries for hosted API
├── js/                      # JavaScript/TypeScript client (npm)
│   ├── src/
│   │   ├── client.ts        # Main VectorEmbedClient class
│   │   ├── errors.ts        # Typed error classes
│   │   └── types.ts         # TypeScript interfaces
│   └── package.json
├── python/                  # Python client (PyPI)
│   ├── src/vector_embed_client/
│   │   ├── client.py        # VectorEmbedClient class
│   │   ├── errors.py        # Exception hierarchy
│   │   └── types.py         # Dataclasses for API types
│   └── pyproject.toml
└── go/                      # Go client
    └── vectorembed/
        ├── client.go        # Client implementation
        ├── errors.go        # Error types
        ├── types.go         # Request/response structs
        └── options.go       # Functional options pattern
```

## Usage

```python
from vector_embed_cache import EmbeddingCache

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

### Running Library Tests
```bash
pytest tests/ -v
```

### Running Server Tests
```bash
cd server
pytest tests/ -v
```

### Library Installation
```bash
# Local models (recommended)
pip install -e ".[local]"

# With OpenAI support
pip install -e ".[openai]"
```

### Server Installation
```bash
cd server
pip install -r requirements.txt

# Set required environment variables
export DATABASE_URL="postgresql://user:pass@localhost/embeddings"
export JWT_SECRET="your-jwt-secret"
export ENCRYPTION_KEY="your-encryption-key"

# Optional: Redis for rate limiting
export REDIS_URL="redis://localhost:6379"

# Start server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Key Design Decisions

**Library:**
- **Cache Key Format**: SHA-256 hash of normalized text + model name
- **Provider Pattern**: `LocalProvider`, `OpenAIProvider` share common interface
- **Lazy Loading**: Models loaded on first use
- **Text Normalization**: Whitespace collapsed, lowercased for cache key generation
- **Preseed Fallback**: Bundled SQLite DB with 3,000 common words (nomic-v1.5 only)

**Server:**
- **Hybrid Compute**: Free tier = BYOK, Paid tier = server GPU
- **Multi-tenant**: Embeddings isolated by tenant_id (user_id)
- **BYOK Security**: API keys encrypted at rest with Fernet, SSRF protection on endpoints
- **Rate Limiting**: Redis-based daily counters (gracefully degrades without Redis)
- **Composite Primary Key**: (text_hash, model, model_version, tenant_id) for versioned caching

**Similarity Search (pgvector):**
- Hash-partitioned embeddings table (32 partitions by tenant_id)
- Partial HNSW indexes per dimension (768, 1536, 384)
- Inner product distance with 0-1 normalized scores
- Multi-tenant isolation via API key

## Server Database Models

| Model | Purpose |
|-------|---------|
| User | Account with email, password_hash, tier |
| ApiKey | API authentication with key_hash, prefix |
| Embedding | Cached vectors with composite key |
| Provider | BYOK provider configuration |
| Usage | Daily usage tracking per user |

## Integration Example: semantic-tarot

The library is tested with [semantic-tarot](../semantic-tarot/), which uses it for:
- 780 card embeddings (78 cards × 2 positions × 5 systems)
- Semantic search queries
- Zero-cost regeneration via cache hits

## Roadmap

### Completed
- [x] Local SQLite caching
- [x] Multiple model support (nomic, OpenAI)
- [x] CLI tool for cache management
- [x] Hosted backend with FastAPI
- [x] BYOK provider support
- [x] JWT authentication
- [x] Rate limiting
- [x] Similarity search on cached embeddings (pgvector)
- [x] Admin dashboard with usage stats
- [x] Pre-seeded common words (3,000 English words)
- [x] Client libraries (JavaScript, Python, Go)
- [x] Cache compression (float16 quantization, ~4x reduction)
