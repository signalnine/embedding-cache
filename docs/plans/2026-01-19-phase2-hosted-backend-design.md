# Phase 2: Hosted Backend Design

**Date**: 2026-01-19
**Status**: Validated

## Overview

Build a self-hosted backend service for vector-embed-cache that provides a shared embedding cache with a hybrid compute model: free tier uses BYOK (Bring Your Own Key) passthrough, paid tier gets server-side GPU compute.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Your Server                          │
│                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   FastAPI    │───▶│  PostgreSQL  │    │   GPU     │ │
│  │   (API)      │    │  + pgvector  │    │  (nomic)  │ │
│  └──────────────┘    └──────────────┘    └───────────┘ │
│         │                                      ▲        │
│         │         ┌────────────────────────────┘        │
│         ▼         ▼                                     │
│  ┌─────────────────────┐                                │
│  │   Compute Router    │                                │
│  │  - Paid → GPU       │                                │
│  │  - Free → BYOK      │                                │
│  └─────────────────────┘                                │
└─────────────────────────────────────────────────────────┘
```

**Request Flow**
1. Client sends embedding request with API key
2. Server checks cache (PostgreSQL) - if hit, return immediately
3. Cache miss → route based on tier:
   - **Paid tier**: Compute locally on GPU, cache result
   - **Free tier**: Proxy to user's configured BYOK provider, cache result
4. Return embedding + cache status

## Decisions

| Question | Decision |
|----------|----------|
| Cache miss compute | Hybrid: Free tier BYOK, paid tier server compute |
| Paid tier backend | Local GPU with nomic-embed-text (CUDA) |
| Free tier BYOK | Any provider via passthrough (user configures endpoint + auth) |
| Pre-seeding | Minimal: English dictionary, common phrases (~100K embeddings) |
| Tenant isolation | Private by default for all tiers, opt-in to shared pool |
| Deployment | Self-hosted bare metal with NVIDIA GPU |
| Packaging | Direct execution (no containers) |
| MVP scope | Full hybrid model with both tiers working |

## API Design

### Endpoints

```
POST /v1/embed
  Request:  { "text": "hello world", "model": "nomic-v1.5" }
  Response: { "embedding": [0.1, 0.2, ...], "cached": true, "dimensions": 768 }

POST /v1/embed/batch
  Request:  { "texts": ["hello", "world"], "model": "nomic-v1.5" }
  Response: { "embeddings": [[...], [...]], "cached": [true, false], "dimensions": 768 }

GET /v1/stats
  Response: { "cache_hits": 42, "cache_misses": 3, "total_cached": 1234 }

POST /v1/provider (Free tier - configure BYOK)
  Request:  {
    "name": "openai",
    "endpoint": "https://api.openai.com/v1/embeddings",
    "api_key": "sk-...",
    "model_field": "model",
    "input_field": "input"
  }
  Response: { "provider_id": "prov_abc123" }

POST /v1/auth/signup
  Request:  { "email": "user@example.com", "password": "..." }
  Response: { "user_id": "usr_abc123", "api_key": "vec_xxxxx..." }

POST /v1/auth/login
  Request:  { "email": "user@example.com", "password": "..." }
  Response: { "token": "jwt_xxxxx...", "expires_in": 86400 }

POST /v1/keys (Authenticated - create additional API key)
  Response: { "api_key": "vec_yyyyy...", "prefix": "vec_yyyyy" }

DELETE /v1/keys/{prefix} (Authenticated - revoke API key)
  Response: { "revoked": true }

GET /v1/health
  Response: { "status": "ok", "version": "1.0.0" }
```

### Authentication

- All requests require `Authorization: Bearer vec_xxxxx` header
- API keys prefixed `vec_` for identification
- Keys stored hashed in database

### Rate Limits

- Free tier: 1,000 requests/day
- Paid tier: 50,000 requests/day

## Database Schema

```sql
-- Embeddings cache (core table)
CREATE TABLE embeddings (
  text_hash TEXT NOT NULL,           -- SHA-256 of normalized text
  model TEXT NOT NULL,               -- 'nomic-v1.5', 'openai:text-embedding-3-small'
  model_version TEXT NOT NULL,       -- For cache invalidation on model updates
  dimensions INTEGER NOT NULL,
  vector BYTEA NOT NULL,             -- Raw bytes, flexible dimensions
  tenant_id TEXT NOT NULL,           -- 'public' or user_id for private
  created_at TIMESTAMP DEFAULT NOW(),
  last_hit_at TIMESTAMP,
  hit_count INTEGER DEFAULT 0,
  PRIMARY KEY (text_hash, model, model_version, tenant_id)
);

CREATE INDEX idx_embeddings_lookup ON embeddings (tenant_id, model, text_hash);

-- Users
CREATE TABLE users (
  id TEXT PRIMARY KEY,               -- 'usr_abc123'
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,       -- bcrypt
  tier TEXT NOT NULL DEFAULT 'free', -- 'free', 'paid'
  created_at TIMESTAMP DEFAULT NOW(),
  email_verified_at TIMESTAMP
);

-- API keys
CREATE TABLE api_keys (
  key_hash TEXT PRIMARY KEY,         -- SHA-256 of full key
  key_prefix TEXT NOT NULL,          -- 'vec_abc12345' (first 12 chars)
  user_id TEXT NOT NULL REFERENCES users(id),
  tier TEXT NOT NULL,                -- 'free', 'paid'
  created_at TIMESTAMP DEFAULT NOW(),
  last_used_at TIMESTAMP,
  revoked_at TIMESTAMP
);

-- BYOK provider configs (free tier)
CREATE TABLE providers (
  id TEXT PRIMARY KEY,               -- 'prov_abc123'
  user_id TEXT NOT NULL,
  name TEXT NOT NULL,                -- 'openai', 'cohere', 'custom'
  endpoint TEXT NOT NULL,
  api_key_encrypted TEXT NOT NULL,   -- Encrypted at rest
  request_template JSONB NOT NULL,   -- How to format requests
  response_path TEXT NOT NULL,       -- JSONPath to extract embedding
  created_at TIMESTAMP DEFAULT NOW()
);

-- Usage tracking
CREATE TABLE usage (
  user_id TEXT NOT NULL,
  date DATE NOT NULL,
  model TEXT NOT NULL,
  cache_hits INTEGER DEFAULT 0,
  cache_misses INTEGER DEFAULT 0,
  compute_ms INTEGER DEFAULT 0,
  PRIMARY KEY (user_id, date, model)
);
```

## Tier System

### Free Tier

- Must configure at least one BYOK provider before embedding
- Cache lookups check: private namespace first, then shared pool
- Cache writes go to private namespace by default
- Can opt to contribute to shared pool: `"public": true` in request
- Rate limit: 1,000 req/day

### Paid Tier

- Server computes embeddings using local GPU (nomic-embed-text)
- Cache lookups check: private namespace first, then shared pool
- Cache writes go to private namespace by default
- Can opt to contribute to shared pool: `"public": true` in request
- Rate limit: 50,000 req/day

### Shared Pool

- Contains pre-seeded content (English dictionary, common phrases)
- Contains opt-in contributions from users who set `"public": true`
- Read access for all tiers
- Write access requires explicit opt-in per request

**Model Compatibility Note**: Shared pool lookups are scoped by model. An embedding cached with `openai:text-embedding-3-small` won't match a query for `nomic-v1.5`. Users benefit from shared pool only when using the same model as other contributors. Pre-seeded content uses `nomic-v1.5`.

## BYOK Passthrough

User configures their provider:
```json
POST /v1/provider
{
  "endpoint": "https://api.openai.com/v1/embeddings",
  "api_key": "sk-...",
  "request_template": { "model": "text-embedding-3-small", "input": "$TEXT" },
  "response_path": "$.data[0].embedding"
}
```

On cache miss, server:
1. Substitutes `$TEXT` with actual text
2. Calls provider endpoint with user's API key
3. Extracts embedding using `response_path`
4. Caches result in user's private namespace (or shared pool if `"public": true`)
5. Returns to user

### Security

- User API keys encrypted with AES-256-GCM
- Keys decrypted only at request time, never logged
- Provider configs isolated per user

### BYOK Endpoint Whitelist (SSRF Protection)

To prevent SSRF attacks, BYOK endpoints are restricted to known embedding providers:

```python
ALLOWED_BYOK_HOSTS = [
    "api.openai.com",
    "api.cohere.ai",
    "api.voyageai.com",
    "api.mistral.ai",
    "api.together.xyz",
]
```

Custom endpoints require admin approval. The server validates:
1. Host is in whitelist
2. Scheme is HTTPS only
3. No private IP ranges (127.x, 10.x, 192.168.x, etc.)

## Project Structure

```
server/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, routes
│   ├── config.py            # Settings, env vars
│   ├── models.py            # Pydantic schemas
│   ├── database.py          # PostgreSQL connection
│   ├── redis.py             # Redis connection for rate limiting
│   ├── auth.py              # API key + JWT validation
│   ├── users.py             # Signup, login, key management
│   ├── cache.py             # Cache lookup/store logic
│   ├── compute.py           # Local GPU embedding (process pool)
│   ├── passthrough.py       # BYOK provider proxy
│   ├── crypto.py            # Key encryption/decryption
│   ├── normalize.py         # Text normalization
│   └── metrics.py           # Prometheus metrics
├── tests/
├── alembic/                 # DB migrations
├── scripts/
│   └── seed.py              # Pre-seeding script
├── requirements.txt
└── README.md
```

## Deployment

### Dependencies

```bash
sudo apt install postgresql postgresql-contrib redis-server
pip install fastapi uvicorn sqlalchemy psycopg2-binary sentence-transformers cryptography redis bcrypt pyjwt prometheus-client
```

### Environment Variables

```bash
export DATABASE_URL="postgresql://user:pass@localhost/embeddings"
export REDIS_URL="redis://localhost:6379"
export ENCRYPTION_KEY="..."  # 32-byte key for BYOK storage
export GPU_DEVICE="cuda:0"
export JWT_SECRET="..."      # For auth tokens
```

### Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Systemd Service (optional)

```ini
[Unit]
Description=Vector Embed Cache API
After=postgresql.service

[Service]
User=gabe
WorkingDirectory=/home/gabe/vector-embed-cache/server
ExecStart=/home/gabe/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

## Cache Invalidation

**Model Version Tracking**

Each embedding is stored with `model_version`. When a model is updated:

1. Increment version in config (e.g., `nomic-v1.5` → `nomic-v1.5.1`)
2. New requests use new version, cache miss, compute fresh embedding
3. Old versions remain queryable but stop receiving new writes
4. Run cleanup job periodically: delete embeddings with `model_version` older than N versions

**TTL-based Eviction**

- Embeddings older than 90 days with `hit_count < 5` are eligible for deletion
- Run nightly cleanup job
- Exempt pre-seeded content (`tenant_id = 'public'`)

## Rate Limiting

**Implementation**: Redis-based sliding window

```python
# Redis key: rate:{user_id}:{window}
# Window = current minute for burst, current day for quota

async def check_rate_limit(user_id: str, tier: str) -> bool:
    daily_key = f"rate:{user_id}:{today()}"
    count = await redis.incr(daily_key)
    if count == 1:
        await redis.expire(daily_key, 86400)

    limit = 1000 if tier == "free" else 50000
    return count <= limit
```

**Dependencies**: Add Redis to deployment requirements.

## GPU Compute

**Non-blocking Inference**

GPU inference runs in a process pool to avoid blocking the async event loop:

```python
from concurrent.futures import ProcessPoolExecutor

executor = ProcessPoolExecutor(max_workers=2)

async def compute_embedding(text: str, model: str) -> list[float]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _compute_sync, text, model)
```

**Batching**

Requests arriving within 50ms window are batched together for efficient GPU utilization.

## Batch Limits

- `/v1/embed/batch` accepts max 100 texts per request
- Total payload size max 1MB
- Returns 400 Bad Request if exceeded

## Text Normalization

Consistent normalization for cache key generation:

```python
def normalize(text: str) -> str:
    text = text.strip()
    text = " ".join(text.split())  # Collapse whitespace
    text = text.lower()
    return text
```

Same algorithm used in library and server. Documented in API docs.

## Observability

**Logging**: Structured JSON logs to stdout
- Request ID, user ID, endpoint, latency, cache hit/miss
- Error details with stack traces

**Metrics** (Prometheus format at `/metrics`):
- `embed_requests_total{tier, cached}`
- `embed_latency_seconds{tier, cached}`
- `cache_size_bytes`
- `gpu_queue_depth`

**Health Check**: `GET /v1/health` returns service status

## Pre-seeding

Minimal initial seed (~100K embeddings):
- English dictionary (common words)
- Common phrases and expressions

Seed script runs once after deployment, populates shared pool with `tenant_id = 'public'`.

## Library Integration

Update the `vector_embed_cache` library to support the hosted backend:

```python
from vector_embed_cache import EmbeddingCache

# Configure to use hosted service
cache = EmbeddingCache(
    remote_url="https://your-server.com/v1",
    api_key="vec_xxxxx"
)

# Works the same as local
embedding = cache.embed("hello world")
```

Add `configure()` function for global setup:
```python
import vector_embed_cache

vector_embed_cache.configure(
    remote_url="https://your-server.com/v1",
    api_key="vec_xxxxx"
)
```
