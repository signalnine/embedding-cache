# Vector Embed Cache Server

Self-hosted embedding cache backend with hybrid compute model.

## Overview

The server provides a multi-tenant embedding cache API with two compute tiers:

| Tier | Compute Method | Rate Limit | Use Case |
|------|----------------|------------|----------|
| **Free** | BYOK (Bring Your Own Key) | 1,000/day | Users provide their own OpenAI/other API key |
| **Paid** | Server GPU (nomic-v1.5) | 50,000/day | Server computes embeddings locally |

## Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL (for production) or SQLite (for testing)
- Redis (optional, for rate limiting)
- NVIDIA GPU (optional, for paid tier compute)

### Installation

```bash
cd server
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables

```bash
# Required
export DATABASE_URL="postgresql://user:pass@localhost/embeddings"
export JWT_SECRET="your-jwt-secret-key"
export ENCRYPTION_KEY="your-32-byte-encryption-key"

# Optional
export REDIS_URL="redis://localhost:6379"  # For rate limiting
export GPU_DEVICE="cuda:0"                  # For paid tier compute (default: cuda:0)
export JWT_EXPIRY_SECONDS="86400"           # Token expiry (default: 24 hours)
```

### Running the Server

```bash
# Development
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Running Tests

```bash
pytest tests/ -v
```

## API Reference

### Authentication

#### Sign Up
```http
POST /v1/auth/signup
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securepassword123"
}
```

**Response:**
```json
{
  "user_id": "usr_abc123...",
  "api_key": "vec_xyz789..."
}
```

#### Login
```http
POST /v1/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securepassword123"
}
```

**Response:**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIs...",
  "expires_in": 86400
}
```

#### Create API Key
```http
POST /v1/keys
Authorization: Bearer <jwt_token>
```

**Response:**
```json
{
  "api_key": "vec_newkey123...",
  "prefix": "vec_newkey12"
}
```

#### Revoke API Key
```http
DELETE /v1/keys/{key_prefix}
Authorization: Bearer <jwt_token>
```

**Response:**
```json
{
  "revoked": true
}
```

### Embeddings

All embedding endpoints require API key authentication:
```http
Authorization: Bearer vec_your_api_key
```

#### Single Embedding
```http
POST /v1/embed
Authorization: Bearer <api_key>
Content-Type: application/json

{
  "text": "Hello world",
  "model": "nomic-v1.5",
  "public": false
}
```

**Response:**
```json
{
  "embedding": [0.123, -0.456, ...],
  "cached": false,
  "dimensions": 768
}
```

#### Batch Embeddings
```http
POST /v1/embed/batch
Authorization: Bearer <api_key>
Content-Type: application/json

{
  "texts": ["Hello", "World", "Foo"],
  "model": "nomic-v1.5",
  "public": false
}
```

**Response:**
```json
{
  "embeddings": [[0.1, ...], [0.2, ...], [0.3, ...]],
  "cached": [false, false, false],
  "dimensions": 768
}
```

### BYOK Provider Configuration (Free Tier)

Free tier users must configure a BYOK provider before using the embed endpoints.

#### Configure Provider
```http
POST /v1/provider
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "name": "openai",
  "endpoint": "https://api.openai.com/v1/embeddings",
  "api_key": "sk-your-openai-key",
  "request_template": {
    "model": "text-embedding-3-small",
    "input": "{{text}}"
  },
  "response_path": "$.data[0].embedding"
}
```

**Response:**
```json
{
  "provider_id": "prv_abc123..."
}
```

**Security Notes:**
- API keys are encrypted at rest using Fernet encryption
- Endpoints are validated against an allowlist (SSRF protection)
- Only HTTPS endpoints are allowed

### Statistics

#### Get Usage Stats
```http
GET /v1/stats
Authorization: Bearer <api_key>
```

**Response:**
```json
{
  "cache_hits": 150,
  "cache_misses": 50,
  "total_cached": 200
}
```

### Health & Metrics

#### Health Check
```http
GET /v1/health
```

**Response:**
```json
{
  "status": "ok",
  "version": "1.0.0"
}
```

#### Prometheus Metrics
```http
GET /metrics
```

Returns Prometheus-format metrics for monitoring.

## Architecture

### Database Schema

```
users
├── id (PK)
├── email (unique)
├── password_hash
├── tier (free/paid)
├── created_at
└── email_verified_at

api_keys
├── key_hash (PK)
├── key_prefix
├── user_id (FK)
├── tier
├── created_at
├── last_used_at
└── revoked_at

embeddings
├── text_hash (PK)
├── model (PK)
├── model_version (PK)
├── tenant_id (PK)
├── dimensions
├── vector (blob)
├── created_at
├── last_hit_at
└── hit_count

providers
├── id (PK)
├── user_id (FK)
├── name
├── endpoint
├── api_key_encrypted
├── request_template (JSON)
├── response_path
└── created_at

usage
├── user_id (PK)
├── date (PK)
├── model (PK)
├── cache_hits
├── cache_misses
└── compute_ms
```

### Request Flow

1. **Authentication**: API key validated, user tier determined
2. **Rate Limiting**: Redis counter checked (if configured)
3. **Cache Lookup**: Check PostgreSQL for existing embedding
4. **Cache Hit**: Return cached embedding immediately
5. **Cache Miss (Paid)**: Compute via GPU in ProcessPoolExecutor
6. **Cache Miss (Free)**: Forward to user's BYOK provider
7. **Store**: Save new embedding to cache

### Security

- **Passwords**: Hashed with bcrypt
- **API Keys**: SHA-256 hashed for storage, only prefix stored in plaintext
- **BYOK Keys**: Encrypted at rest with Fernet (AES-128-CBC)
- **SSRF Protection**: Endpoint allowlist (openai.com, cohere.ai, etc.)
- **Rate Limiting**: Daily counters per user in Redis

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://localhost/embeddings` | Database connection string |
| `REDIS_URL` | `redis://localhost:6379` | Redis URL (optional) |
| `ENCRYPTION_KEY` | *required* | Fernet key for BYOK encryption |
| `JWT_SECRET` | *required* | Secret for JWT signing |
| `JWT_EXPIRY_SECONDS` | `86400` | JWT token expiry (24 hours) |
| `GPU_DEVICE` | `cuda:0` | GPU device for embedding compute |
| `MODEL_VERSION` | `1.0.0` | Version string for cache invalidation |
| `FREE_TIER_DAILY_LIMIT` | `1000` | Free tier rate limit |
| `PAID_TIER_DAILY_LIMIT` | `50000` | Paid tier rate limit |
| `MAX_BATCH_SIZE` | `100` | Maximum texts per batch request |
| `MAX_PAYLOAD_BYTES` | `1000000` | Maximum request payload size |

## Deployment

### With Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY alembic/ alembic/
COPY alembic.ini .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Database Migrations

```bash
# Create migration
alembic revision --autogenerate -m "description"

# Run migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

### Pre-seeding Common Embeddings

```bash
python scripts/seed.py
```

This pre-computes embeddings for common words and phrases to improve cache hit rates.

## Monitoring

The `/metrics` endpoint exposes Prometheus metrics:

- `http_requests_total` - Total HTTP requests by endpoint and status
- `http_request_duration_seconds` - Request latency histogram
- `embedding_cache_hits_total` - Cache hit counter
- `embedding_cache_misses_total` - Cache miss counter
- `embedding_compute_duration_seconds` - Embedding computation time

## Troubleshooting

### "Free tier requires BYOK provider"
Free tier users must configure a provider before using `/v1/embed`. Use `POST /v1/provider` to configure.

### "Rate limit exceeded"
Wait until the daily reset (midnight UTC) or upgrade to paid tier.

### Redis connection errors
Rate limiting gracefully degrades without Redis. The server will continue to function, but rate limits won't be enforced.

### GPU not available
Set `GPU_DEVICE=cpu` for CPU-only compute. Performance will be slower but functional.
