# Vector Embed Cache Server

Self-hosted embedding cache with hybrid compute model.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="postgresql://user:pass@localhost/embeddings"
export REDIS_URL="redis://localhost:6379"
export ENCRYPTION_KEY="your-32-byte-secret-key"
export JWT_SECRET="your-jwt-secret"
export GPU_DEVICE="cuda:0"

# Run migrations
alembic upgrade head

# Start server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

- `POST /v1/auth/signup` - Create account
- `POST /v1/auth/login` - Get JWT token
- `POST /v1/embed` - Get embedding
- `POST /v1/embed/batch` - Get batch embeddings
- `POST /v1/provider` - Configure BYOK provider
- `GET /v1/stats` - Usage statistics
- `GET /v1/health` - Health check
- `GET /metrics` - Prometheus metrics

## Tiers

| Tier | Compute | Rate Limit |
|------|---------|------------|
| Free | BYOK (bring your own key) | 1,000/day |
| Paid | Server GPU (nomic-v1.5) | 50,000/day |

## Pre-seeding

```bash
python scripts/seed.py
```
