# embedding-cache Roadmap

**Date**: 2026-01-18
**Status**: Approved (with open issues noted)

## Vision

embedding-cache becomes the "Cloudflare for embeddings" - a free open-source library backed by a hosted service with a massive shared cache. Developers get instant cache hits for common content, zero setup, and costs that scale with usage.

## Context

**Business Model**: Freemium library with paid hosted backend (like Sentry, PostHog)

**Value Proposition**:
- Shared cache network - Pre-computed embeddings for common text (Wikipedia, docs, common phrases). Instant cache hits even on first request.
- Managed infrastructure - No local model setup, no GPU, no downloads. Just works.

**Target Customers**: Solo developers, indie hackers, and startups building products with embeddings.

**Technical Approach**: Simple monolith (FastAPI, PostgreSQL) deployed on Railway/Fly.io.

**Timeline**: Side project pace - no deadlines, work on whatever's interesting.

## Phases

| Phase | Focus | Outcome |
|-------|-------|---------|
| 1. Library Polish | Make the OSS library production-ready | Publishable to PyPI, attracts early adopters |
| 2. Hosted Backend | Build the remote cache service | Users can opt-in to shared cache network |
| 3. Growth & Monetization | Pre-seed cache, add billing, expand | Sustainable business with network effects |

**Milestone markers** (not deadlines):
- Phase 1 complete: First PyPI release, 10+ GitHub stars
- Phase 2 complete: Hosted service live, first external user
- Phase 3 complete: First paying customer, 1M cached embeddings

---

## Phase 1: Library Polish

**Goal**: Make embedding-cache ready for public release on PyPI.

### 1.1 Packaging & Distribution
- Publish to PyPI as `embedding-cache`
- Verify all extras work: `pip install embedding-cache[local]`, `[openai]`, `[local,openai]`
- Add `py.typed` marker for type checker support
- Test installation on fresh virtualenvs (Linux, macOS, Windows)

### 1.2 Documentation
- Write a quickstart guide (5-minute time-to-value)
- Add docstrings to all public APIs
- Create examples directory:
  - Basic caching
  - RAG pipeline integration
  - Batch processing large datasets
  - Switching between models
- Set up docs site (GitHub Pages or ReadTheDocs)

### 1.3 Developer Experience
- Add CLI tool: `embedding-cache stats`, `embedding-cache clear`, `embedding-cache export`
- Improve error messages (audit for clarity)
- Add `--verbose` logging option for debugging

### 1.4 Quality & Testing
- Reach 90%+ test coverage
- Add property-based tests for edge cases
- Set up CI/CD (GitHub Actions): lint, test, type-check on every PR
- Add badges to README (coverage, PyPI version, downloads)

---

## Phase 2: Hosted Backend

**Goal**: Build the remote cache service that makes the shared cache network real.

### 2.1 API Design

```
POST /v1/embed
  { "text": "hello world", "model": "nomic-v1.5" }
  → { "embedding": [...], "cached": true }

POST /v1/embed/batch
  { "texts": ["hello", "world"], "model": "nomic-v1.5" }
  → { "embeddings": [[...], [...]], "cache_hits": 2 }

GET /v1/stats
  → { "total_cached": 1234567, "your_hits": 42, "your_misses": 3 }
```

- Simple REST API, JSON in/out
- API key authentication (free tier gets a key too, for rate limiting)
- Rate limits: Free tier 1000 req/day, paid tiers higher

### 2.2 Infrastructure
- **Framework**: FastAPI (matches the library, familiar)
- **Database**: PostgreSQL with pgvector extension
- **Hosting**: Railway or Fly.io
- **Caching layer**: Redis for hot embeddings (add when needed)

### 2.3 Schema Design

```sql
-- Core embeddings table
-- Composite key supports multiple models and versions
embeddings (
  text_hash TEXT NOT NULL,           -- SHA-256 of normalized text
  model TEXT NOT NULL,               -- e.g., 'nomic-ai/nomic-embed-text-v1.5'
  model_version TEXT NOT NULL,       -- e.g., '1.5.0' for cache invalidation
  dimensions INTEGER NOT NULL,       -- 768, 1536, etc.
  vector BYTEA NOT NULL,             -- Store as bytes, flexible dimensions
  tenant_id TEXT NOT NULL,           -- User namespace (or 'public' for shared)
  created_at TIMESTAMP DEFAULT NOW(),
  last_hit_at TIMESTAMP,
  hit_count INTEGER DEFAULT 0,
  PRIMARY KEY (text_hash, model, model_version, tenant_id)
)

-- Index for fast lookups
CREATE INDEX idx_embeddings_lookup ON embeddings (tenant_id, model, text_hash);

-- User management
api_keys (
  key_hash TEXT PRIMARY KEY,         -- Store hashed, not plaintext
  key_prefix TEXT NOT NULL,          -- First 8 chars for identification
  user_id TEXT NOT NULL,
  tier TEXT NOT NULL,                -- 'free', 'pro', 'team'
  created_at TIMESTAMP DEFAULT NOW(),
  last_used_at TIMESTAMP,
  revoked_at TIMESTAMP               -- NULL if active
)

-- Usage tracking
usage (
  user_id TEXT NOT NULL,
  date DATE NOT NULL,
  model TEXT NOT NULL,
  cache_hits INTEGER DEFAULT 0,
  cache_misses INTEGER DEFAULT 0,
  compute_ms INTEGER DEFAULT 0,
  PRIMARY KEY (user_id, date, model)
)
```

**Key design decisions:**
- Composite primary key `(text_hash, model, model_version, tenant_id)` enables multi-model support and per-user namespaces
- `BYTEA` for vectors instead of `VECTOR(n)` allows variable dimensions across models
- `tenant_id` enables private caches by default; 'public' namespace for opt-in shared content
- API keys stored hashed with prefix for identification
- `model_version` allows bulk invalidation when models update

### 2.4 Library Integration
- Update `RemoteProvider` to call the hosted API
- Add `embedding_cache.configure(api_key="...")` for easy setup
- Default remote URL points to hosted service
- Graceful fallback if hosted service unavailable

---

## Phase 3: Growth & Monetization

**Goal**: Build network effects through pre-seeding, add billing for sustainability.

### 3.1 Pre-seed the Shared Cache

**Priority datasets:**
1. **Wikipedia abstracts** (~6M articles, first paragraphs)
2. **Common Crawl sample** - popular web content
3. **Programming docs** - Python stdlib, popular libraries
4. **English dictionary** - 100K common words and definitions

**Approach**: Batch jobs to embed datasets, store in hosted DB. Track which pre-seeded content gets hit most, expand those categories.

### 3.2 Pricing & Billing

| Tier | Price | Limits | Target |
|------|-------|--------|--------|
| Free | $0 | 1K req/day, shared cache only | Hobbyists, evaluation |
| Pro | $20/mo | 50K req/day, priority compute | Solo devs, small apps |
| Team | $100/mo | 500K req/day, usage dashboard | Startups, growing apps |

- Stripe integration for payments
- Usage tracking dashboard
- Overage billing or soft caps (user choice)

### 3.3 Growth Mechanics
- "Powered by embedding-cache" badge for free tier
- Cost savings calculator - "You saved $X vs OpenAI"
- Public stats page - "1M embeddings cached, serving 500 developers"
- Referral program - extra quota for invites

---

## Prioritization

| Phase | Workstream | Effort | Impact | Priority |
|-------|------------|--------|--------|----------|
| 1 | 1.1 Packaging & Distribution | Small | High | P0 |
| 1 | 1.2 Documentation | Medium | High | P0 |
| 1 | 1.3 CLI Tools | Small | Medium | P1 |
| 1 | 1.4 Quality & CI/CD | Medium | Medium | P1 |
| 2 | 2.1 API Design | Small | High | P0 |
| 2 | 2.2 Infrastructure | Medium | High | P0 |
| 2 | 2.3 Schema Design | Small | High | P0 |
| 2 | 2.4 Library Integration | Small | High | P0 |
| 3 | 3.1 Pre-seed Cache | Large | High | P0 |
| 3 | 3.2 Pricing & Billing | Medium | High | P1 |
| 3 | 3.3 Growth Mechanics | Medium | Medium | P2 |

## Pick-up-anytime Tasks

For side-project pace, here are small tasks you can do in isolation:
- Write one example script (30 min)
- Add one CLI command (1 hr)
- Design one API endpoint (1 hr)
- Embed one dataset (background job)
- Write one docs page (1 hr)
- Add one test file (30 min)

## Dependencies

- Phase 2 needs Phase 1 packaging (library must be installable)
- Phase 3 billing needs Phase 2 infrastructure running
- Everything else is fairly independent

## Success Metrics

**Phase 1**:
- PyPI package published and installable
- 10+ GitHub stars
- 3+ example scripts in repo

**Phase 2**:
- Hosted service running on Railway/Fly.io
- First external user (not you)
- 99% uptime over first month

**Phase 3**:
- 1M embeddings in shared cache
- First paying customer
- $100+ MRR

---

## Open Issues (From Multi-Agent Review)

Issues identified during design validation that need resolution before/during implementation:

### Critical (Must Address)

1. **Cache miss compute strategy** - Who pays for GPU compute on cache misses?
   - Options: BYOK (Bring Your Own API Key) for free tier, dedicated compute for paid
   - Need cost modeling before finalizing pricing

2. **Pre-seeding economics** - 6M Wikipedia embeddings may not be viable
   - High upfront cost (compute + storage)
   - Invalidation burden when models update
   - Unvalidated assumption users want this specific content
   - Consider: start with no pre-seeding, add based on actual usage patterns

3. **Privacy/tenant isolation** - Shared cache creates inference attack vectors
   - Default to private namespaces per user
   - Opt-in only for public shared pools
   - Need legal review for cached content

### Moderate (Before Paid Launch)

4. **Cache eviction strategy** - Unbounded growth will degrade performance
   - Need TTL or LRU eviction policy
   - Capacity limits per tier

5. **Normalization algorithm** - Must be explicit and documented
   - Whitespace, unicode, case handling
   - Same algorithm in library and server

6. **Legal/licensing review** - ToS for cached content, GDPR, model provider agreements

7. **Observability** - Request logging, error rates, cache hit metrics

### Open Questions

- Is the latency value prop real? Network round-trip may exceed local inference for small models
- What cache hit rate makes this viable? Need threshold for go/no-go
- Should pre-seeding be cut entirely or just delayed?
