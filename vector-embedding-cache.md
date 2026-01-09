---
apple_notes_id: x-coredata://975E8692-B6B0-48E2-97C5-0E5E6F550F8B/ICNote/p1310
---

# Vector Embedding Cache

Created: 2026-01-06
Status: Idea
Priority: Side Project

## Concept

A public, searchable database of string-to-vector embeddings. Like HuggingFace but for pre-computed embeddings.

**Core idea:**
1. Hash incoming string
2. Check if embedding exists in database
3. If exists → return cached vectors (free, fast)
4. If not → compute via OpenAI/local model, cache it, return

## Why This Matters

- **Cost savings**: Embedding API calls cost money, many strings are repeated
- **Speed**: Cache hits are instant vs API latency
- **Public good**: Common phrases/concepts could be pre-embedded for everyone
- **Deduplication**: "hello world" doesn't need to be embedded millions of times

## Technical Approach

### Hashing Strategy
- Hash input string (SHA-256 or similar)
- Normalize first? (lowercase, trim whitespace, etc.)
- Handle near-duplicates? (fuzzy matching adds complexity)

### Storage
- Key: hash of input string
- Value: embedding vector (e.g., 1536 floats for text-embedding-3-small)
- Metadata: model used, timestamp, original string (optional)

### Database Options
- **Postgres + pgvector**: Simple, can do similarity search on stored embeddings
- **Redis**: Fast cache layer
- **Pinecone/Weaviate/Qdrant**: Purpose-built vector DBs
- **SQLite + blob**: Simplest for prototype

### API Design
```
POST /embed
{
  "text": "hello world",
  "model": "text-embedding-3-small"  // optional, default
}

Response (cache hit):
{
  "embedding": [0.1, 0.2, ...],
  "cached": true,
  "hash": "abc123..."
}

Response (cache miss):
{
  "embedding": [0.1, 0.2, ...],
  "cached": false,
  "hash": "abc123..."
}
```

### Batch API
```
POST /embed/batch
{
  "texts": ["hello", "world", "foo"],
  "model": "text-embedding-3-small"
}

Response:
{
  "embeddings": [...],
  "cache_hits": 2,
  "cache_misses": 1
}
```

## Business Model Options

1. **Free tier + paid**: X requests/month free, pay for more
2. **Open source + hosted**: Self-host or use our hosted version
3. **Donation/grant funded**: Public good model
4. **Freemium**: Free for cache hits, pay for cache misses (actual API cost + margin)

## Challenges

- **Model versioning**: text-embedding-3-small v1 vs v2 produce different vectors
- **Storage costs**: 1536 floats × 4 bytes = 6KB per embedding
- **Privacy**: Storing the original strings could be sensitive
- **Cold start**: Empty cache = no value initially
- **Normalization**: "Hello World" vs "hello world" vs "  hello world  "

## MVP Scope

1. Single model support (text-embedding-3-small)
2. Simple hash lookup (exact match only)
3. SQLite storage for prototype
4. Basic REST API
5. OpenAI fallback for cache misses

## Stretch Goals

- Multiple model support
- Similarity search on cached embeddings
- Pre-seed with common phrases/words
- Local embedding model option (no OpenAI dependency)
- Client libraries (Python, JS)
- Self-hostable Docker image
- **Federated network** (see below)

## Federation Model

Federate self-hosted inference nodes into a shared network:

**How it works:**
1. Anyone can run a node (Docker image with nomic-embed + cache)
2. Nodes share their cache via DHT or gossip protocol
3. Query hits local cache first, then asks network peers
4. Cache misses computed locally, shared back to network

**Benefits:**
- Distributed compute - no single point of failure
- Cache grows with network size
- Geographic distribution = lower latency globally
- No central authority needed
- Nodes can be heterogeneous (CPU-only, GPU, different capacities)

**Inspiration:**
- BitTorrent DHT for distributed hash lookup
- IPFS for content-addressed storage
- Mastodon/ActivityPub for federated social
- Folding@home for distributed compute

**Protocol sketch:**
```
1. Node joins network, announces capacity
2. On query:
   - Check local cache → hit? return
   - Hash query to find responsible peer(s)
   - Ask peers → hit? cache locally + return
   - Miss everywhere? compute locally, broadcast to peers
3. Peers cache popular embeddings (LRU/LFU)
4. Optional: proof-of-compute for contributions
```

**Challenges:**
- Consistency (same input should give same embedding)
- Trust (malicious nodes returning garbage)
- Discovery (how nodes find each other)
- Incentives (why run a node?)

**Incentive ideas:**
- Tit-for-tat (serve queries to get queries served)
- Token/credit system for contributions
- Reputation based on uptime + accuracy
- Pure altruism / public good (works for Tor, BOINC)

## Pre-seeding Ideas

- Wikipedia article titles
- Common English phrases
- Programming documentation snippets
- Product descriptions corpus
- Book titles / authors

## Self-Hosted Model Option

**nomic-embed-text-v2** looks promising:
- Open source, can self-host
- 768 dimensions (vs OpenAI's 1536) = smaller storage
- Comparable quality to OpenAI text-embedding-3-small
- No per-request API costs
- Apache 2.0 license
- Runs on modest hardware (CPU inference possible, GPU faster)

This changes the economics significantly:
- No OpenAI dependency
- Cache misses only cost compute, not API fees
- Can scale horizontally
- Full control over the pipeline

## Prior Art / Competition

- OpenAI Embeddings API (not cached)
- HuggingFace Inference API (models, not pre-computed)
- Various vector DBs (storage, not public cache)
- Cohere Embed (similar pricing model)
- Nomic Atlas (visualization, not caching)

## Tech Stack (Proposed)

- **Backend**: Python (FastAPI) or Go
- **Database**: Postgres + pgvector
- **Cache**: Redis for hot embeddings
- **Embedding Model**: nomic-embed-text-v2 (self-hosted) - no API costs
- **Hosting**: Fly.io or Railway for prototype, GPU instance for embeddings

## Next Steps

- [ ] Prototype with SQLite + FastAPI
- [ ] Test with semantic-tarot as first client
- [ ] Measure cache hit rate on real usage
- [ ] Estimate storage/compute costs at scale

---
*Created: 2026-01-06*
