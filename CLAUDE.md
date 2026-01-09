# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Vector Embedding Cache** - A public, searchable database of string-to-vector embeddings. The core idea is to cache pre-computed embeddings to reduce API costs and latency.

**Current Status**: Idea/planning phase. No code implementation exists yet - only design documentation.

## Architecture (Planned)

### Core Flow
1. Hash incoming string (SHA-256)
2. Check if embedding exists in cache database
3. Cache hit → return cached vectors (free, fast)
4. Cache miss → compute via embedding model, cache result, return

### Key Design Decisions

**MVP Scope**:
- Single model support: `text-embedding-3-small` (or self-hosted `nomic-embed-text-v2`)
- Simple hash lookup (exact match only)
- SQLite storage for prototype
- Basic REST API
- OpenAI fallback for cache misses (or local model)

**Proposed Tech Stack**:
- Backend: Python (FastAPI) or Go
- Database: Postgres + pgvector
- Cache layer: Redis for hot embeddings
- Embedding Model: nomic-embed-text-v2 (self-hosted, 768 dimensions, Apache 2.0)
- Hosting: Fly.io or Railway for prototype

### Storage Schema
- Key: hash of input string
- Value: embedding vector (768-1536 floats depending on model)
- Metadata: model used, timestamp, optionally original string

## API Design (Planned)

**Single embedding**:
```
POST /embed
{
  "text": "hello world",
  "model": "text-embedding-3-small"
}
```

**Batch embeddings**:
```
POST /embed/batch
{
  "texts": ["hello", "world", "foo"],
  "model": "text-embedding-3-small"
}
```

## Important Considerations

**Model Versioning**: Different model versions produce different vectors. Cache must be model-specific.

**Normalization**: Need consistent approach to handle "Hello World" vs "hello world" vs "  hello world  ".

**Privacy**: Storing original strings could be sensitive. Consider hash-only storage.

**Storage Costs**: 1536 floats × 4 bytes = 6KB per embedding (or 3KB for nomic-embed's 768 dims).

## Future Extensions

- Multiple model support
- Similarity search on cached embeddings
- Pre-seeded common phrases/words
- Client libraries (Python, JS)
- Self-hostable Docker image
- Federated network of nodes (DHT/gossip protocol)

## Development Notes

When implementing, test with semantic-tarot as the first client to measure real-world cache hit rates.

Self-hosted nomic-embed-text-v2 eliminates per-request API costs and provides full control over the pipeline.
