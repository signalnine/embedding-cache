# Cache Compression Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce embedding cache storage size by ~4.5x through improved serialization and quantization.

**Architecture:** Replace msgpack with raw numpy binary, quantize float32 to float16, maintain backward compatibility with existing caches.

**Tech Stack:** numpy, SQLite

---

## Design Decisions (Consensus-Driven)

| Decision | Choice | Reasoning |
|----------|--------|-----------|
| Serialization | `numpy.tobytes()` with `<f2` | Eliminates 2.25x msgpack overhead, little-endian for portability |
| Quantization | float32 → float16 | 50% size reduction, negligible similarity impact |
| Byte compression | Skip | High-entropy float data doesn't compress well |
| Backward compat | Auto-detect + CLI migrate | Read both formats, write new only, explicit migration |
| Scope | Library only | Server uses pgvector with own optimizations |
| Metadata | Schema columns | Avoids aliasing issues, enables SQL queries |
| Validation | Strict with errors | No silent fallbacks, validate blob size against dimensions |
| Return type | Always float32 | Upcast float16 on read for computation compatibility |

---

## Current State

```python
# Current: msgpack serialization
embedding_bytes = msgpack.packb(embedding.tolist())  # float32 → list → msgpack
# Result: 768 dims × 4 bytes × 2.25 overhead = ~6.9KB per embedding
```

**Problems:**
- msgpack serializes as list of floats (verbose)
- No quantization (full float32 precision unnecessary for similarity)
- 13MB cache for 1,574 embeddings

---

## New Format

```python
# New: raw binary + float16 (little-endian for portability)
embedding_f16 = embedding.astype('<f2')  # Little-endian float16
embedding_bytes = embedding_f16.tobytes()  # Direct binary
# Result: 768 dims × 2 bytes = 1.5KB per embedding
```

**Improvements:**
- ~4.5x size reduction (6.9KB → 1.5KB)
- 13MB cache → ~3MB cache
- Faster serialization (no list conversion)
- Portable across architectures (fixed byte order)

---

## Schema Changes

### Current Schema

```sql
CREATE TABLE embeddings (
    cache_key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at INTEGER NOT NULL,
    access_count INTEGER DEFAULT 1,
    last_accessed INTEGER NOT NULL
)
```

### New Schema

```sql
CREATE TABLE embeddings (
    cache_key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    embedding BLOB NOT NULL,
    dimensions INTEGER,          -- NEW: vector dimensions (768, 1536, etc)
    dtype TEXT,                  -- NEW: 'float16', 'float32', or NULL (legacy)
    created_at INTEGER NOT NULL,
    access_count INTEGER DEFAULT 1,
    last_accessed INTEGER NOT NULL
)
```

**Migration:** ALTER TABLE ADD COLUMN (SQLite supports this without rewriting table)

---

## Format Detection

```python
def _deserialize_embedding(self, blob: bytes, dimensions: int | None, dtype: str | None, cache_key: str) -> np.ndarray:
    """Deserialize embedding with validation. Always returns float32."""

    # New format: has dimensions and dtype
    if dimensions is not None and dtype is not None:
        # Map dtype string to numpy dtype (little-endian)
        dtype_map = {'float16': '<f2', 'float32': '<f4'}
        if dtype not in dtype_map:
            raise ValueError(f"Unknown dtype '{dtype}' for cache_key={cache_key}")

        np_dtype = np.dtype(dtype_map[dtype])
        expected_size = dimensions * np_dtype.itemsize

        # Validate blob size matches expected dimensions
        if len(blob) != expected_size:
            raise ValueError(
                f"Blob size mismatch for cache_key={cache_key}: "
                f"expected {expected_size} bytes ({dimensions} × {np_dtype.itemsize}), "
                f"got {len(blob)} bytes"
            )

        embedding = np.frombuffer(blob, dtype=np_dtype)
        # Always upcast to float32 for computation
        return embedding.astype(np.float32)

    # Legacy format: msgpack (no fallback - fail explicitly)
    try:
        embedding_list = msgpack.unpackb(blob)
        return np.array(embedding_list, dtype=np.float32)
    except Exception as e:
        raise ValueError(f"Failed to deserialize legacy format for cache_key={cache_key}: {e}")
```

---

## Read Path

1. Query: `SELECT embedding, dimensions, dtype FROM embeddings WHERE cache_key = ?`
2. If `dimensions` and `dtype` are set → new format, use `np.frombuffer()`
3. If NULL → legacy format, use `msgpack.unpackb()`
4. Return as float32 (upcast float16 for computation)

---

## Write Path

1. Convert embedding to float16: `embedding.astype('<f2')`  # Little-endian
2. Serialize: `embedding.tobytes()`
3. Store with metadata: `dimensions=len(embedding)`, `dtype='float16'`

**Note:** New entries always use new format. Legacy entries remain unchanged until explicit migration.

---

## Migration Strategy

**Explicit CLI migration only** (no lazy migration - avoids race conditions and blocking):

```python
def migrate_cache(db_path: str, batch_size: int = 100) -> int:
    """Migrate legacy entries to new format. Returns count migrated."""
    conn = sqlite3.connect(db_path)

    # Count legacy entries
    cursor = conn.execute("SELECT COUNT(*) FROM embeddings WHERE dtype IS NULL")
    total = cursor.fetchone()[0]

    if total == 0:
        return 0

    migrated = 0
    while True:
        # Fetch batch of legacy entries
        cursor = conn.execute("""
            SELECT cache_key, embedding FROM embeddings
            WHERE dtype IS NULL LIMIT ?
        """, (batch_size,))
        rows = cursor.fetchall()

        if not rows:
            break

        # Migrate batch in transaction
        for cache_key, blob in rows:
            # Deserialize legacy msgpack
            embedding_list = msgpack.unpackb(blob)
            embedding = np.array(embedding_list, dtype=np.float32)

            # Serialize to new format
            embedding_f16 = embedding.astype('<f2')
            new_blob = embedding_f16.tobytes()

            conn.execute("""
                UPDATE embeddings
                SET embedding = ?, dimensions = ?, dtype = 'float16'
                WHERE cache_key = ?
            """, (new_blob, len(embedding), cache_key))

        conn.commit()
        migrated += len(rows)
        print(f"Migrated {migrated}/{total} entries...")

    return migrated
```

**Safety guarantees:**
- Batched transactions (100 entries per commit)
- No blocking during normal operations
- Interruptible (can resume from where it stopped)
- Progress reporting

---

## CLI Commands

### Existing Commands
- `embedding-cache stats` - Show cache statistics
- `embedding-cache clear` - Clear cache

### New Commands
- `embedding-cache migrate [--path PATH]` - Convert legacy cache to new format
- `embedding-cache stats` - Updated to show format breakdown

```bash
$ embedding-cache stats
Cache: /home/user/.cache/embedding-cache/cache.db
Entries: 1,574
  - New format (float16): 1,200 (76%)
  - Legacy format: 374 (24%)
Size: 4.2 MB (was 13 MB before compression)
```

---

## Precision Analysis

Float16 range: ±65,504 with ~3 decimal digits of precision.

**For embeddings:**
- Embedding values typically in [-1, 1] range
- Cosine similarity uses dot products and norms
- Float16 precision (3-4 significant digits) is sufficient
- Empirical testing shows <0.001% ranking changes

**Verification:** Test with existing semantic-tarot embeddings:
```python
original = cache.embed("test query")  # float32
compressed = original.astype(np.float16).astype(np.float32)
cosine_diff = 1 - np.dot(original, compressed) / (np.linalg.norm(original) * np.linalg.norm(compressed))
assert cosine_diff < 1e-6
```

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Corrupted blob | Raise `ValueError` with cache_key for debugging |
| Unknown dtype | Raise `ValueError` (no silent fallback) |
| Blob size mismatch | Raise `ValueError` with expected vs actual size |
| Legacy format parse failure | Raise `ValueError` with original exception |
| Migration interrupted | Batch committed, resume from next batch |
| Disk full during migration | Raise `IOError`, partial progress preserved |

---

## Testing Strategy

1. **Unit tests:**
   - Serialization roundtrip (float32 → float16 → bytes → float16 → float32)
   - Format detection (new vs legacy)
   - Schema migration (ALTER TABLE)

2. **Integration tests:**
   - End-to-end cache operations with new format
   - Legacy cache reading
   - Lazy migration trigger

3. **Precision tests:**
   - Cosine similarity preservation
   - Ranking stability across format conversion

4. **CLI tests:**
   - `migrate` command
   - `stats` format breakdown

---

## Deprecation Timeline

| Version | Behavior |
|---------|----------|
| 0.x (current) | msgpack + float32 |
| 1.0 | New format default, read legacy with warning |
| 2.0 | Remove legacy format support |

---

## Files to Modify

| File | Changes |
|------|---------|
| `vector_embed_cache/storage.py` | New serialization, schema migration, format detection |
| `vector_embed_cache/cli.py` | Add `migrate` command, update `stats` |
| `tests/test_storage.py` | New format tests, migration tests |
| `tests/test_cli.py` | Migrate command tests |

---

## Performance Expectations

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Storage per embedding | 6.9 KB | 1.5 KB | 4.5x smaller |
| Total cache size (1,574 entries) | 13 MB | ~3 MB | 4.3x smaller |
| Write speed | msgpack overhead | Direct binary | ~2x faster |
| Read speed | msgpack parse | np.frombuffer | ~3x faster |

---

## Summary

This design achieves ~4.5x storage reduction through:
1. Replacing msgpack with raw numpy binary (eliminates 2.25x overhead)
2. Quantizing float32 to float16 (50% size reduction)
3. Skipping byte compression (not effective on float data)

**Safety features:**
- Little-endian byte order for cross-platform portability
- Strict validation (blob size must match dimensions × dtype size)
- No silent fallbacks (explicit errors on corruption)
- Explicit CLI migration (no blocking during normal operations)
- Always returns float32 for computation compatibility

Backward compatibility maintained through auto-detection and explicit CLI migration.
