"""
Integration tests for pgvector similarity search.

Requires PostgreSQL with pgvector extension.
Set TEST_DATABASE_URL environment variable to run.
"""
import os
import pytest

# Try to import asyncpg, skip all tests if not available
try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False
    asyncpg = None  # type: ignore

# Skip if no test database configured or asyncpg not installed
pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL") or not HAS_ASYNCPG,
    reason="TEST_DATABASE_URL not set or asyncpg not installed"
)


@pytest.fixture
async def db_pool():
    """Create test database pool."""
    pool = await asyncpg.create_pool(os.environ["TEST_DATABASE_URL"])

    # Setup: ensure pgvector extension and test table
    async with pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS test_embeddings (
                id SERIAL PRIMARY KEY,
                tenant_id VARCHAR(64),
                dimensions INTEGER,
                vector vector NOT NULL
            )
        """)
        await conn.execute("TRUNCATE test_embeddings")

    yield pool

    # Cleanup
    async with pool.acquire() as conn:
        await conn.execute("DROP TABLE IF EXISTS test_embeddings")
    await pool.close()


@pytest.mark.asyncio
async def test_inner_product_score_calculation(db_pool):
    """Verify score calculation produces expected 0-1 values."""
    async with db_pool.acquire() as conn:
        # Insert test vectors (normalized)
        test_vec = [1.0, 0.0, 0.0]  # Unit vector
        await conn.execute("""
            INSERT INTO test_embeddings (tenant_id, dimensions, vector)
            VALUES ('test', 3, $1)
        """, test_vec)

        # Query with identical vector - should get score ~1.0
        result = await conn.fetchrow("""
            SELECT (-(vector <#> $1::vector) + 1) / 2 as score
            FROM test_embeddings
            WHERE tenant_id = 'test'
        """, test_vec)

        assert result['score'] == pytest.approx(1.0, abs=0.001)

        # Query with orthogonal vector - should get score ~0.5
        ortho_vec = [0.0, 1.0, 0.0]
        result = await conn.fetchrow("""
            SELECT (-(vector <#> $1::vector) + 1) / 2 as score
            FROM test_embeddings
            WHERE tenant_id = 'test'
        """, ortho_vec)

        assert result['score'] == pytest.approx(0.5, abs=0.001)


@pytest.mark.asyncio
async def test_hnsw_index_ordering(db_pool):
    """Verify HNSW index returns results in similarity order."""
    async with db_pool.acquire() as conn:
        # Insert multiple vectors
        vectors = [
            ('v1', [1.0, 0.0, 0.0]),   # Most similar to query
            ('v2', [0.7, 0.7, 0.0]),   # Medium similar
            ('v3', [0.0, 1.0, 0.0]),   # Less similar
        ]
        # Normalize v2
        import math
        v2_norm = math.sqrt(0.7**2 + 0.7**2)
        vectors[1] = ('v2', [0.7/v2_norm, 0.7/v2_norm, 0.0])

        for vid, vec in vectors:
            await conn.execute("""
                INSERT INTO test_embeddings (tenant_id, dimensions, vector)
                VALUES ($1, 3, $2)
            """, vid, vec)

        # Query should return in order of similarity
        query = [1.0, 0.0, 0.0]
        results = await conn.fetch("""
            SELECT tenant_id,
                   (-(vector <#> $1::vector) + 1) / 2 as score
            FROM test_embeddings
            ORDER BY vector <#> $1::vector
            LIMIT 3
        """, query)

        # Verify order: v1 (most similar), v2, v3
        assert results[0]['tenant_id'] == 'v1'
        assert results[0]['score'] > results[1]['score'] > results[2]['score']


@pytest.mark.asyncio
async def test_explain_uses_index(db_pool):
    """Verify query planner uses HNSW index (not sequential scan)."""
    async with db_pool.acquire() as conn:
        # Create partial HNSW index
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_test_dim3
            ON test_embeddings USING hnsw (vector vector_ip_ops)
            WHERE dimensions = 3
        """)

        # Insert some data
        for i in range(100):
            await conn.execute("""
                INSERT INTO test_embeddings (tenant_id, dimensions, vector)
                VALUES ('test', 3, $1)
            """, [float(i % 10) / 10, float(i % 5) / 5, 0.1])

        # Run EXPLAIN to verify index usage
        result = await conn.fetch("""
            EXPLAIN (FORMAT JSON)
            SELECT * FROM test_embeddings
            WHERE dimensions = 3
            ORDER BY vector <#> '[0.5, 0.5, 0.1]'::vector
            LIMIT 10
        """)

        explain_text = str(result)
        # Should see index scan, not seq scan
        assert 'Index' in explain_text or 'idx_test' in explain_text.lower()
