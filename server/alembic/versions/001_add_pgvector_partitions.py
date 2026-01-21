"""Add pgvector extension and partitioned embeddings table

This migration creates:
1. The pgvector extension for native vector support
2. embeddings_v2 - a hash-partitioned table for tenant isolation
3. 32 hash partitions based on tenant_id
4. migration_progress table for tracking data migration

NOTE: HNSW indexes must be created separately using create_hnsw_indexes.py
because CREATE INDEX CONCURRENTLY cannot run inside a transaction.

Revision ID: 001_pgvector
Revises: None (first migration)
Create Date: 2026-01-21
"""
from alembic import op
import sqlalchemy as sa

revision = '001_pgvector'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create new partitioned table with pgvector native type
    op.execute("""
        CREATE TABLE embeddings_v2 (
            text_hash VARCHAR(64) NOT NULL,
            model VARCHAR(100) NOT NULL,
            model_version VARCHAR(20) NOT NULL,
            tenant_id VARCHAR(64) NOT NULL,
            dimensions INTEGER NOT NULL,
            vector vector NOT NULL,
            original_text TEXT CHECK (length(original_text) <= 10240),
            created_at TIMESTAMP DEFAULT NOW(),
            last_hit_at TIMESTAMP,
            hit_count INTEGER DEFAULT 0,
            PRIMARY KEY (tenant_id, text_hash, model, model_version)
        ) PARTITION BY HASH (tenant_id)
    """)

    # Create 32 hash partitions for tenant distribution
    for i in range(32):
        op.execute(f"""
            CREATE TABLE embeddings_p{i} PARTITION OF embeddings_v2
            FOR VALUES WITH (MODULUS 32, REMAINDER {i})
        """)

    # Create btree index for tenant+model lookups (not HNSW - that's separate)
    op.execute("""
        CREATE INDEX idx_embeddings_v2_tenant_model
        ON embeddings_v2 (tenant_id, model, dimensions)
    """)

    # Create migration progress tracking table
    op.execute("""
        CREATE TABLE migration_progress (
            tenant_id VARCHAR(64) PRIMARY KEY,
            last_created_at TIMESTAMP,
            last_text_hash VARCHAR(64),
            rows_migrated INTEGER DEFAULT 0,
            status VARCHAR(20) DEFAULT 'pending',
            started_at TIMESTAMP DEFAULT NOW(),
            completed_at TIMESTAMP
        )
    """)

    # NOTE: HNSW indexes must be created separately using create_hnsw_indexes.py
    # because CREATE INDEX CONCURRENTLY cannot run inside a transaction


def downgrade():
    op.execute("DROP TABLE IF EXISTS migration_progress")
    op.execute("DROP TABLE IF EXISTS embeddings_v2 CASCADE")
    op.execute("DROP EXTENSION IF EXISTS vector")
