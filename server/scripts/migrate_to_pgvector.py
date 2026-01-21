#!/usr/bin/env python3
"""
Migrate embeddings from LargeBinary to pgvector format.

Features:
- Batched processing to avoid memory issues
- Checkpoint/resume capability for interrupted migrations
- Per-tenant progress tracking

Usage:
    python scripts/migrate_to_pgvector.py --database-url postgresql://...
    python scripts/migrate_to_pgvector.py --tenant-id specific-tenant
"""
import argparse
import asyncio
import os
import struct
import sys
from datetime import datetime

import asyncpg

# Model -> dimensions mapping
MODEL_DIMENSIONS = {
    "nomic-ai/nomic-embed-text-v1.5": 768,
    "nomic-v1.5": 768,
    "openai:text-embedding-3-small": 1536,
    "text-embedding-3-small": 1536,
    "all-MiniLM-L6-v2": 384,
}


def decode_vector(blob: bytes) -> list[float]:
    """Decode struct.pack'd vector from LargeBinary."""
    num_floats = len(blob) // 4
    return list(struct.unpack(f'{num_floats}f', blob))


def get_dimensions(model: str, vector: list[float]) -> int:
    """Get dimensions from model name or vector length."""
    if model in MODEL_DIMENSIONS:
        return MODEL_DIMENSIONS[model]
    return len(vector)


async def migrate_tenant(pool: asyncpg.Pool, tenant_id: str, batch_size: int = 1000):
    """Migrate a single tenant's embeddings."""
    async with pool.acquire() as conn:
        # Check existing progress
        progress = await conn.fetchrow("""
            SELECT last_created_at, last_text_hash, rows_migrated, status
            FROM migration_progress
            WHERE tenant_id = $1
        """, tenant_id)

        if progress and progress['status'] == 'completed':
            print(f"  Tenant {tenant_id}: already completed ({progress['rows_migrated']} rows)")
            return progress['rows_migrated']

        last_created_at = progress['last_created_at'] if progress else None
        last_text_hash = progress['last_text_hash'] if progress else None
        total_migrated = progress['rows_migrated'] if progress else 0

        # Initialize progress record
        if not progress:
            await conn.execute("""
                INSERT INTO migration_progress (tenant_id, status)
                VALUES ($1, 'in_progress')
            """, tenant_id)
        else:
            await conn.execute("""
                UPDATE migration_progress SET status = 'in_progress'
                WHERE tenant_id = $1
            """, tenant_id)

        batch_num = 0
        while True:
            # Fetch batch using composite cursor
            # Note: source table 'embeddings' does not have original_text column
            if last_created_at:
                rows = await conn.fetch("""
                    SELECT text_hash, model, model_version, vector,
                           created_at, last_hit_at, hit_count
                    FROM embeddings
                    WHERE tenant_id = $1
                      AND (created_at, text_hash) > ($2, $3)
                    ORDER BY created_at, text_hash
                    LIMIT $4
                """, tenant_id, last_created_at, last_text_hash, batch_size)
            else:
                rows = await conn.fetch("""
                    SELECT text_hash, model, model_version, vector,
                           created_at, last_hit_at, hit_count
                    FROM embeddings
                    WHERE tenant_id = $1
                    ORDER BY created_at, text_hash
                    LIMIT $2
                """, tenant_id, batch_size)

            if not rows:
                break

            # Transform and insert
            insert_data = []
            for row in rows:
                vector = decode_vector(row['vector'])
                dimensions = get_dimensions(row['model'], vector)

                insert_data.append((
                    row['text_hash'],
                    row['model'],
                    row['model_version'],
                    tenant_id,
                    dimensions,
                    vector,  # asyncpg handles list -> vector conversion
                    None,    # original_text not in source table
                    row['created_at'],
                    row['last_hit_at'],
                    row['hit_count']
                ))

            # Batch insert
            await conn.executemany("""
                INSERT INTO embeddings_v2
                (text_hash, model, model_version, tenant_id, dimensions, vector,
                 original_text, created_at, last_hit_at, hit_count)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT DO NOTHING
            """, insert_data)

            # Update checkpoint
            last_created_at = rows[-1]['created_at']
            last_text_hash = rows[-1]['text_hash']
            total_migrated += len(rows)

            await conn.execute("""
                UPDATE migration_progress
                SET last_created_at = $2, last_text_hash = $3, rows_migrated = $4
                WHERE tenant_id = $1
            """, tenant_id, last_created_at, last_text_hash, total_migrated)

            batch_num += 1
            if batch_num % 10 == 0:
                print(f"    Tenant {tenant_id}: {total_migrated} rows migrated...")

        # Mark complete
        await conn.execute("""
            UPDATE migration_progress
            SET status = 'completed', completed_at = NOW()
            WHERE tenant_id = $1
        """, tenant_id)

        print(f"  Tenant {tenant_id}: completed ({total_migrated} rows)")
        return total_migrated


async def migrate_all(database_url: str, batch_size: int = 1000, tenant_id: str = None):
    """Run migration for all tenants or a specific tenant."""
    pool = await asyncpg.create_pool(database_url)

    try:
        if tenant_id:
            tenants = [tenant_id]
        else:
            # Get all distinct tenants
            rows = await pool.fetch("""
                SELECT DISTINCT tenant_id FROM embeddings ORDER BY tenant_id
            """)
            tenants = [r['tenant_id'] for r in rows]

        print(f"Migrating {len(tenants)} tenant(s)...")
        total = 0

        for tid in tenants:
            migrated = await migrate_tenant(pool, tid, batch_size)
            total += migrated

        print(f"\nMigration complete. Total rows: {total}")

    finally:
        await pool.close()


def main():
    parser = argparse.ArgumentParser(description="Migrate embeddings to pgvector")
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="PostgreSQL connection URL"
    )
    parser.add_argument(
        "--tenant-id",
        help="Migrate only this tenant (default: all tenants)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Rows per batch (default: 1000)"
    )

    args = parser.parse_args()

    if not args.database_url:
        print("Error: --database-url or DATABASE_URL required")
        sys.exit(1)

    asyncio.run(migrate_all(args.database_url, args.batch_size, args.tenant_id))


if __name__ == "__main__":
    main()
