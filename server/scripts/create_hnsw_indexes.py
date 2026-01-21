#!/usr/bin/env python3
"""
Create HNSW indexes for pgvector similarity search.

Must be run separately from migrations because CREATE INDEX CONCURRENTLY
cannot run inside a transaction block.

Usage:
    python scripts/create_hnsw_indexes.py --database-url postgresql://...

    # Or use environment variable
    DATABASE_URL=postgresql://... python scripts/create_hnsw_indexes.py
"""
import argparse
import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

SUPPORTED_DIMENSIONS = [768, 1536, 384]
NUM_PARTITIONS = 32
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64


def create_indexes(database_url: str, dimensions: list[int] = None, dry_run: bool = False):
    """Create HNSW partial indexes on all partitions."""
    dims = dimensions or SUPPORTED_DIMENSIONS

    conn = psycopg2.connect(database_url)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    total_indexes = NUM_PARTITIONS * len(dims)
    created = 0

    print(f"Creating {total_indexes} HNSW indexes ({NUM_PARTITIONS} partitions x {len(dims)} dimensions)")
    print(f"HNSW parameters: m={HNSW_M}, ef_construction={HNSW_EF_CONSTRUCTION}")
    print()

    for partition in range(NUM_PARTITIONS):
        for dim in dims:
            index_name = f"idx_p{partition}_dim{dim}"
            table_name = f"embeddings_p{partition}"

            # Check if index already exists
            cur.execute("""
                SELECT 1 FROM pg_indexes
                WHERE indexname = %s
            """, (index_name,))

            if cur.fetchone():
                print(f"  SKIP {index_name} (already exists)")
                created += 1
                continue

            sql = f"""
                CREATE INDEX CONCURRENTLY {index_name}
                ON {table_name} USING hnsw ((vector::vector({dim})) vector_ip_ops)
                WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION})
                WHERE dimensions = {dim}
            """

            if dry_run:
                print(f"  DRY RUN: {index_name}")
            else:
                print(f"  Creating {index_name}...", end=" ", flush=True)
                try:
                    cur.execute(sql)
                    print("OK")
                except Exception as e:
                    print(f"FAILED: {e}")
                    continue

            created += 1

    cur.close()
    conn.close()

    print()
    print(f"Done. {created}/{total_indexes} indexes created.")


def main():
    parser = argparse.ArgumentParser(description="Create HNSW indexes for pgvector")
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="PostgreSQL connection URL"
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=None,
        help=f"Dimensions to create indexes for (default: {SUPPORTED_DIMENSIONS})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without executing"
    )

    args = parser.parse_args()

    if not args.database_url:
        print("Error: --database-url or DATABASE_URL environment variable required")
        sys.exit(1)

    create_indexes(args.database_url, args.dimensions, args.dry_run)


if __name__ == "__main__":
    main()
