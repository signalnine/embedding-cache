#!/usr/bin/env python3
"""
Atomically swap embeddings_v2 to embeddings.

IMPORTANT: Run this AFTER migration is complete and validated.
This is the cutover step that switches live traffic to pgvector.

Usage:
    python scripts/swap_tables.py --database-url postgresql://...
    python scripts/swap_tables.py --dry-run  # Preview only
"""
import argparse
import os
import sys
import psycopg2


def swap_tables(database_url: str, dry_run: bool = False):
    """Atomically swap embeddings tables."""
    conn = psycopg2.connect(database_url)
    cur = conn.cursor()

    # Verify embeddings_v2 exists and has data
    cur.execute("SELECT COUNT(*) FROM embeddings_v2")
    v2_count = cur.fetchone()[0]
    print(f"embeddings_v2 has {v2_count} rows")

    if v2_count == 0:
        print("ERROR: embeddings_v2 is empty. Run migration first.")
        sys.exit(1)

    # Verify migration is complete
    cur.execute("""
        SELECT COUNT(*) FROM migration_progress
        WHERE status != 'completed'
    """)
    incomplete = cur.fetchone()[0]
    if incomplete > 0:
        print(f"ERROR: {incomplete} tenants still migrating. Complete migration first.")
        sys.exit(1)

    swap_sql = """
        BEGIN;

        -- Rename current table to backup
        ALTER TABLE IF EXISTS embeddings RENAME TO embeddings_old;

        -- Rename v2 to primary name
        ALTER TABLE embeddings_v2 RENAME TO embeddings;

        -- Rename partition tables (required for foreign key references)
        -- Note: partition names change from embeddings_pN to match parent

        COMMIT;
    """

    if dry_run:
        print("DRY RUN - Would execute:")
        print(swap_sql)
    else:
        print("Executing atomic table swap...")
        try:
            cur.execute(swap_sql)
            conn.commit()
            print("SUCCESS: Tables swapped.")
            print("  embeddings -> embeddings_old (backup)")
            print("  embeddings_v2 -> embeddings (live)")
            print()
            print("To rollback: python scripts/swap_tables.py --rollback")
        except Exception as e:
            conn.rollback()
            print(f"FAILED: {e}")
            sys.exit(1)

    cur.close()
    conn.close()


def rollback_swap(database_url: str, dry_run: bool = False):
    """Rollback table swap."""
    rollback_sql = """
        BEGIN;
        ALTER TABLE embeddings RENAME TO embeddings_v2;
        ALTER TABLE embeddings_old RENAME TO embeddings;
        COMMIT;
    """

    if dry_run:
        print("DRY RUN - Would execute rollback:")
        print(rollback_sql)
        return

    conn = psycopg2.connect(database_url)
    cur = conn.cursor()
    try:
        cur.execute(rollback_sql)
        conn.commit()
        print("SUCCESS: Rollback complete.")
    except Exception as e:
        conn.rollback()
        print(f"FAILED: {e}")
        sys.exit(1)
    finally:
        cur.close()
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Swap embeddings tables")
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="PostgreSQL connection URL"
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--rollback", action="store_true", help="Undo the swap")

    args = parser.parse_args()

    if not args.database_url:
        print("Error: --database-url or DATABASE_URL required")
        sys.exit(1)

    if args.rollback:
        rollback_swap(args.database_url, args.dry_run)
    else:
        swap_tables(args.database_url, args.dry_run)


if __name__ == "__main__":
    main()
