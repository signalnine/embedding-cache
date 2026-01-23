#!/usr/bin/env python3
"""Command-line interface for vector-embed-cache."""

import argparse
import os
import sys
from pathlib import Path

from .preseed import get_preseed_db_path, preseed_db_exists


def get_cache_dir() -> Path:
    """Get the cache directory path."""
    cache_dir = os.environ.get("EMBEDDING_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir)
    return Path.home() / ".cache" / "embedding-cache"


def _format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def cmd_stats(args):
    """Show cache statistics with format breakdown."""
    cache_dir = get_cache_dir()
    db_path = cache_dir / "cache.db"

    print("Cache Statistics")
    print("=" * 40)
    print(f"Cache directory: {cache_dir}")

    if not db_path.exists():
        print("Status: No cache database found")
        print("Total entries: 0")
        return

    # Get file size
    size_bytes = db_path.stat().st_size
    print(f"Database size: {_format_size(size_bytes)}")

    # Count entries with format breakdown
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)

        # Check if dtype column exists (legacy database)
        cursor = conn.execute("PRAGMA table_info(embeddings)")
        columns = {row[1] for row in cursor.fetchall()}
        has_dtype = "dtype" in columns

        # Total count
        cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
        total = cursor.fetchone()[0]
        print(f"Total entries: {total}")

        if total > 0 and has_dtype:
            # Count by format
            cursor = conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE dtype IS NOT NULL AND dtype != 'failed'"
            )
            new_format = cursor.fetchone()[0]
            legacy = total - new_format

            if new_format > 0:
                pct = (new_format / total) * 100
                print(f"  - New format (float16): {new_format} ({pct:.0f}%)")
            if legacy > 0:
                pct = (legacy / total) * 100
                print(f"  - Legacy format: {legacy} ({pct:.0f}%)")
        elif total > 0:
            print(f"  - Legacy format: {total} (100%)")

        conn.close()
    except Exception as e:
        print(f"Error reading database: {e}")


def cmd_info(args):
    """Show cache configuration."""
    cache_dir = get_cache_dir()

    print("Cache Configuration")
    print("=" * 40)
    print(f"Cache directory: {cache_dir}")
    print(f"Directory exists: {cache_dir.exists()}")

    env_var = os.environ.get("EMBEDDING_CACHE_DIR")
    if env_var:
        print(f"EMBEDDING_CACHE_DIR: {env_var}")
    else:
        print("EMBEDDING_CACHE_DIR: (not set, using default)")

    db_path = cache_dir / "cache.db"
    print(f"Database path: {db_path}")
    print(f"Database exists: {db_path.exists()}")


def cmd_clear(args):
    """Clear the cache."""
    cache_dir = get_cache_dir()
    db_path = cache_dir / "cache.db"

    if not db_path.exists():
        print("No cache to clear.")
        return

    if not args.yes:
        response = input(f"Delete {db_path}? [y/N] ")
        if response.lower() != "y":
            print("Cancelled.")
            return

    db_path.unlink()
    print(f"Deleted: {db_path}")


def _ensure_schema_columns(conn):
    """Ensure dimensions and dtype columns exist (for legacy databases)."""
    cursor = conn.execute("PRAGMA table_info(embeddings)")
    columns = {row[1] for row in cursor.fetchall()}

    if "dimensions" not in columns:
        conn.execute("ALTER TABLE embeddings ADD COLUMN dimensions INTEGER")
    if "dtype" not in columns:
        conn.execute("ALTER TABLE embeddings ADD COLUMN dtype TEXT")
    conn.commit()


def cmd_migrate(args):
    """Migrate legacy cache entries to new float16 format."""
    import sqlite3
    import msgpack
    import numpy as np

    # Validate batch size
    if args.batch_size <= 0:
        print("Error: batch-size must be positive")
        sys.exit(1)

    # Determine database path
    if args.path:
        db_path = Path(args.path)
    else:
        cache_dir = get_cache_dir()
        db_path = cache_dir / "cache.db"

    if not db_path.exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)

    print(f"Migrating: {db_path}")

    conn = sqlite3.connect(db_path)

    # Ensure schema has new columns (for legacy databases)
    _ensure_schema_columns(conn)

    # Count legacy entries
    cursor = conn.execute("SELECT COUNT(*) FROM embeddings WHERE dtype IS NULL")
    total = cursor.fetchone()[0]

    if total == 0:
        print("No legacy entries to migrate.")
        conn.close()
        return

    print(f"Found {total} legacy entries to migrate.")

    batch_size = args.batch_size
    migrated = 0
    failed = 0

    while True:
        # Fetch batch of legacy entries
        cursor = conn.execute("""
            SELECT cache_key, embedding FROM embeddings
            WHERE dtype IS NULL LIMIT ?
        """, (batch_size,))
        rows = cursor.fetchall()

        if not rows:
            break

        batch_migrated = 0
        batch_failed = 0

        # Migrate batch in transaction
        for cache_key, blob in rows:
            try:
                # Deserialize legacy msgpack
                embedding_list = msgpack.unpackb(blob)
                embedding = np.array(embedding_list, dtype=np.float32)

                # Serialize to new format (little-endian float16)
                embedding_f16 = embedding.astype("<f2")
                new_blob = embedding_f16.tobytes()

                conn.execute("""
                    UPDATE embeddings
                    SET embedding = ?, dimensions = ?, dtype = 'float16'
                    WHERE cache_key = ?
                """, (new_blob, len(embedding), cache_key))
                batch_migrated += 1
            except Exception as e:
                # Mark as failed to prevent infinite loop
                conn.execute("""
                    UPDATE embeddings
                    SET dtype = 'failed'
                    WHERE cache_key = ?
                """, (cache_key,))
                print(f"Warning: Failed to migrate {cache_key}: {e}")
                batch_failed += 1

        conn.commit()
        migrated += batch_migrated
        failed += batch_failed
        print(f"Migrated {migrated}/{total} entries...")

    conn.close()
    if failed > 0:
        print(f"Migration complete. {migrated} entries converted, {failed} failed.")
    else:
        print(f"Migration complete. {migrated} entries converted.")


def cmd_preseed_status(args):
    """Show preseed database status."""
    print("Preseed Status")
    print("=" * 40)

    if not preseed_db_exists():
        print("Status: No preseed database found")
        print("The preseed database is not bundled with this installation.")
        return

    preseed_path = get_preseed_db_path()
    print(f"Database: {preseed_path}")

    # Get file size
    size_bytes = preseed_path.stat().st_size
    if size_bytes < 1024:
        size_str = f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        size_str = f"{size_bytes / 1024:.1f} KB"
    else:
        size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
    print(f"Size: {size_str}")

    # Count entries and show metadata
    try:
        import sqlite3

        conn = sqlite3.connect(f"file:{preseed_path}?mode=ro", uri=True)

        cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
        count = cursor.fetchone()[0]
        print(f"Entries: {count}")

        # Show metadata
        cursor = conn.execute("SELECT key, value FROM metadata")
        for key, value in cursor.fetchall():
            print(f"{key}: {value}")

        conn.close()
    except Exception as e:
        print(f"Error reading database: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="vector-embed-cache",
        description="Manage embedding cache",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show cache statistics")
    stats_parser.set_defaults(func=cmd_stats)

    # info command
    info_parser = subparsers.add_parser("info", help="Show cache configuration")
    info_parser.set_defaults(func=cmd_info)

    # clear command
    clear_parser = subparsers.add_parser("clear", help="Clear the cache")
    clear_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    clear_parser.set_defaults(func=cmd_clear)

    # migrate command
    migrate_parser = subparsers.add_parser(
        "migrate", help="Migrate legacy entries to float16 format"
    )
    migrate_parser.add_argument(
        "--path", help="Path to database (default: user cache)"
    )
    migrate_parser.add_argument(
        "--batch-size", type=int, default=100,
        help="Entries per batch (default: 100)"
    )
    migrate_parser.set_defaults(func=cmd_migrate)

    # preseed command group
    preseed_parser = subparsers.add_parser("preseed", help="Manage preseed database")
    preseed_subparsers = preseed_parser.add_subparsers(dest="preseed_command")

    # preseed status
    preseed_status_parser = preseed_subparsers.add_parser("status", help="Show preseed status")
    preseed_status_parser.set_defaults(func=cmd_preseed_status)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Handle preseed subcommands
    if args.command == "preseed":
        if args.preseed_command is None:
            preseed_parser.print_help()
            sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
