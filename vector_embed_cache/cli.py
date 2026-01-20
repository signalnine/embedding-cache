#!/usr/bin/env python3
"""Command-line interface for vector-embed-cache."""

import argparse
import os
import sys
from pathlib import Path


def get_cache_dir() -> Path:
    """Get the cache directory path."""
    cache_dir = os.environ.get("EMBEDDING_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir)
    return Path.home() / ".cache" / "embedding-cache"


def cmd_stats(args):
    """Show cache statistics."""
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
    if size_bytes < 1024:
        size_str = f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        size_str = f"{size_bytes / 1024:.1f} KB"
    else:
        size_str = f"{size_bytes / (1024 * 1024):.1f} MB"

    print(f"Database size: {size_str}")

    # Count entries
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
        count = cursor.fetchone()[0]
        conn.close()
        print(f"Total entries: {count}")
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

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
