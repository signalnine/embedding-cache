"""Tests for CLI tool."""

import subprocess
import sys


def test_cli_stats_command():
    """CLI stats command should run without error."""
    result = subprocess.run(
        [sys.executable, "-m", "vector_embed_cache.cli", "stats"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Cache Statistics" in result.stdout or "hits" in result.stdout.lower()


def test_cli_info_command():
    """CLI info command should show cache location."""
    result = subprocess.run(
        [sys.executable, "-m", "vector_embed_cache.cli", "info"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "cache" in result.stdout.lower()


def test_cli_help():
    """CLI should show help."""
    result = subprocess.run(
        [sys.executable, "-m", "vector_embed_cache.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "embedding-cache" in result.stdout.lower() or "usage" in result.stdout.lower()


def test_cli_migrate_command_exists():
    """CLI migrate command should be available."""
    result = subprocess.run(
        [sys.executable, "-m", "vector_embed_cache.cli", "migrate", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "float16" in result.stdout.lower() or "migrate" in result.stdout.lower()


class TestPreseedCLI:
    def test_preseed_status_command_exists(self):
        """Test that preseed status command is available."""
        result = subprocess.run(
            [sys.executable, "-m", "vector_embed_cache.cli", "preseed", "status"],
            capture_output=True,
            text=True,
        )
        # Should not fail with "invalid choice"
        assert "invalid choice" not in result.stderr
        assert "Preseed" in result.stdout or "preseed" in result.stdout.lower()


class TestMigrateCLI:
    def test_migrate_no_legacy_entries(self, temp_cache_dir):
        """Migrate with no legacy entries should report nothing to migrate."""
        import sqlite3
        from pathlib import Path

        db_path = Path(temp_cache_dir) / "cache.db"

        # Create database with new schema and one new-format entry
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE embeddings (
                cache_key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                embedding BLOB NOT NULL,
                dimensions INTEGER,
                dtype TEXT,
                created_at INTEGER NOT NULL,
                access_count INTEGER DEFAULT 1,
                last_accessed INTEGER NOT NULL
            )
        """)
        conn.execute("""
            INSERT INTO embeddings VALUES
            ('key1', 'model', X'00000000', 2, 'float16', 0, 1, 0)
        """)
        conn.commit()
        conn.close()

        result = subprocess.run(
            [sys.executable, "-m", "vector_embed_cache.cli", "migrate", "--path", str(db_path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "No legacy entries" in result.stdout

    def test_migrate_converts_legacy(self, temp_cache_dir):
        """Migrate should convert legacy msgpack entries to float16."""
        import sqlite3
        import msgpack
        import numpy as np
        from pathlib import Path

        db_path = Path(temp_cache_dir) / "cache.db"

        # Create database with legacy entries
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE embeddings (
                cache_key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                embedding BLOB NOT NULL,
                dimensions INTEGER,
                dtype TEXT,
                created_at INTEGER NOT NULL,
                access_count INTEGER DEFAULT 1,
                last_accessed INTEGER NOT NULL
            )
        """)

        # Insert 3 legacy entries
        for i in range(3):
            legacy_blob = msgpack.packb([0.1 * (i + 1)] * 10)
            conn.execute("""
                INSERT INTO embeddings VALUES
                (?, 'model', ?, NULL, NULL, 0, 1, 0)
            """, (f"key{i}", legacy_blob))
        conn.commit()
        conn.close()

        result = subprocess.run(
            [sys.executable, "-m", "vector_embed_cache.cli", "migrate", "--path", str(db_path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "3 entries converted" in result.stdout

        # Verify entries were converted
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM embeddings WHERE dtype = 'float16'")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 3

    def test_stats_shows_format_breakdown(self, temp_cache_dir):
        """Stats should show new format vs legacy breakdown."""
        import sqlite3
        import msgpack
        from pathlib import Path

        db_path = Path(temp_cache_dir) / "cache.db"

        # Create database with mixed entries
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE embeddings (
                cache_key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                embedding BLOB NOT NULL,
                dimensions INTEGER,
                dtype TEXT,
                created_at INTEGER NOT NULL,
                access_count INTEGER DEFAULT 1,
                last_accessed INTEGER NOT NULL
            )
        """)

        # 2 new format entries
        for i in range(2):
            conn.execute("""
                INSERT INTO embeddings VALUES
                (?, 'model', X'00000000', 2, 'float16', 0, 1, 0)
            """, (f"new{i}",))

        # 1 legacy entry
        legacy_blob = msgpack.packb([0.1, 0.2])
        conn.execute("""
            INSERT INTO embeddings VALUES
            ('legacy1', 'model', ?, NULL, NULL, 0, 1, 0)
        """, (legacy_blob,))
        conn.commit()
        conn.close()

        import os
        env = os.environ.copy()
        env["EMBEDDING_CACHE_DIR"] = temp_cache_dir

        result = subprocess.run(
            [sys.executable, "-m", "vector_embed_cache.cli", "stats"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0
        assert "New format (float16): 2" in result.stdout
        assert "Legacy format: 1" in result.stdout
