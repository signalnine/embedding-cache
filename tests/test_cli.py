"""Tests for CLI tool."""

import subprocess
import sys


def test_cli_stats_command():
    """CLI stats command should run without error."""
    result = subprocess.run(
        [sys.executable, "-m", "embedding_cache.cli", "stats"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Cache Statistics" in result.stdout or "hits" in result.stdout.lower()


def test_cli_info_command():
    """CLI info command should show cache location."""
    result = subprocess.run(
        [sys.executable, "-m", "embedding_cache.cli", "info"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "cache" in result.stdout.lower()


def test_cli_help():
    """CLI should show help."""
    result = subprocess.run(
        [sys.executable, "-m", "embedding_cache.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "embedding-cache" in result.stdout.lower() or "usage" in result.stdout.lower()
