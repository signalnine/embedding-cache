# Phase 1: Library Polish Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make embedding-cache ready for public release on PyPI with professional packaging, documentation, and developer experience.

**Architecture:** No architectural changes - this is polish work on existing functionality. Focus on packaging, docs, CLI tooling, and CI/CD.

**Tech Stack:** Python 3.8+, hatchling (build), pytest (testing), GitHub Actions (CI), click (CLI)

---

## Task 1: Fix Package Metadata

**Files:**
- Modify: `pyproject.toml`

**Step 1: Update author and project metadata**

```toml
[project]
name = "embedding-cache"
version = "0.1.0"
description = "Cache embedding vectors locally with smart fallback to remote compute"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "Gabe", email = "gabe@signalnine.net"}
]
keywords = ["embeddings", "cache", "vectors", "nlp", "machine-learning"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
```

**Step 2: Verify package builds**

Run: `source .venv/bin/activate && pip install build && python -m build --sdist`
Expected: Creates `dist/embedding_cache-0.1.0.tar.gz`

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: update package metadata for PyPI"
```

---

## Task 2: Add Type Hints Marker

**Files:**
- Create: `embedding_cache/py.typed`

**Step 1: Create py.typed marker file**

Create empty file `embedding_cache/py.typed` (signals to type checkers that this package has type hints).

**Step 2: Verify mypy recognizes it**

Run: `source .venv/bin/activate && pip install mypy && python -c "from embedding_cache import embed; print('ok')"`
Expected: No type errors

**Step 3: Commit**

```bash
git add embedding_cache/py.typed
git commit -m "chore: add py.typed marker for type checker support"
```

---

## Task 3: Create Examples Directory

**Files:**
- Create: `examples/basic_usage.py`
- Create: `examples/batch_processing.py`
- Create: `examples/model_comparison.py`

**Step 1: Create basic_usage.py**

```python
#!/usr/bin/env python3
"""Basic usage of embedding-cache."""

from embedding_cache import embed, EmbeddingCache

# Simple function API (uses global singleton)
embedding = embed("Hello, world!")
print(f"Embedding shape: {embedding.shape}")  # (768,)

# Class-based API for more control
cache = EmbeddingCache(model="nomic-ai/nomic-embed-text-v1.5")
embedding = cache.embed("Hello, world!")
print(f"Cache stats: {cache.stats}")

# Second call hits the cache
embedding2 = cache.embed("Hello, world!")
print(f"After cache hit: {cache.stats}")
```

**Step 2: Create batch_processing.py**

```python
#!/usr/bin/env python3
"""Batch processing example for large datasets."""

from embedding_cache import EmbeddingCache

# Sample dataset
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming industries.",
    "Python is a popular programming language.",
    "Vector embeddings capture semantic meaning.",
    "Caching reduces latency and API costs.",
]

# Create cache
cache = EmbeddingCache()

# Batch embed
print(f"Embedding {len(texts)} texts...")
embeddings = cache.embed(texts)

print(f"Generated {len(embeddings)} embeddings")
print(f"Each embedding has {embeddings[0].shape[0]} dimensions")
print(f"Stats: {cache.stats}")

# Second batch call - all cache hits
embeddings2 = cache.embed(texts)
print(f"After re-embedding: {cache.stats}")
```

**Step 3: Create model_comparison.py**

```python
#!/usr/bin/env python3
"""Compare embeddings from different models."""

import numpy as np
from embedding_cache import EmbeddingCache

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Test text
text = "The meaning of life is to find purpose and connection."

# Compare v1.5 and v2-moe (both local, free)
print("Comparing local models...")

cache_v15 = EmbeddingCache(model="nomic-ai/nomic-embed-text-v1.5")
cache_v2 = EmbeddingCache(model="nomic-ai/nomic-embed-text-v2-moe")

emb_v15 = cache_v15.embed(text)
emb_v2 = cache_v2.embed(text)

print(f"v1.5 shape: {emb_v15.shape}")
print(f"v2-moe shape: {emb_v2.shape}")
print(f"Cosine similarity between models: {cosine_similarity(emb_v15, emb_v2):.4f}")

# Note: OpenAI requires API key
# cache_openai = EmbeddingCache(model="openai:text-embedding-3-small")
# emb_openai = cache_openai.embed(text)
# print(f"OpenAI shape: {emb_openai.shape}")  # (1536,)
```

**Step 4: Test examples run**

Run: `source .venv/bin/activate && python examples/basic_usage.py`
Expected: Runs without error, prints embedding shape and stats

**Step 5: Commit**

```bash
git add examples/
git commit -m "docs: add usage examples"
```

---

## Task 4: Add CLI Tool

**Files:**
- Create: `embedding_cache/cli.py`
- Modify: `pyproject.toml` (add entry point)

**Step 1: Write failing test for CLI**

Create `tests/test_cli.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_cli.py -v`
Expected: FAIL (module not found)

**Step 3: Create CLI implementation**

Create `embedding_cache/cli.py`:

```python
#!/usr/bin/env python3
"""Command-line interface for embedding-cache."""

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
    db_path = cache_dir / "embeddings.db"

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

    db_path = cache_dir / "embeddings.db"
    print(f"Database path: {db_path}")
    print(f"Database exists: {db_path.exists()}")


def cmd_clear(args):
    """Clear the cache."""
    cache_dir = get_cache_dir()
    db_path = cache_dir / "embeddings.db"

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
        prog="embedding-cache",
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
```

**Step 4: Add entry point to pyproject.toml**

Add to `pyproject.toml`:

```toml
[project.scripts]
embedding-cache = "embedding_cache.cli:main"
```

**Step 5: Reinstall and run tests**

Run: `source .venv/bin/activate && pip install -e . && pytest tests/test_cli.py -v`
Expected: All tests pass

**Step 6: Commit**

```bash
git add embedding_cache/cli.py tests/test_cli.py pyproject.toml
git commit -m "feat: add CLI tool for cache management"
```

---

## Task 5: Add GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`

**Step 1: Create CI workflow**

```yaml
name: CI

on:
  push:
    branches: [master, main]
  pull_request:
    branches: [master, main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.8", "3.9", "3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev,local]"

      - name: Run tests
        run: pytest tests/ -v --tb=short

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install ruff

      - name: Run linter
        run: ruff check embedding_cache/ tests/
```

**Step 2: Verify YAML is valid**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"`
Expected: No error (or install pyyaml first)

**Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions workflow for testing"
```

---

## Task 6: Add README Badges

**Files:**
- Modify: `README.md`

**Step 1: Add badges to top of README**

Add after the title line:

```markdown
# embedding-cache

[![CI](https://github.com/signalnine/embedding-cache/actions/workflows/ci.yml/badge.svg)](https://github.com/signalnine/embedding-cache/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/embedding-cache.svg)](https://badge.fury.io/py/embedding-cache)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add CI and PyPI badges to README"
```

---

## Task 7: Verify PyPI Readiness

**Files:** None (verification only)

**Step 1: Build distribution**

Run:
```bash
source .venv/bin/activate
pip install build twine
python -m build
```
Expected: Creates `dist/embedding_cache-0.1.0.tar.gz` and `dist/embedding_cache-0.1.0-py3-none-any.whl`

**Step 2: Check distribution**

Run: `twine check dist/*`
Expected: `PASSED` for both files

**Step 3: Test install from wheel**

Run:
```bash
pip uninstall embedding-cache -y
pip install dist/embedding_cache-0.1.0-py3-none-any.whl
python -c "from embedding_cache import embed; print('OK')"
embedding-cache --help
```
Expected: Import works, CLI shows help

**Step 4: Reinstall dev version**

Run: `pip install -e ".[dev,local]"`

**No commit needed** - this is verification only.

---

## Summary

| Task | Description | Files Changed |
|------|-------------|---------------|
| 1 | Fix package metadata | pyproject.toml |
| 2 | Add py.typed marker | embedding_cache/py.typed |
| 3 | Create examples | examples/*.py |
| 4 | Add CLI tool | cli.py, test_cli.py, pyproject.toml |
| 5 | Add GitHub Actions | .github/workflows/ci.yml |
| 6 | Add README badges | README.md |
| 7 | Verify PyPI readiness | (verification only) |

**Total: 7 tasks, ~6 commits**

After completing these tasks:
- Package is ready for `twine upload` to PyPI
- CI runs on every push/PR
- Users can install with `pip install embedding-cache`
- CLI available as `embedding-cache stats|info|clear`
- Examples demonstrate common use cases
