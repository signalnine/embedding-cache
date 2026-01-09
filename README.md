# embedding-cache

Local and remote caching for embedding vectors.

## Installation

```bash
pip install embedding-cache
```

With local model support:

```bash
pip install embedding-cache[local]
```

## Quick Start

```python
from embedding_cache import embed

# Single string
vector = embed("hello world")

# Multiple strings
vectors = embed(["hello", "world"])
```

## Development

```bash
pip install -e .[dev,local]
pytest
```
