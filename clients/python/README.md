# vector-embed-cache-client

Python client for the vector-embed-cache API.

## Installation

```bash
pip install vector-embed-cache-client
```

## Quick Start

```python
from vector_embed_client import VectorEmbedClient

# Initialize the client with your API key
client = VectorEmbedClient(
    api_key="your-api-key",
    base_url="https://api.vector-embed-cache.com"  # Optional: defaults to hosted service
)

# Get embeddings for text
embedding = client.embed("Hello, world!")

# Batch embeddings
embeddings = client.embed_batch(["Hello", "World", "Foo"])

# Async usage
import asyncio

async def main():
    async with VectorEmbedClient(api_key="your-api-key") as client:
        embedding = await client.embed_async("Hello, world!")
        embeddings = await client.embed_batch_async(["Hello", "World"])

asyncio.run(main())
```

## Configuration

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `api_key` | `str` | Yes | - | Your API key |
| `base_url` | `str` | No | `https://api.vector-embed-cache.com` | API base URL |
| `timeout` | `float` | No | `30.0` | Request timeout in seconds |
| `model` | `str` | No | `nomic-ai/nomic-embed-text-v1.5` | Embedding model to use |

## Available Models

| Model | Dimensions | Description |
|-------|-----------|-------------|
| `nomic-ai/nomic-embed-text-v1.5` | 768 | Default choice - stable, precise |
| `nomic-ai/nomic-embed-text-v2-moe` | 768 | Alternative - broader associations |
| `openai:text-embedding-3-small` | 1536 | Highest quality (requires OpenAI key via BYOK) |

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linter
ruff check .

# Run type checker
mypy src/
```

## License

MIT
