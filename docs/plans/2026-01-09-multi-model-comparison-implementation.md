# Multi-Model Comparison Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add OpenAI provider support to embedding-cache and extend semantic-tarot with model comparison capabilities.

**Architecture:** Add OpenAIProvider class following LocalProvider/RemoteProvider pattern with lazy loading. Modify EmbeddingCache to route to appropriate provider based on model prefix. Extend semantic-tarot scripts with --model flags for easy model switching.

**Tech Stack:** Python 3.8+, openai>=1.0.0, pytest, pytest-mock

---

## Task 1: OpenAI Provider - Core Implementation

**Files:**
- Modify: `embedding_cache/providers.py` (add OpenAIProvider class after RemoteProvider)
- Test: `tests/test_providers.py`

**Step 1: Write failing test for OpenAI provider availability check**

```python
# Add to tests/test_providers.py after existing tests

def test_openai_provider_unavailable():
    """Should detect when openai package is not installed."""
    with patch('embedding_cache.providers.OpenAIProvider.is_available', return_value=False):
        from embedding_cache.providers import OpenAIProvider
        provider = OpenAIProvider(model="text-embedding-3-small", api_key="test-key")
        assert provider.is_available() == False


def test_openai_provider_no_api_key():
    """Should detect when API key is not set."""
    import os
    with patch.dict(os.environ, {}, clear=True):
        from embedding_cache.providers import OpenAIProvider
        provider = OpenAIProvider(model="text-embedding-3-small")
        assert provider.is_available() == False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_providers.py::test_openai_provider_unavailable tests/test_providers.py::test_openai_provider_no_api_key -v`

Expected: ImportError (OpenAIProvider doesn't exist yet)

**Step 3: Write minimal OpenAIProvider class**

Add to `embedding_cache/providers.py` after RemoteProvider class:

```python
class OpenAIProvider:
    """OpenAI embedding provider via API."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        """Initialize OpenAI provider.

        Args:
            model: OpenAI model name (e.g., "text-embedding-3-small")
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.model_name = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client: Optional[object] = None

    def is_available(self) -> bool:
        """Check if openai package is installed and API key is set."""
        try:
            import openai
            return self.api_key is not None
        except ImportError:
            return False

    def _load_client(self):
        """Lazy load OpenAI client."""
        if self._client is not None:
            return

        if not self.is_available():
            raise RuntimeError(
                "OpenAI not available. Install with: pip install embedding-cache[openai] "
                "and set OPENAI_API_KEY environment variable"
            )

        from openai import OpenAI
        logger.info(f"Initializing OpenAI client for model {self.model_name}")
        self._client = OpenAI(api_key=self.api_key)

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string via OpenAI API.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        self._load_client()
        response = self._client.embeddings.create(
            input=[text],
            model=self.model_name
        )
        embedding = response.data[0].embedding
        return np.array(embedding, dtype=np.float32)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts (OpenAI supports up to 2048 per request).

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        self._load_client()
        response = self._client.embeddings.create(
            input=texts,
            model=self.model_name
        )
        embeddings = [np.array(item.embedding, dtype=np.float32) for item in response.data]
        return embeddings
```

Add import at top of file:
```python
import os  # Add if not already present
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_providers.py::test_openai_provider_unavailable tests/test_providers.py::test_openai_provider_no_api_key -v`

Expected: PASS (both tests)

**Step 5: Commit**

```bash
git add embedding_cache/providers.py tests/test_providers.py
git commit -m "feat: add OpenAIProvider class with availability checks"
```

---

## Task 2: OpenAI Provider - Mock API Tests

**Files:**
- Test: `tests/test_providers.py`

**Step 1: Write failing test for OpenAI embed single text**

Add to `tests/test_providers.py`:

```python
def test_openai_provider_embed_single():
    """Should embed single text via OpenAI API."""
    from embedding_cache.providers import OpenAIProvider

    # Mock OpenAI client
    mock_client = Mock()
    mock_response = Mock()
    mock_response.data = [Mock(embedding=[0.1] * 1536)]  # OpenAI returns 1536 dims
    mock_client.embeddings.create.return_value = mock_response

    provider = OpenAIProvider(model="text-embedding-3-small", api_key="test-key")

    with patch('embedding_cache.providers.OpenAIProvider._load_client'):
        provider._client = mock_client
        embedding = provider.embed("hello world")

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (1536,)
    assert embedding.dtype == np.float32
    mock_client.embeddings.create.assert_called_once_with(
        input=["hello world"],
        model="text-embedding-3-small"
    )


def test_openai_provider_embed_batch():
    """Should embed multiple texts via OpenAI API."""
    from embedding_cache.providers import OpenAIProvider

    # Mock OpenAI client
    mock_client = Mock()
    mock_response = Mock()
    mock_response.data = [
        Mock(embedding=[0.1] * 1536),
        Mock(embedding=[0.2] * 1536),
    ]
    mock_client.embeddings.create.return_value = mock_response

    provider = OpenAIProvider(model="text-embedding-3-small", api_key="test-key")

    with patch('embedding_cache.providers.OpenAIProvider._load_client'):
        provider._client = mock_client
        embeddings = provider.embed_batch(["hello", "world"])

    assert isinstance(embeddings, list)
    assert len(embeddings) == 2
    assert all(isinstance(e, np.ndarray) for e in embeddings)
    assert all(e.shape == (1536,) for e in embeddings)
    mock_client.embeddings.create.assert_called_once_with(
        input=["hello", "world"],
        model="text-embedding-3-small"
    )
```

**Step 2: Run tests to verify they pass**

Run: `pytest tests/test_providers.py::test_openai_provider_embed_single tests/test_providers.py::test_openai_provider_embed_batch -v`

Expected: PASS (implementation already complete from Task 1)

**Step 3: Commit**

```bash
git add tests/test_providers.py
git commit -m "test: add mock tests for OpenAI provider embed methods"
```

---

## Task 3: Provider Selection in EmbeddingCache

**Files:**
- Modify: `embedding_cache/cache.py:56-57` (update provider initialization)
- Modify: `embedding_cache/providers.py` (add OpenAIProvider to imports in cache.py)
- Test: `tests/test_cache.py`

**Step 1: Write failing test for OpenAI provider selection**

Add to `tests/test_cache.py`:

```python
def test_cache_selects_openai_provider():
    """Should select OpenAIProvider when model starts with 'openai:'."""
    from embedding_cache import EmbeddingCache
    from embedding_cache.providers import OpenAIProvider

    cache = EmbeddingCache(model="openai:text-embedding-3-small")

    assert isinstance(cache.local_provider, OpenAIProvider)
    assert cache.local_provider.model_name == "text-embedding-3-small"


def test_cache_selects_local_provider_for_nomic():
    """Should select LocalProvider for nomic models."""
    from embedding_cache import EmbeddingCache
    from embedding_cache.providers import LocalProvider

    cache = EmbeddingCache(model="nomic-ai/nomic-embed-text-v1.5")

    assert isinstance(cache.local_provider, LocalProvider)
    assert cache.local_provider.model_name == "nomic-ai/nomic-embed-text-v1.5"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cache.py::test_cache_selects_openai_provider tests/test_cache.py::test_cache_selects_local_provider_for_nomic -v`

Expected: FAIL (AssertionError - local_provider is still LocalProvider)

**Step 3: Modify EmbeddingCache.__init__() for provider selection**

In `embedding_cache/cache.py`, update the imports at the top:

```python
from .providers import LocalProvider, RemoteProvider, OpenAIProvider
```

Then modify lines 55-57 in `__init__`:

```python
        # Initialize providers based on model prefix
        if model.startswith("openai:"):
            # Extract actual model name: "openai:text-embedding-3-small" -> "text-embedding-3-small"
            actual_model = model.split(":", 1)[1]
            self.local_provider = OpenAIProvider(model=actual_model)
        else:
            # Local model (nomic-embed or other sentence-transformers model)
            self.local_provider = LocalProvider(model=model)

        self.remote_provider = RemoteProvider(remote_url, timeout=timeout, model=model) if remote_url else None
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cache.py::test_cache_selects_openai_provider tests/test_cache.py::test_cache_selects_local_provider_for_nomic -v`

Expected: PASS

**Step 5: Commit**

```bash
git add embedding_cache/cache.py embedding_cache/providers.py tests/test_cache.py
git commit -m "feat: add provider selection based on model prefix in EmbeddingCache"
```

---

## Task 4: Add OpenAI to Optional Dependencies

**Files:**
- Modify: `pyproject.toml:21-26` (add openai extra)

**Step 1: Add openai to optional dependencies**

In `pyproject.toml`, update the `[project.optional-dependencies]` section:

```toml
[project.optional-dependencies]
local = [
    "sentence-transformers>=2.3",
    "torch>=2.0",
    "einops>=0.8",  # Required by nomic-embed models
]
openai = [
    "openai>=1.0.0",
]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "pytest-mock>=3.10",
]
```

**Step 2: Verify installation command works**

Run: `pip install -e ".[openai]" --dry-run`

Expected: Shows openai>=1.0.0 would be installed

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add openai as optional dependency"
```

---

## Task 5: Update README with OpenAI Usage

**Files:**
- Modify: `README.md` (add OpenAI section after "With local model support")

**Step 1: Add OpenAI installation and usage section**

In `README.md`, after line 62 (the `pip install embedding-cache[local]` section), add:

```markdown
With OpenAI support:

\```bash
pip install embedding-cache[openai]
\```

With both local and OpenAI support:

\```bash
pip install embedding-cache[local,openai]
\```

## Usage with OpenAI

Set your API key:

\```bash
export OPENAI_API_KEY=your-key-here
\```

Use OpenAI embeddings:

\```python
from embedding_cache import EmbeddingCache

# Create cache with OpenAI model (note the "openai:" prefix)
cache = EmbeddingCache(model="openai:text-embedding-3-small")

# Use it like any other provider
vector = cache.embed("hello world")
print(vector.shape)  # (1536,)

# Subsequent calls hit the cache
vector2 = cache.embed("hello world")
print(cache.stats)
# {'hits': 1, 'misses': 1, 'remote_hits': 0}
\```

The OpenAI provider:
- Uses the OpenAI API with automatic retries
- Caches embeddings locally just like other providers
- Requires `OPENAI_API_KEY` environment variable
- Supports batch embedding (up to 2048 texts per request)
- Returns 1536-dimensional embeddings for text-embedding-3-small

```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add OpenAI usage section to README"
```

---

## Task 6: Semantic-Tarot Model Selection - generate_embeddings_cached.py

**Files:**
- Modify: `/home/gabe/semantic-tarot/generate_embeddings_cached.py`

**Step 1: Add --model argument parsing**

In `generate_embeddings_cached.py`, locate the `main()` function and add after imports (around line 27):

```python
# Model name mappings
MODEL_MAP = {
    'v1.5': 'nomic-ai/nomic-embed-text-v1.5',
    'v2-moe': 'nomic-ai/nomic-embed-text-v2-moe',
    'openai': 'openai:text-embedding-3-small'
}
```

Then modify the `main()` function to add argument parsing before `print("=" * 70)` line:

```python
def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate embeddings with model selection'
    )
    parser.add_argument(
        '--model',
        choices=['v1.5', 'v2-moe', 'openai'],
        default=os.environ.get('EMBEDDING_MODEL', 'v1.5'),
        help='Embedding model to use (default: v1.5, or EMBEDDING_MODEL env var)'
    )
    args = parser.parse_args()

    # Get full model name
    model_name = MODEL_MAP[args.model]
    output_file = f'card_embeddings_{args.model.replace("-", "_")}.json'

    print("=" * 70)
    print(f"Tarot Card Embedding Generator (with embedding-cache)")
    print(f"Using model: {model_name}")
    print("=" * 70)
```

**Step 2: Update output file path**

Change line near the bottom where `EMBEDDINGS_OUTPUT_FILE` is used (around line 227):

```python
# Replace:
EMBEDDINGS_OUTPUT_FILE = 'card_embeddings_cached.json'

# With (at the top, line 27):
# Remove EMBEDDINGS_OUTPUT_FILE constant - we'll use output_file from args
```

Then update the `generate_embeddings()` call to pass the model:

```python
def generate_embeddings(cards: List[Dict], interpretations: Dict, model: str) -> List[Dict]:
    """Generate embeddings for all cards (both upright and reversed) for each interpretation system.

    Args:
        cards: List of card dictionaries
        interpretations: Interpretation data
        model: Full model name (e.g., "nomic-ai/nomic-embed-text-v1.5")

    This version uses embedding-cache which:
    - Computes embeddings locally (no API cost for local models)
    - Caches results automatically
    - Works offline once model is downloaded

    Returns:
        List of embedding records with metadata
    """
```

Update the embed() call around line 178:

```python
    try:
        # This single call handles everything: normalization, caching, local computation
        # Pass model explicitly to use the selected model
        from embedding_cache import EmbeddingCache
        cache = EmbeddingCache(model=model)
        embeddings = cache.embed(texts_to_embed)
```

Update the main() function to pass model and output_file:

```python
    # Generate embeddings
    print("Generating embeddings...")
    print(f"(Creating {len(cards) * 2 * 5} embeddings: {len(cards)} cards × 2 positions × 5 systems)")
    print("Systems: Traditional, Crowley, Jungian, Modern, Combined")
    print()
    embeddings_data = generate_embeddings(cards, interpretations, model_name)

    # Save results
    save_embeddings(embeddings_data, output_file)
```

**Step 3: Test the changes**

Run: `cd /home/gabe/semantic-tarot && python3 generate_embeddings_cached.py --help`

Expected: Help message showing --model option with choices [v1.5, v2-moe, openai]

**Step 4: Commit**

```bash
cd /home/gabe/semantic-tarot
git add generate_embeddings_cached.py
git commit -m "feat: add --model flag to generate_embeddings_cached.py for model selection"
```

---

## Task 7: Semantic-Tarot Model Selection - search_cards_cached.py

**Files:**
- Modify: `/home/gabe/semantic-tarot/search_cards_cached.py`

**Step 1: Add model argument and auto-detect embeddings file**

In `search_cards_cached.py`, add after imports (around line 52):

```python
# Model name mappings
MODEL_MAP = {
    'v1.5': 'nomic-ai/nomic-embed-text-v1.5',
    'v2-moe': 'nomic-ai/nomic-embed-text-v2-moe',
    'openai': 'openai:text-embedding-3-small'
}
```

Update the `EMBEDDINGS_FILE` line (around line 50) to be removed, as we'll compute it dynamically.

Modify `get_query_embedding()` to accept model parameter (around line 103):

```python
def get_query_embedding(query: str, model: str) -> List[float]:
    """
    Generate embedding for a search query using embedding-cache.

    This version:
    - Uses local or OpenAI model (no API cost for local)
    - Caches queries automatically
    - Works offline (for local models)

    Args:
        query: Search query text
        model: Full model name

    Returns:
        Embedding vector as list
    """
    from embedding_cache import EmbeddingCache
    cache = EmbeddingCache(model=model)
    embedding = cache.embed(query)
    return embedding.tolist()
```

Update `search_cards()` function signature (around line 123):

```python
def search_cards(
    query: str,
    embeddings_data: List[Dict],
    model: str,
    top_k: int = 5,
    position_filter: str = None,
    system_filter: str = None
) -> List[Tuple[str, str, float]]:
```

Update the call to `get_query_embedding()` inside `search_cards()`:

```python
    # Get query embedding (cached if seen before!)
    query_embedding = get_query_embedding(query, model)
```

Update `find_similar_cards()` to not need model (it uses precomputed embeddings).

In `main()` function, add argument parsing:

```python
def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Search tarot cards using semantic embeddings (with embedding-cache)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('query', nargs='?', help='Search query')
    parser.add_argument(
        '--model',
        choices=['v1.5', 'v2-moe', 'openai'],
        default=os.environ.get('EMBEDDING_MODEL', 'v1.5'),
        help='Embedding model to use (default: v1.5, or EMBEDDING_MODEL env var)'
    )
    parser.add_argument('--similar', metavar='CARD', help='Find cards similar to this card')
    parser.add_argument('--reversed', action='store_true', help='Use reversed position for similarity search')
    parser.add_argument('--top', type=int, default=5, help='Number of results to return (default: 5)')
    parser.add_argument('--ascii', '--art', action='store_true', help='Show ASCII art for cards')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    parser.add_argument('--yaml', action='store_true', help='Output results as YAML')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive search mode')

    args = parser.parse_args()

    # Get full model name and embeddings file
    model_name = MODEL_MAP[args.model]
    embeddings_file = f'card_embeddings_{args.model.replace("-", "_")}.json'

    # Check if embeddings file exists
    if not os.path.exists(embeddings_file):
        print(f"Error: {embeddings_file} not found.", file=sys.stderr)
        print(f"Generate embeddings first: python3 generate_embeddings_cached.py --model {args.model}", file=sys.stderr)
        sys.exit(1)

    # Load data
    try:
        with open(embeddings_file, 'r') as f:
            embeddings_data = json.load(f)
        cards = load_cards()
        interpretations = load_interpretations()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Using model: {model_name}")
```

Update the calls to `search_cards()` throughout to pass `model_name`:

```python
    # Semantic search
    if not args.query:
        parser.print_help()
        sys.exit(1)

    results = search_cards(args.query, embeddings_data, model_name, top_k=args.top)
    print(format_results(results, cards, interpretations, args.ascii, format_type))
```

And in interactive mode:

```python
            else:
                # Semantic search
                results = search_cards(query, embeddings_data, model_name, top_k=5)
                print(format_results(results, cards, interpretations))
```

**Step 2: Test the changes**

Run: `cd /home/gabe/semantic-tarot && python3 search_cards_cached.py --help`

Expected: Help message showing --model option

**Step 3: Commit**

```bash
cd /home/gabe/semantic-tarot
git add search_cards_cached.py
git commit -m "feat: add --model flag to search_cards_cached.py with auto-detect embeddings file"
```

---

## Task 8: Integration Test - Generate Embeddings with v2-moe

**Files:**
- Test in semantic-tarot

**Step 1: Generate small test with v2-moe model**

This will verify the full integration works. First, check if v2-moe model works:

Run:
```bash
cd /home/gabe/semantic-tarot
# Test with just a few cards
python3 -c "from embedding_cache import EmbeddingCache; cache = EmbeddingCache(model='nomic-ai/nomic-embed-text-v2-moe'); print(cache.embed('test').shape)"
```

Expected: `(768,)` (v2-moe also produces 768-dim embeddings)

**Step 2: Generate full embeddings with v2-moe**

Run:
```bash
cd /home/gabe/semantic-tarot
python3 generate_embeddings_cached.py --model v2-moe
```

Expected: Creates `card_embeddings_v2_moe.json` with all 780 embeddings

**Step 3: Test search with v2-moe**

Run:
```bash
cd /home/gabe/semantic-tarot
python3 search_cards_cached.py --model v2-moe "new beginnings" --top 3
```

Expected: Returns top 3 cards (likely The Fool, Ace of Wands, etc.)

**Step 4: Verify cache isolation**

Run:
```bash
# Check cache stats show both models are cached separately
python3 -c "
from embedding_cache import EmbeddingCache
cache_v15 = EmbeddingCache(model='nomic-ai/nomic-embed-text-v1.5')
cache_v2 = EmbeddingCache(model='nomic-ai/nomic-embed-text-v2-moe')
result_v15 = cache_v15.embed('test')
result_v2 = cache_v2.embed('test')
print(f'v1.5 shape: {result_v15.shape}, v2-moe shape: {result_v2.shape}')
print(f'v1.5 != v2-moe: {not np.allclose(result_v15[:100], result_v2[:100])}')
"
```

Expected: Shows different embeddings for same text with different models

**Step 5: Document success**

No commit needed - this was a manual integration test.

---

## Task 9: Documentation Update - Model Comparison Summary

**Files:**
- Modify: `README.md` (add Model Comparison section)

**Step 1: Add comparison section before "Architecture"**

In `README.md`, add after the OpenAI usage section:

```markdown
## Model Comparison

embedding-cache supports multiple embedding models. Here's how they compare:

| Model | Dimensions | Provider | Cost | Speed | Use Case |
|-------|-----------|----------|------|-------|----------|
| nomic-ai/nomic-embed-text-v1.5 | 768 | Local | $0 | Fast (cached) | General purpose, offline |
| nomic-ai/nomic-embed-text-v2-moe | 768 | Local | $0 | Medium (MoE) | Higher quality, offline |
| openai:text-embedding-3-small | 1536 | API | $0.0001/1K tokens | Fast (API) | Highest quality, online |

### Switching Models

\```python
from embedding_cache import EmbeddingCache

# Use v1.5 (default, fast and reliable)
cache_v15 = EmbeddingCache(model="nomic-ai/nomic-embed-text-v1.5")

# Use v2-moe (newer, potentially higher quality)
cache_v2 = EmbeddingCache(model="nomic-ai/nomic-embed-text-v2-moe")

# Use OpenAI (highest quality, requires API key)
cache_openai = EmbeddingCache(model="openai:text-embedding-3-small")
\```

All models benefit from the same caching layer, so repeated queries are instant regardless of which model you choose.

```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add model comparison section to README"
```

---

## Task 10: Update semantic-tarot Integration Docs

**Files:**
- Modify: `/home/gabe/semantic-tarot/EMBEDDING_CACHE_INTEGRATION.md`

**Step 1: Add model selection section**

In `/home/gabe/semantic-tarot/EMBEDDING_CACHE_INTEGRATION.md`, add after the "Usage" section (around line 48):

```markdown
### Model Selection

You can choose between different embedding models:

\```bash
# Use v1.5 (default, stable)
python3 generate_embeddings_cached.py --model v1.5

# Use v2-moe (newer Mixture of Experts model)
python3 generate_embeddings_cached.py --model v2-moe

# Use OpenAI (requires API key)
export OPENAI_API_KEY=your-key-here
python3 generate_embeddings_cached.py --model openai
\```

Search with the same model:

\```bash
# Search with v2-moe
python3 search_cards_cached.py --model v2-moe "transformation"

# Search with OpenAI
python3 search_cards_cached.py --model openai "new beginnings"
\```

**Model Comparison:**

- **v1.5**: Stable, well-tested, 768 dimensions, ~400MB download
- **v2-moe**: Newer architecture, potentially higher quality, 768 dimensions
- **openai**: Highest quality, 1536 dimensions, requires API key ($0.0001/1K tokens)

All models use the same cache layer, so switching models requires regenerating embeddings but subsequent searches are instant.

```

**Step 2: Commit**

```bash
cd /home/gabe/semantic-tarot
git add EMBEDDING_CACHE_INTEGRATION.md
git commit -m "docs: add model selection section to integration guide"
```

---

## Summary

This implementation plan adds multi-model support to embedding-cache and extends semantic-tarot with easy model switching:

**Phase 1 (Tasks 1-5):** Add OpenAI provider to embedding-cache
- OpenAIProvider class with lazy loading
- Provider selection based on model prefix
- Optional dependency and documentation

**Phase 2 (Tasks 6-8):** Extend semantic-tarot with model flags
- `--model` flag for generate_embeddings_cached.py
- `--model` flag for search_cards_cached.py
- Auto-detect embeddings file based on model
- Integration testing with v2-moe

**Phase 3 (Tasks 9-10):** Documentation
- Model comparison table in README
- Usage examples for all models
- Integration guide updates

**Testing Strategy:**
- Unit tests with mocks for OpenAI API (Tasks 1-2)
- Provider selection tests (Task 3)
- Integration tests with real models (Task 8)
- Manual verification of semantic-tarot workflow

**Next Steps:**
After this plan, you can implement the benchmark and interactive comparison tools (Phase 3-5 from design document).
