# Multi-Model Embedding Comparison Design

**Date:** 2026-01-09
**Status:** Approved
**Purpose:** Add support for comparing embedding models (nomic v1.5, v2-moe, OpenAI) using semantic-tarot as test case

## Overview

We extend embedding-cache to support multiple embedding models through a unified interface, then use semantic-tarot as a test harness to compare:
- **nomic-ai/nomic-embed-text-v1.5** (current default, 768 dims)
- **nomic-ai/nomic-embed-text-v2-moe** (newer MoE model, 768 dims)
- **OpenAI text-embedding-3-small** (API-based, 1536 dims)

The comparison measures search quality, performance, cost, and usability across all three models.

## Goals

1. **Multi-model support in embedding-cache:** Add OpenAI provider alongside existing local model support
2. **Flexible model selection:** Allow users to switch between models via command-line flags
3. **Comprehensive comparison:** Measure search quality, performance, cost, and edge case behavior
4. **Automated benchmarking:** Create reproducible benchmark suite with ground truth data
5. **Interactive exploration:** Build tool for qualitative model comparison

## Architecture

### High-Level Structure

```
embedding-cache/
├── embedding_cache/
│   └── providers.py          # Add OpenAIProvider class
└── setup.py                  # Add openai as optional dependency

semantic-tarot/
├── generate_embeddings_cached.py  # Add --model flag
├── search_cards_cached.py         # Add --model flag
├── card_embeddings_v1.5.json      # Model-specific outputs
├── card_embeddings_v2_moe.json
├── card_embeddings_openai.json
└── benchmark/
    ├── test_queries.yaml          # Ground truth data
    ├── compare_models.py          # Automated benchmark
    ├── interactive_compare.py     # Side-by-side query tool
    └── results/
        ├── 2026-01-09_comparison.md
        └── metrics.json
```

### Data Flow

1. User specifies model via `--model v1.5|v2-moe|openai` flag or `EMBEDDING_MODEL` env var
2. embedding-cache routes to appropriate provider:
   - `v1.5` → LocalProvider with `nomic-ai/nomic-embed-text-v1.5`
   - `v2-moe` → LocalProvider with `nomic-ai/nomic-embed-text-v2-moe`
   - `openai` → OpenAIProvider with `text-embedding-3-small`
3. Cache layer works identically for all providers (model name is part of cache key)
4. Benchmark tools aggregate results across all models and generate reports

### Cache Isolation

Different models produce different embeddings, so cache keys must include model name to prevent collisions.

**Current implementation:** The `generate_cache_key()` function uses JSON encoding:
```python
combined = json.dumps({"text": text, "model": model}, sort_keys=True)
cache_key = hashlib.sha256(combined.encode()).hexdigest()
```

Each combination produces a unique key:
- "hello world" + "nomic-ai/nomic-embed-text-v1.5" → unique key
- "hello world" + "nomic-ai/nomic-embed-text-v2-moe" → different key
- "hello world" + "openai:text-embedding-3-small" → another key

All models share one SQLite database without collision.

## Component Design

### 1. OpenAI Provider Implementation

**New class in `embedding_cache/providers.py`:**

```python
class OpenAIProvider:
    """OpenAI embedding provider via API."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        """Initialize OpenAI provider.

        Args:
            model: OpenAI model name
            api_key: API key (defaults to OPENAI_API_KEY env var)
        """
        self.model_name = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client: Optional[object] = None  # Lazy load

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
        """Embed a single text string via OpenAI API."""
        self._load_client()
        response = self._client.embeddings.create(
            input=[text],
            model=self.model_name
        )
        embedding = response.data[0].embedding
        return np.array(embedding, dtype=np.float32)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts (OpenAI supports up to 2048 per request)."""
        self._load_client()
        response = self._client.embeddings.create(
            input=texts,
            model=self.model_name
        )
        embeddings = [np.array(item.embedding, dtype=np.float32) for item in response.data]
        return embeddings
```

**Provider selection in `EmbeddingCache.__init__()`:**

```python
def __init__(self, model: str = "nomic-ai/nomic-embed-text-v1.5", ...):
    self.model = model

    # Determine provider based on model prefix
    if model.startswith("openai:"):
        actual_model = model.split(":", 1)[1]  # "openai:text-embedding-3-small" -> "text-embedding-3-small"
        self.local_provider = OpenAIProvider(model=actual_model)
    else:
        # Local model (nomic-embed)
        self.local_provider = LocalProvider(model=model)

    # ... rest of initialization
```

**Dependencies:**
- Add to `setup.py`: `"openai": ["openai>=1.0.0"]`
- Install with: `pip install embedding-cache[openai]` or `pip install embedding-cache[local,openai]`

**Error handling:**
- If API key is missing: "Set OPENAI_API_KEY environment variable"
- If quota is exceeded: Include OpenAI error message
- OpenAI client provides built-in retry logic (no custom rate limiting for MVP)

### 2. Semantic-Tarot Integration

**Modify `generate_embeddings_cached.py`:**

Add argument parsing:
```python
parser.add_argument(
    '--model',
    choices=['v1.5', 'v2-moe', 'openai'],
    default='v1.5',
    help='Embedding model to use (default: v1.5)'
)
```

Map short names to full identifiers:
```python
MODEL_MAP = {
    'v1.5': 'nomic-ai/nomic-embed-text-v1.5',
    'v2-moe': 'nomic-ai/nomic-embed-text-v2-moe',
    'openai': 'openai:text-embedding-3-small'
}
```

Update output filename:
```python
output_file = f'card_embeddings_{args.model.replace("-", "_")}.json'
# v1.5 -> card_embeddings_v1_5.json (but we'll use v1.5 for readability)
```

**Modify `search_cards_cached.py`:**

Add same `--model` argument, auto-detect embeddings file:
```python
embeddings_file = f'card_embeddings_{args.model.replace("-", "_")}.json'
if not os.path.exists(embeddings_file):
    print(f"Error: {embeddings_file} not found. Generate embeddings first: generate_embeddings_cached.py --model {args.model}")
    sys.exit(1)
```

Display model in header:
```python
print(f"Using model: {MODEL_MAP[args.model]}")
```

**Model selection priority:**
1. `--model` command-line flag (highest)
2. `EMBEDDING_MODEL` environment variable
3. Default: `v1.5` (most stable)

### 3. Automated Benchmark Runner

**New file: `benchmark/compare_models.py`**

**Purpose:** Runs standardized tests across all models and generates comparison reports.

**Command interface:**
```bash
# Run full comparison
python3 benchmark/compare_models.py

# Compare specific models
python3 benchmark/compare_models.py --models v1.5 v2-moe

# Include cost analysis
python3 benchmark/compare_models.py --include-cost

# Skip cache (measure cold performance)
python3 benchmark/compare_models.py --no-cache
```

**Workflow:**
1. Load `test_queries.yaml` with ground truth
2. For each model:
   - Load embeddings file
   - Run all test queries
   - Measure: query time, cache hits, memory usage
   - Calculate IR metrics against ground truth
3. Generate markdown report to `results/YYYY-MM-DD_comparison.md`
4. Save raw metrics to `results/metrics.json`

**Report sections:**
1. **Executive Summary:** Winner per metric (quality, speed, cost)
2. **Search Quality Table:** Precision@5, nDCG@5, MRR by category
3. **Performance Table:** Query time (cold/warm), memory, throughput
4. **Cost Analysis:** API costs, compute estimates, break-even point
5. **Edge Case Analysis:** Significant model disagreements
6. **Recommendations:** Model selection by use case

**Metrics collected:**

*Search Quality (Information Retrieval):*
- **Precision@5:** % of top-5 results in expected_top5
- **nDCG@5:** Normalized Discounted Cumulative Gain (rewards correct cards ranked higher)
- **MRR:** Mean Reciprocal Rank (speed to first relevant card)
- **Agreement:** % overlap in top-5 results across models

*Performance:*
- **Cold query time:** First query with empty cache (ms)
- **Warm query time:** Repeated query with cache hit (ms)
- **Memory usage:** RSS memory during query (MB)
- **Cache hit rate:** % of queries hitting cache
- **Throughput:** Queries per second

*Cost:*
- **OpenAI:** $0.0001/1K tokens × actual token count
- **Local models:** AWS EC2 equivalent GPU time estimate
- **Break-even:** Queries needed to recoup local model setup cost

### 4. Interactive Comparison Tool

**New file: `benchmark/interactive_compare.py`**

**Purpose:** Explores model differences qualitatively with live queries.

**UI design:**
```
=== Interactive Model Comparison ===
Models loaded: v1.5 ✓  v2-moe ✓  openai ✓

Query: new beginnings

┌─ v1.5 (89ms) ──────────────────┬─ v2-moe (124ms) ────────────────┬─ openai (203ms) ────────────────┐
│ 1. The Fool (0.8234)           │ 1. The Fool (0.8456)            │ 1. The Fool (0.8821)            │
│ 2. Ace of Wands (0.7891)       │ 2. Ace of Wands (0.8012)        │ 2. The Magician (0.8234)        │
│ 3. The Magician (0.7654)       │ 3. The Magician (0.7823)        │ 3. Ace of Wands (0.8012)        │
│ 4. The Star (0.7234)           │ 4. The Star (0.7456)            │ 4. Ace of Cups (0.7789)         │
│ 5. Ace of Cups (0.7123)        │ 5. Ace of Pentacles (0.7234)    │ 5. The Star (0.7656)            │
└────────────────────────────────┴─────────────────────────────────┴─────────────────────────────────┘

Agreement: 4/5 cards in common (The Fool, Ace of Wands, The Magician, The Star)
Unique to openai: Ace of Cups at #4

Commands: <query> | similar <card> | quit
```

**Features:**
- Real-time query input
- Side-by-side top-5 results with similarity scores
- Timing display for each model
- Agreement analysis (overlapping cards)
- Highlighted unique results
- Card descriptions with `--verbose` option

**Implementation notes:**
- Colored output improves readability (optional, degrades gracefully)
- Cache all embeddings in memory on startup for speed
- Support both semantic search and similarity search modes

## Evaluation Methodology

### Ground Truth Data

**File: `benchmark/test_queries.yaml`**

Structure:
```yaml
queries:
  # Direct meanings (5 queries)
  - query: "new beginnings"
    expected_top5:
      - {card: "The Fool", position: "upright", rank: 1}
      - {card: "Ace of Wands", position: "upright", rank: 2}
      - {card: "The Magician", position: "upright", rank: 3}
      - {card: "The Star", position: "upright", rank: 4}
      - {card: "Ace of Pentacles", position: "upright", rank: 5}
    category: "direct_meaning"
    notes: "Clear connection to fresh starts and initiation"

  - query: "love and relationships"
    expected_top5: [...]
    category: "direct_meaning"

  # Emotional states (5 queries)
  - query: "feeling stuck and trapped"
    expected_top5: [...]
    category: "emotional_state"

  - query: "overwhelming grief"
    expected_top5: [...]
    category: "emotional_state"

  # Life situations (5 queries)
  - query: "career change decision"
    expected_top5: [...]
    category: "life_situation"

  - query: "financial hardship"
    expected_top5: [...]
    category: "life_situation"

  # Abstract concepts (5 queries)
  - query: "transformation through adversity"
    expected_top5: [...]
    category: "abstract_concept"

  - query: "spiritual awakening"
    expected_top5: [...]
    category: "abstract_concept"

  # Edge cases (5-10 queries)
  - query: "not"
    expected_top5: [...]
    category: "edge_case"
    notes: "Very short query - tests handling of minimal input"

  - query: "I feel overwhelmed by too many responsibilities and don't know which way to turn"
    expected_top5: [...]
    category: "edge_case"
    notes: "Long descriptive query - tests handling of verbose input"

  - query: "bank"
    expected_top5: [...]
    category: "edge_case"
    notes: "Ambiguous term - financial institution vs river bank"

  - query: "love and loss together"
    expected_top5: [...]
    category: "edge_case"
    notes: "Multi-concept query with conflicting emotions"
```

**Creation process:**
1. Select 25-30 diverse queries covering all categories
2. For each query, manually rank top 5 cards based on traditional tarot meanings
3. 2-3 people independently rank, then reconcile differences through discussion
4. Document reasoning in `notes` field for controversial rankings
5. Version control ground truth to track changes over time

### Evaluation Workflow

1. **Generate embeddings** for all 780 cards (78 cards × 2 positions × 5 systems) using all three models
2. **Run automated benchmark:** `python3 benchmark/compare_models.py --include-cost`
3. **Manual review:** Examine edge cases where models disagree significantly (>0.2 similarity difference)
4. **Interactive exploration:** Use `interactive_compare.py` for qualitative assessment
5. **Document findings:** Review generated report, add commentary about model characteristics
6. **Iterate if needed:** Refine ground truth or model configuration if results reveal issues

### Success Criteria

**Minimum acceptable performance:**
- **Search quality:** All models achieve >0.6 nDCG@5
- **Performance:** Cached queries <10ms, cold queries <500ms
- **Cost:** Clear cost comparison showing break-even point
- **Completeness:** All 25-30 test queries evaluated without crashes

**Ideal outcomes:**
- One model wins clearly on quality with <10% cost premium
- All models handle edge cases gracefully (no catastrophic failures)
- Clear recommendations emerge for different use cases:
  - Budget-conscious: Best cost/performance ratio
  - Quality-focused: Highest nDCG even if slower or pricier
  - Offline: Best local model for air-gapped environments

## Implementation Plan

### Phase 1: OpenAI Provider (embedding-cache)
1. Add `OpenAIProvider` class to `providers.py`
2. Modify `EmbeddingCache.__init__()` for provider selection
3. Add `openai>=1.0.0` to optional dependencies
4. Write tests for OpenAI provider (mock API calls)
5. Update README with OpenAI usage

### Phase 2: Model Selection (semantic-tarot)
1. Add `--model` flag to `generate_embeddings_cached.py`
2. Add `--model` flag to `search_cards_cached.py`
3. Test generating embeddings with all three models
4. Verify cache isolation

### Phase 3: Ground Truth Data
1. Create `benchmark/` directory
2. Select 25-30 test queries across categories
3. Manually rank top 5 cards per query
4. Write `test_queries.yaml` with rankings and notes
5. Review with 2-3 people, reconcile differences

### Phase 4: Automated Benchmark
1. Write `compare_models.py` skeleton
2. Implement query evaluation against ground truth
3. Add IR metrics calculation (precision, nDCG, MRR)
4. Add performance metrics collection
5. Add cost analysis
6. Implement markdown report generation
7. Test with all three models

### Phase 5: Interactive Tool
1. Write `interactive_compare.py` skeleton
2. Implement side-by-side query display
3. Add agreement analysis
4. Add timing display
5. Polish UI
6. Add `--verbose` mode for card descriptions

### Phase 6: Evaluation & Documentation
1. Run full comparison with `compare_models.py`
2. Explore edge cases with `interactive_compare.py`
3. Review results, add commentary
4. Update embedding-cache README with model comparison summary
5. Update semantic-tarot EMBEDDING_CACHE_INTEGRATION.md with findings
6. Create example comparison report

## Testing Strategy

**Unit tests (embedding-cache):**
- Mock OpenAIProvider API calls
- Test provider selection logic in EmbeddingCache
- Test error handling for missing API key
- Test dimension handling (768 vs 1536)

**Integration tests (semantic-tarot):**
- Generate embeddings with all three models
- Verify output files
- Test search with each model
- Verify cache isolation (no cross-model pollution)

**Benchmark validation:**
- Run on small test set (5 queries) to verify metrics calculation
- Verify report generation produces valid markdown
- Verify cost calculations

**Manual testing:**
- Interactive tool UI/UX
- Edge case handling
- Report clarity

## Risk Analysis

**Risks:**

1. **OpenAI API rate limits:** Generating 780 embeddings may hit rate limits
   - *Mitigation:* Implement exponential backoff, optimize request batching

2. **Model v2-moe resource requirements:** MoE model may demand more memory and compute
   - *Mitigation:* Document requirements, test on target hardware first

3. **Ground truth subjectivity:** Tarot interpretations are subjective
   - *Mitigation:* Multi-person review, document reasoning, accept inevitable ambiguity

4. **Cost of OpenAI testing:** Repeated benchmark runs incur API costs
   - *Mitigation:* Cache OpenAI embeddings, skip OpenAI during development with `--models v1.5 v2-moe`

5. **Dimension mismatch issues:** 768 vs 1536 dims may cause subtle bugs
   - *Mitigation:* Cache keys include model name, test thoroughly

## Future Enhancements

**Not in MVP:**
- Support for additional models (Cohere, Voyage, etc.)
- A/B testing framework for production deployment
- Automated reranking combining multiple models
- Model ensemble strategies
- Fine-tuning evaluation for domain-specific models
- Continuous benchmark tracking

## Success Metrics

This design succeeds if:

1. **Technical success:** All three models work reliably through unified API
2. **Evaluation success:** Comprehensive comparison identifies clear winner(s)
3. **Usability success:** Users can easily switch models and understand trade-offs
4. **Documentation success:** Results guide future users to the right model

## Conclusion

This design extends embedding-cache to support multiple embedding models and uses semantic-tarot as a real-world test case to compare them. Automated benchmarks and interactive exploration will reveal which model works best for semantic search in different scenarios.

Cache abstraction makes provider differences transparent, enabling fair model comparison while maintaining the same user experience.
