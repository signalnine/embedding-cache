"""
Similarity search logic for pgvector integration.

Score Calculation:
- pgvector inner product (<#>) returns NEGATIVE similarity for index ordering
- For normalized vectors: <#> returns -1 (identical) to 1 (opposite)
- We convert to 0-1 scale: (-(raw_distance) + 1) / 2
"""

from dataclasses import dataclass
from typing import Optional, Any

SUPPORTED_DIMENSIONS = {768, 1536, 384}

# Model name -> expected dimensions mapping
MODEL_DIMENSIONS = {
    "nomic-ai/nomic-embed-text-v1.5": 768,
    "nomic-v1.5": 768,
    "openai:text-embedding-3-small": 1536,
    "text-embedding-3-small": 1536,
    "all-MiniLM-L6-v2": 384,
}


def calculate_similarity_score(raw_distance: float) -> float:
    """
    Convert pgvector inner product distance to 0-1 similarity score.

    Args:
        raw_distance: Result of vector <#> query (ranges -1 to 1)

    Returns:
        Similarity score from 0 (opposite) to 1 (identical)
    """
    return (-raw_distance + 1) / 2


def get_dimensions_for_model(model: str) -> int:
    """
    Get expected dimensions for a model name.

    Raises:
        ValueError: If model is not recognized
    """
    if model in MODEL_DIMENSIONS:
        return MODEL_DIMENSIONS[model]
    raise ValueError(f"Unknown model: {model}. Known models: {list(MODEL_DIMENSIONS.keys())}")


def validate_dimensions(dimensions: int) -> None:
    """
    Validate that dimensions are supported (have partial HNSW index).

    Raises:
        ValueError: If dimensions not supported
    """
    if dimensions not in SUPPORTED_DIMENSIONS:
        raise ValueError(
            f"Unsupported dimension {dimensions}. "
            f"Supported: {sorted(SUPPORTED_DIMENSIONS)}"
        )


@dataclass
class SearchParams:
    query_vector: list[float]
    tenant_id: str
    model: str
    dimensions: int
    top_k: int
    min_score: Optional[float]


async def similarity_search(db: Any, params: SearchParams) -> list[dict]:
    """
    Execute similarity search using pgvector HNSW index.

    Args:
        db: Database connection with transaction() and fetch() methods
        params: Search parameters

    Returns:
        List of matching embeddings with scores

    Raises:
        ValueError: If dimensions not supported
    """
    validate_dimensions(params.dimensions)

    # SAFETY: dimensions is validated against SUPPORTED_DIMENSIONS whitelist above
    # pgvector ::vector(N) cast requires literal integer, cannot use $param
    # This is safe because validate_dimensions() only allows known integers
    dim = params.dimensions
    assert dim in SUPPORTED_DIMENSIONS, "Dimension bypass attempt"

    async with db.transaction():
        await db.execute("SET LOCAL hnsw.ef_search = 40")

        # Build query with dimension-specific casts for partial index
        # The ::vector(N) casts are REQUIRED to use the partial HNSW index
        # Note: dim is safe to interpolate (validated integer from whitelist)
        query = f"""
            SELECT
                text_hash,
                original_text,
                model,
                (-(vector::vector({dim}) <#> $1::vector({dim})) + 1) / 2 as score,
                hit_count
            FROM embeddings
            WHERE tenant_id = $2
              AND dimensions = $5
              AND model = $3
            ORDER BY vector::vector({dim}) <#> $1::vector({dim})
            LIMIT $4
        """

        results = await db.fetch(
            query,
            params.query_vector,
            params.tenant_id,
            params.model,
            params.top_k,
            params.dimensions  # $5 - parameterized for WHERE clause
        )

    # Apply min_score filter if specified (post-query for simplicity)
    if params.min_score is not None:
        results = [r for r in results if r['score'] >= params.min_score]

    return results
