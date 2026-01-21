"""
Similarity search logic for pgvector integration.

Score Calculation:
- pgvector inner product (<#>) returns NEGATIVE similarity for index ordering
- For normalized vectors: <#> returns -1 (identical) to 1 (opposite)
- We convert to 0-1 scale: (-(raw_distance) + 1) / 2
"""

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
