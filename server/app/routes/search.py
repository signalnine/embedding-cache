# server/app/routes/search.py
"""Search API endpoint for similarity search on cached embeddings."""
import time
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.schemas import SearchRequest, SearchResponse, SearchResult
from app.similarity import (
    similarity_search,
    SearchParams,
    get_dimensions_for_model,
    validate_dimensions,
)
from app.auth import get_current_api_key
from app.database import get_db
from app.compute import compute_embedding
from app.models import ApiKey

router = APIRouter(prefix="/v1", tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search_embeddings(
    request: SearchRequest,
    api_key: ApiKey = Depends(get_current_api_key),
    db: Session = Depends(get_db),
):
    """
    Search for similar embeddings in the cache.

    Supports two query modes:
    - query_text: Text is converted to embedding, then searched
    - query_vector: Pre-computed embedding vector for direct search

    Returns top_k most similar cached embeddings.
    """
    start_time = time.perf_counter()

    # Validate model
    try:
        dimensions = get_dimensions_for_model(request.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Validate dimensions are supported
    try:
        validate_dimensions(dimensions)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Get query vector
    if request.query_text is not None:
        # Compute embedding from text
        query_vector = await compute_embedding(request.query_text, request.model)
    else:
        # Use provided vector
        query_vector = request.query_vector
        if len(query_vector) != dimensions:
            raise HTTPException(
                status_code=400,
                detail=f"Query vector dimension {len(query_vector)} does not match "
                f"model {request.model} (expected {dimensions})",
            )

    # Build search parameters
    params = SearchParams(
        query_vector=query_vector,
        tenant_id=api_key.user_id,
        model=request.model,
        dimensions=dimensions,
        top_k=request.top_k,
        min_score=request.min_score,
    )

    # Execute similarity search
    raw_results = await similarity_search(db=db, params=params)

    # Transform results
    results = []
    for row in raw_results:
        result = SearchResult(
            text_hash=row["text_hash"],
            score=row["score"],
            text=row.get("original_text") if request.include_text else None,
            vector=row.get("vector") if request.include_vectors else None,
            model=row["model"],
            hit_count=row.get("hit_count", 0),
        )
        results.append(result)

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)

    return SearchResponse(
        results=results,
        total=len(results),
        search_time_ms=elapsed_ms,
        model=request.model,
        dimensions=dimensions,
    )
