# server/app/routes/embed.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import ApiKey, Provider
from app.schemas import (
    EmbedRequest, EmbedResponse,
    BatchEmbedRequest, BatchEmbedResponse,
    StatsResponse,
)
from app.auth import get_current_api_key
from app.cache import get_cached_embedding, store_embedding
from app.normalize import generate_text_hash
from app.rate_limit import check_rate_limit
from app.compute import compute_embedding, compute_batch
from app.passthrough import call_byok_provider
from app.config import settings

router = APIRouter(prefix="/v1", tags=["embed"])


@router.post("/embed", response_model=EmbedResponse)
async def embed(
    request: EmbedRequest,
    api_key: ApiKey = Depends(get_current_api_key),
    db: Session = Depends(get_db),
):
    """Get embedding for text."""
    # Check rate limit
    await check_rate_limit(api_key.user_id, api_key.tier)

    # Check cache
    text_hash = generate_text_hash(request.text)
    cached = get_cached_embedding(
        db=db,
        text_hash=text_hash,
        model=request.model,
        model_version=settings.model_version,
        tenant_id=api_key.user_id,
    )

    if cached:
        return EmbedResponse(
            embedding=cached,
            cached=True,
            dimensions=len(cached),
        )

    # Cache miss - compute embedding
    if api_key.tier == "paid":
        # Use local GPU
        embedding = await compute_embedding(request.text, request.model)
    else:
        # Use BYOK provider
        provider = db.query(Provider).filter(
            Provider.user_id == api_key.user_id
        ).first()

        if not provider:
            raise HTTPException(
                status_code=400,
                detail="Free tier requires BYOK provider. Configure at POST /v1/provider"
            )

        embedding = await call_byok_provider(
            endpoint=provider.endpoint,
            api_key_encrypted=provider.api_key_encrypted,
            request_template=provider.request_template,
            response_path=provider.response_path,
            text=request.text,
        )

    # Store in cache
    store_embedding(
        db=db,
        text=request.text,
        model=request.model,
        vector=embedding,
        tenant_id=api_key.user_id,
        public=request.public,
    )

    return EmbedResponse(
        embedding=embedding,
        cached=False,
        dimensions=len(embedding),
    )


@router.post("/embed/batch", response_model=BatchEmbedResponse)
async def embed_batch(
    request: BatchEmbedRequest,
    api_key: ApiKey = Depends(get_current_api_key),
    db: Session = Depends(get_db),
):
    """Get embeddings for multiple texts."""
    if len(request.texts) > settings.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds limit of {settings.max_batch_size}"
        )

    # Check rate limit (counts as N requests)
    for _ in request.texts:
        await check_rate_limit(api_key.user_id, api_key.tier)

    embeddings = []
    cached_flags = []

    # Check cache for each text
    texts_to_compute = []
    indices_to_compute = []

    for i, text in enumerate(request.texts):
        text_hash = generate_text_hash(text)
        cached = get_cached_embedding(
            db=db,
            text_hash=text_hash,
            model=request.model,
            model_version=settings.model_version,
            tenant_id=api_key.user_id,
        )

        if cached:
            embeddings.append(cached)
            cached_flags.append(True)
        else:
            embeddings.append(None)  # Placeholder
            cached_flags.append(False)
            texts_to_compute.append(text)
            indices_to_compute.append(i)

    # Compute missing embeddings
    if texts_to_compute:
        if api_key.tier == "paid":
            computed = await compute_batch(texts_to_compute, request.model)
        else:
            # BYOK doesn't support batch - compute one at a time
            provider = db.query(Provider).filter(
                Provider.user_id == api_key.user_id
            ).first()

            if not provider:
                raise HTTPException(
                    status_code=400,
                    detail="Free tier requires BYOK provider"
                )

            computed = []
            for text in texts_to_compute:
                emb = await call_byok_provider(
                    endpoint=provider.endpoint,
                    api_key_encrypted=provider.api_key_encrypted,
                    request_template=provider.request_template,
                    response_path=provider.response_path,
                    text=text,
                )
                computed.append(emb)

        # Fill in computed embeddings and cache them
        for idx, text, emb in zip(indices_to_compute, texts_to_compute, computed):
            embeddings[idx] = emb
            store_embedding(
                db=db,
                text=text,
                model=request.model,
                vector=emb,
                tenant_id=api_key.user_id,
                public=request.public,
            )

    return BatchEmbedResponse(
        embeddings=embeddings,
        cached=cached_flags,
        dimensions=len(embeddings[0]) if embeddings else 0,
    )


@router.get("/stats", response_model=StatsResponse)
async def stats(
    api_key: ApiKey = Depends(get_current_api_key),
    db: Session = Depends(get_db),
):
    """Get usage statistics."""
    from sqlalchemy import func
    from app.models import Embedding, Usage

    # Get user's cache stats
    total_cached = db.query(func.count(Embedding.text_hash)).filter(
        Embedding.tenant_id == api_key.user_id
    ).scalar() or 0

    # Get today's usage
    from datetime import date
    usage = db.query(Usage).filter(
        Usage.user_id == api_key.user_id,
        Usage.date == date.today(),
    ).first()

    return StatsResponse(
        cache_hits=usage.cache_hits if usage else 0,
        cache_misses=usage.cache_misses if usage else 0,
        total_cached=total_cached,
    )
