# server/app/cache.py
import struct
from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import or_
from app.models import Embedding
from app.normalize import generate_text_hash
from app.config import settings


def vector_to_bytes(vector: list[float]) -> bytes:
    """Convert float vector to bytes for storage."""
    return struct.pack(f"{len(vector)}f", *vector)


def bytes_to_vector(data: bytes) -> list[float]:
    """Convert bytes back to float vector."""
    count = len(data) // 4
    return list(struct.unpack(f"{count}f", data))


def get_cached_embedding(
    db: Session,
    text_hash: str,
    model: str,
    model_version: str,
    tenant_id: str,
    check_public: bool = True,
) -> Optional[list[float]]:
    """Look up cached embedding.

    Checks private namespace first, then public if check_public=True.
    Updates hit_count and last_hit_at on hit.
    """
    # Build query for private + optional public
    query = db.query(Embedding).filter(
        Embedding.text_hash == text_hash,
        Embedding.model == model,
        Embedding.model_version == model_version,
    )

    if check_public:
        query = query.filter(
            or_(Embedding.tenant_id == tenant_id, Embedding.tenant_id == "public")
        )
        # Prefer private over public
        query = query.order_by(
            (Embedding.tenant_id == tenant_id).desc()
        )
    else:
        query = query.filter(Embedding.tenant_id == tenant_id)

    embedding = query.first()

    if embedding:
        # Update hit stats
        embedding.hit_count += 1
        embedding.last_hit_at = datetime.utcnow()
        db.commit()
        return bytes_to_vector(embedding.vector)

    return None


def store_embedding(
    db: Session,
    text: str,
    model: str,
    vector: list[float],
    tenant_id: str,
    public: bool = False,
) -> None:
    """Store embedding in cache."""
    text_hash = generate_text_hash(text)
    target_tenant = "public" if public else tenant_id

    embedding = Embedding(
        text_hash=text_hash,
        model=model,
        model_version=settings.model_version,
        tenant_id=target_tenant,
        dimensions=len(vector),
        vector=vector_to_bytes(vector),
    )

    db.merge(embedding)  # Upsert
    db.commit()
