# server/app/models.py
from datetime import datetime
from sqlalchemy import Boolean, Column, String, Text, Integer, LargeBinary, DateTime, ForeignKey, Index, JSON
from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    tier = Column(String, nullable=False, default="free")
    is_admin = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    email_verified_at = Column(DateTime, nullable=True)


class ApiKey(Base):
    __tablename__ = "api_keys"

    key_hash = Column(String, primary_key=True)
    key_prefix = Column(String, nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    tier = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)
    revoked_at = Column(DateTime, nullable=True)


class Embedding(Base):
    __tablename__ = "embeddings"

    text_hash = Column(String, primary_key=True)
    model = Column(String, primary_key=True)
    model_version = Column(String, primary_key=True)
    tenant_id = Column(String, primary_key=True)
    dimensions = Column(Integer, nullable=False)
    vector = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_hit_at = Column(DateTime, nullable=True)
    hit_count = Column(Integer, default=0)

    __table_args__ = (
        Index("idx_embeddings_lookup", "tenant_id", "model", "text_hash"),
    )


class Provider(Base):
    __tablename__ = "providers"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    endpoint = Column(String, nullable=False)
    api_key_encrypted = Column(Text, nullable=False)
    request_template = Column(JSON, nullable=False)
    response_path = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Usage(Base):
    __tablename__ = "usage"

    user_id = Column(String, primary_key=True)
    date = Column(DateTime, primary_key=True)
    model = Column(String, primary_key=True)
    cache_hits = Column(Integer, default=0)
    cache_misses = Column(Integer, default=0)
    compute_ms = Column(Integer, default=0)
