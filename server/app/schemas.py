# server/app/schemas.py
from typing import Any, Optional

from pydantic import BaseModel, EmailStr, Field, model_validator


# Auth schemas
class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class SignupResponse(BaseModel):
    user_id: str
    api_key: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    token: str
    expires_in: int


class CreateKeyResponse(BaseModel):
    api_key: str
    prefix: str


class RevokeKeyResponse(BaseModel):
    revoked: bool


# Embed schemas
class EmbedRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    model: str = "nomic-v1.5"
    public: bool = False


class EmbedResponse(BaseModel):
    embedding: list[float]
    cached: bool
    dimensions: int


class BatchEmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=100)
    model: str = "nomic-v1.5"
    public: bool = False


class BatchEmbedResponse(BaseModel):
    embeddings: list[list[float]]
    cached: list[bool]
    dimensions: int


# Provider schemas
class ProviderRequest(BaseModel):
    name: str
    endpoint: str
    api_key: str
    request_template: dict[str, Any]
    response_path: str


class ProviderResponse(BaseModel):
    provider_id: str


# Stats schemas
class StatsResponse(BaseModel):
    cache_hits: int
    cache_misses: int
    total_cached: int


# Health schemas
class HealthResponse(BaseModel):
    status: str
    version: str


# Search schemas
class SearchRequest(BaseModel):
    query_text: Optional[str] = None
    query_vector: Optional[list[float]] = None
    model: str
    top_k: int = Field(default=10, ge=1, le=100)
    min_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    include_vectors: bool = False
    include_text: bool = True

    @model_validator(mode="after")
    def validate_query(self):
        if self.query_text is not None and self.query_vector is not None:
            raise ValueError("query_text and query_vector are mutually exclusive")
        if self.query_text is None and self.query_vector is None:
            raise ValueError("Either query_text or query_vector is required")
        return self


class SearchResult(BaseModel):
    text_hash: str
    score: float
    text: Optional[str] = None
    vector: Optional[list[float]] = None
    model: str
    hit_count: int = 0


class SearchResponse(BaseModel):
    results: list[SearchResult]
    total: int
    search_time_ms: int
    model: str
    dimensions: int
