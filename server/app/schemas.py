# server/app/schemas.py
from pydantic import BaseModel, EmailStr, Field
from typing import Optional


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
    request_template: dict
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
