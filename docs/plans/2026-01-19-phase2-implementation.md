# Phase 2: Hosted Backend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a self-hosted FastAPI backend that provides cached embeddings with hybrid compute (BYOK for free tier, GPU for paid tier).

**Architecture:** FastAPI server with PostgreSQL for cache storage, Redis for rate limiting, and sentence-transformers for GPU inference. Users authenticate via API keys, configure BYOK providers, and get embeddings with automatic caching.

**Tech Stack:** FastAPI, SQLAlchemy, PostgreSQL, Redis, sentence-transformers, bcrypt, PyJWT, cryptography

---

## Task 1: Project Setup

**Files:**
- Create: `server/requirements.txt`
- Create: `server/app/__init__.py`
- Create: `server/app/config.py`

**Step 1: Create server directory structure**

```bash
mkdir -p server/app server/tests server/alembic server/scripts
touch server/app/__init__.py
```

**Step 2: Create requirements.txt**

```
# server/requirements.txt
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
redis>=5.0.0
bcrypt>=4.1.0
pyjwt>=2.8.0
cryptography>=42.0.0
sentence-transformers>=2.3.0
prometheus-client>=0.19.0
httpx>=0.26.0
jsonpath-ng>=1.6.0
pydantic>=2.5.0
pydantic-settings>=2.1.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

**Step 3: Create config.py**

```python
# server/app/config.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://localhost/embeddings"
    redis_url: str = "redis://localhost:6379"
    encryption_key: str = ""  # 32-byte hex key for BYOK encryption
    jwt_secret: str = ""
    jwt_expiry_seconds: int = 86400
    gpu_device: str = "cuda:0"
    model_version: str = "1.0.0"

    # Rate limits
    free_tier_daily_limit: int = 1000
    paid_tier_daily_limit: int = 50000

    # Batch limits
    max_batch_size: int = 100
    max_payload_bytes: int = 1_000_000

    class Config:
        env_file = ".env"


settings = Settings()
```

**Step 4: Verify setup**

```bash
cd server && pip install -r requirements.txt
python -c "from app.config import settings; print(settings.database_url)"
```

Expected: `postgresql://localhost/embeddings`

**Step 5: Commit**

```bash
git add server/
git commit -m "feat(server): initial project setup with config"
```

---

## Task 2: Database Models

**Files:**
- Create: `server/app/database.py`
- Create: `server/app/models.py`
- Create: `server/tests/test_models.py`

**Step 1: Write database connection**

```python
# server/app/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import settings

engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

**Step 2: Write SQLAlchemy models**

```python
# server/app/models.py
from datetime import datetime
from sqlalchemy import Column, String, Text, Integer, LargeBinary, DateTime, ForeignKey, Index
from sqlalchemy.dialects.postgresql import JSONB
from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    tier = Column(String, nullable=False, default="free")
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
    request_template = Column(JSONB, nullable=False)
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
```

**Step 3: Write model tests**

```python
# server/tests/test_models.py
import pytest
from app.models import User, ApiKey, Embedding, Provider, Usage


def test_user_model_has_required_fields():
    """User model should have id, email, password_hash, tier."""
    user = User(
        id="usr_123",
        email="test@example.com",
        password_hash="hash",
        tier="free"
    )
    assert user.id == "usr_123"
    assert user.email == "test@example.com"
    assert user.tier == "free"


def test_api_key_model_has_required_fields():
    """ApiKey model should have key_hash, key_prefix, user_id, tier."""
    key = ApiKey(
        key_hash="hash123",
        key_prefix="vec_abc12345",
        user_id="usr_123",
        tier="free"
    )
    assert key.key_prefix == "vec_abc12345"
    assert key.user_id == "usr_123"


def test_embedding_model_has_composite_key():
    """Embedding should have composite primary key."""
    emb = Embedding(
        text_hash="hash",
        model="nomic-v1.5",
        model_version="1.0.0",
        tenant_id="usr_123",
        dimensions=768,
        vector=b"\x00" * 768 * 4
    )
    assert emb.text_hash == "hash"
    assert emb.model == "nomic-v1.5"
    assert emb.tenant_id == "usr_123"
```

**Step 4: Run tests**

```bash
cd server && pytest tests/test_models.py -v
```

Expected: 3 tests PASS

**Step 5: Commit**

```bash
git add server/app/database.py server/app/models.py server/tests/test_models.py
git commit -m "feat(server): add SQLAlchemy database models"
```

---

## Task 3: Text Normalization

**Files:**
- Create: `server/app/normalize.py`
- Create: `server/tests/test_normalize.py`

**Step 1: Write failing test**

```python
# server/tests/test_normalize.py
import pytest
from app.normalize import normalize_text


def test_normalize_strips_whitespace():
    assert normalize_text("  hello  ") == "hello"


def test_normalize_collapses_internal_whitespace():
    assert normalize_text("hello    world") == "hello world"


def test_normalize_lowercases():
    assert normalize_text("HELLO World") == "hello world"


def test_normalize_handles_newlines():
    assert normalize_text("hello\n\nworld") == "hello world"


def test_normalize_empty_string():
    assert normalize_text("") == ""
```

**Step 2: Run tests to verify failure**

```bash
cd server && pytest tests/test_normalize.py -v
```

Expected: FAIL with "cannot import name 'normalize_text'"

**Step 3: Implement normalize.py**

```python
# server/app/normalize.py
import hashlib


def normalize_text(text: str) -> str:
    """Normalize text for cache key generation.

    - Strip leading/trailing whitespace
    - Collapse internal whitespace to single space
    - Convert to lowercase
    """
    text = text.strip()
    text = " ".join(text.split())
    text = text.lower()
    return text


def generate_text_hash(text: str) -> str:
    """Generate SHA-256 hash of normalized text."""
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode()).hexdigest()
```

**Step 4: Run tests**

```bash
cd server && pytest tests/test_normalize.py -v
```

Expected: 5 tests PASS

**Step 5: Commit**

```bash
git add server/app/normalize.py server/tests/test_normalize.py
git commit -m "feat(server): add text normalization for cache keys"
```

---

## Task 4: Crypto Utilities

**Files:**
- Create: `server/app/crypto.py`
- Create: `server/tests/test_crypto.py`

**Step 1: Write failing tests**

```python
# server/tests/test_crypto.py
import pytest
from app.crypto import encrypt_api_key, decrypt_api_key, hash_api_key, generate_api_key


def test_encrypt_decrypt_roundtrip():
    """Encrypted key should decrypt to original."""
    original = "sk-test123456789"
    encrypted = encrypt_api_key(original)
    decrypted = decrypt_api_key(encrypted)
    assert decrypted == original


def test_encrypted_differs_from_original():
    """Encrypted value should not equal original."""
    original = "sk-test123456789"
    encrypted = encrypt_api_key(original)
    assert encrypted != original


def test_hash_api_key_consistent():
    """Same key should produce same hash."""
    key = "vec_abc123456789"
    hash1 = hash_api_key(key)
    hash2 = hash_api_key(key)
    assert hash1 == hash2


def test_hash_api_key_different_keys():
    """Different keys should produce different hashes."""
    hash1 = hash_api_key("vec_abc123456789")
    hash2 = hash_api_key("vec_xyz987654321")
    assert hash1 != hash2


def test_generate_api_key_format():
    """Generated key should start with vec_ prefix."""
    key = generate_api_key()
    assert key.startswith("vec_")
    assert len(key) == 36  # vec_ + 32 chars
```

**Step 2: Run tests to verify failure**

```bash
cd server && pytest tests/test_crypto.py -v
```

Expected: FAIL with import error

**Step 3: Implement crypto.py**

```python
# server/app/crypto.py
import hashlib
import secrets
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from app.config import settings


def _get_fernet() -> Fernet:
    """Get Fernet instance from encryption key."""
    if not settings.encryption_key:
        raise ValueError("ENCRYPTION_KEY not set")

    # Derive a proper Fernet key from our secret
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b"vec-embed-cache-salt",  # Fixed salt is OK for this use case
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(settings.encryption_key.encode()))
    return Fernet(key)


def encrypt_api_key(plaintext: str) -> str:
    """Encrypt a BYOK API key for storage."""
    f = _get_fernet()
    return f.encrypt(plaintext.encode()).decode()


def decrypt_api_key(ciphertext: str) -> str:
    """Decrypt a BYOK API key."""
    f = _get_fernet()
    return f.decrypt(ciphertext.encode()).decode()


def hash_api_key(key: str) -> str:
    """Hash an API key for storage (one-way)."""
    return hashlib.sha256(key.encode()).hexdigest()


def generate_api_key() -> str:
    """Generate a new API key with vec_ prefix."""
    random_part = secrets.token_hex(16)
    return f"vec_{random_part}"


def get_key_prefix(key: str) -> str:
    """Get the prefix of an API key for identification."""
    return key[:12]
```

**Step 4: Set test encryption key and run tests**

```bash
cd server && ENCRYPTION_KEY=test-secret-key-123 pytest tests/test_crypto.py -v
```

Expected: 5 tests PASS

**Step 5: Commit**

```bash
git add server/app/crypto.py server/tests/test_crypto.py
git commit -m "feat(server): add crypto utilities for key encryption and hashing"
```

---

## Task 5: Pydantic Schemas

**Files:**
- Create: `server/app/schemas.py`

**Step 1: Create request/response schemas**

```python
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
```

**Step 2: Verify schemas import**

```bash
cd server && python -c "from app.schemas import EmbedRequest, EmbedResponse; print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
git add server/app/schemas.py
git commit -m "feat(server): add Pydantic request/response schemas"
```

---

## Task 6: Authentication

**Files:**
- Create: `server/app/auth.py`
- Create: `server/tests/test_auth.py`

**Step 1: Write failing tests**

```python
# server/tests/test_auth.py
import pytest
from unittest.mock import MagicMock, patch
from app.auth import hash_password, verify_password, create_jwt, decode_jwt


def test_hash_password_not_plaintext():
    """Hashed password should not equal original."""
    password = "mysecretpassword"
    hashed = hash_password(password)
    assert hashed != password


def test_verify_password_correct():
    """Correct password should verify."""
    password = "mysecretpassword"
    hashed = hash_password(password)
    assert verify_password(password, hashed) is True


def test_verify_password_incorrect():
    """Incorrect password should not verify."""
    hashed = hash_password("mysecretpassword")
    assert verify_password("wrongpassword", hashed) is False


def test_create_jwt_returns_string():
    """JWT should be a string."""
    token = create_jwt(user_id="usr_123")
    assert isinstance(token, str)
    assert len(token) > 0


def test_decode_jwt_roundtrip():
    """Decoded JWT should contain original user_id."""
    token = create_jwt(user_id="usr_123")
    payload = decode_jwt(token)
    assert payload["user_id"] == "usr_123"
```

**Step 2: Run tests to verify failure**

```bash
cd server && pytest tests/test_auth.py -v
```

Expected: FAIL with import error

**Step 3: Implement auth.py**

```python
# server/app/auth.py
import bcrypt
import jwt
from datetime import datetime, timedelta
from typing import Optional
from fastapi import HTTPException, Header, Depends
from sqlalchemy.orm import Session
from app.config import settings
from app.database import get_db
from app.models import ApiKey
from app.crypto import hash_api_key


def hash_password(password: str) -> str:
    """Hash a password with bcrypt."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    return bcrypt.checkpw(password.encode(), hashed.encode())


def create_jwt(user_id: str) -> str:
    """Create a JWT token for a user."""
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(seconds=settings.jwt_expiry_seconds),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def decode_jwt(token: str) -> dict:
    """Decode and verify a JWT token."""
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_current_api_key(
    authorization: str = Header(...),
    db: Session = Depends(get_db)
) -> ApiKey:
    """Validate API key from Authorization header."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    key = authorization[7:]  # Remove "Bearer "

    if not key.startswith("vec_"):
        raise HTTPException(status_code=401, detail="Invalid API key format")

    key_hash = hash_api_key(key)
    api_key = db.query(ApiKey).filter(
        ApiKey.key_hash == key_hash,
        ApiKey.revoked_at.is_(None)
    ).first()

    if not api_key:
        raise HTTPException(status_code=401, detail="Invalid or revoked API key")

    # Update last_used_at
    api_key.last_used_at = datetime.utcnow()
    db.commit()

    return api_key
```

**Step 4: Run tests**

```bash
cd server && JWT_SECRET=test-jwt-secret pytest tests/test_auth.py -v
```

Expected: 5 tests PASS

**Step 5: Commit**

```bash
git add server/app/auth.py server/tests/test_auth.py
git commit -m "feat(server): add authentication utilities"
```

---

## Task 7: Redis Rate Limiting

**Files:**
- Create: `server/app/redis_client.py`
- Create: `server/app/rate_limit.py`
- Create: `server/tests/test_rate_limit.py`

**Step 1: Create Redis client**

```python
# server/app/redis_client.py
import redis.asyncio as redis
from app.config import settings

redis_client: redis.Redis | None = None


async def get_redis() -> redis.Redis:
    """Get Redis connection."""
    global redis_client
    if redis_client is None:
        redis_client = redis.from_url(settings.redis_url)
    return redis_client


async def close_redis():
    """Close Redis connection."""
    global redis_client
    if redis_client:
        await redis_client.close()
        redis_client = None
```

**Step 2: Write rate limit tests**

```python
# server/tests/test_rate_limit.py
import pytest
from unittest.mock import AsyncMock, patch
from app.rate_limit import check_rate_limit, RateLimitExceeded


@pytest.mark.asyncio
async def test_rate_limit_allows_under_limit():
    """Requests under limit should be allowed."""
    mock_redis = AsyncMock()
    mock_redis.incr.return_value = 5

    with patch("app.rate_limit.get_redis", return_value=mock_redis):
        result = await check_rate_limit("usr_123", "free")
        assert result is True


@pytest.mark.asyncio
async def test_rate_limit_blocks_over_limit():
    """Requests over limit should raise RateLimitExceeded."""
    mock_redis = AsyncMock()
    mock_redis.incr.return_value = 1001  # Over free tier limit

    with patch("app.rate_limit.get_redis", return_value=mock_redis):
        with pytest.raises(RateLimitExceeded):
            await check_rate_limit("usr_123", "free")


@pytest.mark.asyncio
async def test_paid_tier_has_higher_limit():
    """Paid tier should allow more requests."""
    mock_redis = AsyncMock()
    mock_redis.incr.return_value = 5000  # Over free, under paid

    with patch("app.rate_limit.get_redis", return_value=mock_redis):
        result = await check_rate_limit("usr_123", "paid")
        assert result is True
```

**Step 3: Run tests to verify failure**

```bash
cd server && pytest tests/test_rate_limit.py -v
```

Expected: FAIL with import error

**Step 4: Implement rate_limit.py**

```python
# server/app/rate_limit.py
from datetime import date
from fastapi import HTTPException
from app.redis_client import get_redis
from app.config import settings


class RateLimitExceeded(HTTPException):
    def __init__(self, limit: int, reset_time: str):
        super().__init__(
            status_code=429,
            detail=f"Rate limit exceeded. Limit: {limit}/day. Resets at: {reset_time}"
        )


async def check_rate_limit(user_id: str, tier: str) -> bool:
    """Check if user is within rate limit.

    Raises RateLimitExceeded if over limit.
    Returns True if allowed.
    """
    redis = await get_redis()
    today = date.today().isoformat()
    key = f"rate:{user_id}:{today}"

    count = await redis.incr(key)

    # Set expiry on first request of the day
    if count == 1:
        await redis.expire(key, 86400)

    limit = settings.paid_tier_daily_limit if tier == "paid" else settings.free_tier_daily_limit

    if count > limit:
        raise RateLimitExceeded(limit=limit, reset_time=f"{today} 00:00 UTC")

    return True


async def get_usage_count(user_id: str) -> int:
    """Get current day's request count for user."""
    redis = await get_redis()
    today = date.today().isoformat()
    key = f"rate:{user_id}:{today}"

    count = await redis.get(key)
    return int(count) if count else 0
```

**Step 5: Run tests**

```bash
cd server && pytest tests/test_rate_limit.py -v
```

Expected: 3 tests PASS

**Step 6: Commit**

```bash
git add server/app/redis_client.py server/app/rate_limit.py server/tests/test_rate_limit.py
git commit -m "feat(server): add Redis-based rate limiting"
```

---

## Task 8: Cache Logic

**Files:**
- Create: `server/app/cache.py`
- Create: `server/tests/test_cache.py`

**Step 1: Write failing tests**

```python
# server/tests/test_cache.py
import pytest
import struct
from unittest.mock import MagicMock
from app.cache import (
    get_cached_embedding,
    store_embedding,
    vector_to_bytes,
    bytes_to_vector,
)


def test_vector_to_bytes_roundtrip():
    """Vector should survive bytes conversion."""
    original = [0.1, 0.2, 0.3, 0.4]
    as_bytes = vector_to_bytes(original)
    recovered = bytes_to_vector(as_bytes)
    assert recovered == pytest.approx(original)


def test_vector_to_bytes_length():
    """Bytes should be 4 * len(vector) for float32."""
    vector = [0.1] * 768
    as_bytes = vector_to_bytes(vector)
    assert len(as_bytes) == 768 * 4


def test_get_cached_embedding_miss():
    """Cache miss should return None."""
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = None

    result = get_cached_embedding(
        db=mock_db,
        text_hash="abc123",
        model="nomic-v1.5",
        model_version="1.0.0",
        tenant_id="usr_123"
    )
    assert result is None
```

**Step 2: Run tests to verify failure**

```bash
cd server && pytest tests/test_cache.py -v
```

Expected: FAIL with import error

**Step 3: Implement cache.py**

```python
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
```

**Step 4: Run tests**

```bash
cd server && pytest tests/test_cache.py -v
```

Expected: 3 tests PASS

**Step 5: Commit**

```bash
git add server/app/cache.py server/tests/test_cache.py
git commit -m "feat(server): add cache lookup and storage logic"
```

---

## Task 9: GPU Compute

**Files:**
- Create: `server/app/compute.py`
- Create: `server/tests/test_compute.py`

**Step 1: Write failing tests**

```python
# server/tests/test_compute.py
import pytest
from unittest.mock import patch, MagicMock


def test_compute_embedding_returns_list():
    """Compute should return list of floats."""
    # Mock sentence-transformers to avoid GPU requirement in tests
    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1] * 768]

    with patch("app.compute._get_model", return_value=mock_model):
        from app.compute import compute_embedding_sync
        result = compute_embedding_sync("hello world", "nomic-v1.5")
        assert isinstance(result, list)
        assert len(result) == 768


def test_compute_batch_returns_multiple():
    """Batch compute should return multiple embeddings."""
    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1] * 768, [0.2] * 768]

    with patch("app.compute._get_model", return_value=mock_model):
        from app.compute import compute_batch_sync
        result = compute_batch_sync(["hello", "world"], "nomic-v1.5")
        assert len(result) == 2
        assert len(result[0]) == 768
```

**Step 2: Run tests to verify failure**

```bash
cd server && pytest tests/test_compute.py -v
```

Expected: FAIL with import error

**Step 3: Implement compute.py**

```python
# server/app/compute.py
import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import Optional
from app.config import settings

# Model cache (per-process)
_model_cache: dict = {}
_executor: Optional[ProcessPoolExecutor] = None


def _get_executor() -> ProcessPoolExecutor:
    """Get or create process pool executor."""
    global _executor
    if _executor is None:
        _executor = ProcessPoolExecutor(max_workers=2)
    return _executor


def _get_model(model_name: str):
    """Load model (cached per process)."""
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer

        # Map model names to HuggingFace IDs
        model_map = {
            "nomic-v1.5": "nomic-ai/nomic-embed-text-v1.5",
            "nomic-v2-moe": "nomic-ai/nomic-embed-text-v2-moe",
        }
        hf_name = model_map.get(model_name, model_name)
        _model_cache[model_name] = SentenceTransformer(hf_name, device=settings.gpu_device)

    return _model_cache[model_name]


def compute_embedding_sync(text: str, model: str) -> list[float]:
    """Compute embedding synchronously (runs in process pool)."""
    model_instance = _get_model(model)
    # nomic models expect 'search_query: ' or 'search_document: ' prefix
    prefixed_text = f"search_document: {text}"
    embedding = model_instance.encode([prefixed_text])[0]
    return embedding.tolist()


def compute_batch_sync(texts: list[str], model: str) -> list[list[float]]:
    """Compute batch embeddings synchronously."""
    model_instance = _get_model(model)
    prefixed_texts = [f"search_document: {t}" for t in texts]
    embeddings = model_instance.encode(prefixed_texts)
    return [e.tolist() for e in embeddings]


async def compute_embedding(text: str, model: str) -> list[float]:
    """Compute embedding asynchronously using process pool."""
    loop = asyncio.get_event_loop()
    executor = _get_executor()
    return await loop.run_in_executor(executor, compute_embedding_sync, text, model)


async def compute_batch(texts: list[str], model: str) -> list[list[float]]:
    """Compute batch embeddings asynchronously."""
    loop = asyncio.get_event_loop()
    executor = _get_executor()
    return await loop.run_in_executor(executor, compute_batch_sync, texts, model)
```

**Step 4: Run tests**

```bash
cd server && pytest tests/test_compute.py -v
```

Expected: 2 tests PASS

**Step 5: Commit**

```bash
git add server/app/compute.py server/tests/test_compute.py
git commit -m "feat(server): add GPU compute with process pool"
```

---

## Task 10: BYOK Passthrough

**Files:**
- Create: `server/app/passthrough.py`
- Create: `server/tests/test_passthrough.py`

**Step 1: Write failing tests**

```python
# server/tests/test_passthrough.py
import pytest
from app.passthrough import validate_endpoint, is_allowed_host, extract_embedding


def test_is_allowed_host_openai():
    """OpenAI should be allowed."""
    assert is_allowed_host("api.openai.com") is True


def test_is_allowed_host_localhost_blocked():
    """Localhost should be blocked."""
    assert is_allowed_host("localhost") is False
    assert is_allowed_host("127.0.0.1") is False


def test_is_allowed_host_private_ip_blocked():
    """Private IPs should be blocked."""
    assert is_allowed_host("192.168.1.1") is False
    assert is_allowed_host("10.0.0.1") is False


def test_validate_endpoint_valid():
    """Valid OpenAI endpoint should pass."""
    validate_endpoint("https://api.openai.com/v1/embeddings")


def test_validate_endpoint_http_rejected():
    """HTTP (not HTTPS) should be rejected."""
    with pytest.raises(ValueError, match="HTTPS required"):
        validate_endpoint("http://api.openai.com/v1/embeddings")


def test_extract_embedding_jsonpath():
    """Should extract embedding using JSONPath."""
    response = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
    result = extract_embedding(response, "$.data[0].embedding")
    assert result == [0.1, 0.2, 0.3]
```

**Step 2: Run tests to verify failure**

```bash
cd server && pytest tests/test_passthrough.py -v
```

Expected: FAIL with import error

**Step 3: Implement passthrough.py**

```python
# server/app/passthrough.py
import ipaddress
import httpx
from urllib.parse import urlparse
from jsonpath_ng import parse as jsonpath_parse
from app.crypto import decrypt_api_key


ALLOWED_HOSTS = [
    "api.openai.com",
    "api.cohere.ai",
    "api.voyageai.com",
    "api.mistral.ai",
    "api.together.xyz",
]


def is_allowed_host(host: str) -> bool:
    """Check if host is in whitelist and not a private IP."""
    # Check against whitelist
    if host in ALLOWED_HOSTS:
        return True

    # Block localhost
    if host in ("localhost", "127.0.0.1", "::1"):
        return False

    # Block private IP ranges
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local:
            return False
    except ValueError:
        # Not an IP address, check if it's in whitelist
        pass

    return False


def validate_endpoint(endpoint: str) -> None:
    """Validate BYOK endpoint URL.

    Raises ValueError if invalid.
    """
    parsed = urlparse(endpoint)

    if parsed.scheme != "https":
        raise ValueError("HTTPS required for BYOK endpoints")

    if not is_allowed_host(parsed.hostname):
        raise ValueError(f"Host not in allowed list: {parsed.hostname}")


def extract_embedding(response: dict, json_path: str) -> list[float]:
    """Extract embedding from response using JSONPath."""
    expr = jsonpath_parse(json_path)
    matches = expr.find(response)

    if not matches:
        raise ValueError(f"No match for JSONPath: {json_path}")

    return matches[0].value


async def call_byok_provider(
    endpoint: str,
    api_key_encrypted: str,
    request_template: dict,
    response_path: str,
    text: str,
) -> list[float]:
    """Call BYOK provider and return embedding."""
    # Decrypt API key
    api_key = decrypt_api_key(api_key_encrypted)

    # Build request body from template
    body = _substitute_template(request_template, text)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            endpoint,
            json=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
        response.raise_for_status()

    return extract_embedding(response.json(), response_path)


def _substitute_template(template: dict, text: str) -> dict:
    """Replace $TEXT placeholders in template."""
    result = {}
    for key, value in template.items():
        if isinstance(value, str):
            result[key] = value.replace("$TEXT", text)
        elif isinstance(value, dict):
            result[key] = _substitute_template(value, text)
        elif isinstance(value, list):
            result[key] = [
                v.replace("$TEXT", text) if isinstance(v, str) else v
                for v in value
            ]
        else:
            result[key] = value
    return result
```

**Step 4: Run tests**

```bash
cd server && pytest tests/test_passthrough.py -v
```

Expected: 6 tests PASS

**Step 5: Commit**

```bash
git add server/app/passthrough.py server/tests/test_passthrough.py
git commit -m "feat(server): add BYOK passthrough with SSRF protection"
```

---

## Task 11: User Management Routes

**Files:**
- Create: `server/app/routes/users.py`
- Create: `server/tests/test_routes_users.py`

**Step 1: Create routes directory**

```bash
mkdir -p server/app/routes
touch server/app/routes/__init__.py
```

**Step 2: Implement user routes**

```python
# server/app/routes/users.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import User, ApiKey
from app.schemas import (
    SignupRequest, SignupResponse,
    LoginRequest, LoginResponse,
    CreateKeyResponse, RevokeKeyResponse,
)
from app.auth import hash_password, verify_password, create_jwt, decode_jwt
from app.crypto import generate_api_key, hash_api_key, get_key_prefix
import secrets

router = APIRouter(prefix="/v1/auth", tags=["auth"])


def generate_user_id() -> str:
    return f"usr_{secrets.token_hex(12)}"


@router.post("/signup", response_model=SignupResponse)
def signup(request: SignupRequest, db: Session = Depends(get_db)):
    """Create new user account."""
    # Check if email exists
    existing = db.query(User).filter(User.email == request.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create user
    user = User(
        id=generate_user_id(),
        email=request.email,
        password_hash=hash_password(request.password),
        tier="free",
    )
    db.add(user)

    # Create initial API key
    api_key = generate_api_key()
    key_record = ApiKey(
        key_hash=hash_api_key(api_key),
        key_prefix=get_key_prefix(api_key),
        user_id=user.id,
        tier=user.tier,
    )
    db.add(key_record)
    db.commit()

    return SignupResponse(user_id=user.id, api_key=api_key)


@router.post("/login", response_model=LoginResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    """Login and get JWT token."""
    user = db.query(User).filter(User.email == request.email).first()

    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_jwt(user_id=user.id)

    return LoginResponse(
        token=token,
        expires_in=86400,
    )


# Key management router (requires JWT auth)
keys_router = APIRouter(prefix="/v1/keys", tags=["keys"])


def get_current_user_from_jwt(authorization: str, db: Session) -> User:
    """Extract user from JWT token."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization")

    token = authorization[7:]
    payload = decode_jwt(token)
    user = db.query(User).filter(User.id == payload["user_id"]).first()

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user


@keys_router.post("", response_model=CreateKeyResponse)
def create_key(
    authorization: str = Depends(lambda: ""),
    db: Session = Depends(get_db),
):
    """Create additional API key."""
    from fastapi import Header
    # Note: This needs proper dependency injection in main.py
    raise HTTPException(status_code=501, detail="Use /v1/auth endpoints")


@keys_router.delete("/{prefix}", response_model=RevokeKeyResponse)
def revoke_key(
    prefix: str,
    db: Session = Depends(get_db),
):
    """Revoke an API key."""
    raise HTTPException(status_code=501, detail="Use /v1/auth endpoints")
```

**Step 3: Write basic test**

```python
# server/tests/test_routes_users.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


def test_signup_creates_user():
    """Signup should create user and return API key."""
    # This will be a full integration test once main.py is set up
    pass  # Placeholder for now


def test_login_returns_token():
    """Login should return JWT token."""
    pass  # Placeholder
```

**Step 4: Commit**

```bash
git add server/app/routes/
git commit -m "feat(server): add user authentication routes"
```

---

## Task 12: Embed Routes

**Files:**
- Create: `server/app/routes/embed.py`

**Step 1: Implement embed routes**

```python
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
```

**Step 2: Commit**

```bash
git add server/app/routes/embed.py
git commit -m "feat(server): add embed routes with cache and compute"
```

---

## Task 13: Provider Routes

**Files:**
- Create: `server/app/routes/providers.py`

**Step 1: Implement provider routes**

```python
# server/app/routes/providers.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import secrets
from app.database import get_db
from app.models import ApiKey, Provider
from app.schemas import ProviderRequest, ProviderResponse
from app.auth import get_current_api_key
from app.crypto import encrypt_api_key
from app.passthrough import validate_endpoint

router = APIRouter(prefix="/v1", tags=["providers"])


@router.post("/provider", response_model=ProviderResponse)
def create_provider(
    request: ProviderRequest,
    api_key: ApiKey = Depends(get_current_api_key),
    db: Session = Depends(get_db),
):
    """Configure BYOK provider."""
    # Validate endpoint
    try:
        validate_endpoint(request.endpoint)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Check if user already has a provider (limit 1 for now)
    existing = db.query(Provider).filter(
        Provider.user_id == api_key.user_id
    ).first()

    if existing:
        # Update existing
        existing.name = request.name
        existing.endpoint = request.endpoint
        existing.api_key_encrypted = encrypt_api_key(request.api_key)
        existing.request_template = request.request_template
        existing.response_path = request.response_path
        db.commit()
        return ProviderResponse(provider_id=existing.id)

    # Create new
    provider = Provider(
        id=f"prov_{secrets.token_hex(12)}",
        user_id=api_key.user_id,
        name=request.name,
        endpoint=request.endpoint,
        api_key_encrypted=encrypt_api_key(request.api_key),
        request_template=request.request_template,
        response_path=request.response_path,
    )
    db.add(provider)
    db.commit()

    return ProviderResponse(provider_id=provider.id)
```

**Step 2: Commit**

```bash
git add server/app/routes/providers.py
git commit -m "feat(server): add BYOK provider configuration route"
```

---

## Task 14: Main Application

**Files:**
- Create: `server/app/main.py`

**Step 1: Implement FastAPI app**

```python
# server/app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
from app.database import engine, Base
from app.redis_client import close_redis
from app.routes import users, embed, providers
from app.schemas import HealthResponse
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    Base.metadata.create_all(bind=engine)
    yield
    # Shutdown
    await close_redis()


app = FastAPI(
    title="Vector Embed Cache API",
    description="Cached embedding service with hybrid compute",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Include routers
app.include_router(users.router)
app.include_router(users.keys_router)
app.include_router(embed.router)
app.include_router(providers.router)


@app.get("/v1/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    return HealthResponse(status="ok", version="1.0.0")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Step 2: Test server starts**

```bash
cd server
DATABASE_URL=sqlite:///./test.db \
REDIS_URL=redis://localhost:6379 \
ENCRYPTION_KEY=test-encryption-key-32bytes \
JWT_SECRET=test-jwt-secret \
python -c "from app.main import app; print('App loaded successfully')"
```

Expected: `App loaded successfully`

**Step 3: Commit**

```bash
git add server/app/main.py
git commit -m "feat(server): add main FastAPI application"
```

---

## Task 15: Alembic Migrations

**Files:**
- Create: `server/alembic.ini`
- Create: `server/alembic/env.py`
- Create: `server/alembic/versions/001_initial.py`

**Step 1: Initialize Alembic**

```bash
cd server && pip install alembic && alembic init alembic
```

**Step 2: Configure alembic/env.py**

```python
# server/alembic/env.py
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
import os
import sys

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.database import Base
from app.models import User, ApiKey, Embedding, Provider, Usage

config = context.config

# Set database URL from environment
config.set_main_option("sqlalchemy.url", os.environ.get("DATABASE_URL", "postgresql://localhost/embeddings"))

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

**Step 3: Create initial migration**

```bash
cd server && alembic revision --autogenerate -m "initial"
```

**Step 4: Commit**

```bash
git add server/alembic/ server/alembic.ini
git commit -m "feat(server): add Alembic database migrations"
```

---

## Task 16: Pre-seeding Script

**Files:**
- Create: `server/scripts/seed.py`

**Step 1: Create seed script**

```python
#!/usr/bin/env python3
# server/scripts/seed.py
"""Pre-seed the embedding cache with common words."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.database import SessionLocal
from app.cache import store_embedding
from app.compute import compute_batch_sync


# Common English words to pre-seed
COMMON_WORDS = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "hello", "world", "yes", "no", "please", "thank", "sorry", "help", "good", "bad",
]


def seed_common_words(model: str = "nomic-v1.5", batch_size: int = 50):
    """Seed cache with common words."""
    print(f"Seeding {len(COMMON_WORDS)} common words...")

    db = SessionLocal()
    try:
        # Process in batches
        for i in range(0, len(COMMON_WORDS), batch_size):
            batch = COMMON_WORDS[i:i + batch_size]
            print(f"  Computing batch {i // batch_size + 1}...")

            embeddings = compute_batch_sync(batch, model)

            for word, embedding in zip(batch, embeddings):
                store_embedding(
                    db=db,
                    text=word,
                    model=model,
                    vector=embedding,
                    tenant_id="public",  # Shared pool
                    public=True,
                )

            print(f"  Stored {len(batch)} embeddings")

        print(f"Done! Seeded {len(COMMON_WORDS)} words to public cache.")

    finally:
        db.close()


if __name__ == "__main__":
    seed_common_words()
```

**Step 2: Make executable**

```bash
chmod +x server/scripts/seed.py
```

**Step 3: Commit**

```bash
git add server/scripts/
git commit -m "feat(server): add pre-seeding script for common words"
```

---

## Task 17: README and Final Documentation

**Files:**
- Create: `server/README.md`

**Step 1: Create server README**

```markdown
# Vector Embed Cache Server

Self-hosted embedding cache with hybrid compute model.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="postgresql://user:pass@localhost/embeddings"
export REDIS_URL="redis://localhost:6379"
export ENCRYPTION_KEY="your-32-byte-secret-key"
export JWT_SECRET="your-jwt-secret"
export GPU_DEVICE="cuda:0"

# Run migrations
alembic upgrade head

# Start server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

- `POST /v1/auth/signup` - Create account
- `POST /v1/auth/login` - Get JWT token
- `POST /v1/embed` - Get embedding
- `POST /v1/embed/batch` - Get batch embeddings
- `POST /v1/provider` - Configure BYOK provider
- `GET /v1/stats` - Usage statistics
- `GET /v1/health` - Health check
- `GET /metrics` - Prometheus metrics

## Tiers

| Tier | Compute | Rate Limit |
|------|---------|------------|
| Free | BYOK (bring your own key) | 1,000/day |
| Paid | Server GPU (nomic-v1.5) | 50,000/day |

## Pre-seeding

```bash
python scripts/seed.py
```
```

**Step 2: Commit**

```bash
git add server/README.md
git commit -m "docs(server): add README with setup instructions"
```

---

## Summary

**Total Tasks:** 17

**Files Created:**
- `server/requirements.txt`
- `server/app/__init__.py`
- `server/app/config.py`
- `server/app/database.py`
- `server/app/models.py`
- `server/app/normalize.py`
- `server/app/crypto.py`
- `server/app/schemas.py`
- `server/app/auth.py`
- `server/app/redis_client.py`
- `server/app/rate_limit.py`
- `server/app/cache.py`
- `server/app/compute.py`
- `server/app/passthrough.py`
- `server/app/routes/__init__.py`
- `server/app/routes/users.py`
- `server/app/routes/embed.py`
- `server/app/routes/providers.py`
- `server/app/main.py`
- `server/alembic/*`
- `server/scripts/seed.py`
- `server/README.md`

**Tests Created:**
- `server/tests/test_models.py`
- `server/tests/test_normalize.py`
- `server/tests/test_crypto.py`
- `server/tests/test_auth.py`
- `server/tests/test_rate_limit.py`
- `server/tests/test_cache.py`
- `server/tests/test_compute.py`
- `server/tests/test_passthrough.py`

**After completing all tasks:**
1. Run full test suite: `cd server && pytest tests/ -v`
2. Start server: `uvicorn app.main:app`
3. Run pre-seeding: `python scripts/seed.py`
4. Test with curl or the Python library
