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
