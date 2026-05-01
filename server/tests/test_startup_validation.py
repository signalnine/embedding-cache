# server/tests/test_startup_validation.py
"""Test that validate_secrets() is invoked at startup (bd: embedding-cache-bo0)."""
import pytest

from app import main
from app.config import Settings


@pytest.mark.asyncio
async def test_lifespan_raises_when_jwt_secret_missing(monkeypatch):
    """Startup must fail when JWT_SECRET is empty."""
    monkeypatch.setattr(main.settings, "jwt_secret", "")
    monkeypatch.setattr(main.settings, "encryption_key", "valid-key")

    with pytest.raises(ValueError, match="JWT_SECRET"):
        async with main.lifespan(main.app):
            pass


@pytest.mark.asyncio
async def test_lifespan_raises_when_encryption_key_missing(monkeypatch):
    """Startup must fail when ENCRYPTION_KEY is empty."""
    monkeypatch.setattr(main.settings, "jwt_secret", "valid-secret")
    monkeypatch.setattr(main.settings, "encryption_key", "")

    with pytest.raises(ValueError, match="ENCRYPTION_KEY"):
        async with main.lifespan(main.app):
            pass


@pytest.mark.asyncio
async def test_lifespan_raises_when_both_secrets_missing(monkeypatch):
    """Startup must fail when both secrets are empty, listing both."""
    monkeypatch.setattr(main.settings, "jwt_secret", "")
    monkeypatch.setattr(main.settings, "encryption_key", "")

    with pytest.raises(ValueError, match="JWT_SECRET.*ENCRYPTION_KEY"):
        async with main.lifespan(main.app):
            pass


def test_validate_secrets_raises_for_empty_jwt():
    """Settings.validate_secrets() raises when jwt_secret is empty."""
    s = Settings(jwt_secret="", encryption_key="x")
    with pytest.raises(ValueError, match="JWT_SECRET"):
        s.validate_secrets()


def test_validate_secrets_raises_for_empty_encryption_key():
    """Settings.validate_secrets() raises when encryption_key is empty."""
    s = Settings(jwt_secret="x", encryption_key="")
    with pytest.raises(ValueError, match="ENCRYPTION_KEY"):
        s.validate_secrets()


def test_validate_secrets_passes_when_both_set():
    """Settings.validate_secrets() returns None when both set."""
    s = Settings(jwt_secret="x", encryption_key="y")
    assert s.validate_secrets() is None
