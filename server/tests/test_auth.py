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
