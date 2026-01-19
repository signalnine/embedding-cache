# server/tests/test_crypto.py
import pytest
from app.crypto import (
    encrypt_api_key,
    decrypt_api_key,
    hash_api_key,
    generate_api_key,
    get_key_prefix,
)


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


def test_get_key_prefix():
    """Key prefix should return first 12 characters."""
    key = "vec_abc123456789xyz"
    prefix = get_key_prefix(key)
    assert prefix == "vec_abc12345"
    assert len(prefix) == 12
