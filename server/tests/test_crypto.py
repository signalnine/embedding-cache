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


# bd: embedding-cache-lz1 -- _get_fernet must derive the Fernet key once
# (PBKDF2 100k iterations) and cache the result. decrypt_api_key is on the
# hot path of every BYOK request and cannot afford a per-call key derivation.

def test_fernet_derived_once_across_many_calls():
    """Repeated encrypt/decrypt calls must not re-run PBKDF2."""
    from unittest.mock import patch
    import app.crypto as crypto_mod

    crypto_mod._get_fernet_cached.cache_clear()

    with patch("app.crypto.PBKDF2HMAC", wraps=crypto_mod.PBKDF2HMAC) as spy:
        for _ in range(50):
            ciphertext = crypto_mod.encrypt_api_key("sk-test")
            crypto_mod.decrypt_api_key(ciphertext)

    # Exactly one KDF derivation across 50 round-trips.
    assert spy.call_count == 1


def test_fernet_changes_when_encryption_key_changes(monkeypatch):
    """Cache key is the encryption_key, so a different key produces a new Fernet."""
    import app.crypto as crypto_mod

    crypto_mod._get_fernet_cached.cache_clear()

    monkeypatch.setattr(crypto_mod.settings, "encryption_key", "key-one")
    f1 = crypto_mod._get_fernet()
    monkeypatch.setattr(crypto_mod.settings, "encryption_key", "key-two")
    f2 = crypto_mod._get_fernet()

    assert f1 is not f2
