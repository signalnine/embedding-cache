# server/app/crypto.py
import hashlib
import secrets
import base64
from functools import lru_cache
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from app.config import settings


@lru_cache(maxsize=1)
def _get_fernet_cached(encryption_key: str) -> Fernet:
    # PBKDF2 is the expensive step (100k iterations); cache the derived
    # Fernet instance so it is computed once per encryption_key for the
    # lifetime of the process. encryption_key is the cache key so a test
    # that monkeypatches settings.encryption_key still gets a fresh Fernet.
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b"vec-embed-cache-salt",
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(encryption_key.encode()))
    return Fernet(key)


def _get_fernet() -> Fernet:
    """Get Fernet instance from encryption key."""
    if not settings.encryption_key:
        raise ValueError("ENCRYPTION_KEY not set")
    return _get_fernet_cached(settings.encryption_key)


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
