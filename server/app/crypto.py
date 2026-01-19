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
