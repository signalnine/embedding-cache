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
