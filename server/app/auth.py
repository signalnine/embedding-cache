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
