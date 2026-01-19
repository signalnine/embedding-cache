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
