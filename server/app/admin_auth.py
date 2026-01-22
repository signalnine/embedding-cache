"""Admin dashboard authentication - JWT cookies and CSRF tokens."""
from datetime import datetime, timedelta, timezone
from typing import Optional
import jwt
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from fastapi import Request, HTTPException, Depends
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models import User


# JWT for auth cookie
def create_admin_jwt(user_id: str, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT with minimal claims (sub, exp only)."""
    if expires_delta is None:
        expires_delta = timedelta(minutes=30)

    expire = datetime.now(timezone.utc) + expires_delta
    payload = {"sub": user_id, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def verify_admin_jwt(token: str) -> dict:
    """Verify JWT and return payload. Raises on invalid/expired."""
    return jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])


# CSRF tokens
def generate_csrf_token(user_id: str) -> str:
    """Generate CSRF token tied to user."""
    serializer = URLSafeTimedSerializer(settings.jwt_secret)
    return serializer.dumps(user_id, salt="csrf")


def verify_csrf_token(token: str, user_id: str, max_age: int = 3600) -> bool:
    """Verify CSRF token. Returns False on any failure."""
    serializer = URLSafeTimedSerializer(settings.jwt_secret)
    try:
        data = serializer.loads(token, salt="csrf", max_age=max_age)
        return data == user_id
    except (BadSignature, SignatureExpired):
        return False


# Auth redirect helper (handles HTMX)
def auth_redirect(request: Request):
    """Raise appropriate exception for auth failure."""
    if request.headers.get("HX-Request"):
        # For HTMX: 200 response with HX-Redirect header
        raise HTTPException(
            status_code=200,
            headers={"HX-Redirect": "/admin/login/"}
        )
    # For regular requests: 302 redirect
    raise HTTPException(status_code=302, headers={"Location": "/admin/login/"})


# Dependencies
async def get_current_admin_user(
    request: Request,
    db: Session = Depends(get_db)
) -> User:
    """Get current user from JWT cookie. Raises HTTPException on failure."""
    token = request.cookies.get("auth_token")
    if not token:
        auth_redirect(request)  # Now raises, not returns

    try:
        payload = verify_admin_jwt(token)
    except jwt.PyJWTError:
        auth_redirect(request)

    user = db.query(User).filter(User.id == payload["sub"]).first()
    if not user:
        auth_redirect(request)

    return user


async def require_admin(
    request: Request,
    user: User = Depends(get_current_admin_user)
) -> User:
    """Require admin privileges. Raises HTTPException if not admin."""
    if not user.is_admin:
        raise HTTPException(403, detail="Admin access required")
    return user
