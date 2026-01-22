"""Admin dashboard routes."""
import os
from typing import Optional

from fastapi import APIRouter, Request, Depends, Form
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User
from app.auth import verify_password
from app.admin_auth import (
    create_admin_jwt,
    generate_csrf_token,
    get_current_admin_user,
)

router = APIRouter(prefix="/admin", tags=["admin"])
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates"))


def get_csrf_token_for_user(user_id: Optional[str] = None) -> str:
    """Generate CSRF token, using 'anonymous' for unauthenticated users."""
    return generate_csrf_token(user_id or "anonymous")


@router.get("/login/", response_class=HTMLResponse)
async def login_page(request: Request):
    """Show login form."""
    csrf_token = get_csrf_token_for_user()
    return templates.TemplateResponse(request, "admin/login.html", {
        "csrf_token": csrf_token,
        "error": None
    })


@router.post("/login/")
async def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    """Process login form."""
    # Find user
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.password_hash):
        return templates.TemplateResponse(request, "admin/login.html", {
            "csrf_token": get_csrf_token_for_user(),
            "error": "Invalid email or password."
        }, status_code=200)

    # Create JWT and set cookies
    token = create_admin_jwt(user.id)
    csrf = generate_csrf_token(user.id)

    response = RedirectResponse(url="/admin/", status_code=302)
    response.set_cookie(
        "auth_token", token,
        httponly=True, secure=False,  # Set secure=True in production
        samesite="lax", max_age=1800
    )
    response.set_cookie(
        "csrf_token", csrf,
        httponly=False,  # JS needs to read this
        secure=False, samesite="lax", max_age=1800
    )
    return response


@router.post("/logout/")
async def logout(request: Request):
    """Clear auth cookies."""
    response = Response(content='{"status": "logged out"}', media_type="application/json")
    response.delete_cookie("auth_token")
    response.delete_cookie("csrf_token")
    return response


@router.get("/", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Main dashboard page."""
    csrf_token = generate_csrf_token(user.id)

    # Placeholder stats (will be implemented in Task 7)
    stats = {
        "total_users": 0,
        "total_embeddings": 0,
        "hit_rate": 0,
        "chart_data": {"labels": [], "datasets": []}
    }

    return templates.TemplateResponse(request, "admin/dashboard.html", {
        "user": user,
        "csrf_token": csrf_token,
        "stats": stats
    })
