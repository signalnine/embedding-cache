"""Admin dashboard routes."""
import os
from datetime import date, timedelta
from typing import Optional

from fastapi import APIRouter, Request, Depends, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.database import get_db
from app.models import User, Embedding, Usage
from app.auth import verify_password
from app.admin_auth import (
    create_admin_jwt,
    generate_csrf_token,
    get_current_admin_user,
    require_admin,
)

router = APIRouter(prefix="/admin", tags=["admin"])
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates"))


def get_csrf_token_for_user(user_id: Optional[str] = None) -> str:
    """Generate CSRF token, using 'anonymous' for unauthenticated users."""
    return generate_csrf_token(user_id or "anonymous")


def _is_postgres(db: Session) -> bool:
    """Check if the database is PostgreSQL."""
    return "postgresql" in str(db.get_bind().url)


def get_dashboard_stats(db: Session, user_id: Optional[str], is_admin: bool) -> dict:
    """Get dashboard summary statistics."""
    # Total users (admin only)
    if is_admin:
        total_users = db.execute(text("SELECT COUNT(*) FROM users")).scalar()
    else:
        total_users = 1

    # Total embeddings
    if is_admin:
        if _is_postgres(db):
            # Use estimate for large table (PostgreSQL only)
            total_embeddings = db.execute(text(
                "SELECT COALESCE(reltuples::bigint, 0) FROM pg_class WHERE relname = 'embeddings'"
            )).scalar() or 0
        else:
            # For SQLite and other databases, use COUNT
            total_embeddings = db.execute(text(
                "SELECT COUNT(*) FROM embeddings"
            )).scalar() or 0
    else:
        total_embeddings = db.execute(text(
            "SELECT COUNT(*) FROM embeddings WHERE tenant_id = :user_id"
        ), {"user_id": user_id}).scalar() or 0

    # Hit rate (last 30 days)
    start_date = date.today() - timedelta(days=30)
    if is_admin:
        result = db.execute(text("""
            SELECT
                COALESCE(SUM(cache_hits), 0) as hits,
                COALESCE(SUM(cache_misses), 0) as misses
            FROM usage WHERE date >= :start_date
        """), {"start_date": start_date}).fetchone()
    else:
        result = db.execute(text("""
            SELECT
                COALESCE(SUM(cache_hits), 0) as hits,
                COALESCE(SUM(cache_misses), 0) as misses
            FROM usage WHERE user_id = :user_id AND date >= :start_date
        """), {"user_id": user_id, "start_date": start_date}).fetchone()

    total = (result.hits or 0) + (result.misses or 0)
    hit_rate = (result.hits or 0) / total if total > 0 else 0

    return {
        "total_users": total_users,
        "total_embeddings": total_embeddings,
        "hit_rate": hit_rate
    }


def get_usage_chart_data(
    db: Session,
    start_date: date,
    user_id: Optional[str],
    is_admin: bool
) -> dict:
    """Get usage data for charts with date gap filling."""
    if _is_postgres(db):
        # PostgreSQL: Use generate_series for date gap filling
        if is_admin:
            query = text("""
                WITH date_range AS (
                    SELECT generate_series(
                        :start_date::date,
                        CURRENT_DATE,
                        '1 day'::interval
                    )::date AS date
                )
                SELECT
                    d.date,
                    COALESCE(SUM(u.cache_hits), 0) as hits,
                    COALESCE(SUM(u.cache_misses), 0) as misses
                FROM date_range d
                LEFT JOIN usage u ON u.date = d.date
                GROUP BY d.date
                ORDER BY d.date
            """)
            result = db.execute(query, {"start_date": start_date}).fetchall()
        else:
            query = text("""
                WITH date_range AS (
                    SELECT generate_series(
                        :start_date::date,
                        CURRENT_DATE,
                        '1 day'::interval
                    )::date AS date
                )
                SELECT
                    d.date,
                    COALESCE(u.cache_hits, 0) as hits,
                    COALESCE(u.cache_misses, 0) as misses
                FROM date_range d
                LEFT JOIN usage u ON u.date = d.date AND u.user_id = :user_id
                ORDER BY d.date
            """)
            result = db.execute(query, {"start_date": start_date, "user_id": user_id}).fetchall()
    else:
        # SQLite/other: Generate dates in Python and fill gaps
        if is_admin:
            query = text("""
                SELECT
                    date,
                    COALESCE(SUM(cache_hits), 0) as hits,
                    COALESCE(SUM(cache_misses), 0) as misses
                FROM usage
                WHERE date >= :start_date
                GROUP BY date
                ORDER BY date
            """)
            db_result = db.execute(query, {"start_date": start_date}).fetchall()
        else:
            query = text("""
                SELECT
                    date,
                    COALESCE(cache_hits, 0) as hits,
                    COALESCE(cache_misses, 0) as misses
                FROM usage
                WHERE user_id = :user_id AND date >= :start_date
                ORDER BY date
            """)
            db_result = db.execute(query, {"start_date": start_date, "user_id": user_id}).fetchall()

        # Build a dict of existing data
        usage_by_date = {}
        for row in db_result:
            row_date = row.date
            # Handle datetime objects, date objects, and strings
            if isinstance(row_date, str):
                # Parse string date (SQLite returns strings)
                from datetime import datetime
                row_date = datetime.fromisoformat(row_date.split()[0]).date()
            elif hasattr(row_date, 'date'):
                row_date = row_date.date()
            usage_by_date[row_date] = (row.hits, row.misses)

        # Fill gaps by generating all dates in range
        result = []
        current_date = start_date
        today = date.today()
        while current_date <= today:
            hits, misses = usage_by_date.get(current_date, (0, 0))
            result.append(type('Row', (), {'date': current_date, 'hits': hits, 'misses': misses})())
            current_date += timedelta(days=1)

    labels = [row.date.strftime("%b %d") for row in result]
    hits = [row.hits for row in result]
    misses = [row.misses for row in result]

    return {
        "labels": labels,
        "datasets": [
            {"name": "Hits", "values": hits},
            {"name": "Misses", "values": misses}
        ]
    }


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

    # Get stats
    stats = get_dashboard_stats(db, user.id, user.is_admin)

    # Get chart data (last 30 days)
    start_date = date.today() - timedelta(days=30)
    stats["chart_data"] = get_usage_chart_data(db, start_date, user.id, user.is_admin)

    return templates.TemplateResponse(request, "admin/dashboard.html", {
        "user": user,
        "csrf_token": csrf_token,
        "stats": stats
    })


@router.get("/users/", response_class=HTMLResponse)
async def users_list(
    request: Request,
    page: int = 1,
    q: Optional[str] = None,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """List all users (admin only)."""
    csrf_token = generate_csrf_token(user.id)

    # Pagination
    per_page = 50
    offset = (page - 1) * per_page

    # Base query - use LIKE for SQLite compatibility (ILIKE is PostgreSQL-specific)
    is_postgres = _is_postgres(db)
    like_op = "ILIKE" if is_postgres else "LIKE"

    query = f"""
        SELECT
            u.id, u.email, u.tier, u.is_admin, u.created_at,
            (SELECT MAX(last_used_at) FROM api_keys WHERE user_id = u.id) as last_active
        FROM users u
    """
    params = {"limit": per_page, "offset": offset}

    # Search filter
    if q:
        query += f" WHERE u.email {like_op} :search"
        params["search"] = f"%{q}%"

    query += " ORDER BY u.created_at DESC LIMIT :limit OFFSET :offset"

    users = db.execute(text(query), params).fetchall()

    # Total count for pagination
    count_query = "SELECT COUNT(*) FROM users"
    if q:
        count_query += f" WHERE email {like_op} :search"
    total = db.execute(text(count_query), {"search": f"%{q}%"} if q else {}).scalar()

    total_pages = (total + per_page - 1) // per_page

    # For HTMX partial updates
    if request.headers.get("HX-Request"):
        return templates.TemplateResponse(request, "admin/users_table.html", {
            "users": users,
            "page": page,
            "total_pages": total_pages,
            "q": q
        })

    return templates.TemplateResponse(request, "admin/users.html", {
        "user": user,
        "csrf_token": csrf_token,
        "users": users,
        "page": page,
        "total_pages": total_pages,
        "q": q
    })


@router.post("/users/{user_id}/toggle-admin")
async def toggle_admin(
    request: Request,
    user_id: str,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Toggle admin status for a user."""
    target = db.query(User).filter(User.id == user_id).first()
    if not target:
        raise HTTPException(404, "User not found")

    # Prevent self-demotion (last admin protection)
    if target.id == admin.id and target.is_admin:
        admin_count = db.query(User).filter(User.is_admin == True).count()
        if admin_count <= 1:
            raise HTTPException(400, "Cannot remove last admin")

    target.is_admin = not target.is_admin
    db.commit()

    return {"status": "ok", "is_admin": target.is_admin}


@router.post("/users/{user_id}/tier")
async def change_tier(
    request: Request,
    user_id: str,
    tier: str = Form(...),
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Change user tier."""
    if tier not in ("free", "paid"):
        raise HTTPException(400, "Invalid tier")

    target = db.query(User).filter(User.id == user_id).first()
    if not target:
        raise HTTPException(404, "User not found")

    target.tier = tier
    db.commit()

    return {"status": "ok", "tier": target.tier}


@router.get("/keys/", response_class=HTMLResponse)
async def api_keys_list(
    request: Request,
    user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """List API keys for current user."""
    csrf_token = generate_csrf_token(user.id)

    query = """
        SELECT key_prefix, created_at, last_used_at, revoked_at
        FROM api_keys
        WHERE user_id = :user_id
        ORDER BY created_at DESC
    """
    keys = db.execute(text(query), {"user_id": user.id}).fetchall()

    # Check for newly created key in session
    new_key = request.session.pop("new_key", None) if hasattr(request, 'session') else None

    return templates.TemplateResponse(request, "admin/api_keys.html", {
        "user": user,
        "csrf_token": csrf_token,
        "keys": keys,
        "new_key": new_key
    })


@router.post("/keys/create")
async def create_api_key(
    request: Request,
    user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Create new API key."""
    from app.crypto import generate_api_key, hash_api_key, get_key_prefix

    raw_key = generate_api_key()
    key_hash = hash_api_key(raw_key)
    key_prefix = get_key_prefix(raw_key)

    db.execute(text("""
        INSERT INTO api_keys (key_hash, key_prefix, user_id, tier, created_at)
        VALUES (:key_hash, :key_prefix, :user_id, :tier, CURRENT_TIMESTAMP)
    """), {
        "key_hash": key_hash,
        "key_prefix": key_prefix,
        "user_id": user.id,
        "tier": user.tier
    })
    db.commit()

    # Store in session to show once
    if hasattr(request, 'session'):
        request.session["new_key"] = raw_key

    return RedirectResponse("/admin/keys/", status_code=303)


@router.post("/keys/{key_prefix}/revoke")
async def revoke_api_key(
    request: Request,
    key_prefix: str,
    user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Revoke an API key."""
    # SQLite doesn't support RETURNING, so we use UPDATE then SELECT
    result = db.execute(text("""
        UPDATE api_keys
        SET revoked_at = CURRENT_TIMESTAMP
        WHERE key_prefix = :key_prefix AND user_id = :user_id AND revoked_at IS NULL
    """), {"key_prefix": key_prefix, "user_id": user.id})

    if result.rowcount == 0:
        raise HTTPException(404, "Key not found or already revoked")

    db.commit()

    # Fetch the updated row
    updated_key = db.execute(text("""
        SELECT key_prefix, created_at, last_used_at, revoked_at
        FROM api_keys
        WHERE key_prefix = :key_prefix AND user_id = :user_id
    """), {"key_prefix": key_prefix, "user_id": user.id}).fetchone()

    # Return updated row HTML for HTMX
    return templates.TemplateResponse(request, "admin/api_key_row.html", {
        "key": updated_key
    })


@router.get("/users/{user_id}/", response_class=HTMLResponse)
async def user_detail(
    request: Request,
    user_id: str,
    user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """View user details."""
    # Non-admins can only view themselves
    if not user.is_admin and user.id != user_id:
        raise HTTPException(403, "Access denied")

    csrf_token = generate_csrf_token(user.id)

    target = db.query(User).filter(User.id == user_id).first()
    if not target:
        raise HTTPException(404, "User not found")

    # Get user's API keys
    keys = db.execute(text("""
        SELECT key_prefix, created_at, last_used_at, revoked_at
        FROM api_keys
        WHERE user_id = :user_id
        ORDER BY created_at DESC
    """), {"user_id": user_id}).fetchall()

    # Get user's usage stats
    start_date = date.today() - timedelta(days=30)
    stats = get_dashboard_stats(db, user_id, is_admin=False)
    stats["chart_data"] = get_usage_chart_data(db, start_date, user_id, is_admin=False)

    return templates.TemplateResponse(request, "admin/user_detail.html", {
        "user": user,
        "csrf_token": csrf_token,
        "target": target,
        "keys": keys,
        "stats": stats
    })


@router.get("/me/", response_class=HTMLResponse)
async def my_profile(
    request: Request,
    user: User = Depends(get_current_admin_user)
):
    """Redirect to own user detail page."""
    return RedirectResponse(f"/admin/users/{user.id}/", status_code=302)
