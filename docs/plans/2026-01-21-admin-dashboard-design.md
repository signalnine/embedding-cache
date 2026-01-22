# Admin Dashboard Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an admin dashboard for vector-embed-cache server with hybrid access (admins see all tenants, users see own stats).

**Architecture:** HTMX + Jinja2 server-rendered templates with Frappe Charts for visualization. JWT cookie authentication with CSRF protection.

**Tech Stack:** FastAPI, Jinja2, HTMX, Frappe Charts, PostgreSQL

---

## Design Decisions

| Decision | Answer | Source |
|----------|--------|--------|
| Target users | Hybrid - admins see all, users see own | consensus |
| Frontend | HTMX + Jinja templates | consensus |
| Core views | Usage stats + User management + API keys | consensus |
| Authentication | JWT in HttpOnly cookie + CSRF | consensus |
| Admin designation | is_admin boolean column | split (chose YAGNI) |
| Charting | Frappe Charts | consensus (2/3) |

---

## Tenant Model

**In this system, `user_id` = `tenant_id`.** Each user is their own tenant.

For consistency, all dashboard code uses `user_id`. The `embeddings.tenant_id` column maps 1:1 with `users.id`. Queries filter by `user_id` for clarity.

---

## Architecture Overview

**Access Model:**
- **Admins** (`is_admin=true`): See all users, all usage stats, all API keys
- **Users** (`is_admin=false`): See only their own stats and API keys

**File Structure:**
```
server/app/
├── templates/
│   ├── base.html           # Layout with nav, HTMX/Frappe includes
│   ├── login.html          # Login form
│   ├── dashboard.html      # Usage charts
│   ├── users.html          # User list (admin only)
│   ├── user_detail.html    # Single user view
│   └── api_keys.html       # API key management
├── routes/
│   └── admin.py            # Dashboard routes
├── admin_auth.py           # Dashboard auth (JWT cookie, CSRF)
└── static/
    ├── css/
    │   └── admin.css       # Minimal styling
    └── js/
        ├── htmx.min.js     # Vendored HTMX (no CDN)
        └── frappe-charts.min.js  # Vendored Frappe Charts
```

**Database Change:**
- Add `is_admin BOOLEAN DEFAULT false` to `users` table

---

## Security Specifications

### Password Hashing

Uses existing `app/auth.py` which implements **bcrypt** via passlib:

```python
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)
```

### API Key Storage

Uses existing pattern - keys are **SHA-256 hashed** before storage:

```python
import hashlib
import secrets

def generate_api_key() -> tuple[str, str]:
    """Returns (raw_key, hashed_key). Raw key shown once to user."""
    raw_key = f"vec_{secrets.token_urlsafe(32)}"
    hashed = hashlib.sha256(raw_key.encode()).hexdigest()
    return raw_key, hashed

# Storage: api_keys.key_hash = SHA-256 hash
# Display: api_keys.key_prefix = first 12 chars of raw key
```

### Admin Bootstrapping

First admin created via CLI command (run once during deployment):

```bash
# Create first admin user
python -m app.cli create-admin --email admin@example.com --password <password>
```

Implementation in `app/cli.py`:

```python
import click
from app.database import SessionLocal
from app.models import User
from app.auth import hash_password
import uuid

@click.command()
@click.option('--email', required=True)
@click.option('--password', required=True)
def create_admin(email: str, password: str):
    """Create an admin user (for initial setup)."""
    db = SessionLocal()
    try:
        user = User(
            id=str(uuid.uuid4()),
            email=email,
            password_hash=hash_password(password),
            tier="paid",
            is_admin=True
        )
        db.add(user)
        db.commit()
        click.echo(f"Admin user created: {email}")
    except Exception as e:
        click.echo(f"Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    create_admin()
```

---

## Dashboard Views

### 1. Main Dashboard (`/admin/`)
- Cache hit/miss ratio chart (line, last 30 days)
- Total requests chart (bar, by day)
- Summary cards: Total users, Total cached embeddings, Hit rate %
- Admin view: Aggregate across all tenants
- User view: Filtered to own user_id only

### 2. Users Page (`/admin/users/`) - Admin Only
- Table: email, tier (free/paid), created_at, last_active, is_admin
- Search/filter by email
- Pagination: 50 users per page
- Click row → user detail page
- Actions: Change tier, toggle admin (via HTMX partial updates)

### 3. User Detail (`/admin/users/{id}/`)
- User info card
- Their usage stats (hits/misses over time)
- Their API keys list
- Admin: Can view any user
- User: Can only view self (redirects to `/admin/me/`)

### 4. API Keys Page (`/admin/keys/`)
- Table: key_prefix (12 chars), created_at, last_used_at, status
- Actions: Create new key, Revoke key
- New key shown once in modal, then only prefix visible
- Revoked keys shown with strikethrough, filtered by default

### Navigation
- Dashboard | Users (admin only) | API Keys | Logout

---

## Authentication & Authorization

### Login Flow
1. User visits `/admin/login/`
2. Submit email/password form (with CSRF token)
3. Server validates input and verifies credentials (with rate limiting)
4. Server issues JWT with minimal claims
5. JWT stored in cookie: `HttpOnly=true`, `Secure=true`, `SameSite=Lax`
6. CSRF token stored in separate cookie
7. Redirect to `/admin/`

### JWT Claims (Minimal)
```json
{
  "sub": "user_id",
  "exp": 1737500000
}
```

**Note:** `is_admin` and `tier` are NOT stored in JWT. Queried from database on each request.

### Input Validation

All form inputs validated server-side:

```python
from pydantic import BaseModel, EmailStr, constr

class LoginForm(BaseModel):
    email: EmailStr
    password: constr(min_length=8, max_length=128)

class UserUpdateForm(BaseModel):
    tier: Literal["free", "paid"]

# In route handler
@router.post("/admin/login/")
async def login(form: LoginForm = Depends()):
    # Pydantic validates before handler runs
    ...
```

### Route Protection
```python
from fastapi import Request, HTTPException, Depends, Response
from sqlalchemy.orm import Session

def auth_redirect(request: Request) -> Response:
    """Return appropriate redirect for auth failure (handles HTMX)."""
    # HTMX requests need HX-Redirect header, not 302
    # (302 would inject login page HTML into DOM fragment)
    if request.headers.get("HX-Request"):
        return Response(status_code=200, headers={"HX-Redirect": "/admin/login/"})
    raise HTTPException(302, headers={"Location": "/admin/login/"})

async def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    """Get current user from JWT cookie. Queries DB for fresh is_admin/tier."""
    token = request.cookies.get("auth_token")
    if not token:
        return auth_redirect(request)

    try:
        payload = verify_jwt(token)  # Only contains sub, exp
    except JWTError:
        return auth_redirect(request)

    user = db.query(User).filter(User.id == payload["sub"]).first()
    if not user:
        return auth_redirect(request)

    return user  # Fresh is_admin/tier from DB

async def require_admin(request: Request, user: User = Depends(get_current_user)) -> User:
    """Require admin privileges. Checked against DB, not JWT."""
    if not user.is_admin:
        if request.headers.get("HX-Request"):
            raise HTTPException(403, detail="Admin access required")
        raise HTTPException(403, "Admin access required")
    return user
```

### Rate Limiting on Login
```python
from app.rate_limit import check_rate_limit

@router.post("/admin/login/")
async def login(request: Request, form: LoginForm = Depends()):
    # Rate limit: 5 attempts per minute per IP
    client_ip = request.client.host
    if not await check_rate_limit(f"login:{client_ip}", limit=5, window=60):
        raise HTTPException(429, "Too many login attempts. Try again later.")

    # Verify credentials...
```

### CSRF Protection (HTMX-Compatible)

**Problem:** HTMX buttons (`hx-post`, `hx-delete`) don't submit form fields.

**Solution:** CSRF token in header via `htmx:configRequest` event.

```python
from itsdangerous import URLSafeTimedSerializer

def generate_csrf_token(user_id: str) -> str:
    serializer = URLSafeTimedSerializer(settings.JWT_SECRET)
    return serializer.dumps(user_id, salt="csrf")

def verify_csrf_token(token: str, user_id: str, max_age: int = 3600) -> bool:
    serializer = URLSafeTimedSerializer(settings.JWT_SECRET)
    try:
        data = serializer.loads(token, salt="csrf", max_age=max_age)
        return data == user_id
    except:
        return False
```

**Base template CSRF injection:**
```html
<head>
    <meta name="csrf-token" content="{{ csrf_token }}">
    <script src="/static/js/htmx.min.js"></script>
    <script>
        document.body.addEventListener('htmx:configRequest', function(event) {
            event.detail.headers['X-CSRF-Token'] =
                document.querySelector('meta[name="csrf-token"]').content;
        });
    </script>
</head>
```

**CSRF validation middleware:**
```python
@app.middleware("http")
async def csrf_middleware(request: Request, call_next):
    if request.method in ("POST", "PUT", "DELETE", "PATCH"):
        if request.url.path.startswith("/admin/") and request.url.path != "/admin/login/":
            csrf_token = request.headers.get("X-CSRF-Token")
            auth_token = request.cookies.get("auth_token")

            if not csrf_token or not auth_token:
                return JSONResponse({"detail": "CSRF validation failed"}, 403)

            payload = verify_jwt(auth_token)
            if not verify_csrf_token(csrf_token, payload["sub"]):
                return JSONResponse({"detail": "CSRF validation failed"}, 403)

    return await call_next(request)
```

### Token Expiry & Refresh
- **Access token:** 30 minutes
- **Refresh:** On each authenticated request, if token expires in < 5 minutes, issue new token
- **Logout:** Clear both `auth_token` and `csrf_token` cookies

```python
@router.post("/admin/logout/")
async def logout(response: Response):
    """Clear all auth cookies."""
    response.delete_cookie("auth_token")
    response.delete_cookie("csrf_token")
    return {"status": "logged out"}
```

### Session Middleware (for Flash Messages)

```python
from starlette.middleware.sessions import SessionMiddleware

app.add_middleware(
    SessionMiddleware,
    secret_key=settings.JWT_SECRET,
    session_cookie="admin_session",
    max_age=1800,  # 30 minutes
    same_site="lax",
    https_only=True
)

# Usage in routes
@router.post("/admin/keys/create")
async def create_key(request: Request, user: User = Depends(get_current_user)):
    raw_key, hashed = generate_api_key()
    # ... save to DB ...
    request.session["flash"] = {"type": "success", "message": "API key created"}
    request.session["new_key"] = raw_key  # Show once
    return RedirectResponse("/admin/keys/", status_code=303)

# In template
{% if session.flash %}
<div class="flash {{ session.flash.type }}">{{ session.flash.message }}</div>
{% endif %}
```

---

## Database Changes & Queries

### Schema Migration
```sql
ALTER TABLE users ADD COLUMN is_admin BOOLEAN NOT NULL DEFAULT false;
```

### Dashboard Stats Query (with date gap filling)

**Problem:** Days with zero traffic create gaps in charts.

**Solution:** Use `generate_series` to fill gaps:

```sql
-- Dashboard stats (admin view) - fills date gaps with zeros
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
ORDER BY d.date;

-- Dashboard stats (user view) - filtered by user_id
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
ORDER BY d.date;
```

### Other Queries

```python
# User list (admin only) - with pagination
SELECT id, email, tier, is_admin, created_at,
       (SELECT MAX(last_used_at) FROM api_keys WHERE user_id = users.id) as last_active
FROM users
ORDER BY created_at DESC
LIMIT :limit OFFSET :offset;

# API keys for user
SELECT key_prefix, created_at, last_used_at, revoked_at
FROM api_keys
WHERE user_id = :user_id
ORDER BY created_at DESC;
```

### Summary Card Aggregates

```python
# Total users (exact - small table)
SELECT COUNT(*) FROM users;

# Total cached embeddings (estimate for large table)
SELECT reltuples::bigint AS estimate
FROM pg_class WHERE relname = 'embeddings';

# User embedding count (accept exact count - filtered queries are fast)
SELECT COUNT(*) FROM embeddings WHERE tenant_id = :user_id;

# Hit rate (last 30 days)
SELECT
    SUM(cache_hits)::float / NULLIF(SUM(cache_hits + cache_misses), 0)
FROM usage WHERE date >= :start_date;
```

---

## Frontend & HTMX Patterns

### Base Template (`base.html`)
```html
<!DOCTYPE html>
<html>
<head>
    <title>{% block title %}Admin{% endblock %}</title>
    <meta name="csrf-token" content="{{ csrf_token }}">
    <script src="/static/js/htmx.min.js"></script>
    <script src="/static/js/frappe-charts.min.js"></script>
    <link rel="stylesheet" href="/static/css/admin.css">
    <script>
        document.body.addEventListener('htmx:configRequest', function(event) {
            event.detail.headers['X-CSRF-Token'] =
                document.querySelector('meta[name="csrf-token"]').content;
        });
    </script>
</head>
<body>
    {% if session.flash %}
    <div class="flash {{ session.flash.type }}">{{ session.flash.message }}</div>
    {% endif %}
    <nav>
        <a href="/admin/">Dashboard</a>
        {% if user.is_admin %}<a href="/admin/users/">Users</a>{% endif %}
        <a href="/admin/keys/">API Keys</a>
        <button hx-post="/admin/logout/" hx-swap="none"
                hx-on::after-request="window.location='/admin/login/'">Logout</button>
    </nav>
    <main>{% block content %}{% endblock %}</main>
</body>
</html>
```

### HTMX Patterns

```html
<!-- Revoke API key (inline update) -->
<button hx-post="/admin/keys/{{ key.prefix }}/revoke"
        hx-target="closest tr"
        hx-swap="outerHTML">
    Revoke
</button>

<!-- Toggle admin status (with confirmation) -->
<input type="checkbox"
       hx-post="/admin/users/{{ user.id }}/toggle-admin"
       hx-confirm="Change admin status for {{ user.email }}?"
       hx-swap="none"
       {% if user.is_admin %}checked{% endif %}>

<!-- Search users (debounced) -->
<input type="search" name="q"
       hx-get="/admin/users/"
       hx-trigger="keyup changed delay:300ms"
       hx-target="#user-table"
       hx-push-url="true">

<!-- Pagination -->
<a hx-get="/admin/users/?page={{ page + 1 }}"
   hx-target="#user-table"
   hx-push-url="true">Next</a>
```

### Chart Initialization
```html
<div id="hits-chart"></div>
<script>
// Data is server-escaped via tojson filter (safe from XSS)
new frappe.Chart("#hits-chart", {
    data: {{ chart_data | tojson }},
    type: 'line',
    colors: ['#2ecc71', '#e74c3c'],
    axisOptions: {
        xAxisMode: 'tick'
    }
});
</script>
```

---

## Testing Strategy

### Unit Tests

```python
# test_admin_auth.py
- test_login_valid_credentials → sets cookie, redirects to /admin/
- test_login_invalid_credentials → returns 401, no cookie
- test_login_invalid_email_format → returns 422 (validation error)
- test_login_rate_limited → returns 429 after 5 attempts
- test_protected_route_without_cookie → redirects to login
- test_protected_route_htmx_without_cookie → returns HX-Redirect header (not 302)
- test_csrf_token_required_on_post → 403 without token
- test_csrf_token_in_header_accepted → 200 with X-CSRF-Token header
- test_jwt_expiry → redirects to login after expiry
- test_jwt_refresh_when_expiring → new token issued
- test_logout_clears_all_cookies → both auth and csrf cookies deleted

# test_admin_views.py
- test_dashboard_shows_own_stats_for_user
- test_dashboard_shows_all_stats_for_admin
- test_dashboard_fills_date_gaps_with_zeros
- test_users_page_forbidden_for_non_admin
- test_users_page_accessible_for_admin
- test_users_page_pagination
- test_user_detail_own_profile_allowed
- test_user_detail_other_profile_forbidden_for_user

# test_admin_actions.py
- test_create_api_key → returns key once, stores hash
- test_revoke_api_key → sets revoked_at
- test_toggle_admin_requires_admin
- test_change_tier_requires_admin
- test_flash_message_shown_after_action
```

### E2E Tests (Playwright)

```python
# test_admin_e2e.py
- test_login_flow_complete → login form → dashboard
- test_htmx_revoke_key → click revoke, row updates without reload
- test_htmx_toggle_admin → checkbox toggles, persists on reload
- test_htmx_user_search → type in search, table filters
- test_htmx_pagination → click next, URL updates, table changes
- test_logout_clears_session → logout button → redirects to login
- test_flash_message_disappears → message shown, then gone on next page
```

### Error Handling

| Error | Response |
|-------|----------|
| Not authenticated | Redirect to `/admin/login/` |
| Not authorized (non-admin) | 403 page with message |
| Validation error | 422 with field errors |
| User not found | 404 page |
| CSRF validation failed | 403 "Invalid request" |
| Rate limited | 429 "Too many attempts" |
| Database error | 500 page, log details |

---

## Dependencies

Add to `requirements.txt`:
```
jinja2>=3.1.0
python-multipart>=0.0.6  # Form parsing
itsdangerous>=2.1.0      # CSRF tokens
starlette>=0.27.0        # Session middleware (included with FastAPI)
click>=8.0.0             # CLI for admin bootstrap
```

**Vendored JS (download to /static/js/):**
- htmx.min.js from https://unpkg.com/htmx.org@1.9/dist/htmx.min.js
- frappe-charts.min.js from https://cdn.jsdelivr.net/npm/frappe-charts/dist/frappe-charts.min.umd.js

---

## Security Checklist

- [x] Passwords hashed with bcrypt (existing auth.py)
- [x] API keys stored as SHA-256 hashes (existing pattern)
- [x] JWT contains only `sub` and `exp` (no mutable claims)
- [x] `is_admin`/`tier` queried from DB on each request
- [x] CSRF token injected via `htmx:configRequest` for all HTMX requests
- [x] Rate limiting on login (5 attempts/minute/IP)
- [x] Cookies: `HttpOnly`, `Secure`, `SameSite=Lax`
- [x] Logout clears both auth and CSRF cookies
- [x] Vendored JS assets (no CDN in production)
- [x] Input validation via Pydantic on all forms
- [x] Admin bootstrapping via CLI (not web signup)
- [x] Tenant isolation: queries always filter by user_id
- [x] HTMX auth redirects use HX-Redirect header (not 302)

---

## Deployment Checklist

1. Run migration: `alembic upgrade head`
2. Create first admin: `python -m app.cli create-admin --email admin@example.com --password <secure>`
3. Download vendored JS to `/static/js/`
4. Set `JWT_SECRET` environment variable (min 32 chars)
5. Verify HTTPS is configured (required for Secure cookies)
