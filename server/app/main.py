# server/app/main.py
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from prometheus_client import make_asgi_app
from starlette.middleware.sessions import SessionMiddleware
from app.database import engine, Base
from app.redis_client import close_redis
from app.routes import users, embed, providers, search, admin
from app.schemas import HealthResponse
from app.config import settings
from app.admin_auth import verify_admin_jwt, verify_csrf_token


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    Base.metadata.create_all(bind=engine)
    yield
    # Shutdown
    await close_redis()


app = FastAPI(
    title="Vector Embed Cache API",
    description="Cached embedding service with hybrid compute",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def csrf_middleware(request: Request, call_next):
    """Validate CSRF token for admin POST/PUT/DELETE/PATCH requests.

    Checks X-CSRF-Token header against the authenticated user.
    Exempts /admin/login/ (no auth yet) and /admin/logout/ (convenience).
    Only applies to /admin/* routes.
    """
    if request.method in ("POST", "PUT", "DELETE", "PATCH"):
        path = request.url.path

        # Only apply to admin routes, skip login and logout
        if path.startswith("/admin/") and path not in ("/admin/login/", "/admin/logout/"):
            csrf_token = request.headers.get("X-CSRF-Token")
            auth_token = request.cookies.get("auth_token")

            if not csrf_token or not auth_token:
                return JSONResponse(
                    {"detail": "CSRF validation failed"},
                    status_code=403
                )

            try:
                payload = verify_admin_jwt(auth_token)
                if not verify_csrf_token(csrf_token, payload["sub"]):
                    return JSONResponse(
                        {"detail": "CSRF validation failed"},
                        status_code=403
                    )
            except Exception:
                return JSONResponse(
                    {"detail": "CSRF validation failed"},
                    status_code=403
                )

    return await call_next(request)


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session middleware (for flash messages)
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.jwt_secret,
    session_cookie="admin_session",
    max_age=1800,
    same_site="lax",
    https_only=False,  # Set True in production
)

# Static files
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

# Templates
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# Mount Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Include routers
app.include_router(users.router)
app.include_router(users.keys_router)
app.include_router(embed.router)
app.include_router(providers.router)
app.include_router(search.router)
app.include_router(admin.router)


@app.get("/v1/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    return HealthResponse(status="ok", version="1.0.0")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
