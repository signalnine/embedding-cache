# server/app/routes/__init__.py
from app.routes.users import router as users_router, keys_router
from app.routes.embed import router as embed_router

__all__ = ["users_router", "keys_router", "embed_router"]
