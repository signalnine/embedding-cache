# server/app/routes/__init__.py
from app.routes.users import router as users_router, keys_router
from app.routes.embed import router as embed_router
from app.routes.providers import router as provider_router
from app.routes.search import router as search_router

__all__ = ["users_router", "keys_router", "embed_router", "provider_router", "search_router"]
