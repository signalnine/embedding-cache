# server/app/redis_client.py
import redis.asyncio as redis
from app.config import settings

redis_client: redis.Redis | None = None
redis_available: bool | None = None


async def get_redis() -> redis.Redis | None:
    """Get Redis connection, or None if not configured."""
    global redis_client, redis_available

    # If we know Redis is not available, return None
    if redis_available is False:
        return None

    # Check if Redis URL is configured
    if not settings.redis_url or not settings.redis_url.startswith(("redis://", "rediss://", "unix://")):
        redis_available = False
        return None

    if redis_client is None:
        redis_client = redis.from_url(settings.redis_url)
        redis_available = True
    return redis_client


async def close_redis():
    """Close Redis connection."""
    global redis_client
    if redis_client:
        await redis_client.close()
        redis_client = None
