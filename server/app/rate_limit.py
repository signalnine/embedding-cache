# server/app/rate_limit.py
from datetime import date
from fastapi import HTTPException
from app.redis_client import get_redis
from app.config import settings


class RateLimitExceeded(HTTPException):
    def __init__(self, limit: int, reset_time: str):
        super().__init__(
            status_code=429,
            detail=f"Rate limit exceeded. Limit: {limit}/day. Resets at: {reset_time}"
        )


async def check_rate_limit(user_id: str, tier: str) -> bool:
    """Check if user is within rate limit.

    Raises RateLimitExceeded if over limit.
    Returns True if allowed.
    Skips rate limiting if Redis is not configured.
    """
    redis = await get_redis()

    # Skip rate limiting if Redis is not configured
    if redis is None:
        return True

    today = date.today().isoformat()
    key = f"rate:{user_id}:{today}"

    count = await redis.incr(key)

    # Set expiry on first request of the day
    if count == 1:
        await redis.expire(key, 86400)

    limit = settings.paid_tier_daily_limit if tier == "paid" else settings.free_tier_daily_limit

    if count > limit:
        raise RateLimitExceeded(limit=limit, reset_time=f"{today} 00:00 UTC")

    return True


async def get_usage_count(user_id: str) -> int:
    """Get current day's request count for user."""
    redis = await get_redis()

    # Return 0 if Redis is not configured
    if redis is None:
        return 0

    today = date.today().isoformat()
    key = f"rate:{user_id}:{today}"

    count = await redis.get(key)
    return int(count) if count else 0
