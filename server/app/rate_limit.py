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


async def check_rate_limit(user_id: str, tier: str, count: int = 1) -> bool:
    """Check if user is within rate limit, charging `count` against the daily quota.

    The increment is applied atomically: if it would exceed the limit, the
    counter is rolled back to its prior value and RateLimitExceeded is raised.

    Args:
        user_id: User identifier
        tier: "paid" or "free"
        count: Number of units to charge (default 1)

    Raises:
        RateLimitExceeded: If charging `count` would exceed the daily limit.

    Returns:
        True if charged successfully.

    Skips rate limiting if Redis is not configured or if count is 0.
    """
    if count <= 0:
        return True

    redis = await get_redis()

    # Skip rate limiting if Redis is not configured
    if redis is None:
        return True

    today = date.today().isoformat()
    key = f"rate:{user_id}:{today}"

    new_count = await redis.incrby(key, count)

    # Set expiry on first request of the day
    if new_count == count:
        await redis.expire(key, 86400)

    limit = settings.paid_tier_daily_limit if tier == "paid" else settings.free_tier_daily_limit

    if new_count > limit:
        # Roll back the increment so partial charges don't stick.
        await redis.decrby(key, count)
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
