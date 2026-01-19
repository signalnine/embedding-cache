# server/tests/test_rate_limit.py
import pytest
from unittest.mock import AsyncMock, patch
from app.rate_limit import check_rate_limit, RateLimitExceeded


@pytest.mark.asyncio
async def test_rate_limit_allows_under_limit():
    """Requests under limit should be allowed."""
    mock_redis = AsyncMock()
    mock_redis.incr.return_value = 5

    with patch("app.rate_limit.get_redis", return_value=mock_redis):
        result = await check_rate_limit("usr_123", "free")
        assert result is True


@pytest.mark.asyncio
async def test_rate_limit_blocks_over_limit():
    """Requests over limit should raise RateLimitExceeded."""
    mock_redis = AsyncMock()
    mock_redis.incr.return_value = 1001  # Over free tier limit

    with patch("app.rate_limit.get_redis", return_value=mock_redis):
        with pytest.raises(RateLimitExceeded):
            await check_rate_limit("usr_123", "free")


@pytest.mark.asyncio
async def test_paid_tier_has_higher_limit():
    """Paid tier should allow more requests."""
    mock_redis = AsyncMock()
    mock_redis.incr.return_value = 5000  # Over free, under paid

    with patch("app.rate_limit.get_redis", return_value=mock_redis):
        result = await check_rate_limit("usr_123", "paid")
        assert result is True
