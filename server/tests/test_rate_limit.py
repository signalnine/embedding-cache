# server/tests/test_rate_limit.py
import pytest
from unittest.mock import AsyncMock, patch
from app.rate_limit import check_rate_limit, RateLimitExceeded


@pytest.mark.asyncio
async def test_rate_limit_allows_under_limit():
    """Requests under limit should be allowed."""
    mock_redis = AsyncMock()
    mock_redis.incrby.return_value = 5

    with patch("app.rate_limit.get_redis", return_value=mock_redis):
        result = await check_rate_limit("usr_123", "free")
        assert result is True
        mock_redis.incrby.assert_awaited_once_with("rate:usr_123:" + __import__("datetime").date.today().isoformat(), 1)
        mock_redis.decrby.assert_not_called()


@pytest.mark.asyncio
async def test_rate_limit_blocks_over_limit():
    """Requests over limit should raise RateLimitExceeded and roll back the increment."""
    mock_redis = AsyncMock()
    mock_redis.incrby.return_value = 1001  # Over free tier limit

    with patch("app.rate_limit.get_redis", return_value=mock_redis):
        with pytest.raises(RateLimitExceeded):
            await check_rate_limit("usr_123", "free")
        mock_redis.decrby.assert_awaited_once()
        # Rollback amount must equal the requested charge
        rollback_args = mock_redis.decrby.await_args
        assert rollback_args[0][1] == 1


@pytest.mark.asyncio
async def test_paid_tier_has_higher_limit():
    """Paid tier should allow more requests."""
    mock_redis = AsyncMock()
    mock_redis.incrby.return_value = 5000  # Over free, under paid

    with patch("app.rate_limit.get_redis", return_value=mock_redis):
        result = await check_rate_limit("usr_123", "paid")
        assert result is True


@pytest.mark.asyncio
async def test_zero_count_is_noop():
    """check_rate_limit with count=0 should not touch Redis or raise."""
    mock_redis = AsyncMock()
    with patch("app.rate_limit.get_redis", return_value=mock_redis):
        result = await check_rate_limit("usr_123", "free", count=0)
        assert result is True
        mock_redis.incrby.assert_not_called()


@pytest.mark.asyncio
async def test_batch_charges_in_a_single_atomic_increment():
    """A batch charge should call incrby once with the full count."""
    mock_redis = AsyncMock()
    mock_redis.incrby.return_value = 50

    with patch("app.rate_limit.get_redis", return_value=mock_redis):
        result = await check_rate_limit("usr_123", "free", count=10)
        assert result is True
        mock_redis.incrby.assert_awaited_once()
        assert mock_redis.incrby.await_args[0][1] == 10
        mock_redis.decrby.assert_not_called()


@pytest.mark.asyncio
async def test_over_limit_batch_rolls_back_full_count():
    """An over-limit batch must decrby by the same count it tried to add."""
    mock_redis = AsyncMock()
    # 990 prior + 20 charge = 1010, over the 1000 free tier limit
    mock_redis.incrby.return_value = 1010

    with patch("app.rate_limit.get_redis", return_value=mock_redis):
        with pytest.raises(RateLimitExceeded):
            await check_rate_limit("usr_123", "free", count=20)
        mock_redis.decrby.assert_awaited_once()
        assert mock_redis.decrby.await_args[0][1] == 20
