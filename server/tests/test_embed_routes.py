"""Test that /v1/embed and /v1/embed/batch only charge rate limit for uncached items.

bd: embedding-cache-exm
"""
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as main_mod
from app.auth import get_current_api_key
from app.database import get_db
from app.models import ApiKey
from app.rate_limit import RateLimitExceeded


@pytest.fixture
def fake_api_key():
    key = MagicMock(spec=ApiKey)
    key.user_id = "usr_test"
    key.tier = "paid"
    return key


@pytest.fixture
def client_with_overrides(fake_api_key):
    """TestClient with auth + db dependencies overridden."""
    main_mod.app.dependency_overrides[get_current_api_key] = lambda: fake_api_key
    main_mod.app.dependency_overrides[get_db] = lambda: MagicMock()
    yield TestClient(main_mod.app)
    main_mod.app.dependency_overrides.clear()


def test_single_embed_cache_hit_does_not_charge_rate_limit(client_with_overrides):
    """A cached single embed must not consume rate-limit quota."""
    cached_vec = [0.1] * 768
    with patch("app.routes.embed.get_cached_embedding", return_value=cached_vec), \
         patch("app.routes.embed.check_rate_limit", new_callable=AsyncMock) as mock_rate:
        resp = client_with_overrides.post(
            "/v1/embed",
            json={"text": "already-cached"},
        )

    assert resp.status_code == 200
    assert resp.json()["cached"] is True
    mock_rate.assert_not_awaited()


def test_batch_all_cached_does_not_charge_rate_limit(client_with_overrides):
    """An all-cached batch must consume zero rate-limit quota."""
    cached_vec = [0.1] * 768
    with patch("app.routes.embed.get_cached_embedding", return_value=cached_vec), \
         patch("app.routes.embed.check_rate_limit", new_callable=AsyncMock) as mock_rate:
        resp = client_with_overrides.post(
            "/v1/embed/batch",
            json={"texts": ["a", "b", "c"]},
        )

    assert resp.status_code == 200
    assert resp.json()["cached"] == [True, True, True]
    mock_rate.assert_not_awaited()


def test_batch_mixed_charges_only_miss_count(client_with_overrides):
    """A mixed batch should call check_rate_limit once with len(misses)."""
    cached_vec = [0.1] * 768
    miss_vec = np.array([0.2] * 768, dtype=np.float32)

    # First two cached, last two are misses
    cache_results = [cached_vec, cached_vec, None, None]

    with patch("app.routes.embed.get_cached_embedding", side_effect=cache_results), \
         patch("app.routes.embed.compute_batch", new_callable=AsyncMock, return_value=[miss_vec, miss_vec]), \
         patch("app.routes.embed.store_embedding"), \
         patch("app.routes.embed.check_rate_limit", new_callable=AsyncMock) as mock_rate:
        resp = client_with_overrides.post(
            "/v1/embed/batch",
            json={"texts": ["cached1", "cached2", "fresh1", "fresh2"]},
        )

    assert resp.status_code == 200
    mock_rate.assert_awaited_once()
    # check_rate_limit(user_id, tier, count=2)
    assert mock_rate.await_args.kwargs.get("count") == 2 or (
        len(mock_rate.await_args.args) >= 3 and mock_rate.await_args.args[2] == 2
    )


def test_batch_over_limit_returns_429(client_with_overrides):
    """An over-limit batch should return 429."""
    miss_vec = np.array([0.2] * 768, dtype=np.float32)

    with patch("app.routes.embed.get_cached_embedding", return_value=None), \
         patch("app.routes.embed.compute_batch", new_callable=AsyncMock, return_value=[miss_vec] * 3), \
         patch("app.routes.embed.store_embedding"), \
         patch(
             "app.routes.embed.check_rate_limit",
             new_callable=AsyncMock,
             side_effect=RateLimitExceeded(limit=1000, reset_time="tomorrow"),
         ):
        resp = client_with_overrides.post(
            "/v1/embed/batch",
            json={"texts": ["a", "b", "c"]},
        )

    assert resp.status_code == 429
