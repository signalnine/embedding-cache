"""Tests for HTTP layer with retry logic."""

from __future__ import annotations

import sys

from vector_embed_client.errors import (
    AuthenticationError,
    RateLimitError,
    ServerError,
    ValidationError,
)
from vector_embed_client.http import (
    RETRY_CONFIG,
    build_headers,
    calculate_delay,
    map_status_to_error,
)


def test_retry_config_defaults():
    assert RETRY_CONFIG["base_delay"] == 1.0
    assert RETRY_CONFIG["max_delay"] == 30.0
    assert RETRY_CONFIG["jitter_factor"] == 0.2
    assert 429 in RETRY_CONFIG["retryable_statuses"]
    assert 500 in RETRY_CONFIG["retryable_statuses"]


def test_calculate_delay_exponential():
    # Attempt 0: base_delay * 2^0 = 1.0 (±jitter)
    delay0 = calculate_delay(0)
    assert 0.8 <= delay0 <= 1.2  # 1.0 ± 20%

    # Attempt 1: base_delay * 2^1 = 2.0 (±jitter)
    delay1 = calculate_delay(1)
    assert 1.6 <= delay1 <= 2.4  # 2.0 ± 20%

    # Attempt 2: base_delay * 2^2 = 4.0 (±jitter)
    delay2 = calculate_delay(2)
    assert 3.2 <= delay2 <= 4.8  # 4.0 ± 20%


def test_calculate_delay_capped():
    # Large attempt should be capped at max_delay
    delay = calculate_delay(10)  # Would be 1024 without cap
    assert delay <= RETRY_CONFIG["max_delay"] * 1.2  # max + jitter


def test_map_status_400_validation_error():
    error = map_status_to_error(400, {"error": "bad input"}, {})
    assert isinstance(error, ValidationError)
    assert error.status == 400


def test_map_status_401_authentication_error():
    error = map_status_to_error(401, {"error": "unauthorized"}, {})
    assert isinstance(error, AuthenticationError)
    assert error.status == 401


def test_map_status_429_rate_limit_error():
    error = map_status_to_error(429, {"error": "rate limited"}, {"retry-after": "30"})
    assert isinstance(error, RateLimitError)
    assert error.status == 429
    assert error.retry_after == 30


def test_map_status_429_no_retry_after():
    error = map_status_to_error(429, {"error": "rate limited"}, {})
    assert isinstance(error, RateLimitError)
    assert error.retry_after is None


def test_map_status_500_server_error():
    error = map_status_to_error(500, {"error": "internal"}, {})
    assert isinstance(error, ServerError)
    assert error.status == 500


def test_map_status_503_server_error():
    error = map_status_to_error(503, {"error": "unavailable"}, {})
    assert isinstance(error, ServerError)
    assert error.status == 503


def test_build_headers():
    headers = build_headers("test_api_key")
    assert headers["Authorization"] == "Bearer test_api_key"
    assert headers["Content-Type"] == "application/json"
    assert headers["Accept"] == "application/json"
    assert "vector-embed-cache-client" in headers["User-Agent"]
    assert f"python/{sys.version_info.major}.{sys.version_info.minor}" in headers["User-Agent"]
