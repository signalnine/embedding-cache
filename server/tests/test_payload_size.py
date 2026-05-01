"""Test that requests exceeding settings.max_payload_bytes are rejected with HTTP 413.

bd: embedding-cache-5ut
"""
from fastapi.testclient import TestClient

import app.main as main_mod


def test_oversize_request_returns_413(monkeypatch):
    """A POST whose Content-Length exceeds max_payload_bytes is rejected with 413."""
    # Patch the live settings reference used inside main, since other test
    # modules reload app.config and replace the module-level singleton.
    monkeypatch.setattr(main_mod.settings, "max_payload_bytes", 100)
    client = TestClient(main_mod.app)

    big_body = "a" * 200
    resp = client.post(
        "/v1/embed",
        content=big_body,
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 413
    assert "max size" in resp.json()["detail"].lower()


def test_request_under_limit_passes_middleware(monkeypatch):
    """Requests under the limit should not be rejected by payload-size middleware."""
    monkeypatch.setattr(main_mod.settings, "max_payload_bytes", 1_000_000)
    client = TestClient(main_mod.app)

    # No auth header - we expect a 401, not 413, proving the middleware allowed it through.
    resp = client.post(
        "/v1/embed",
        json={"text": "hello"},
    )
    assert resp.status_code != 413
