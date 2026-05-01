# server/tests/test_cors.py
"""Test CORS configuration honors the spec invariant that wildcard
allow_origins is incompatible with allow_credentials=True (bd: embedding-cache-fcx)."""
import importlib

import pytest
from fastapi.testclient import TestClient

from app.config import Settings


def _reload_main():
    """Reload app.config and app.main so the CORS middleware re-reads
    settings (Settings is a module-level singleton; reloading config
    rebuilds it from the current environment)."""
    import app.config as config_mod
    importlib.reload(config_mod)
    import app.main as main_mod
    return importlib.reload(main_mod)


def test_cors_origins_list_empty_when_unset():
    s = Settings(cors_allowed_origins="", jwt_secret="x", encryption_key="y")
    assert s.cors_origins_list == []


def test_cors_origins_list_parses_comma_separated():
    s = Settings(
        cors_allowed_origins="https://a.example, https://b.example ,https://c.example",
        jwt_secret="x",
        encryption_key="y",
    )
    assert s.cors_origins_list == [
        "https://a.example",
        "https://b.example",
        "https://c.example",
    ]


def test_cors_origins_list_drops_blank_entries():
    s = Settings(
        cors_allowed_origins=",,https://only.example,,",
        jwt_secret="x",
        encryption_key="y",
    )
    assert s.cors_origins_list == ["https://only.example"]


def test_cors_unset_serves_wildcard_without_credentials(monkeypatch):
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "")
    main = _reload_main()
    client = TestClient(main.app)

    resp = client.options(
        "/health",
        headers={
            "Origin": "https://anywhere.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.headers.get("access-control-allow-origin") == "*"
    # Wildcard MUST NOT be combined with credentials per CORS spec.
    assert resp.headers.get("access-control-allow-credentials") != "true"


def test_cors_set_echoes_allowed_origin_with_credentials(monkeypatch):
    monkeypatch.setenv(
        "CORS_ALLOWED_ORIGINS",
        "https://admin.example,https://other.example",
    )
    main = _reload_main()
    client = TestClient(main.app)

    resp = client.options(
        "/health",
        headers={
            "Origin": "https://admin.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.headers.get("access-control-allow-origin") == "https://admin.example"
    assert resp.headers.get("access-control-allow-credentials") == "true"


def test_cors_set_rejects_unlisted_origin(monkeypatch):
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://admin.example")
    main = _reload_main()
    client = TestClient(main.app)

    resp = client.options(
        "/health",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    # Starlette omits the allow-origin header entirely when origin doesn't match.
    assert resp.headers.get("access-control-allow-origin") in (None, "")


@pytest.fixture(autouse=True)
def _reset_main_after_test():
    """Reload main with default test env after each test so other suites
    aren't affected by the CORS env mutations above."""
    yield
    _reload_main()
