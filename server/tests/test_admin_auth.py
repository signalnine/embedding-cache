import pytest
from datetime import datetime, timedelta
from app.admin_auth import (
    create_admin_jwt,
    verify_admin_jwt,
    generate_csrf_token,
    verify_csrf_token,
)


class TestAdminJWT:
    def test_create_and_verify_jwt(self):
        token = create_admin_jwt("user-123")
        payload = verify_admin_jwt(token)
        assert payload["sub"] == "user-123"
        assert "exp" in payload

    def test_expired_jwt_raises(self):
        token = create_admin_jwt("user-123", expires_delta=timedelta(seconds=-1))
        with pytest.raises(Exception):
            verify_admin_jwt(token)

    def test_invalid_jwt_raises(self):
        with pytest.raises(Exception):
            verify_admin_jwt("invalid-token")


class TestCSRF:
    def test_generate_and_verify_csrf(self):
        token = generate_csrf_token("user-123")
        assert verify_csrf_token(token, "user-123")

    def test_csrf_wrong_user_fails(self):
        token = generate_csrf_token("user-123")
        assert not verify_csrf_token(token, "user-456")

    def test_csrf_expired_fails(self):
        token = generate_csrf_token("user-123")
        # max_age=-1 ensures the token is always considered expired
        assert not verify_csrf_token(token, "user-123", max_age=-1)

    def test_csrf_invalid_token_fails(self):
        assert not verify_csrf_token("invalid", "user-123")
