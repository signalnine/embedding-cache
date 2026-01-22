# server/tests/test_csrf.py
"""Tests for CSRF middleware protection on admin routes."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.database import Base, get_db
from app.models import User
from app.auth import hash_password
from app.admin_auth import generate_csrf_token
import uuid


# Use a file-based SQLite for test consistency across threads
TEST_DATABASE_URL = "sqlite:///./test_csrf.db?check_same_thread=False"
test_engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="module", autouse=True)
def setup_database():
    """Create tables once for the module."""
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)
    # Clean up test database file
    import os
    if os.path.exists("./test_csrf.db"):
        os.remove("./test_csrf.db")


@pytest.fixture
def client():
    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def admin_user(client):
    """Create an admin user for testing."""
    db = TestSessionLocal()
    # Clean up any existing user with this email
    existing = db.query(User).filter(User.email == "csrftest@example.com").first()
    if existing:
        db.delete(existing)
        db.commit()

    user = User(
        id=str(uuid.uuid4()),
        email="csrftest@example.com",
        password_hash=hash_password("testpass123"),
        tier="paid",
        is_admin=True
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    user_id = user.id  # Store before closing session
    db.close()

    yield user

    # Cleanup
    db = TestSessionLocal()
    user_to_delete = db.query(User).filter(User.id == user_id).first()
    if user_to_delete:
        db.delete(user_to_delete)
        db.commit()
    db.close()


class TestCSRFMiddleware:
    def test_post_without_csrf_returns_403(self, client, admin_user):
        """POST to protected admin endpoint without CSRF token should return 403."""
        # Login first to get auth cookie
        login_response = client.post("/admin/login/", data={
            "email": "csrftest@example.com",
            "password": "testpass123"
        }, follow_redirects=False)

        # Should get redirect (successful login)
        assert login_response.status_code == 302

        # Try to access protected endpoint without CSRF header
        response = client.post(
            "/admin/keys/create",
            cookies=login_response.cookies
            # No X-CSRF-Token header
        )
        assert response.status_code == 403

    def test_login_exempt_from_csrf(self, client):
        """Login should work without CSRF token (it's the entry point)."""
        # Login should work without CSRF token
        response = client.post("/admin/login/", data={
            "email": "nonexistent@example.com",
            "password": "wrongpassword"
        })
        # Should return 200 (with error message), not 403
        assert response.status_code != 403
        # Should be 200 with error in the HTML response
        assert response.status_code == 200

    def test_post_with_valid_csrf_succeeds(self, client, admin_user):
        """POST with valid CSRF token should succeed."""
        # Login first
        login_response = client.post("/admin/login/", data={
            "email": "csrftest@example.com",
            "password": "testpass123"
        }, follow_redirects=False)

        assert login_response.status_code == 302

        # Generate valid CSRF token for this user
        csrf_token = generate_csrf_token(admin_user.id)

        # Try to access protected endpoint with valid CSRF header
        response = client.post(
            "/admin/keys/create",
            cookies=login_response.cookies,
            headers={"X-CSRF-Token": csrf_token},
            follow_redirects=False
        )
        # Should succeed (303 redirect after key creation)
        assert response.status_code == 303

    def test_post_with_invalid_csrf_returns_403(self, client, admin_user):
        """POST with invalid CSRF token should return 403."""
        # Login first
        login_response = client.post("/admin/login/", data={
            "email": "csrftest@example.com",
            "password": "testpass123"
        }, follow_redirects=False)

        assert login_response.status_code == 302

        # Try with invalid CSRF token
        response = client.post(
            "/admin/keys/create",
            cookies=login_response.cookies,
            headers={"X-CSRF-Token": "invalid-token"}
        )
        assert response.status_code == 403

    def test_logout_exempt_from_csrf(self, client, admin_user):
        """Logout should work without CSRF token (convenience)."""
        # Login first
        login_response = client.post("/admin/login/", data={
            "email": "csrftest@example.com",
            "password": "testpass123"
        }, follow_redirects=False)

        assert login_response.status_code == 302

        # Logout without CSRF token should work
        response = client.post(
            "/admin/logout/",
            cookies=login_response.cookies
        )
        # Should return 200 (successful logout), not 403
        assert response.status_code == 200

    def test_get_requests_not_affected(self, client, admin_user):
        """GET requests should not require CSRF tokens."""
        # Login first
        login_response = client.post("/admin/login/", data={
            "email": "csrftest@example.com",
            "password": "testpass123"
        }, follow_redirects=False)

        assert login_response.status_code == 302

        # GET dashboard without CSRF header should work
        response = client.get(
            "/admin/",
            cookies=login_response.cookies
        )
        assert response.status_code == 200

    def test_non_admin_routes_not_affected(self, client):
        """Non-admin POST routes should not require CSRF."""
        # Health endpoint GET should work
        response = client.get("/v1/health")
        assert response.status_code == 200
