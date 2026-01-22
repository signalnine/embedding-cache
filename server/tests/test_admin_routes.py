# server/tests/test_admin_routes.py
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.database import Base, get_db
from app.models import User
from app.auth import hash_password
import uuid


# Use a file-based SQLite for test consistency across threads
TEST_DATABASE_URL = "sqlite:///./test_admin.db?check_same_thread=False"
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
    if os.path.exists("./test_admin.db"):
        os.remove("./test_admin.db")


@pytest.fixture
def client():
    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def test_user(client):
    """Create a test admin user in the database."""
    db = TestSessionLocal()
    # Clean up any existing user with this email
    existing = db.query(User).filter(User.email == "testadmin@example.com").first()
    if existing:
        db.delete(existing)
        db.commit()

    user = User(
        id=str(uuid.uuid4()),
        email="testadmin@example.com",
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


@pytest.fixture
def non_admin_user(client):
    """Create a non-admin test user in the database."""
    db = TestSessionLocal()
    # Clean up any existing user with this email
    existing = db.query(User).filter(User.email == "nonadmin@example.com").first()
    if existing:
        db.delete(existing)
        db.commit()

    user = User(
        id=str(uuid.uuid4()),
        email="nonadmin@example.com",
        password_hash=hash_password("testpass123"),
        tier="free",
        is_admin=False
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


class TestLoginRoute:
    def test_login_page_loads(self, client):
        response = client.get("/admin/login/")
        assert response.status_code == 200
        assert "Login" in response.text

    def test_login_success_sets_cookie(self, client, test_user):
        response = client.post("/admin/login/", data={
            "email": "testadmin@example.com",
            "password": "testpass123",
            "csrf_token": "test"  # Skipped for login
        }, follow_redirects=False)
        assert response.status_code == 302
        assert "auth_token" in response.cookies

    def test_login_invalid_credentials(self, client, test_user):
        response = client.post("/admin/login/", data={
            "email": "testadmin@example.com",
            "password": "wrongpassword"
        })
        assert response.status_code == 200
        assert "Invalid" in response.text or "error" in response.text.lower()


class TestLogoutRoute:
    def test_logout_clears_cookies(self, client, test_user):
        # First login
        login_response = client.post("/admin/login/", data={
            "email": "testadmin@example.com",
            "password": "testpass123"
        }, follow_redirects=False)

        # Then logout
        cookies = login_response.cookies
        response = client.post("/admin/logout/", cookies=cookies)
        assert response.status_code == 200


class TestProtectedRoutes:
    def test_dashboard_requires_auth(self, client):
        response = client.get("/admin/", follow_redirects=False)
        assert response.status_code == 302
        assert "/admin/login/" in response.headers.get("location", "")

    def test_dashboard_htmx_returns_hx_redirect(self, client):
        response = client.get("/admin/", headers={"HX-Request": "true"})
        assert response.status_code == 200
        assert response.headers.get("HX-Redirect") == "/admin/login/"

    def test_non_admin_can_access_dashboard(self, client, non_admin_user):
        """Non-admin users should be able to access the dashboard (they see their own stats)."""
        # Login as non-admin user
        login_response = client.post("/admin/login/", data={
            "email": "nonadmin@example.com",
            "password": "testpass123"
        }, follow_redirects=False)
        assert login_response.status_code == 302
        assert "auth_token" in login_response.cookies

        # Access dashboard with auth cookie
        dashboard_response = client.get("/admin/", cookies=login_response.cookies)
        assert dashboard_response.status_code == 200
        # Should show dashboard content, not redirect to login
        assert "Dashboard" in dashboard_response.text or "dashboard" in dashboard_response.text.lower()


class TestUsersPage:
    def test_users_page_requires_admin(self, client, test_user, non_admin_user):
        # Login as non-admin
        login_response = client.post("/admin/login/", data={
            "email": non_admin_user.email,
            "password": "testpass123"
        }, follow_redirects=False)

        # Try to access users page
        response = client.get("/admin/users/", cookies=login_response.cookies)
        assert response.status_code == 403

    def test_users_page_accessible_for_admin(self, client, test_user):
        login_response = client.post("/admin/login/", data={
            "email": "testadmin@example.com",
            "password": "testpass123"
        }, follow_redirects=False)

        response = client.get("/admin/users/", cookies=login_response.cookies)
        assert response.status_code == 200
        assert "Users" in response.text
