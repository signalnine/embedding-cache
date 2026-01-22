# server/tests/test_admin_stats.py
"""Tests for admin dashboard stats queries."""
import pytest
from datetime import date, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database import Base, get_db
from app.models import User, Embedding, Usage
from app.auth import hash_password
import uuid


# Use a file-based SQLite for test consistency
TEST_DATABASE_URL = "sqlite:///./test_admin_stats.db?check_same_thread=False"
test_engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="module", autouse=True)
def setup_database():
    """Create tables once for the module."""
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)
    # Clean up test database file
    import os
    if os.path.exists("./test_admin_stats.db"):
        os.remove("./test_admin_stats.db")


@pytest.fixture
def db_session():
    """Provide a database session for tests."""
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def admin_user(db_session):
    """Create an admin user for testing."""
    # Clean up any existing user
    existing = db_session.query(User).filter(User.email == "statsadmin@example.com").first()
    if existing:
        db_session.delete(existing)
        db_session.commit()

    user = User(
        id=str(uuid.uuid4()),
        email="statsadmin@example.com",
        password_hash=hash_password("testpass123"),
        tier="paid",
        is_admin=True
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    user_id = user.id

    yield user

    # Cleanup
    db = TestSessionLocal()
    user_to_delete = db.query(User).filter(User.id == user_id).first()
    if user_to_delete:
        db.delete(user_to_delete)
        db.commit()
    db.close()


@pytest.fixture
def regular_user(db_session):
    """Create a non-admin user for testing."""
    # Clean up any existing user
    existing = db_session.query(User).filter(User.email == "statsuser@example.com").first()
    if existing:
        db_session.delete(existing)
        db_session.commit()

    user = User(
        id=str(uuid.uuid4()),
        email="statsuser@example.com",
        password_hash=hash_password("testpass123"),
        tier="free",
        is_admin=False
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    user_id = user.id

    yield user

    # Cleanup
    db = TestSessionLocal()
    user_to_delete = db.query(User).filter(User.id == user_id).first()
    if user_to_delete:
        db.delete(user_to_delete)
        db.commit()
    db.close()


class TestDashboardStats:
    def test_get_dashboard_stats_returns_dict(self, db_session, admin_user):
        """Test that get_dashboard_stats returns required keys."""
        from app.routes.admin import get_dashboard_stats

        stats = get_dashboard_stats(db_session, user_id=None, is_admin=True)
        assert "total_users" in stats
        assert "total_embeddings" in stats
        assert "hit_rate" in stats

    def test_get_dashboard_stats_admin_sees_all_users(self, db_session, admin_user, regular_user):
        """Test that admin sees total user count."""
        from app.routes.admin import get_dashboard_stats

        stats = get_dashboard_stats(db_session, user_id=admin_user.id, is_admin=True)
        # Should count at least the two test users
        assert stats["total_users"] >= 2

    def test_get_dashboard_stats_non_admin_sees_one(self, db_session, regular_user):
        """Test that non-admin users see total_users as 1 (just themselves)."""
        from app.routes.admin import get_dashboard_stats

        stats = get_dashboard_stats(db_session, user_id=regular_user.id, is_admin=False)
        assert stats["total_users"] == 1

    def test_get_dashboard_stats_calculates_hit_rate(self, db_session, regular_user):
        """Test that hit rate is calculated correctly."""
        from app.routes.admin import get_dashboard_stats

        # Create usage data directly in this test to ensure proper session
        today = date.today()
        for i in range(5):
            usage_date = today - timedelta(days=i)
            usage = Usage(
                user_id=regular_user.id,
                date=usage_date,
                model="test-model-hitrate",
                cache_hits=10 + i,
                cache_misses=2 + i,
                compute_ms=100 * i
            )
            db_session.add(usage)
        db_session.commit()

        stats = get_dashboard_stats(db_session, user_id=regular_user.id, is_admin=False)
        # With sample data: hits = 10+11+12+13+14=60, misses = 2+3+4+5+6=20
        # hit_rate = 60 / 80 = 0.75
        assert 0 <= stats["hit_rate"] <= 1
        assert stats["hit_rate"] == pytest.approx(0.75, rel=0.01)

    def test_get_dashboard_stats_handles_no_usage(self, db_session, admin_user):
        """Test that hit rate is 0 when there's no usage data."""
        from app.routes.admin import get_dashboard_stats

        # Query for admin user's own data (which should be empty)
        # Note: Don't delete all usage data as it affects other tests
        stats = get_dashboard_stats(db_session, user_id=admin_user.id, is_admin=False)
        # Admin user has no usage data of their own
        assert stats["hit_rate"] == 0


class TestUsageChartData:
    def test_get_usage_chart_data_returns_structure(self, db_session, admin_user):
        """Test that get_usage_chart_data returns correct structure."""
        from app.routes.admin import get_usage_chart_data

        start_date = date.today() - timedelta(days=7)
        data = get_usage_chart_data(db_session, start_date, user_id=None, is_admin=True)
        assert "labels" in data
        assert "datasets" in data

    def test_get_usage_chart_data_fills_gaps(self, db_session, admin_user):
        """Test that date gaps are filled with zeros."""
        from app.routes.admin import get_usage_chart_data

        start_date = date.today() - timedelta(days=7)
        data = get_usage_chart_data(db_session, start_date, user_id=None, is_admin=True)

        # Should have 8 days (today + 7 days back)
        assert len(data["labels"]) == 8

    def test_get_usage_chart_data_has_correct_datasets(self, db_session, admin_user):
        """Test that datasets include hits and misses."""
        from app.routes.admin import get_usage_chart_data

        start_date = date.today() - timedelta(days=7)
        data = get_usage_chart_data(db_session, start_date, user_id=None, is_admin=True)

        assert len(data["datasets"]) == 2
        dataset_names = [ds["name"] for ds in data["datasets"]]
        assert "Hits" in dataset_names
        assert "Misses" in dataset_names

    def test_get_usage_chart_data_user_specific(self, db_session, regular_user):
        """Test that non-admin sees only their own usage data."""
        from app.routes.admin import get_usage_chart_data

        # Create usage data directly in this test to ensure proper session
        today = date.today()
        for i in range(5):
            usage_date = today - timedelta(days=i)
            usage = Usage(
                user_id=regular_user.id,
                date=usage_date,
                model="test-model-specific",
                cache_hits=10 + i,
                cache_misses=2 + i,
                compute_ms=100 * i
            )
            db_session.add(usage)
        db_session.commit()

        start_date = today - timedelta(days=7)
        data = get_usage_chart_data(db_session, start_date, user_id=regular_user.id, is_admin=False)

        # Should have data
        assert len(data["labels"]) == 8

        # Find the hits dataset
        hits_dataset = next(ds for ds in data["datasets"] if ds["name"] == "Hits")
        # Should have some non-zero values from sample data
        assert sum(hits_dataset["values"]) > 0
