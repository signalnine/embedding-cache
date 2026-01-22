import pytest
from click.testing import CliRunner
from app.cli import create_admin
from app.database import SessionLocal, Base, engine
from app.models import User


@pytest.fixture(autouse=True)
def setup_database():
    """Create tables before each test and drop after."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


class TestCreateAdminCLI:
    def test_create_admin_success(self):
        runner = CliRunner()
        result = runner.invoke(create_admin, [
            '--email', 'admin@test.com',
            '--password', 'testpass123'
        ])
        assert result.exit_code == 0
        assert 'Admin user created' in result.output

        # Verify in database
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.email == 'admin@test.com').first()
            assert user is not None
            assert user.is_admin is True
            assert user.tier == 'paid'
            # Clean up
            db.delete(user)
            db.commit()
        finally:
            db.close()

    def test_create_admin_duplicate_email(self):
        runner = CliRunner()
        # First creation
        runner.invoke(create_admin, ['--email', 'dupe@test.com', '--password', 'pass123'])
        # Second creation should fail
        result = runner.invoke(create_admin, ['--email', 'dupe@test.com', '--password', 'pass456'])
        assert 'Error' in result.output

        # Clean up
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.email == 'dupe@test.com').first()
            if user:
                db.delete(user)
                db.commit()
        finally:
            db.close()
