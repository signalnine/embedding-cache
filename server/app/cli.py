"""CLI commands for admin dashboard management."""
import uuid
import click
from app.database import SessionLocal
from app.models import User
from app.auth import hash_password


@click.command()
@click.option('--email', required=True, help='Admin email address')
@click.option('--password', required=True, help='Admin password')
def create_admin(email: str, password: str):
    """Create an admin user for initial setup."""
    db = SessionLocal()
    try:
        # Check if email already exists
        existing = db.query(User).filter(User.email == email).first()
        if existing:
            click.echo(f"Error: User with email {email} already exists")
            return

        user = User(
            id=str(uuid.uuid4()),
            email=email,
            password_hash=hash_password(password),
            tier="paid",
            is_admin=True
        )
        db.add(user)
        db.commit()
        click.echo(f"Admin user created: {email}")
    except Exception as e:
        click.echo(f"Error: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    create_admin()
