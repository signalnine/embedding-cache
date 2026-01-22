"""Add is_admin column to users table

Revision ID: 002_add_is_admin
Revises: 001_pgvector
Create Date: 2026-01-21
"""
from alembic import op
import sqlalchemy as sa

revision = '002_add_is_admin'
down_revision = '001_pgvector'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('users', sa.Column('is_admin', sa.Boolean(), nullable=False, server_default='false'))


def downgrade() -> None:
    op.drop_column('users', 'is_admin')
