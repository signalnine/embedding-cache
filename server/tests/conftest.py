# server/tests/conftest.py
# Set environment variables BEFORE any imports that read them
import os
os.environ["ENCRYPTION_KEY"] = "test-secret-key-for-testing-only"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["JWT_SECRET"] = "test-jwt-secret-for-testing-only"

import pytest
