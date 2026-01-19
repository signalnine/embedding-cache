# server/app/config.py
import warnings
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://localhost/embeddings"
    redis_url: str = "redis://localhost:6379"
    encryption_key: str = ""  # Required for BYOK encryption
    jwt_secret: str = ""  # Required for JWT authentication
    jwt_expiry_seconds: int = 86400
    gpu_device: str = "cuda:0"
    model_version: str = "1.0.0"

    # Rate limits
    free_tier_daily_limit: int = 1000
    paid_tier_daily_limit: int = 50000

    # Batch limits
    max_batch_size: int = 100
    max_payload_bytes: int = 1_000_000

    class Config:
        env_file = ".env"

    def validate_secrets(self) -> None:
        """Validate that required secrets are configured.

        Call this at application startup to fail fast if secrets are missing.
        """
        missing = []
        if not self.jwt_secret:
            missing.append("JWT_SECRET")
        if not self.encryption_key:
            missing.append("ENCRYPTION_KEY")
        if missing:
            raise ValueError(
                f"Required secrets not configured: {', '.join(missing)}. "
                "Set these environment variables before starting the server."
            )


settings = Settings()

# Warn at import time if secrets are not configured (don't fail yet for tests)
if not settings.jwt_secret or not settings.encryption_key:
    warnings.warn(
        "JWT_SECRET and/or ENCRYPTION_KEY not configured. "
        "The server will fail to start. Set these environment variables.",
        UserWarning,
    )
