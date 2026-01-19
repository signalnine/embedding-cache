# server/app/config.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://localhost/embeddings"
    redis_url: str = "redis://localhost:6379"
    encryption_key: str = ""  # 32-byte hex key for BYOK encryption
    jwt_secret: str = ""
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


settings = Settings()
