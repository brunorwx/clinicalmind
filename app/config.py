# app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    database_url: str = "postgresql+asyncpg://clinical:clinical@localhost/clinicalmind"
    database_url_sync: str = "postgresql://clinical:clinical@localhost/clinicalmind"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-4o"
    openai_embed_model: str = "text-embedding-3-small"

    # Auth
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # AWS S3
    s3_bucket: str = "clinicalmind-documents"
    aws_region: str = "us-east-1"

settings = Settings()