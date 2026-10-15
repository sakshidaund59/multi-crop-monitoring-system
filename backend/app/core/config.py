import os
from typing import List, Union
from pydantic import AnyHttpUrl, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Multi-Crop Monitoring System"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8100
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]
    # JWT Security Configuration
    SECRET_KEY: str = "crop-monitoring-secret-key-development-32chars"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    # Postgres / PostGIS Connection Details
    POSTGRES_SERVER: str = "127.0.0.1"
    POSTGRES_PORT: int = 15432
    POSTGRES_USER: str = "crop_user"
    POSTGRES_PASSWORD: str = "crop_password_dev"
    POSTGRES_DB: str = "crop_monitoring_db"

    DATABASE_URL: str = (
        "postgresql+asyncpg://crop_user:crop_password_dev@127.0.0.1:15432/crop_monitoring_db"
    )
    SYNC_DATABASE_URL: str = (
        "postgresql+psycopg2://crop_user:crop_password_dev@127.0.0.1:15432/crop_monitoring_db"
    )

    # Redis Connection Details
    REDIS_HOST: str = "127.0.0.1"
    REDIS_PORT: int = 6379
    REDIS_URL: str = "redis://127.0.0.1:6379/0"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
