"""Configuration settings for Ananke2."""

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings."""

    # Neo4j settings
    NEO4J_URI: str = "bolt://0.0.0.0:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"

    # Chroma settings
    CHROMA_HOST: str = "0.0.0.0"
    CHROMA_PORT: int = 8000
    CHROMA_COLLECTION: str = "ananke2"

    # MySQL settings
    MYSQL_HOST: str = "0.0.0.0"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = "password"
    MYSQL_DATABASE: str = "ananke2"

    # Redis settings
    REDIS_HOST: str = "0.0.0.0"  # Bind to all interfaces
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    # Celery settings
    CELERY_BROKER_URL: str = "redis://0.0.0.0:6379/0"  # Using local Redis instance
    CELERY_RESULT_BACKEND: str = "redis://0.0.0.0:6379/0"

    class Config:
        """Pydantic config."""
        env_file = ".env"
