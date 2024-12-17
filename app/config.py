"""Configuration settings for Ananke2."""

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings."""

    # Neo4j settings
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"

    # Chroma settings
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    CHROMA_COLLECTION: str = "ananke2"

    # MySQL settings
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = "password"
    MYSQL_DATABASE: str = "ananke2"

    class Config:
        """Pydantic config."""
        env_file = ".env"
