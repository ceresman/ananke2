"""Configuration settings for Ananke2."""

from pydantic_settings import BaseSettings
from pydantic import Field, computed_field
import os

class Settings(BaseSettings):
    """Application settings."""

    # Docker network flag
    DOCKER_NETWORK: bool = Field(default=False, description="Whether to use Docker networking")

    # Neo4j settings
    NEO4J_URI: str = "bolt://neo4j:7687"  # Use container name
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "testpassword123"  # Match docker-compose.yml

    # Chroma settings
    CHROMA_HOST: str = "chromadb"  # Use container name in Docker
    CHROMA_PORT: int = 8000
    CHROMA_COLLECTION: str = "ananke2"

    # MySQL settings
    MYSQL_HOST: str = "mysql"  # Use container name
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "ananke"
    MYSQL_PASSWORD: str = "anankepass"
    MYSQL_DATABASE: str = "ananke"

    # Redis settings
    REDIS_HOST: str = "redis"  # Use container name
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    # Celery settings
    CELERY_BROKER_URL: str = Field(
        default="redis://redis:6379/0",
        description="Celery broker URL"
    )
    CELERY_RESULT_BACKEND: str = Field(
        default="redis://redis:6379/0",
        description="Celery result backend URL"
    )

    # Qwen API settings
    QWEN_API_KEY: str = Field(
        default="sk-46e78b90eb8e4d6ebef79f265891f238",
        description="API key for Qwen model"
    )

    def get_neo4j_uri(self) -> str:
        """Get Neo4j URI based on environment."""
        host = "neo4j" if self.DOCKER_NETWORK else "localhost"
        return f"bolt://{host}:7687"

    def get_mysql_host(self) -> str:
        """Get MySQL host based on environment."""
        return "mysql" if self.DOCKER_NETWORK else "localhost"

    def get_redis_host(self) -> str:
        """Get Redis host based on environment."""
        return "redis" if self.DOCKER_NETWORK else "localhost"

    def get_chroma_host(self) -> str:
        """Get ChromaDB host based on environment."""
        return "chromadb" if self.DOCKER_NETWORK else "localhost"

    def get_celery_broker_url(self) -> str:
        """Get Celery broker URL based on environment."""
        if self.DOCKER_NETWORK:
            return f"redis://{self.get_redis_host()}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return os.getenv("CELERY_BROKER_URL", "sqlite:///celery-broker.sqlite")

    def get_celery_result_backend(self) -> str:
        """Get Celery result backend URL based on environment."""
        if self.DOCKER_NETWORK:
            return f"redis://{self.get_redis_host()}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return os.getenv("CELERY_RESULT_BACKEND", "sqlite:///celery-results.sqlite")

    @computed_field
    @property
    def neo4j_uri(self) -> str:
        """Get Neo4j URI."""
        return self.get_neo4j_uri()

    @computed_field
    @property
    def neo4j_user(self) -> str:
        """Get Neo4j username."""
        return self.NEO4J_USER

    @computed_field
    @property
    def neo4j_password(self) -> str:
        """Get Neo4j password."""
        return self.NEO4J_PASSWORD

    @computed_field
    @property
    def chroma_uri(self) -> str:
        """Get ChromaDB URI."""
        return f"{self.get_chroma_host()}:{self.CHROMA_PORT}"

    @computed_field
    @property
    def mysql_uri(self) -> str:
        """Get MySQL URI."""
        return f"mysql+aiomysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}@{self.get_mysql_host()}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"

    @computed_field
    @property
    def qwen_api_key(self) -> str:
        """Get Qwen API key."""
        return self.QWEN_API_KEY

    @computed_field
    @property
    def broker_url(self) -> str:
        """Get Celery broker URL."""
        return self.get_celery_broker_url()

    @computed_field
    @property
    def result_backend(self) -> str:
        """Get Celery result backend URL."""
        return self.get_celery_result_backend()

    class Config:
        """Pydantic config."""
        env_file = ".env"

# Create settings instance
settings = Settings()
