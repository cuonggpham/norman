# Application Configuration
# Load from .env using pydantic-settings

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # App
    app_name: str = "Norman - Japanese Legal RAG"
    app_version: str = "0.1.0"
    debug: bool = False
    
    # OpenAI
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2048
    
    # Qdrant Cloud
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "japanese_laws"
    
    # Neo4j (Phase 4)
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    
    # API Settings
    cors_origins: list[str] = ["*"]
    api_prefix: str = "/api"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra env vars


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
