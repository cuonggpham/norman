"""
Dependency Injection for FastAPI.

Provides factory functions for services and pipelines.
Uses lru_cache for singleton behavior.
"""

from functools import lru_cache
from typing import Any

from app.core.config import get_settings
from app.core.protocols import VectorStore
from app.llm.openai_provider import OpenAIProvider
from app.llm.query_translator import QueryTranslator
from app.services.embedding import EmbeddingService
from app.db.qdrant import get_qdrant_client, search as qdrant_search, get_collection_name
from app.pipelines.rag import RAGPipeline


class QdrantVectorStore:
    """
    Wrapper around Qdrant client to match VectorStore protocol.
    """
    
    def __init__(self, client: Any, collection_name: str | None = None):
        self.client = client
        self.collection_name = collection_name or get_collection_name()
    
    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[dict]:
        """Search Qdrant collection."""
        return qdrant_search(
            client=self.client,
            query_vector=query_vector,
            top_k=top_k,
            collection_name=self.collection_name,
            filter_conditions=filters,
        )


class EmbeddingAdapter:
    """
    Adapter to match EmbeddingService to EmbeddingProvider protocol.
    
    EmbeddingService uses embed_text(), protocol expects embed().
    """
    
    def __init__(self, service: EmbeddingService):
        self._service = service
    
    def embed(self, text: str) -> list[float]:
        """Embed single text."""
        return self._service.embed_text(text)
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        return self._service.embed_batch(texts)


@lru_cache
def get_embedding_service() -> EmbeddingAdapter:
    """Get cached embedding adapter."""
    settings = get_settings()
    service = EmbeddingService(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
    )
    return EmbeddingAdapter(service)


@lru_cache
def get_llm_provider() -> OpenAIProvider:
    """Get cached LLM provider."""
    settings = get_settings()
    return OpenAIProvider(
        api_key=settings.openai_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )


@lru_cache
def get_vector_store() -> QdrantVectorStore:
    """Get cached vector store."""
    settings = get_settings()
    client = get_qdrant_client()
    return QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection_name,
    )


@lru_cache
def get_query_translator() -> QueryTranslator:
    """Get cached query translator for cross-lingual search."""
    return QueryTranslator(llm=get_llm_provider())


@lru_cache
def get_rag_pipeline() -> RAGPipeline:
    """
    Get cached RAG pipeline.
    
    This is the main entry point for RAG operations.
    """
    return RAGPipeline(
        embedding=get_embedding_service(),
        vector_store=get_vector_store(),
        llm=get_llm_provider(),
        reranker=None,  # Add in Phase 3
        translator=get_query_translator(),  # Vietnamese → Japanese
    )


# For FastAPI Depends()
def get_pipeline() -> RAGPipeline:
    """Dependency for FastAPI routes."""
    return get_rag_pipeline()
