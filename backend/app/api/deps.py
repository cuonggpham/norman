"""
Dependency Injection for FastAPI.

Provides factory functions for services and pipelines.
Uses lru_cache for singleton behavior.
"""

import logging
from functools import lru_cache
from typing import Any

from app.core.config import get_settings
from app.core.protocols import VectorStore
from app.llm.openai_provider import OpenAIProvider
from app.llm.query_translator import QueryTranslator
from app.services.embedding import EmbeddingService
from app.db.qdrant import get_qdrant_client, search as qdrant_search, get_collection_name
from app.pipelines.rag import RAGPipeline
from app.pipelines.graph_rag import GraphRAGPipeline

logger = logging.getLogger(__name__)


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
def get_sparse_embedding_service():
    """
    Get cached sparse embedding service for hybrid search.
    
    ⚠️ LAZY LOADED: Only initializes when actually called.
    Model (~500MB) loads on first embed() call.
    """
    from app.services.sparse_embedding import SparseEmbeddingService
    logger.info("[LAZY LOAD] Initializing sparse embedding service")
    return SparseEmbeddingService(model_name="Qdrant/bm25")


@lru_cache
def get_hybrid_vector_store():
    """
    Get cached hybrid vector store.
    
    ⚠️ LAZY LOADED: Only initializes when use_hybrid_search=True.
    """
    from app.db.hybrid_store import QdrantHybridStore
    client = get_qdrant_client()
    logger.info("[LAZY LOAD] Initializing hybrid vector store")
    return QdrantHybridStore(
        client=client,
        prefetch_limit=20,  # Balance between quality and speed
    )


@lru_cache
def get_reranker():
    """
    Get cached BGE reranker (optional).
    
    ⚠️ MEMORY HEAVY: BGE model loads ~1-2GB into RAM.
    Returns None if disabled or if loading fails.
    """
    settings = get_settings()
    
    # Check if explicitly disabled (default is disabled for now)
    if not getattr(settings, 'reranker_enabled', False):  # Changed default to False
        logger.info("✓ Reranker disabled - skipping model load (saves ~560MB RAM)")
        return None
    
    try:
        from app.services.reranker import BGEReranker
        logger.warning("[HEAVY LOAD] Loading BGE reranker v2-m3 (~560MB RAM)...")
        return BGEReranker()
    except Exception as e:
        logger.warning(f"Failed to load reranker, continuing without: {e}")
        return None


@lru_cache
def get_rag_pipeline() -> RAGPipeline:
    """
    Get cached RAG pipeline (vector-only, for backwards compatibility).
    
    Memory optimization:
    - Only loads hybrid components when use_hybrid_search=True
    - Reranker disabled to save ~1-2GB RAM
    - Sparse embedding lazy loaded (~500MB on first use)
    """
    settings = get_settings()
    use_hybrid = getattr(settings, 'use_hybrid_search', True)
    
    # Conditional loading: only load hybrid components if enabled
    if use_hybrid:
        logger.info("Hybrid search enabled - loading sparse embedding and hybrid store")
        sparse_emb = get_sparse_embedding_service()
        hybrid_st = get_hybrid_vector_store()
    else:
        logger.info("Hybrid search disabled - using vector-only search (saves ~500MB RAM)")
        sparse_emb = None
        hybrid_st = None
    
    return RAGPipeline(
        embedding=get_embedding_service(),
        vector_store=get_vector_store(),
        llm=get_llm_provider(),
        reranker=get_reranker(),  # ✅ Use bge-reranker-v2-m3 for better multilingual
        translator=get_query_translator(),  # Vietnamese → Japanese
        # Hybrid search configuration
        use_hybrid_search=use_hybrid,
        sparse_embedding=sparse_emb,
        hybrid_store=hybrid_st,
    )


@lru_cache
def get_graphrag_pipeline() -> GraphRAGPipeline:
    """
    Get cached GraphRAG pipeline (graph + vector search).
    
    This is the recommended pipeline that combines:
    - Graph search for entity lookups (第32条, 労働基準法)
    - Vector search for semantic similarity
    - Intelligent query routing to choose the best strategy
    
    Performance: ~3-8s faster than vector-only on entity queries.
    """
    settings = get_settings()
    use_hybrid = getattr(settings, 'use_hybrid_search', True)
    
    # Load hybrid components if enabled
    if use_hybrid:
        sparse_emb = get_sparse_embedding_service()
        hybrid_st = get_hybrid_vector_store()
    else:
        sparse_emb = None
        hybrid_st = None
    
    logger.info("✅ GraphRAG pipeline initialized (graph + vector search)")
    
    return GraphRAGPipeline(
        embedding=get_embedding_service(),
        vector_store=get_vector_store(),
        llm=get_llm_provider(),
        reranker=get_reranker(),  # ✅ Use bge-reranker-v2-m3 for better multilingual
        translator=get_query_translator(),
        sparse_embedding=sparse_emb,
        hybrid_store=hybrid_st,
        use_hybrid_search=use_hybrid,
        use_graph=True,  # Enable graph search
        graph_weight=1.2,  # Boost graph results slightly
    )


# For FastAPI Depends()
def get_pipeline() -> GraphRAGPipeline:
    """
    Dependency for FastAPI routes.
    
    Uses GraphRAGPipeline which intelligently routes queries:
    - ENTITY_LOOKUP → Graph search (e.g., "第32条 là gì?")
    - SEMANTIC → Vector search (e.g., "Thời gian làm việc tối đa?")  
    - HYBRID → Both combined (e.g., "労働基準法の规定について")
    """
    return get_graphrag_pipeline()


