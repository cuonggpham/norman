"""
Core Protocols/Interfaces for the RAG system.

Using Python's Protocol for duck typing - allows easy swapping of implementations
(e.g., OpenAI -> LangChain) without changing consumer code.
"""

from typing import Protocol, runtime_checkable, Any


@runtime_checkable
class LLMProvider(Protocol):
    """Interface for LLM providers (OpenAI, LangChain, etc.)"""
    
    def generate(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """
        Generate response from chat messages.
        
        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": "..."}
            **kwargs: Provider-specific options (temperature, max_tokens, etc.)
            
        Returns:
            Generated text response
        """
        ...
    
    def generate_with_context(
        self,
        query: str,
        context: list[str],
        system_prompt: str | None = None,
    ) -> str:
        """
        Generate response with RAG context.
        
        Args:
            query: User's question
            context: List of retrieved document chunks
            system_prompt: Optional system prompt override
            
        Returns:
            Generated answer with citations
        """
        ...


@runtime_checkable  
class EmbeddingProvider(Protocol):
    """Interface for embedding providers."""
    
    def embed(self, text: str) -> list[float]:
        """Embed single text to vector."""
        ...
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts to vectors."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Interface for vector stores (Qdrant, Pinecone, etc.)"""
    
    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query embedding
            top_k: Number of results
            filters: Metadata filters
            
        Returns:
            List of results with id, score, payload
        """
        ...


@runtime_checkable
class SparseEmbeddingProvider(Protocol):
    """Interface for sparse embedding providers (BM25, SPLADE, etc.)."""
    
    def embed(self, text: str) -> dict[str, list]:
        """
        Embed single text to sparse vector.
        
        Returns:
            Dict with 'indices' and 'values' lists
        """
        ...
    
    def embed_batch(self, texts: list[str]) -> list[dict[str, list]]:
        """Embed multiple texts to sparse vectors."""
        ...


@runtime_checkable
class HybridVectorStore(Protocol):
    """Interface for hybrid vector stores supporting dense + sparse search."""
    
    def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: dict[str, list],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Hybrid search combining dense and sparse vectors.
        
        Args:
            dense_vector: Dense query embedding
            sparse_vector: Sparse query vector with 'indices' and 'values'
            top_k: Number of results
            filters: Metadata filters
            
        Returns:
            List of results with id, score, payload
        """
        ...


@runtime_checkable
class Reranker(Protocol):
    """Interface for rerankers (Phase 3)."""
    
    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Rerank documents by relevance to query.
        
        Args:
            query: Search query
            documents: List of documents with text
            top_k: Number of results to return
            
        Returns:
            Reranked documents
        """
        ...
