"""
Qdrant Hybrid Store Wrapper.

Provides a HybridVectorStore protocol implementation for Qdrant.
"""

from typing import Any, Optional

from qdrant_client import QdrantClient

from app.db.qdrant import (
    hybrid_search as qdrant_hybrid_search,
    get_hybrid_collection_name,
)


class QdrantHybridStore:
    """
    Wrapper class for Qdrant hybrid search.
    
    Implements the HybridVectorStore protocol for use with RAGPipeline.
    
    Example:
        from app.db.qdrant import get_qdrant_client
        
        client = get_qdrant_client()
        hybrid_store = QdrantHybridStore(client)
        
        results = hybrid_store.hybrid_search(
            dense_vector=dense_vec,
            sparse_vector=sparse_vec,
            top_k=10,
        )
    """
    
    def __init__(
        self,
        client: QdrantClient,
        collection_name: Optional[str] = None,
        prefetch_limit: int = 20,
    ):
        """
        Initialize hybrid store.
        
        Args:
            client: Qdrant client instance
            collection_name: Hybrid collection name (default from env)
            prefetch_limit: Number of results to prefetch from each search
        """
        self.client = client
        self.collection_name = collection_name or get_hybrid_collection_name()
        self.prefetch_limit = prefetch_limit
    
    def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: dict[str, list],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Perform hybrid search combining dense and sparse vectors.
        
        Uses RRF (Reciprocal Rank Fusion) to merge results.
        
        Args:
            dense_vector: Dense query embedding
            sparse_vector: Sparse query vector with 'indices' and 'values'
            top_k: Number of results to return
            filters: Metadata filters
            
        Returns:
            List of results with id, score, payload
        """
        return qdrant_hybrid_search(
            client=self.client,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            top_k=top_k,
            collection_name=self.collection_name,
            filter_conditions=filters,
            prefetch_limit=self.prefetch_limit,
        )
