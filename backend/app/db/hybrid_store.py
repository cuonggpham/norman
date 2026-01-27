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


def _normalize_rrf_scores(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Normalize RRF scores to 0-1 range.
    
    Qdrant RRF uses 1/(k + rank) formula which gives scores like:
    - Rank 1: 0.5 (with k=1)
    - Rank 2: 0.333
    - Rank 3: 0.25
    
    This normalizes them so first result = 1.0 and scales others proportionally.
    """
    if not results:
        return results
    
    # Get max score for normalization
    max_score = max(r.get("score", 0) for r in results)
    
    if max_score <= 0:
        return results
    
    # Normalize all scores to 0-1 range relative to max
    normalized = []
    for r in results:
        r_copy = r.copy()
        original_score = r.get("score", 0)
        r_copy["score"] = original_score / max_score
        r_copy["original_rrf_score"] = original_score
        normalized.append(r_copy)
    
    return normalized


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
        Scores are normalized to 0-1 range (highest = 1.0).
        
        Args:
            dense_vector: Dense query embedding
            sparse_vector: Sparse query vector with 'indices' and 'values'
            top_k: Number of results to return
            filters: Metadata filters
            
        Returns:
            List of results with id, score, payload
        """
        results = qdrant_hybrid_search(
            client=self.client,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            top_k=top_k,
            collection_name=self.collection_name,
            filter_conditions=filters,
            prefetch_limit=self.prefetch_limit,
        )
        
        # Normalize RRF scores to 0-1 range
        return _normalize_rrf_scores(results)

