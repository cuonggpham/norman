"""
Sparse Embedding Service using FastEmbed BM25.

Provides sparse vector generation for hybrid search combining
dense (semantic) + sparse (keyword) vectors.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SparseEmbeddingService:
    """
    BM25 sparse embedding service using FastEmbed.
    
    Generates sparse vectors for keyword-based search,
    complementing dense vectors for semantic search.
    
    Example:
        service = SparseEmbeddingService()
        sparse = service.embed("労働基準法 第一条")
        # Returns: {"indices": [123, 456, ...], "values": [0.5, 0.3, ...]}
    """
    
    def __init__(self, model_name: str = "Qdrant/bm25"):
        """
        Initialize sparse embedding service.
        
        Args:
            model_name: FastEmbed sparse model name. Options:
                - "Qdrant/bm25" (default, general purpose)
                - "prithivida/Splade_PP_en_v1" (SPLADE, English)
        """
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self):
        """
        Lazy load model on first use.
        
        ⚠️ MEMORY IMPACT: ~500MB loaded on first call.
        """
        if self._model is None:
            try:
                from fastembed import SparseTextEmbedding
                logger.warning(f"[HEAVY LOAD] Loading sparse embedding model: {self.model_name} (~500MB)")
                self._model = SparseTextEmbedding(model_name=self.model_name)
                logger.info("✓ Sparse embedding model loaded successfully")
            except ImportError:
                raise ImportError(
                    "fastembed is required for sparse embeddings. "
                    "Install with: pip install fastembed>=0.3.0"
                )
        return self._model
    
    def embed(self, text: str) -> dict[str, list]:
        """
        Embed single text to sparse vector.
        
        Args:
            text: Text to embed
            
        Returns:
            Dict with 'indices' and 'values' lists for sparse vector
        """
        embeddings = list(self.model.embed([text]))
        if not embeddings:
            return {"indices": [], "values": []}
        
        return {
            "indices": embeddings[0].indices.tolist(),
            "values": embeddings[0].values.tolist(),
        }
    
    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[dict[str, list]]:
        """
        Batch embed texts to sparse vectors.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            List of sparse vector dicts
        """
        if not texts:
            return []
        
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = list(self.model.embed(batch))
            
            for emb in embeddings:
                results.append({
                    "indices": emb.indices.tolist(),
                    "values": emb.values.tolist(),
                })
        
        return results
    
    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "loaded": self._model is not None,
        }
