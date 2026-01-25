"""
BGE Reranker Service using FlagEmbedding.

Uses BAAI/bge-reranker-large cross-encoder for reranking search results.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Lazy import to avoid loading torch at module import time
_reranker_model = None


class BGEReranker:
    """
    BGE-based reranker for Japanese legal documents.
    
    Uses BAAI/bge-reranker-large cross-encoder model.
    Runs on CPU for systems without GPU.
    
    Example:
        reranker = BGEReranker()
        docs = [{"payload": {"text": "労働時間は1日8時間"}, "score": 0.8}]
        reranked = reranker.rerank("working hours", docs, top_k=5)
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        use_fp16: bool = False,  # CPU doesn't support fp16 well
        device: str = "cpu",
    ):
        """
        Initialize BGE reranker.
        
        Args:
            model_name: HuggingFace model name
            use_fp16: Use half precision (only for GPU)
            device: "cuda" or "cpu"
        """
        global _reranker_model
        
        if _reranker_model is None:
            logger.warning(f"[HEAVY LOAD] Loading BGE reranker: {model_name} (device={device}) - ~1-2GB RAM")
            from FlagEmbedding import FlagReranker
            _reranker_model = FlagReranker(
                model_name,
                use_fp16=use_fp16,
                device=device,
            )
            logger.info("✓ BGE reranker loaded successfully")
        else:
            logger.info("Using existing BGE reranker instance (already in memory)")
        
        self.model = _reranker_model
    
    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Rerank documents by relevance to query.
        
        Args:
            query: Search query (Vietnamese or Japanese)
            documents: List of retrieved documents with 'payload' containing 'text'
            top_k: Number of top results to return
            
        Returns:
            Reranked documents with updated scores
        """
        if not documents:
            return []
        
        # Extract texts for reranking
        texts = [doc.get("payload", {}).get("text", "") for doc in documents]
        
        # Filter out empty texts
        valid_pairs = [(i, doc, text) for i, (doc, text) in enumerate(zip(documents, texts)) if text.strip()]
        
        if not valid_pairs:
            logger.warning("No valid documents to rerank")
            return documents[:top_k]
        
        # Create query-document pairs
        pairs = [[query, text] for _, _, text in valid_pairs]
        
        # Compute scores
        try:
            scores = self.model.compute_score(pairs, normalize=True)
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return documents[:top_k]
        
        # Handle single document case
        if not isinstance(scores, list):
            scores = [scores]
        
        # Combine with original documents
        scored_docs = []
        for (orig_idx, doc, _), score in zip(valid_pairs, scores):
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = float(score)
            doc_copy["original_score"] = doc.get("score", 0)
            doc_copy["score"] = float(score)  # Update main score
            scored_docs.append(doc_copy)
        
        # Sort by rerank score
        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        logger.info(f"Reranked {len(scored_docs)} documents, returning top {top_k}")
        
        return scored_docs[:top_k]


# Convenience function for singleton access
_singleton_reranker = None


def get_bge_reranker() -> BGEReranker:
    """Get singleton BGE reranker instance."""
    global _singleton_reranker
    if _singleton_reranker is None:
        _singleton_reranker = BGEReranker()
    return _singleton_reranker
