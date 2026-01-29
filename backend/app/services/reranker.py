"""
Reranker Service using sentence-transformers CrossEncoder.

Uses cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 for reranking search results.
"""

import logging
from typing import Any
import math

logger = logging.getLogger(__name__)

# Lazy import to avoid loading torch at module import time
_reranker_model = None


class BGEReranker:
    """
    Reranker for Japanese/Multilingual documents using CrossEncoder.
    
    Uses cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 model.
    Runs on CPU for systems without GPU.
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        use_fp16: bool = False,
        device: str = "cpu",
    ):
        """
        Initialize Reranker.
        
        Args:
            model_name: HuggingFace model name
            use_fp16: Use half precision (only for GPU)
            device: "cuda" or "cpu"
        """
        global _reranker_model
        
        if _reranker_model is None:
            logger.warning(f"[HEAVY LOAD] Loading Reranker: {model_name} (device={device})")
            from sentence_transformers import CrossEncoder
            _reranker_model = CrossEncoder(
                model_name,
                device=device,
                # default automodel checks available resources
            )
            logger.info("✓ Reranker loaded successfully")
        else:
            logger.info("Using existing Reranker instance (already in memory)")
        
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
            query: Search query
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
            # CrossEncoder predict returns logits by default for this model
            scores = self.model.predict(pairs)
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return documents[:top_k]
        
        # Handle single document case
        if not isinstance(scores, (list, tuple)) and not getattr(scores, 'shape', None):
             # numpy scalar or float
            scores = [scores]
        elif hasattr(scores, 'tolist'):
            scores = scores.tolist()
            
        if not isinstance(scores, list):
             scores = [scores]

        def sigmoid(x):
            return 1 / (1 + math.exp(-x))
        
        # Apply sigmoid to normalize logits to probabilities [0, 1]
        probs = [sigmoid(s) for s in scores]
        
        # Combine with original documents
        scored_docs = []
        for (orig_idx, doc, _), score in zip(valid_pairs, probs):
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = float(score)
            doc_copy["original_score"] = doc.get("score", 0)
            doc_copy["score"] = float(score)  # Update main score
            scored_docs.append(doc_copy)
        
        # Sort by rerank score
        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        # Normalize scores to 0-1 range (best result = 1.0)
        if scored_docs:
            max_score = scored_docs[0]["rerank_score"]
            if max_score > 0:
                for doc in scored_docs:
                    doc["raw_rerank_score"] = doc["rerank_score"]
                    doc["score"] = doc["rerank_score"] / max_score
        
        logger.info(f"Reranked {len(scored_docs)} documents, returning top {top_k}")
        
        return scored_docs[:top_k]


# Convenience function for singleton access
_singleton_reranker = None


def get_bge_reranker() -> BGEReranker:
    """Get singleton Reranker instance."""
    global _singleton_reranker
    if _singleton_reranker is None:
        _singleton_reranker = BGEReranker()
    return _singleton_reranker
