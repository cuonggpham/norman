"""
Base Pipeline - Abstract base class for RAG pipelines.

Provides shared logic for:
- Query translation
- Embedding (dense + sparse batch)
- Vector search (standard + hybrid)
- Result reranking
- Context building with citations
- LLM response generation
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from app.core.protocols import (
    LLMProvider,
    EmbeddingProvider,
    VectorStore,
    Reranker,
    SparseEmbeddingProvider,
    HybridVectorStore,
)
from app.models.schemas import ChatResponse, SourceDocument

logger = logging.getLogger(__name__)


@runtime_checkable
class QueryTranslator(Protocol):
    """Protocol for query translator."""
    def translate(self, query: str) -> str: ...


@dataclass
class BasePipeline(ABC):
    """
    Abstract base class for RAG pipelines.
    
    Provides shared functionality:
    - Query translation
    - Batch embedding (dense + sparse)
    - Vector search (standard + hybrid)
    - Reranking
    - Context building with citations
    - LLM generation
    
    Subclasses must implement:
    - chat(): Main entry point for the pipeline
    """
    
    # Required dependencies
    embedding: EmbeddingProvider
    vector_store: VectorStore
    llm: LLMProvider
    
    # Optional components
    reranker: Reranker | None = None
    translator: QueryTranslator | None = None
    sparse_embedding: SparseEmbeddingProvider | None = None
    hybrid_store: HybridVectorStore | None = None
    
    # Config
    default_top_k: int = 10
    retrieval_multiplier: int = 4
    min_score_threshold: float = 0.25
    use_hybrid_search: bool = False
    use_multi_query: bool = True
    
    @abstractmethod
    def chat(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        **kwargs,
    ) -> ChatResponse:
        """Main entry point - must be implemented by subclasses."""
        ...
    
    def _translate_query(self, query: str) -> str:
        """
        Translate query using translator if available.
        
        Falls back to original query if translation fails or no translator.
        """
        if not self.translator:
            return query
        
        try:
            translated = self.translator.translate(query)
            if translated and translated != query:
                logger.info(f"Query translated: '{query}' → '{translated}'")
            return translated
        except Exception as e:
            logger.warning(f"Translation failed, using original query: {e}")
            return query
    
    def _get_search_texts(self, query: str) -> list[str]:
        """
        Get search texts - either multi-query expansion or single translation.
        
        Returns:
            List of search queries (Japanese translated)
        """
        if self.use_multi_query and self.translator and hasattr(self.translator, 'get_all_search_texts'):
            search_texts = self.translator.get_all_search_texts(query)
            return search_texts[:3]  # Limit to 3 for performance
        else:
            return [self._translate_query(query)]
    
    def _embed_queries(self, search_texts: list[str], can_hybrid: bool) -> tuple[list, list | None]:
        """
        Batch embed all search texts at once.
        
        Args:
            search_texts: List of queries to embed
            can_hybrid: Whether to also compute sparse embeddings
            
        Returns:
            Tuple of (dense_vectors, sparse_vectors or None)
        """
        all_dense_vectors = self.embedding.embed_batch(search_texts)
        all_sparse_vectors = None
        
        if can_hybrid and self.sparse_embedding:
            all_sparse_vectors = self.sparse_embedding.embed_batch(search_texts)
        
        return all_dense_vectors, all_sparse_vectors
    
    def _vector_search_multi(
        self,
        search_texts: list[str],
        all_dense_vectors: list,
        all_sparse_vectors: list | None,
        retrieve_k: int,
        filters: dict[str, Any] | None,
        can_hybrid: bool,
    ) -> dict[str, dict]:
        """
        Perform vector search with multiple queries and deduplicate.
        
        Returns:
            Dict of chunk_id -> result (deduplicated, highest score kept)
        """
        all_results = {}
        
        for idx, search_text in enumerate(search_texts):
            query_vector = all_dense_vectors[idx]
            
            if can_hybrid and all_sparse_vectors:
                sparse_vector = all_sparse_vectors[idx]
                results = self.hybrid_store.hybrid_search(
                    dense_vector=query_vector,
                    sparse_vector=sparse_vector,
                    top_k=retrieve_k,
                    filters=filters,
                )
                logger.info(f"           Hybrid search {idx+1}/{len(search_texts)}: {len(results)} results")
            else:
                results = self.vector_store.search(
                    query_vector=query_vector,
                    top_k=retrieve_k,
                    filters=filters,
                )
                logger.info(f"           Vector search {idx+1}/{len(search_texts)}: {len(results)} results")
            
            # Merge results, keeping highest score for each chunk
            for r in results:
                chunk_id = r.get("id", str(r.get("payload", {}).get("chunk_id", "")))
                existing_score = all_results.get(chunk_id, {}).get("score", 0)
                if chunk_id not in all_results or r.get("score", 0) > existing_score:
                    r["source"] = "vector"
                    all_results[chunk_id] = r
        
        return all_results
    
    def _filter_and_sort_results(self, all_results: dict[str, dict]) -> list[dict]:
        """
        Sort by score and filter by threshold.
        
        Returns:
            Sorted and filtered list of results
        """
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x.get("score", 0),
            reverse=True
        )
        
        filtered_results = [
            r for r in sorted_results
            if r.get("score", 0) >= self.min_score_threshold
        ]
        
        if not filtered_results:
            logger.warning(f"No results above threshold {self.min_score_threshold}, using top 3")
            filtered_results = sorted_results[:3]
        
        return filtered_results
    
    def _rerank_results(
        self,
        query: str,
        results: list[dict],
        top_k: int,
    ) -> list[dict]:
        """
        Rerank results if reranker is available.
        
        Returns:
            Reranked (or truncated) results
        """
        if self.reranker:
            return self.reranker.rerank(query, results, top_k)
        return results[:top_k]
    
    def _build_context(self, results: list[dict[str, Any]]) -> list[str]:
        """Build context strings from search results with source numbering for citations."""
        context = []
        for idx, r in enumerate(results, start=1):
            payload = r.get("payload", {})
            
            # Use text_with_context if available, otherwise text
            text = payload.get("text_with_context") or payload.get("text", "")
            
            # Add metadata for citation with source number
            law_title = payload.get("law_title", "")
            article_title = payload.get("article_title", "")
            
            # Format: [1]【法律名 条文】\n内容
            if law_title and article_title:
                context_str = f"[{idx}]【{law_title} {article_title}】\n{text}"
            else:
                context_str = f"[{idx}] {text}"
            
            context.append(context_str)
        
        return context
    
    def _generate_response(self, query: str, context: list[str]) -> str:
        """Generate LLM response with context."""
        return self.llm.generate_with_context(query, context)
    
    def _to_source_document(self, result: dict[str, Any]) -> SourceDocument:
        """Convert raw result to SourceDocument for ChatResponse."""
        payload = result.get("payload", {})
        highlight_path = payload.get("highlight_path", {})
        
        return SourceDocument(
            law_title=payload.get("law_title", ""),
            article=payload.get("article_title", ""),
            text=payload.get("text", "")[:500],  # Truncate for response
            score=result.get("score", 0.0),
            highlight_path=highlight_path if isinstance(highlight_path, dict) else {},
            # Additional structured metadata
            law_id=payload.get("law_id", ""),
            chapter_title=payload.get("chapter_title", ""),
            article_caption=payload.get("article_caption", ""),
            paragraph_num=payload.get("paragraph_num", ""),
        )
    
    def _can_hybrid(self) -> bool:
        """Check if hybrid search is possible."""
        return self.use_hybrid_search and self.sparse_embedding and self.hybrid_store
