"""
RAG Pipeline - Orchestrates retrieval and generation.

Coordinates between embedding, vector search, optional reranking, and LLM generation.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from app.core.protocols import (
    LLMProvider, 
    EmbeddingProvider, 
    VectorStore, 
    Reranker,
    SparseEmbeddingProvider,
    HybridVectorStore,
)
from app.models.schemas import ChatResponse, SearchResult, SourceDocument

logger = logging.getLogger(__name__)


@runtime_checkable
class QueryTranslator(Protocol):
    """Protocol for query translator."""
    def translate(self, query: str) -> str: ...


@dataclass
class RAGPipeline:
    """
    RAG Pipeline orchestrator.
    
    Coordinates the full RAG flow:
    1. Embed query (dense + optional sparse)
    2. Vector search (standard or hybrid)
    3. (Optional) Rerank
    4. Generate response with LLM
    
    Example:
        pipeline = RAGPipeline(
            embedding=EmbeddingService(...),
            vector_store=QdrantVectorStore(...),
            llm=OpenAIProvider(...),
        )
        response = pipeline.chat("労働時間の規定は？")
        
        # With hybrid search:
        pipeline = RAGPipeline(
            embedding=EmbeddingService(...),
            vector_store=QdrantVectorStore(...),
            hybrid_store=QdrantHybridStore(...),
            sparse_embedding=SparseEmbeddingService(),
            llm=OpenAIProvider(...),
            use_hybrid_search=True,
        )
    """
    
    embedding: EmbeddingProvider
    vector_store: VectorStore
    llm: LLMProvider
    reranker: Reranker | None = None
    translator: QueryTranslator | None = None  # Cross-lingual translation
    
    # Hybrid search components (optional)
    sparse_embedding: SparseEmbeddingProvider | None = None
    hybrid_store: HybridVectorStore | None = None
    use_hybrid_search: bool = False  # Toggle for hybrid search mode
    
    # Default settings
    default_top_k: int = 10  # Increased for better context recall with reranker
    retrieval_multiplier: int = 4  # Increased from 3 for larger rerank pool
    min_score_threshold: float = 0.25  # Lowered for broader coverage
    
    def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        use_hybrid: bool | None = None,
    ) -> list[SearchResult]:
        """
        Vector search only (no LLM generation).
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Metadata filters
            use_hybrid: Override for hybrid search mode (None = use default)
            
        Returns:
            List of search results
        """
        start_time = time.time()
        top_k = top_k or self.default_top_k
        
        # Determine search mode
        hybrid_mode = use_hybrid if use_hybrid is not None else self.use_hybrid_search
        
        # Translate query if translator available (Vietnamese → Japanese)
        search_query = self._translate_query(query)
        
        # Embed query (dense)
        query_vector = self.embedding.embed(search_query)
        
        # Perform search (hybrid or vector-only)
        if hybrid_mode and self.sparse_embedding and self.hybrid_store:
            # Hybrid search: dense + sparse with RRF fusion
            sparse_vector = self.sparse_embedding.embed(search_query)
            raw_results = self.hybrid_store.hybrid_search(
                dense_vector=query_vector,
                sparse_vector=sparse_vector,
                top_k=top_k,
                filters=filters,
            )
            logger.debug(f"Hybrid search returned {len(raw_results)} results")
        else:
            # Standard vector search
            raw_results = self.vector_store.search(
                query_vector=query_vector,
                top_k=top_k,
                filters=filters,
            )
        
        # Convert to SearchResult
        results = [self._to_search_result(r) for r in raw_results]
        
        elapsed = (time.time() - start_time) * 1000
        for r in results:
            r.processing_time_ms = elapsed
        
        return results
    
    def chat(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        use_multi_query: bool = True,
        auto_filter: bool = False,  # Disabled: Qdrant only has '労働' category
        use_hybrid: bool | None = None,
    ) -> ChatResponse:
        """
        Full RAG: retrieve + generate.
        
        Args:
            query: User's question
            top_k: Number of sources to use
            filters: Metadata filters
            use_multi_query: If True, use multi-query retrieval for better recall
            auto_filter: If True, auto-detect category and add filters
            use_hybrid: Override for hybrid search mode (None = use default)
            
        Returns:
            ChatResponse with answer and sources
        """
        start_time = time.time()
        top_k = top_k or self.default_top_k
        
        # Determine search mode
        hybrid_mode = use_hybrid if use_hybrid is not None else self.use_hybrid_search
        can_hybrid = hybrid_mode and self.sparse_embedding and self.hybrid_store
        
        # 0a. Auto-detect category and merge with user filters
        final_filters = dict(filters) if filters else {}
        if auto_filter:
            from app.llm.query_analyzer import get_query_analyzer
            analyzer = get_query_analyzer()
            analysis = analyzer.analyze(query)
            if analysis.detected_category and analysis.confidence >= 0.6:
                logger.info(f"Auto-detected category: {analysis.detected_category} (confidence: {analysis.confidence:.2f})")
                # Only add category filter if user didn't specify
                if "category" not in final_filters:
                    final_filters["category"] = analysis.detected_category
        
        # 0b. Get search queries (multi-query or single)
        if use_multi_query and self.translator and hasattr(self.translator, 'get_all_search_texts'):
            search_texts = self.translator.get_all_search_texts(query)
            search_texts = search_texts[:3]  # ⚡ EMERGENCY FIX: Reduce from 5 to 3 queries (-5s)
            logger.info(f"Multi-query retrieval with {len(search_texts)} queries (limited for performance)")
        else:
            search_query = self._translate_query(query)
            search_texts = [search_query]
        
        # 1. Retrieve with multiple queries and deduplicate
        retrieve_k = top_k * self.retrieval_multiplier if self.reranker else top_k * 2
        
        # ⚡ OPTIMIZATION: Batch embed all search texts at once (reduces 6 calls to 2)
        logger.debug(f"Batch embedding {len(search_texts)} search queries")
        all_dense_vectors = self.embedding.embed_batch(search_texts)
        all_sparse_vectors = None
        if can_hybrid:
            all_sparse_vectors = self.sparse_embedding.embed_batch(search_texts)
        
        all_results = {}  # chunk_id -> result (for deduplication)
        
        for idx, search_text in enumerate(search_texts):
            # Use pre-computed embeddings
            query_vector = all_dense_vectors[idx]
            
            if can_hybrid:
                # Hybrid search: dense + sparse with RRF fusion
                sparse_vector = all_sparse_vectors[idx]
                results = self.hybrid_store.hybrid_search(
                    dense_vector=query_vector,
                    sparse_vector=sparse_vector,
                    top_k=retrieve_k,
                    filters=final_filters if final_filters else None,
                )
            else:
                # Standard vector search
                results = self.vector_store.search(
                    query_vector=query_vector,
                    top_k=retrieve_k,
                    filters=final_filters if final_filters else None,
                )
            
            # Merge results, keeping highest score for each chunk
            for r in results:
                chunk_id = r.get("id", str(r.get("payload", {}).get("chunk_id", "")))
                if chunk_id not in all_results or r.get("score", 0) > all_results[chunk_id].get("score", 0):
                    all_results[chunk_id] = r
        
        # Sort by score and take top results
        raw_results = sorted(all_results.values(), key=lambda x: x.get("score", 0), reverse=True)
        
        # 2. Filter by score threshold
        filtered_results = [r for r in raw_results if r.get("score", 0) >= self.min_score_threshold]
        
        if not filtered_results:
            logger.warning(f"No results above threshold {self.min_score_threshold} for: {query}")
            # Fallback: use top 3 results anyway
            filtered_results = raw_results[:3]
        
        # 3. Rerank (if available)
        if self.reranker:
            raw_results = self.reranker.rerank(query, filtered_results, top_k)
        else:
            raw_results = filtered_results[:top_k]
        
        # 4. Build context from results
        context = self._build_context(raw_results)
        
        # 4. Generate response
        answer = self.llm.generate_with_context(query, context)
        
        # 5. Format response
        sources = [self._to_source_document(r) for r in raw_results]
        elapsed = (time.time() - start_time) * 1000
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            query=query,
            processing_time_ms=elapsed,
        )
    
    def _build_context(self, results: list[dict[str, Any]]) -> list[str]:
        """Build context strings from search results."""
        context = []
        for r in results:
            payload = r.get("payload", {})
            
            # Use text_with_context if available, otherwise text
            text = payload.get("text_with_context") or payload.get("text", "")
            
            # Add metadata for citation
            law_title = payload.get("law_title", "")
            article_title = payload.get("article_title", "")
            
            if law_title and article_title:
                context_str = f"【{law_title} {article_title}】\n{text}"
            else:
                context_str = text
            
            context.append(context_str)
        
        return context
    
    def _to_search_result(self, result: dict[str, Any]) -> SearchResult:
        """Convert raw result to SearchResult."""
        payload = result.get("payload", {})
        return SearchResult(
            chunk_id=str(result.get("id", "")),
            text=payload.get("text", ""),
            score=result.get("score", 0.0),
            law_id=payload.get("law_id", ""),
            law_title=payload.get("law_title", ""),
            article_title=payload.get("article_title", ""),
            article_caption=payload.get("article_caption", ""),
            chapter_title=payload.get("chapter_title", ""),
            paragraph_num=payload.get("paragraph_num", ""),
        )
    
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
        )
    
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
