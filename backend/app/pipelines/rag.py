"""
RAG Pipeline - Orchestrates retrieval and generation.

Inherits from BasePipeline for shared logic.
Adds unique features:
- search() method for vector-only search
- Multi-query retrieval with deduplication
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

from app.pipelines.base import BasePipeline, QueryTranslator
from app.models.schemas import ChatResponse, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class RAGPipeline(BasePipeline):
    """
    RAG Pipeline orchestrator.
    
    Inherits from BasePipeline and adds:
    - search(): Vector search only (no LLM)
    - Multi-query retrieval with auto-filter support
    
    Example:
        pipeline = RAGPipeline(
            embedding=EmbeddingService(...),
            vector_store=QdrantVectorStore(...),
            llm=OpenAIProvider(...),
        )
        response = pipeline.chat("労働時間の規定は？")
    """
    
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
        can_hybrid = (use_hybrid if use_hybrid is not None else self.use_hybrid_search) and self._can_hybrid()
        
        # Translate query
        search_query = self._translate_query(query)
        
        # Embed query (single)
        query_vector = self.embedding.embed(search_query)
        
        # Perform search
        if can_hybrid:
            sparse_vector = self.sparse_embedding.embed(search_query)
            raw_results = self.hybrid_store.hybrid_search(
                dense_vector=query_vector,
                sparse_vector=sparse_vector,
                top_k=top_k,
                filters=filters,
            )
            logger.debug(f"Hybrid search returned {len(raw_results)} results")
        else:
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
        use_multi_query: bool | None = None,
        auto_filter: bool = False,
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
        multi_query = use_multi_query if use_multi_query is not None else self.use_multi_query
        can_hybrid = (use_hybrid if use_hybrid is not None else self.use_hybrid_search) and self._can_hybrid()
        
        logger.info("="*60)
        logger.info(f"[RAG] Starting RAG Pipeline for query: {query[:50]}...")
        logger.info("="*60)
        logger.info(f"[RAG] Config: top_k={top_k}, hybrid_mode={can_hybrid}, multi_query={multi_query}")
        
        # Prepare filters
        final_filters = dict(filters) if filters else {}
        if auto_filter:
            self._apply_auto_filter(query, final_filters)
        
        # Step 1: Query Translation & Expansion
        step_start = time.time()
        logger.info(f"[STEP 1/5] Query Translation & Expansion...")
        if multi_query:
            search_texts = self._get_search_texts(query)
        else:
            search_texts = [self._translate_query(query)]
        logger.info(f"[STEP 1/5] Done: {len(search_texts)} queries ({(time.time()-step_start)*1000:.0f}ms)")
        for i, t in enumerate(search_texts):
            logger.info(f"           Query {i+1}: {t[:60]}...")
        
        # Step 2: Batch Embedding
        step_start = time.time()
        logger.info(f"[STEP 2/5] Embedding {len(search_texts)} queries...")
        all_dense_vectors, all_sparse_vectors = self._embed_queries(search_texts, can_hybrid)
        logger.info(f"[STEP 2/5] Done: Embeddings complete ({(time.time()-step_start)*1000:.0f}ms)")
        
        # Step 3: Vector Search with multi-query
        step_start = time.time()
        retrieve_k = top_k * self.retrieval_multiplier if self.reranker else top_k * 2
        logger.info(f"[STEP 3/5] Vector Search (retrieve_k={retrieve_k})...")
        all_results = self._vector_search_multi(
            search_texts, all_dense_vectors, all_sparse_vectors,
            retrieve_k, final_filters if final_filters else None, can_hybrid
        )
        logger.info(f"[STEP 3/5] Done: {len(all_results)} unique results ({(time.time()-step_start)*1000:.0f}ms)")
        
        # Filter and sort
        filtered_results = self._filter_and_sort_results(all_results)
        logger.info(f"[RAG] After score filter (>={self.min_score_threshold}): {len(filtered_results)} results")
        
        # Step 4: Rerank
        step_start = time.time()
        if self.reranker:
            logger.info(f"[STEP 4/5] Reranking {len(filtered_results)} results...")
            final_results = self._rerank_results(query, filtered_results, top_k)
            logger.info(f"[STEP 4/5] Done: Reranking complete ({(time.time()-step_start)*1000:.0f}ms)")
        else:
            final_results = filtered_results[:top_k]
            logger.info(f"[STEP 4/5] Skipped: Reranker disabled, using top {top_k} results")
        
        # Build context
        context = self._build_context(final_results)
        logger.info(f"[RAG] Context built from {len(final_results)} sources")
        
        # Step 5: Generate response
        step_start = time.time()
        logger.info(f"[STEP 5/5] Generating LLM response...")
        answer = self._generate_response(query, context)
        logger.info(f"[STEP 5/5] Done: LLM response generated ({(time.time()-step_start)*1000:.0f}ms)")
        
        # Format response
        sources = [self._to_source_document(r) for r in final_results]
        elapsed = (time.time() - start_time) * 1000
        
        logger.info("="*60)
        logger.info(f"[RAG] Pipeline complete in {elapsed:.0f}ms ({elapsed/1000:.2f}s)")
        logger.info("="*60)
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            query=query,
            processing_time_ms=elapsed,
        )
    
    def _apply_auto_filter(self, query: str, filters: dict) -> None:
        """Apply auto-detected category filter."""
        try:
            from app.llm.query_analyzer import get_query_analyzer
            analyzer = get_query_analyzer()
            analysis = analyzer.analyze(query)
            if analysis.detected_category and analysis.confidence >= 0.6:
                logger.info(f"[RAG] Auto-detected category: {analysis.detected_category} (confidence: {analysis.confidence:.2f})")
                if "category" not in filters:
                    filters["category"] = analysis.detected_category
        except Exception as e:
            logger.warning(f"Auto-filter failed: {e}")
    
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
