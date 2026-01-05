"""
RAG Pipeline - Orchestrates retrieval and generation.

Coordinates between embedding, vector search, optional reranking, and LLM generation.
"""

import time
from dataclasses import dataclass, field
from typing import Any

from app.core.protocols import LLMProvider, EmbeddingProvider, VectorStore, Reranker
from app.models.schemas import ChatResponse, SearchResult, SourceDocument


@dataclass
class RAGPipeline:
    """
    RAG Pipeline orchestrator.
    
    Coordinates the full RAG flow:
    1. Embed query
    2. Vector search
    3. (Optional) Rerank
    4. Generate response with LLM
    
    Example:
        pipeline = RAGPipeline(
            embedding=EmbeddingService(...),
            vector_store=QdrantVectorStore(...),
            llm=OpenAIProvider(...),
        )
        response = pipeline.chat("労働時間の規定は？")
    """
    
    embedding: EmbeddingProvider
    vector_store: VectorStore
    llm: LLMProvider
    reranker: Reranker | None = None
    
    # Default settings
    default_top_k: int = 5
    retrieval_multiplier: int = 3  # Retrieve more for reranking
    
    def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Vector search only (no LLM generation).
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Metadata filters
            
        Returns:
            List of search results
        """
        start_time = time.time()
        top_k = top_k or self.default_top_k
        
        # Embed query
        query_vector = self.embedding.embed(query)
        
        # Search
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
    ) -> ChatResponse:
        """
        Full RAG: retrieve + generate.
        
        Args:
            query: User's question
            top_k: Number of sources to use
            filters: Metadata filters
            
        Returns:
            ChatResponse with answer and sources
        """
        start_time = time.time()
        top_k = top_k or self.default_top_k
        
        # 1. Retrieve (get more if reranking)
        retrieve_k = top_k * self.retrieval_multiplier if self.reranker else top_k
        
        query_vector = self.embedding.embed(query)
        raw_results = self.vector_store.search(
            query_vector=query_vector,
            top_k=retrieve_k,
            filters=filters,
        )
        
        # 2. Rerank (if available)
        if self.reranker:
            raw_results = self.reranker.rerank(query, raw_results, top_k)
        else:
            raw_results = raw_results[:top_k]
        
        # 3. Build context from results
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
