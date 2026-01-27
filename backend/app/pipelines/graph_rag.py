"""
GraphRAG Pipeline - Hybrid retrieval combining graph + vector search.

Inherits from BasePipeline for shared logic.
Adds unique features:
- Query routing (entity extraction)
- Graph search via Neo4j
- Result fusion (graph + vector)
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional, List

from app.pipelines.base import BasePipeline
from app.models.schemas import ChatResponse
from app.services.graph_service import GraphService, GraphResult, get_graph_service
from app.services.query_router import QueryRouter, QueryType, get_query_router

logger = logging.getLogger(__name__)


@dataclass
class GraphRAGPipeline(BasePipeline):
    """
    GraphRAG Pipeline - Hybrid graph + vector retrieval.
    
    Inherits from BasePipeline and adds:
    - Query routing with entity extraction
    - Graph search via Neo4j
    - Result fusion with graph boost
    
    Flow:
    1. Query Translation
    2. Query Routing (entity extraction)
    3. Graph search for entity lookups
    4. Vector search for semantic matching
    5. Fuse results and rerank
    6. Generate response with LLM
    """
    
    # Graph-specific components
    graph_service: Optional[GraphService] = None
    query_router: Optional[QueryRouter] = None
    
    # Graph-specific config
    use_graph: bool = True
    graph_weight: float = 1.2  # Boost for graph results
    
    def __post_init__(self):
        """Initialize default components if not provided."""
        if self.graph_service is None and self.use_graph:
            try:
                self.graph_service = get_graph_service()
            except Exception as e:
                logger.warning(f"Could not initialize graph service: {e}")
                self.use_graph = False
        
        if self.query_router is None:
            self.query_router = get_query_router()
    
    def chat(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
        use_graph: Optional[bool] = None,
        use_hybrid: Optional[bool] = None,
        use_multi_query: Optional[bool] = None,
    ) -> ChatResponse:
        """
        Full GraphRAG: translate → route → retrieve (graph + vector) → generate.
        
        Args:
            query: User's question
            top_k: Number of sources
            filters: Metadata filters
            use_graph: Override graph search
            use_hybrid: Override hybrid vector search
            use_multi_query: Override multi-query retrieval
        """
        start_time = time.time()
        top_k = top_k or self.default_top_k
        
        # Determine modes
        graph_mode = use_graph if use_graph is not None else self.use_graph
        multi_query_mode = use_multi_query if use_multi_query is not None else self.use_multi_query
        can_hybrid = (use_hybrid if use_hybrid is not None else self.use_hybrid_search) and self._can_hybrid()
        
        logger.info("="*60)
        logger.info(f"[GraphRAG] Starting GraphRAG Pipeline for query: {query[:50]}...")
        logger.info("="*60)
        logger.info(f"[GraphRAG] Config: top_k={top_k}, graph={graph_mode}, hybrid={can_hybrid}, multi_query={multi_query_mode}")
        
        # Storage
        all_results = {}
        graph_results: List[GraphResult] = []
        final_filters = dict(filters) if filters else {}
        
        # Step 1: Query Translation & Expansion
        step_start = time.time()
        logger.info(f"[STEP 1/7] Query Translation & Expansion...")
        if multi_query_mode:
            search_texts = self._get_search_texts(query)
        else:
            search_texts = [self._translate_query(query)]
        logger.info(f"[STEP 1/7] Done: {len(search_texts)} queries ({(time.time()-step_start)*1000:.0f}ms)")
        
        routing_query = search_texts[0] if search_texts else query
        
        # Step 2: Query Routing
        step_start = time.time()
        logger.info(f"[STEP 2/7] Query Routing...")
        routed = self.query_router.route(routing_query) if self.query_router else None
        
        if routed:
            logger.info(f"[STEP 2/7] Done: type={routed.query_type.value}, entities={len(routed.entities)} ({(time.time()-step_start)*1000:.0f}ms)")
            if routed.query_type == QueryType.SEMANTIC:
                graph_mode = False
            elif routed.query_type == QueryType.ENTITY_LOOKUP:
                graph_mode = True
        else:
            logger.info(f"[STEP 2/7] Skipped: No routing ({(time.time()-step_start)*1000:.0f}ms)")
        
        # Step 3: Graph Search
        step_start = time.time()
        if graph_mode and self.graph_service and routed and routed.entities:
            logger.info(f"[STEP 3/7] Graph Search ({len(routed.entities)} entities)...")
            graph_results = self._graph_search(routed.entities)
            logger.info(f"[STEP 3/7] Done: {len(graph_results)} results ({(time.time()-step_start)*1000:.0f}ms)")
            
            for gr in graph_results:
                if gr.chunk_id:
                    all_results[gr.chunk_id] = {
                        "id": gr.chunk_id,
                        "score": gr.relevance * self.graph_weight,
                        "payload": {
                            "law_id": gr.law_id,
                            "law_title": gr.law_title,
                            "article_title": f"第{gr.article_num}条",
                            "article_caption": gr.article_caption or "",
                            "text": gr.article_caption or gr.article_title or "",
                            "highlight_path": {"law": gr.law_title, "article": f"第{gr.article_num}条"},
                        },
                        "source": "graph",
                    }
        else:
            logger.info(f"[STEP 3/7] Skipped: Graph search")
        
        # Step 4: Batch Embedding
        step_start = time.time()
        logger.info(f"[STEP 4/7] Embedding {len(search_texts)} queries...")
        all_dense_vectors, all_sparse_vectors = self._embed_queries(search_texts, can_hybrid)
        logger.info(f"[STEP 4/7] Done: ({(time.time()-step_start)*1000:.0f}ms)")
        
        # Step 5: Vector Search
        step_start = time.time()
        retrieve_k = top_k * self.retrieval_multiplier if self.reranker else top_k * 2
        logger.info(f"[STEP 5/7] Vector Search (retrieve_k={retrieve_k})...")
        vector_results = self._vector_search_multi(
            search_texts, all_dense_vectors, all_sparse_vectors,
            retrieve_k, final_filters if final_filters else None, can_hybrid
        )
        # Merge with graph results
        for chunk_id, r in vector_results.items():
            if chunk_id not in all_results or r.get("score", 0) > all_results[chunk_id].get("score", 0):
                all_results[chunk_id] = r
        logger.info(f"[STEP 5/7] Done: {len(all_results)} unique ({(time.time()-step_start)*1000:.0f}ms)")
        
        # Filter and sort
        filtered_results = self._filter_and_sort_results(all_results)
        logger.info(f"[GraphRAG] After filter: {len(filtered_results)} results (graph={len(graph_results)})")
        
        # Step 6: Rerank
        step_start = time.time()
        if self.reranker:
            logger.info(f"[STEP 6/7] Reranking {len(filtered_results)} results...")
            final_results = self._rerank_results(query, filtered_results, top_k)
            logger.info(f"[STEP 6/7] Done: ({(time.time()-step_start)*1000:.0f}ms)")
        else:
            final_results = filtered_results[:top_k]
            logger.info(f"[STEP 6/7] Skipped: Using top {top_k}")
        
        # Step 7: Generate
        step_start = time.time()
        context = self._build_context(final_results)
        logger.info(f"[STEP 7/7] Generating LLM response...")
        answer = self._generate_response(query, context)
        logger.info(f"[STEP 7/7] Done: ({(time.time()-step_start)*1000:.0f}ms)")
        
        # Format response
        sources = [self._to_source_document(r) for r in final_results]
        elapsed = (time.time() - start_time) * 1000
        
        graph_count = sum(1 for r in final_results if r.get("source") == "graph")
        logger.info("="*60)
        logger.info(f"[GraphRAG] Complete: {elapsed:.0f}ms (graph={graph_count}, vector={len(final_results)-graph_count})")
        logger.info("="*60)
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            query=query,
            processing_time_ms=elapsed,
        )
    
    def _graph_search(self, entities: List[tuple]) -> List[GraphResult]:
        """Perform graph search based on extracted entities."""
        results = []
        
        for entity, entity_type in entities:
            if entity_type == 'law_article':
                ref = self.query_router.parse_law_article_reference(entity)
                if ref:
                    law_name, article_num = ref
                    result = self.graph_service.find_article(law_name, article_num)
                    if result:
                        results.append(result)
                        related = self.graph_service.find_related_articles(
                            result.law_id, article_num, depth=2, limit=3
                        )
                        results.extend(related)
            
            elif entity_type == 'law':
                keyword_results = self.graph_service.search_by_keyword(entity, limit=5)
                results.extend(keyword_results)
            
            elif entity_type == 'article':
                article_num = self.query_router.parse_article_reference(entity)
                if article_num:
                    keyword_results = self.graph_service.search_by_keyword(
                        f"第{article_num}条", limit=3
                    )
                    results.extend(keyword_results)
        
        # Deduplicate
        seen = set()
        unique = []
        for r in results:
            key = (r.law_id, r.article_num)
            if key not in seen:
                seen.add(key)
                unique.append(r)
        
        return unique


def create_graphrag_pipeline(
    embedding,
    vector_store,
    llm,
    **kwargs
) -> GraphRAGPipeline:
    """Factory function to create GraphRAG pipeline."""
    return GraphRAGPipeline(
        embedding=embedding,
        vector_store=vector_store,
        llm=llm,
        **kwargs
    )
