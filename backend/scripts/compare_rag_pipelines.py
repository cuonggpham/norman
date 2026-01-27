#!/usr/bin/env python3
"""
Compare Original RAG vs GraphRAG performance.
Tests both pipelines with the same queries and compares results.

Run: python -m scripts.compare_rag_pipelines
"""

import sys
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.api.deps import (
    get_embedding_service,
    get_vector_store,
    get_llm_provider,
    get_query_translator,
    get_sparse_embedding_service,
    get_hybrid_vector_store,
)
from app.pipelines.rag import RAGPipeline
from app.pipelines.graph_rag import GraphRAGPipeline


def create_original_rag() -> RAGPipeline:
    """Create original RAG pipeline."""
    return RAGPipeline(
        embedding=get_embedding_service(),
        vector_store=get_vector_store(),
        llm=get_llm_provider(),
        translator=get_query_translator(),
        use_hybrid_search=True,
        sparse_embedding=get_sparse_embedding_service(),
        hybrid_store=get_hybrid_vector_store(),
    )


def create_graph_rag() -> GraphRAGPipeline:
    """Create GraphRAG pipeline."""
    return GraphRAGPipeline(
        embedding=get_embedding_service(),
        vector_store=get_vector_store(),
        llm=get_llm_provider(),
        translator=get_query_translator(),
        use_hybrid_search=True,
        sparse_embedding=get_sparse_embedding_service(),
        hybrid_store=get_hybrid_vector_store(),
        use_graph=True,
    )


def print_separator():
    print("\n" + "=" * 70)


def compare_responses(query: str, original_rag: RAGPipeline, graph_rag: GraphRAGPipeline):
    """Compare responses from both pipelines."""
    print_separator()
    print(f"QUERY: {query}")
    print_separator()
    
    # Original RAG
    print("\n🔵 ORIGINAL RAG")
    print("-" * 50)
    start = time.time()
    try:
        original_response = original_rag.chat(query)
        original_time = (time.time() - start) * 1000
        
        print(f"Answer: {original_response.answer[:300]}...")
        print(f"\nSources ({len(original_response.sources)}):")
        for i, src in enumerate(original_response.sources[:3], 1):
            print(f"  {i}. {src.law_title} - {src.article}")
            print(f"     Score: {src.score:.4f}")
        print(f"\n⏱️ Time: {original_time:.0f}ms")
    except Exception as e:
        print(f"❌ Error: {e}")
        original_time = 0
    
    # GraphRAG
    print("\n🟢 GRAPH RAG")
    print("-" * 50)
    start = time.time()
    try:
        graph_response = graph_rag.chat(query)
        graph_time = (time.time() - start) * 1000
        
        print(f"Answer: {graph_response.answer[:300]}...")
        print(f"\nSources ({len(graph_response.sources)}):")
        for i, src in enumerate(graph_response.sources[:3], 1):
            print(f"  {i}. {src.law_title} - {src.article}")
            print(f"     Score: {src.score:.4f}")
        print(f"\n⏱️ Time: {graph_time:.0f}ms")
    except Exception as e:
        print(f"❌ Error: {e}")
        graph_time = 0
    
    # Comparison
    if original_time > 0 and graph_time > 0:
        diff = graph_time - original_time
        print(f"\n📊 Comparison: GraphRAG is {'+' if diff > 0 else ''}{diff:.0f}ms vs Original")


def main():
    print("=" * 70)
    print("RAG vs GraphRAG Comparison Test")
    print("=" * 70)
    
    # Initialize pipelines
    print("\n📦 Initializing pipelines...")
    original_rag = create_original_rag()
    print("✅ Original RAG ready")
    
    graph_rag = create_graph_rag()
    print("✅ GraphRAG ready")
    
    # Test queries - mix of entity-based and semantic
    test_queries = [
        # Entity lookup (should benefit from graph)
        "所得税法の規定について教えてください",
        
        # Semantic query (should use vector)
        "Thời gian làm việc tối đa theo luật pháp Nhật Bản là bao nhiêu?",
        
        # Mixed query
        "租税特別措置法の控除について説明してください",
    ]
    
    for query in test_queries:
        compare_responses(query, original_rag, graph_rag)
    
    print_separator()
    print("✅ Comparison complete!")
    print_separator()


if __name__ == "__main__":
    main()
