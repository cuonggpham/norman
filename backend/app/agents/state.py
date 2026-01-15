"""
LangGraph Agent State for Japanese Legal RAG.

Defines the shared state across all graph nodes.
"""

from typing import TypedDict, Annotated
from operator import add


class LegalRAGState(TypedDict, total=False):
    """
    State shared across all nodes in the LangGraph.
    
    Attributes:
        query: Original user query (Vietnamese or Japanese)
        translated_query: Japanese translation for search
        search_queries: Multiple search queries for multi-query retrieval
        documents: Retrieved documents from vector store
        document_grades: Relevance grades for each document
        reranked_documents: Documents after BGE reranking
        rewrite_count: Number of query rewrites (for loop control)
        answer: Final generated answer
        sources: Source documents for citations
        error: Error message if any step fails
    """
    # Input
    query: str
    
    # Translation
    translated_query: str
    search_queries: list[str]
    
    # Retrieval
    documents: list[dict]
    document_grades: list[str]  # "relevant" or "not_relevant"
    reranked_documents: list[dict]
    
    # Control
    rewrite_count: int
    
    # Output
    answer: str
    sources: list[dict]
    error: str | None
