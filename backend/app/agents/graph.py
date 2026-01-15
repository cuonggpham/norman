"""
LangGraph-based RAG Agent for Japanese Legal Documents.

Provides an alternative to RAGPipeline with:
- Self-correction loop when documents are weak
- Document grading for quality control
- Modular node design for extensibility
"""

import logging
import time
from typing import Any

from langgraph.graph import StateGraph, START, END

from app.agents.state import LegalRAGState
from app.agents.nodes import (
    translate_node,
    retrieve_node,
    grade_documents_node,
    rerank_node,
    generate_node,
    rewrite_query_node,
    should_rewrite,
)

logger = logging.getLogger(__name__)


def build_legal_rag_graph():
    """
    Build and compile the Legal RAG graph.
    
    Graph structure:
        START → translate → retrieve → grade → [rerank | rewrite → retrieve]
                                                     ↓
                                              generate → END
    
    Returns:
        Compiled StateGraph ready for invocation
    """
    graph = StateGraph(LegalRAGState)
    
    # Add nodes
    graph.add_node("translate", translate_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_documents_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)
    graph.add_node("rewrite", rewrite_query_node)
    
    # Add edges
    graph.add_edge(START, "translate")
    graph.add_edge("translate", "retrieve")
    graph.add_edge("retrieve", "grade")
    
    # Conditional edge: grade → rerank OR grade → rewrite
    graph.add_conditional_edges(
        "grade",
        should_rewrite,
        {
            "rerank": "rerank",
            "rewrite": "rewrite",
        }
    )
    
    # Rewrite loops back to retrieve
    graph.add_edge("rewrite", "retrieve")
    
    # Rerank goes to generate
    graph.add_edge("rerank", "generate")
    
    # Generate ends
    graph.add_edge("generate", END)
    
    return graph.compile()


class LegalRAGAgent:
    """
    High-level wrapper around the LangGraph agent.
    
    Provides a simple interface similar to RAGPipeline.
    
    Example:
        agent = LegalRAGAgent()
        response = agent.chat("Thời gian làm việc tối đa mỗi tuần?")
    """
    
    def __init__(self):
        """Initialize the agent with compiled graph."""
        self.graph = build_legal_rag_graph()
        logger.info("LegalRAGAgent initialized with LangGraph")
    
    def chat(self, query: str) -> dict[str, Any]:
        """
        Run RAG pipeline with self-correction.
        
        Args:
            query: User's question (Vietnamese or Japanese)
            
        Returns:
            Dict with 'answer', 'sources', 'query', 'processing_time_ms'
        """
        start_time = time.time()
        
        # Initial state
        initial_state = {
            "query": query,
            "rewrite_count": 0,
        }
        
        try:
            # Run graph
            result = self.graph.invoke(initial_state)
            
            elapsed = (time.time() - start_time) * 1000
            
            return {
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
                "query": query,
                "processing_time_ms": elapsed,
            }
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            elapsed = (time.time() - start_time) * 1000
            return {
                "answer": f"エラーが発生しました: {str(e)}",
                "sources": [],
                "query": query,
                "processing_time_ms": elapsed,
                "error": str(e),
            }
    
    def get_graph_diagram(self) -> str:
        """Get ASCII diagram of the graph structure."""
        try:
            return self.graph.get_graph().draw_ascii()
        except Exception:
            return "Graph diagram not available"


# Singleton instance
_agent_instance = None


def get_legal_rag_agent() -> LegalRAGAgent:
    """Get singleton LegalRAGAgent instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = LegalRAGAgent()
    return _agent_instance
