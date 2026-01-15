"""
Node implementations for LangGraph Legal RAG Agent.

Each node is a function that takes state and returns state updates.
"""

import logging
from typing import Literal

from app.agents.state import LegalRAGState

logger = logging.getLogger(__name__)


def translate_node(state: LegalRAGState) -> dict:
    """
    Translate Vietnamese query to Japanese.
    
    Uses QueryTranslator from deps.py to:
    1. Translate main query
    2. Generate multiple search queries
    """
    from app.api.deps import get_query_translator
    
    query = state["query"]
    translator = get_query_translator()
    
    # Get translated query and multiple search texts
    translated = translator.translate(query)
    search_queries = translator.get_all_search_texts(query)
    
    logger.info(f"Translated query: '{query}' → '{translated}'")
    logger.info(f"Generated {len(search_queries)} search queries")
    
    return {
        "translated_query": translated,
        "search_queries": search_queries,
    }


def retrieve_node(state: LegalRAGState) -> dict:
    """
    Vector search with multi-query retrieval.
    
    Uses EmbeddingService and QdrantVectorStore from deps.py.
    Retrieves top-k * 4 documents for reranking.
    """
    from app.api.deps import get_embedding_service, get_vector_store
    
    embedding = get_embedding_service()
    vector_store = get_vector_store()
    
    search_queries = state.get("search_queries", [state.get("translated_query", "")])
    
    # Multi-query retrieval with deduplication
    all_results = {}
    retrieve_k = 40  # Get more for reranking
    
    for search_text in search_queries:
        query_vector = embedding.embed(search_text)
        results = vector_store.search(query_vector=query_vector, top_k=retrieve_k)
        
        for r in results:
            chunk_id = r.get("id", str(r.get("payload", {}).get("chunk_id", "")))
            if chunk_id not in all_results or r.get("score", 0) > all_results[chunk_id].get("score", 0):
                all_results[chunk_id] = r
    
    # Sort by score
    documents = sorted(all_results.values(), key=lambda x: x.get("score", 0), reverse=True)
    
    logger.info(f"Retrieved {len(documents)} unique documents")
    
    return {"documents": documents}


def grade_documents_node(state: LegalRAGState) -> dict:
    """
    LLM grades each document's relevance to the query.
    
    Returns list of "relevant" or "not_relevant" grades.
    """
    from app.api.deps import get_llm_provider
    
    query = state["query"]
    documents = state.get("documents", [])[:10]  # Grade top 10 only
    
    llm = get_llm_provider()
    grades = []
    
    for doc in documents:
        text = doc.get("payload", {}).get("text", "")[:500]
        
        messages = [
            {"role": "system", "content": """Bạn là chuyên gia đánh giá tài liệu pháp lý.
Đánh giá tài liệu có liên quan đến câu hỏi không.
Trả lời CHỈ MỘT từ: "relevant" hoặc "not_relevant"."""},
            {"role": "user", "content": f"Câu hỏi: {query}\n\nTài liệu: {text}"},
        ]
        
        try:
            grade = llm.generate(messages, temperature=0, max_tokens=10).strip().lower()
            grades.append("relevant" if "relevant" in grade and "not" not in grade else "not_relevant")
        except Exception as e:
            logger.warning(f"Grading failed: {e}")
            grades.append("relevant")  # Default to relevant on error
    
    relevant_count = sum(1 for g in grades if g == "relevant")
    logger.info(f"Document grading: {relevant_count}/{len(grades)} relevant")
    
    return {"document_grades": grades}


def rerank_node(state: LegalRAGState) -> dict:
    """
    BGE reranker for final ranking.
    
    Uses BGEReranker from deps.py.
    """
    from app.api.deps import get_reranker
    
    query = state["query"]
    documents = state.get("documents", [])
    reranker = get_reranker()
    
    if reranker and documents:
        reranked = reranker.rerank(query, documents, top_k=10)
        logger.info(f"Reranked {len(documents)} documents → top 10")
    else:
        reranked = documents[:10]
        logger.info("No reranker available, using top 10 by vector score")
    
    return {"reranked_documents": reranked}


def generate_node(state: LegalRAGState) -> dict:
    """
    Generate answer with citations.
    
    Uses LLM provider and prompt templates.
    """
    from app.api.deps import get_llm_provider
    from app.llm.prompts import LEGAL_ASSISTANT_SYSTEM
    
    query = state["query"]
    documents = state.get("reranked_documents", [])[:5]  # Use top 5
    
    llm = get_llm_provider()
    
    # Build context
    context_parts = []
    sources = []
    
    for doc in documents:
        payload = doc.get("payload", {})
        text = payload.get("text_with_context") or payload.get("text", "")
        law_title = payload.get("law_title", "")
        article_title = payload.get("article_title", "")
        
        if law_title and article_title:
            context_parts.append(f"【{law_title} {article_title}】\n{text}")
        else:
            context_parts.append(text)
        
        sources.append({
            "law_title": law_title,
            "article": article_title,
            "text": payload.get("text", "")[:500],
            "score": doc.get("score", 0),
            "highlight_path": payload.get("highlight_path", {}),
        })
    
    context = "\n\n---\n\n".join(context_parts)
    
    messages = [
        {"role": "system", "content": LEGAL_ASSISTANT_SYSTEM},
        {"role": "user", "content": f"Context:\n{context}\n\n---\n\nCâu hỏi: {query}"},
    ]
    
    answer = llm.generate(messages)
    
    logger.info(f"Generated answer with {len(sources)} sources")
    
    return {"answer": answer, "sources": sources}


def rewrite_query_node(state: LegalRAGState) -> dict:
    """
    Rewrite query for better retrieval.
    
    Called when document grading shows low relevance.
    """
    from app.api.deps import get_llm_provider, get_query_translator
    
    query = state["query"]
    rewrite_count = state.get("rewrite_count", 0) + 1
    
    llm = get_llm_provider()
    translator = get_query_translator()
    
    messages = [
        {"role": "system", "content": """Bạn là chuyên gia pháp luật Nhật Bản.
Viết lại câu hỏi để tìm kiếm tốt hơn trong cơ sở dữ liệu pháp luật.
Thêm từ khóa pháp lý cụ thể, điều luật liên quan.
Trả lời CHỈ câu hỏi đã viết lại bằng tiếng Việt."""},
        {"role": "user", "content": f"Câu hỏi gốc: {query}"},
    ]
    
    rewritten = llm.generate(messages, temperature=0.3, max_tokens=200).strip()
    
    # Re-translate
    translated = translator.translate(rewritten)
    search_queries = translator.get_all_search_texts(rewritten)
    
    logger.info(f"Query rewrite #{rewrite_count}: '{query}' → '{rewritten}'")
    
    return {
        "query": rewritten,  # Update query for next iteration
        "translated_query": translated,
        "search_queries": search_queries,
        "rewrite_count": rewrite_count,
    }


def should_rewrite(state: LegalRAGState) -> Literal["rerank", "rewrite"]:
    """
    Decide whether to rewrite query or proceed to reranking.
    
    Conditions for rewrite:
    - Less than 2 relevant documents
    - AND rewrite_count < 2
    """
    grades = state.get("document_grades", [])
    rewrite_count = state.get("rewrite_count", 0)
    
    relevant_count = sum(1 for g in grades if g == "relevant")
    
    if relevant_count >= 2 or rewrite_count >= 2:
        logger.info(f"Proceeding to rerank (relevant={relevant_count}, rewrites={rewrite_count})")
        return "rerank"
    else:
        logger.info(f"Triggering query rewrite (relevant={relevant_count}, rewrites={rewrite_count})")
        return "rewrite"
