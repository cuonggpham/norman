# API Routes
# Search and Chat endpoints for Japanese Legal RAG

from fastapi import APIRouter, Depends, HTTPException
from app.api.deps import get_pipeline
from app.pipelines.rag import RAGPipeline
from app.models.schemas import (
    SearchQuery,
    SearchResponse,
    ChatQuery,
    ChatResponse,
    HealthResponse,
)

router = APIRouter(prefix="/api", tags=["rag"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        services={
            "qdrant": "connected",
            "openai": "configured",
        }
    )


@router.post("/search", response_model=SearchResponse)
async def search(
    query: SearchQuery,
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """
    Vector search endpoint.
    
    Returns relevant legal documents without LLM generation.
    Useful for exploring the law database.
    """
    try:
        results = pipeline.search(
            query=query.query,
            top_k=query.top_k,
            filters=query.filters,
        )
        
        processing_time = results[0].processing_time_ms if results else 0.0
        
        return SearchResponse(
            results=results,
            query=query.query,
            total=len(results),
            processing_time_ms=processing_time,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
async def chat(
    query: ChatQuery,
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """
    RAG Chat endpoint.
    
    Retrieves relevant documents and generates an answer in Vietnamese
    with Japanese legal term annotations and citations.
    """
    try:
        response = pipeline.chat(
            query=query.query,
            top_k=query.top_k,
            filters=query.filters,
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
