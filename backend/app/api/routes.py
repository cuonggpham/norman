# API Routes
# Search and Chat endpoints for Japanese Legal RAG

import os
import time
from fastapi import APIRouter, Depends, HTTPException
import httpx
from dotenv import load_dotenv
from app.api.deps import get_pipeline

# Load .env file
load_dotenv()
from app.pipelines.rag import RAGPipeline
from app.core.config import get_settings
from app.models.schemas import (
    SearchQuery,
    SearchResponse,
    ChatQuery,
    ChatResponse,
    SourceDocument,
    HealthResponse,
    TranslateRequest,
    TranslateResponse,
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
    
    Set use_agent=true for LangGraph agent with self-correction loop.
    """
    try:
        if query.use_agent:
            # Use LangGraph agent (with self-correction)
            from app.agents.graph import get_legal_rag_agent
            agent = get_legal_rag_agent()
            result = agent.chat(query.query)
            return ChatResponse(
                answer=result["answer"],
                sources=[
                    SourceDocument(**s) for s in result.get("sources", [])
                ],
                query=result["query"],
                processing_time_ms=result["processing_time_ms"],
            )
        else:
            # Use RAGPipeline (default)
            response = pipeline.chat(
                query=query.query,
                top_k=query.top_k,
                filters=query.filters,
            )
            return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/translate", response_model=TranslateResponse)
async def translate(request: TranslateRequest):
    """
    Translate Japanese legal text to Vietnamese using Gemini 2.5 Flash.
    
    Optimized for legal terminology with context preservation.
    """
    start_time = time.time()
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    if not gemini_api_key:
        raise HTTPException(
            status_code=500,
            detail="Gemini API key is not configured. Please set GEMINI_API_KEY in .env"
        )
    
    prompt = f"""Bạn là một chuyên gia dịch thuật pháp luật Nhật Bản. 
Hãy dịch đoạn văn bản luật tiếng Nhật sau sang tiếng Việt.

Yêu cầu:
- Giữ nguyên các thuật ngữ pháp lý quan trọng trong ngoặc []
- Dịch chính xác, đúng ngữ cảnh pháp luật
- Giữ cấu trúc câu rõ ràng

Văn bản tiếng Nhật:
{request.text}

Bản dịch tiếng Việt:"""

    try:
        from google import genai
        
        client = genai.Client(api_key=gemini_api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={
                "temperature": 0.2,
                "max_output_tokens": 2048,
            }
        )
        
        translated_text = response.text.strip()
        processing_time = (time.time() - start_time) * 1000
        
        return TranslateResponse(
            original=request.text,
            translated=translated_text,
            processing_time_ms=processing_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
