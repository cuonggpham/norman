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
    
    prompt = f"""Bạn là chuyên gia dịch thuật pháp luật Nhật Bản sang tiếng Việt.

## QUY TẮC DỊCH BẮT BUỘC:

### 1. Thuật ngữ pháp lý
- Giữ nguyên tiếng Nhật kèm dịch nghĩa trong ngoặc đơn cho thuật ngữ quan trọng
- Ví dụ: 居住者 (cư dân/người cư trú), 所得税 (thuế thu nhập), 予定納税 (nộp thuế tạm tính)

### 2. Số hiệu pháp luật
- Giữ nguyên format: Điều 104 Khoản 1 (第百四条第一項)
- Năm Chiêu Hòa (昭和) → ghi cả hai: "năm Chiêu Hòa 42 (1967)"

### 3. Cấu trúc văn bản
- Tách các mục/khoản thành dòng riêng với đánh số rõ ràng
- Nếu có "一、二、三" hoặc "１、２、３" → format thành "1., 2., 3."
- Câu dài → ngắt dòng hợp lý, giữ logic pháp lý

### 4. Giải thích ngữ cảnh
- Các cụm như "以下「...」という" → "(sau đây gọi là '...')"
- Điều kiện ngoại lệ → làm rõ bằng dấu gạch đầu dòng

### 5. Format output
- Dịch tự nhiên, dễ đọc cho người Việt
- KHÔNG thêm giải thích ngoài văn bản gốc
- KHÔNG bỏ sót nội dung nào

---
VĂN BẢN GỐC (日本語):
{request.text}

---
BẢN DỊCH TIẾNG VIỆT:"""

    try:
        from google import genai
        
        client = genai.Client(api_key=gemini_api_key)
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt,
            config={
                "temperature": 0.2,
                "max_output_tokens": 2048,
            }
        )
        
        # Check if response.text is None (can happen if content is filtered)
        if response.text is None:
            raise HTTPException(
                status_code=500,
                detail="Gemini API returned empty response. Content may have been filtered or request was invalid."
            )
        
        translated_text = response.text.strip()
        processing_time = (time.time() - start_time) * 1000
        
        return TranslateResponse(
            original=request.text,
            translated=translated_text,
            processing_time_ms=processing_time
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
