# Pydantic Schemas for API Request/Response

from pydantic import BaseModel, Field
from typing import Optional


# ============== Request Schemas ==============

class SearchQuery(BaseModel):
    """Search request body."""
    query: str = Field(..., description="Search query in Japanese or Vietnamese")
    top_k: int = Field(default=5, ge=1, le=50)
    filters: Optional[dict] = Field(default=None, description="Metadata filters")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "労働時間の制限",
                "top_k": 5,
                "filters": {"category": "労働"}
            }
        }


class ChatQuery(BaseModel):
    """Chat request body for RAG."""
    query: str = Field(..., description="Question in Japanese or Vietnamese")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of sources to use")
    filters: Optional[dict] = Field(default=None, description="Metadata filters")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "残業代の計算方法を教えてください",
                "top_k": 5
            }
        }


# ============== Response Schemas ==============

class SearchResult(BaseModel):
    """Single search result."""
    chunk_id: str
    text: str
    score: float
    law_id: str
    law_title: str
    article_title: str
    article_caption: Optional[str] = None
    chapter_title: Optional[str] = None
    paragraph_num: Optional[str] = None
    processing_time_ms: Optional[float] = None


class SearchResponse(BaseModel):
    """Search endpoint response."""
    results: list[SearchResult]
    query: str
    total: int = Field(default=0)
    processing_time_ms: float = Field(default=0.0)


class SourceDocument(BaseModel):
    """Source document in chat response."""
    law_title: str
    article: str
    text: str
    score: float
    highlight_path: dict = Field(default_factory=dict)


class ChatResponse(BaseModel):
    """Chat endpoint response with answer and sources."""
    answer: str
    sources: list[SourceDocument]
    query: str
    processing_time_ms: float = Field(default=0.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Theo quy định tại Điều 32 [第三十二条] của Luật Tiêu chuẩn Lao động [労働基準法]...",
                "sources": [
                    {
                        "law_title": "労働基準法",
                        "article": "第三十二条",
                        "text": "使用者は、労働者に...",
                        "score": 0.85,
                        "highlight_path": {
                            "law": "労働基準法",
                            "chapter": "第四章",
                            "article": "第三十二条"
                        }
                    }
                ],
                "query": "残業代の計算方法",
                "processing_time_ms": 1250.5
            }
        }


# ============== Health Check ==============

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    version: str = "0.1.0"
    services: dict = Field(default_factory=dict)
