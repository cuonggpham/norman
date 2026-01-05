# Pydantic Schemas for API Request/Response
# TODO: Implement for Phase 3

from pydantic import BaseModel, Field
from typing import Optional


class SearchQuery(BaseModel):
    """Search request body."""
    query: str = Field(..., description="Search query in Japanese")
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


class HighlightPath(BaseModel):
    """Path for highlighting in law structure."""
    law: str
    chapter: Optional[str] = None
    article: str
    paragraph: Optional[str] = None


class SearchSource(BaseModel):
    """Single search result source."""
    law_title: str
    article: str
    text: str
    highlight_path: HighlightPath
    score: float


class SearchResponse(BaseModel):
    """Search response with sources."""
    answer: str
    sources: list[SearchSource]
    query: str
    processing_time_ms: float
