"""
Norman - Japanese Legal RAG System

FastAPI application for searching and querying Japanese legal documents.
Provides Vietnamese responses with Japanese legal term annotations.
"""

import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.api.routes import router as api_router

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    settings = get_settings()
    print(f"🚀 Starting {settings.app_name} v{settings.app_version}")
    print(f"📚 Qdrant collection: {settings.qdrant_collection_name}")
    print(f"🤖 LLM model: {settings.llm_model}")
    
    yield
    
    # Shutdown
    print("👋 Shutting down...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="""
        Japanese Legal RAG System - Tìm kiếm và tra cứu pháp luật Nhật Bản.
        
        Hệ thống sử dụng:
        - Vector search với Qdrant Cloud
        - LLM (GPT-4o-mini) để sinh câu trả lời
        - Output tiếng Việt với chú thích tiếng Nhật
        """,
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(api_router)
    
    @app.get("/")
    async def root():
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
        }
    
    return app


# Create app instance
app = create_app()
