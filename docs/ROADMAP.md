# Japanese Legal RAG System - Roadmap

Lộ trình phát triển hệ thống RAG cho văn bản pháp luật Nhật Bản.

**Last Updated**: 2026-01-10

---

## ✅ Phase 0: Data Collection & Processing (Complete)
- [x] Download XML từ e-Gov API (233 laws)
- [x] Parse XML → JSON với cấu trúc hierarchical
- [x] Smart chunking với hierarchical context preservation
- [x] Embedding chunks với OpenAI text-embedding-3-large (15,629 chunks → 192 MB)

---

## ✅ Phase 1: Vector Search with Qdrant (Complete)

### 1.1 Setup Infrastructure
- [x] Sử dụng **Qdrant Cloud Free Tier** thay vì Docker self-hosted
- [x] Cấu hình `.env` với QDRANT_URL và QDRANT_API_KEY

### 1.2 Embedding với OpenAI
- [x] Model: `text-embedding-3-large` (3072 dimensions)
- [x] Batch processing để tối ưu API calls
- [x] Caching embeddings trong `data/embeddings/`

### 1.3 Indexing Pipeline
- [x] Implement `app/db/qdrant.py` - Qdrant client functions
- [x] Implement `scripts/indexer.py` - Batch upload với retry logic
- [x] Upload 15,629 vectors lên Qdrant Cloud
- [x] Create payload indexes (`category`, `law_title`) for filtering

### 1.4 Search Implementation
- [x] Vector similarity search với `search()` function
- [x] Metadata filtering (by law_id, category, etc.)
- [ ] Hybrid search (vector + keyword) - *optional*

---

## ✅ Phase 2: Response Generation with API (Complete)

### 2.1 LLM Integration
- [x] Sử dụng retrieved chunks làm context
- [x] Generate answer với citations (OpenAI GPT-4o-mini)

### 2.2 Query Processing
- [x] Query translation Vietnamese → Japanese (`query_translator.py`)
- [x] Query analysis for category detection (`query_analyzer.py`)
- [x] Multi-query retrieval (original + translated queries)

### 2.3 Response Format with Highlighting
```json
{
  "answer": "Theo Điều 1...",
  "sources": [
    {
      "law_title": "労働基準法",
      "article": "第一条",
      "text": "...",
      "highlight_path": ["労働基準法", "第一章", "第一条"]
    }
  ]
}
```

### 2.4 API Endpoints
- [x] Implement `POST /api/search` - Vector search endpoint
- [x] Implement `POST /api/chat` - RAG chat với LLM
- [x] Health check endpoint

### 2.5 RAG Pipeline
- [x] Implement `app/llm/` - LLM providers (`OpenAIProvider`, `BaseLLM`, prompts)
- [x] Implement `app/pipelines/rag.py` - Full RAG pipeline
- [x] Score filtering (min_score threshold)
- [x] Deduplication of retrieved chunks

### 2.6 Frontend UI
- [x] React + Vite setup with HMR
- [x] Chat interface với message history
- [x] Source cards với expandable content
- [x] Loading states và error handling
- [x] Responsive dark theme design

---

## ✅ Phase 3: Reranking (Complete)

### 3.1 Reranker Implementation
| Model | Type | Status |
|-------|------|--------|
| **BAAI/bge-reranker-large** | Local (CPU) | ✅ Implemented |
| Cohere Rerank 3.5 | API | ❌ Skipped (cost) |

**Completed:**
- [x] Install torch CPU-only (~200MB) + FlagEmbedding
- [x] Implement `BGEReranker` class in `app/services/reranker.py`
- [x] Integrate into RAGPipeline via `deps.py`
- [x] Verify improvement: 60% queries improved, scores +10-20%

### 3.2 Two-Stage Retrieval
```
Query → Vector Search (top 10) → BGE Rerank (CPU) → Final Results (top 5)
```

### 3.3 Test Results
| Query | Top Changed? | Score Improvement |
|-------|--------------|-------------------|
| Thời gian nghỉ giữa ca | ✅ Yes | 0.50 → 0.66 (+32%) |
| Làm thêm giờ gấp đôi | No | 0.59 → 0.64 (+8%) |
| Sa thải thử việc | ✅ Yes | Reordered |

---

## 📋 Phase 3.5: LangGraph Agent (Next)

### Architecture
```
┌─────────┐    ┌──────────┐    ┌────────┐    ┌──────────┐
│Translate│ →  │ Retrieve │ →  │ Rerank │ →  │ Generate │
└─────────┘    └──────────┘    └────────┘    └──────────┘
                    ↑                              │
                    └──────── Self-Correction ─────┘
```

**Tasks:**
- [ ] Add `langchain`, `langgraph` dependencies
- [ ] Implement `LegalRAGAgent` with graph nodes
- [ ] Multi-step reasoning with retry loop

---

## 📋 Phase 4: Graph RAG with Neo4j (Future)

### 4.1 Graph Schema
```
(Law) -[:HAS_CHAPTER]-> (Chapter) -[:HAS_ARTICLE]-> (Article)
(Article) -[:REFERENCES]-> (Article)
(Article) -[:DEFINES]-> (LegalTerm)
(Law) -[:AMENDS]-> (Law)
```

### 4.2 Use Cases
- Tìm tất cả điều liên quan đến một điều cụ thể
- Trace lịch sử sửa đổi của một điều
- Tìm định nghĩa thuật ngữ pháp lý

---

## 📋 Phase 5: Production Deployment (Future)

### 5.1 API Server
- FastAPI backend
- Rate limiting
- Caching layer (Redis)

### 5.2 Enhancements
- Conversation memory (multi-turn chat)
- User feedback collection
- A/B testing for prompts

---

## Tech Stack Summary

| Component | Technology | Status |
|-----------|------------|--------|
| Embedding | OpenAI text-embedding-3-large | ✅ Done |
| Vector DB | Qdrant Cloud (Free Tier) | ✅ Done |
| LLM | OpenAI GPT-4o-mini | ✅ Done |
| Query Translation | OpenAI (Vietnamese → Japanese) | ✅ Done |
| Backend | FastAPI (Python 3.12) | ✅ Done |
| Frontend | React 18 + Vite | ✅ Done |
| Reranker | BGE-reranker-large (CPU) | ✅ Done |
| Agent Framework | LangGraph | 📋 Next |
| Graph DB | Neo4j | ⬜ Future |

---

## Timeline & Progress

| Phase | Duration | Status |
|-------|----------|--------|
| Phase 0 | 1 day | ✅ Complete |
| Phase 1 | 2 days | ✅ Complete |
| Phase 2 | 3-5 days | ✅ Complete |
| Phase 3 | 2-3 days | ✅ Complete |
| Phase 3.5 | 2-3 days | 📋 Next |
| Phase 4 | 5-7 days | ⬜ Pending |
