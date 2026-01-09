# Japanese Legal RAG System - Roadmap

Lộ trình phát triển hệ thống RAG cho văn bản pháp luật Nhật Bản.

---

## ✅ Phase 0: Data Collection & Processing (Complete)
- [x] Download XML từ e-Gov API
- [x] Parse XML → JSON với cấu trúc hierarchical
- [x] Chunking data để chuẩn bị cho embedding
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

### 1.4 Search Implementation
- [x] Vector similarity search với `search()` function
- [x] Metadata filtering (by law_id, category, etc.)
- [ ] Hybrid search (vector + keyword) - *optional*

---

## ✅ Phase 2: Response Generation with API (Complete)

### 2.1 LLM Integration
- [x] Sử dụng retrieved chunks làm context
- [x] Generate answer với citations (OpenAI GPT-4o-mini)

### 2.2 Response Format with Highlighting
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

### 2.3 API Endpoints
- [x] Implement `POST /api/search` - Vector search endpoint
- [x] Implement `POST /api/chat` - RAG chat với LLM

### 2.4 Tasks
- [x] Implement `app/llm/` - LLM providers (`OpenAIProvider`, `BaseLLM`, prompts)
- [x] Implement `app/pipelines/rag.py` - Full RAG pipeline với query translation
- [x] Implement `app/api/routes.py` - FastAPI routes
- [x] Setup FastAPI app với CORS, health check (`app/main.py`)

---

## 📋 Phase 3: Reranking Integration (Next)

### 3.1 Reranker Options
| Model | Type | Pros | Cons |
|-------|------|------|------|
| Cohere Rerank | API | Chất lượng cao | Trả phí |
| BAAI/bge-reranker-large | Local | Miễn phí, đa ngôn ngữ | Cần GPU |
| cross-encoder/ms-marco | Local | Nhanh | Chủ yếu English |

### 3.2 Two-Stage Retrieval
```
Query → Vector Search (top 50) → Rerank → Final Results (top 5)
```

### 3.3 Tasks
- [ ] Chọn reranker phù hợp (Cohere API hoặc BGE local)
- [ ] Implement `app/services/reranker.py`
- [ ] Tích hợp vào search pipeline

---

## 📋 Phase 4: Graph RAG with Neo4j

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

### 5.2 Frontend
- Search UI với highlight
- Law browser với navigation

---

## Tech Stack Summary

| Component | Technology | Status |
|-----------|------------|--------|
| Embedding | OpenAI text-embedding-3-large | ✅ Done |
| Vector DB | Qdrant Cloud (Free Tier) | ✅ Done |
| LLM | OpenAI GPT-4o-mini | ✅ Done |
| Backend | FastAPI (Python) | ✅ Done |
| Frontend | React + Vite | ✅ Done |
| Reranker | TBD (Cohere/BGE) | 📋 Next |
| Graph DB | Neo4j | ⬜ Future |

---

## Timeline & Progress

| Phase | Duration | Status |
|-------|----------|--------|
| Phase 0 | 1 day | ✅ Complete |
| Phase 1 | 2 days | ✅ Complete |
| Phase 2 | 3-5 days | ✅ Complete |
| Phase 3 | 2-3 days | 📋 Next |
| Phase 4 | 5-7 days | ⬜ Pending |
