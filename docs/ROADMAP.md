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

## 📋 Phase 2: Reranking Integration (Next)

### 2.1 Reranker Options
| Model | Type | Pros | Cons |
|-------|------|------|------|
| Cohere Rerank | API | Chất lượng cao | Trả phí |
| BAAI/bge-reranker-large | Local | Miễn phí, đa ngôn ngữ | Cần GPU |
| cross-encoder/ms-marco | Local | Nhanh | Chủ yếu English |

### 2.2 Two-Stage Retrieval
```
Query → Vector Search (top 50) → Rerank → Final Results (top 5)
```

### 2.3 Tasks
- [ ] Chọn reranker phù hợp (Cohere API hoặc BGE local)
- [ ] Implement `app/services/reranker.py`
- [ ] Tích hợp vào search pipeline

---

## 📋 Phase 3: Response Generation with Highlighting

### 3.1 LLM Integration
- [ ] Sử dụng retrieved chunks làm context
- [ ] Generate answer với citations

### 3.2 Highlight Response Format
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

### 3.3 API Design
- [ ] Implement `POST /api/search` endpoint
- [ ] Implement `POST /api/chat` endpoint (RAG với LLM)

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
| Reranker | TBD (Cohere/BGE) | 📋 Next |
| Graph DB | Neo4j | ⬜ Future |
| Backend | FastAPI (Python) | 🔧 In Progress |
| LLM | OpenAI GPT-4o-mini | 📋 Phase 3 |

---

## Timeline & Progress

| Phase | Duration | Status |
|-------|----------|--------|
| Phase 0 | 1 day | ✅ Complete |
| Phase 1 | 2 days | ✅ Complete |
| Phase 2 | 2-3 days | 📋 Next |
| Phase 3 | 3-5 days | ⬜ Pending |
| Phase 4 | 5-7 days | ⬜ Pending |
